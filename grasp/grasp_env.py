import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mengine')))

import numpy as np
import pybullet as p
import gymnasium as gym
import mengine as m


class GraspEnv(gym.Env):
    """Phase 1 of hierarchical pick-and-place: reach the object, grasp it, and lift 20 cm.

    Gripper one-shot-close rule
    ---------------------------
    The gripper starts fully open every episode.  The lock is committed only
    after GRASP_CONFIRM_STEPS (3) consecutive simulation steps in which:
      • the gripper finger/fingertip links (26/27/29/30) are in contact with
        the object  (non-gripper body contact is ignored),
      • gripper_angle < GRASP_CLOSE_THRESH, and
      • dist(EE, obj) < GRASP_DIST.
    If any condition breaks the counter resets, preventing false locks from
    momentary touches.  Once locked, action[7] is ignored and the gripper
    stays closed for the rest of the episode.

    Episodes terminate early (terminated=True) the step r_success fires.
    The Gymnasium time-limit wrapper also ends them at max_episode_steps (300).

    Observation (18 dims):
        ee_pos_local       (3)  – end-effector in robot base frame
        obj_pos_local      (3)  – object in robot base frame
        diff               (3)  – obj - ee (local)
        joint_obs          (5)  – lift, arm_sum, wrist_yaw, wrist_pitch, wrist_roll
        gripper_angle      (1)  – mean gripper joint (0 = closed, 0.6 = open)
        obj_above_table    (1)  – object z above table surface (signed, metres)
        in_contact         (1)  – 1.0 if robot touches object
        gripper_locked     (1)  – 1.0 once the one-shot close is committed

    Action (8 dims, all in [-1, 1]):
        [0:2]  base wheel velocities (delta)
        [2]    lift (delta)
        [3]    arm extension, shared across 4 prismatic joints (delta)
        [4:7]  wrist yaw / pitch / roll (delta)
        [7]    gripper open/close target (+1=open, -1=closed) — ignored after lock

    Reward design — dense where possible, sparse for key events:

    PHASE A  gripper NOT locked yet:
        r_reach   = clip((prev_dist - dist) * 20, -2, 2)
                    Potential-based shaping: rewards progress toward the object,
                    penalises moving away.  Zero reward for hovering in place.
                    This suppresses the base-spinning exploit.
        r_orient  = (dot(ee_x_axis, to_obj_unit) + 1) / 2 * 1.0
                    Active within ALIGN_DIST (0.30 m).  EE local x-axis is the
                    confirmed arm approach direction (not z, which points up).
                    Maps [-1,1] → [0,1] so the policy gets gradient even when
                    pointing the wrong way.

    PHASE B  gripper locked (requires 3 consecutive gripper-contact steps):
        r_grasp   = +10.0  one-time sparse bonus when the lock first triggers
        r_lift    = clip(obj_lift / LIFT_TARGET, 0, 1) * 5.0
                    Dense proportional lift measured relative to _obj_init_z
                    (object COM at reset), not table_height.
        r_success = +100.0  one-time sparse bonus when obj_lift >= LIFT_TARGET
                    Episode terminates immediately after this bonus.

    Always:
        r_step = -0.05  (time pressure)
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    # ── Constants ─────────────────────────────────────────────────────────────
    GRIPPER_OPEN        = 0.6    # pybullet joint angle: open
    GRIPPER_CLOSE       = 0.0    # pybullet joint angle: closed
    GRASP_DIST          = 0.10   # metres: EE must be within this to trigger lock
    GRASP_CLOSE_THRESH  = 0.25   # gripper angle below this = "closed enough" to lock
    GRASP_CONFIRM_STEPS = 3      # consecutive gripper-contact steps required to commit lock
    LIFT_TARGET_M       = 0.20   # metres above initial object z = success (20 cm)
    ALIGN_DIST          = 0.30   # metres: orientation reward active within this distance

    # PyBullet link indices for the gripper fingers and fingertips only.
    # Contacts from any other robot link (arm, base, …) are ignored for lock/reward.
    #   26 = link_gripper_finger_right   27 = link_gripper_fingertip_right
    #   29 = link_gripper_finger_left    30 = link_gripper_fingertip_left
    _GRIPPER_LINK_INDICES = frozenset([26, 27, 29, 30])

    # controllable_joints = [0, 1, 4, 6, 7, 8, 9, 10, 12, 13, 26, 29]
    # indices:               0  1  2  3  4  5  6   7   8   9  10  11
    _IDX_LIFT      = 2
    _IDX_ARM_START = 3
    _IDX_ARM_END   = 7
    _IDX_WRIST     = slice(7, 10)
    _IDX_GRIP_L    = 10
    _IDX_GRIP_R    = 11

    _CAM_EYE    = [0.8, -1.5, 1.6]
    _CAM_TARGET = [-0.6, 0.0, 0.85]
    _CAM_W, _CAM_H = 480, 320

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode   = render_mode
        self.env           = m.Env(gravity=[0, 0, -9.81], render=render_mode == 'human')
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(18,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.table_height: float     = 0.85
        self.object                  = None
        self.robot                   = None
        self._gripper_locked: bool   = False
        self._grasp_confirm: int     = 0      # consecutive valid-contact steps toward lock
        self._prev_dist: float       = 1.0
        self._obj_init_z: float      = 0.0    # object z at episode start (lift reference)
        self._success_rewarded: bool = False
        self._step_info: dict        = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        ee_pos,  _ = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos, _ = self.object.get_base_pos_orient()

        ee_local,  _ = self.robot.global_to_local_coordinate_frame(ee_pos)
        obj_local, _ = self.robot.global_to_local_coordinate_frame(obj_pos)
        diff = obj_local - ee_local

        ja      = self.robot.get_joint_angles(self.robot.controllable_joints)
        lift    = ja[self._IDX_LIFT]
        arm_sum = ja[self._IDX_ARM_START:self._IDX_ARM_END].sum()
        wrist   = ja[self._IDX_WRIST]
        gripper = float((ja[self._IDX_GRIP_L] + ja[self._IDX_GRIP_R]) / 2.0)

        all_contacts = self.robot.get_contact_points(bodyB=self.object)
        in_contact   = 1.0 if (
            all_contacts and
            any(c['linkA'] in self._GRIPPER_LINK_INDICES for c in all_contacts)
        ) else 0.0
        obj_above    = float(obj_pos[2] - self._obj_init_z)

        return np.concatenate([
            ee_local,
            obj_local,
            diff,
            [lift, arm_sum, wrist[0], wrist[1], wrist[2]],
            [gripper],
            [obj_above],
            [in_contact],
            [1.0 if self._gripper_locked else 0.0],   # ← lock flag
        ]).astype(np.float32)

    def _get_info(self) -> dict:
        return dict(self._step_info)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def seed(self, seed):
        np.random.seed(seed)

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        view = p.computeViewMatrix(
            self._CAM_EYE, self._CAM_TARGET, [0, 0, 1],
            physicsClientId=self.env.id,
        )
        proj = p.computeProjectionMatrixFOV(
            60, self._CAM_W / self._CAM_H, 0.1, 10.0,
            physicsClientId=self.env.id,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            self._CAM_W, self._CAM_H, view, proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.env.id,
        )
        return np.array(rgba, dtype=np.uint8).reshape(self._CAM_H, self._CAM_W, 4)[:, :, :3]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)
        self.env.reset()
        self._gripper_locked   = False
        self._grasp_confirm    = 0
        self._success_rewarded = False
        self._prev_dist        = 1.0    # overwritten after scene settles
        self._obj_init_z       = 0.0    # overwritten after scene settles
        self._step_info        = {}

        _ = m.Ground()
        height_offset     = np.random.uniform(-0.2, 0.2)
        self.table_height = 0.85 + height_offset

        _ = m.URDF(
            filename=os.path.join(m.directory, 'table', 'table.urdf'),
            static=True,
            position=[-1.3, 0, height_offset],
            orientation=[0, 0, 0, 1],
        )
        self.object = m.Shape(
            m.Mesh(filename=os.path.join(m.directory, 'ycb', 'mustard.obj'), scale=[1, 1, 1]),
            static=False, mass=0.5,
            position=[
                -0.6 - np.random.uniform(0.0, 0.2),
                np.random.uniform(-0.2, 0.2),
                self.table_height,
            ],
            orientation=[0, 0, 0, 1],
            rgba=None, visual=True, collision=True,
        )

        pos_x = np.random.uniform(-0.1, 0.1)
        pos_y = np.random.uniform(-0.1, 0.1)
        theta = np.random.uniform(-np.pi / 4, np.pi / 4)
        self.robot = m.Robot.Stretch3(position=[pos_x, pos_y, 0], orientation=[0, 0, theta])
        self.robot.set_joint_angles(angles=[0.9], joints=[4])

        # Always start open
        self.robot.set_gripper_position(
            [self.GRIPPER_OPEN, self.GRIPPER_OPEN], set_instantly=True
        )

        m.step_simulation(steps=20, realtime=False)

        # Snapshot initial positions so step-1 gets no spurious distance bonus.
        ee_pos0, _  = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos0, _ = self.object.get_base_pos_orient()
        self._prev_dist  = float(np.linalg.norm(np.array(ee_pos0) - np.array(obj_pos0)))
        self._obj_init_z = float(obj_pos0[2])    # lift is measured relative to this

        return self._get_obs(), self._get_info()

    def step(self, action):
        scale  = 0.025
        action = np.asarray(action, dtype=np.float32)

        if self._gripper_locked:
            grip_target = self.GRIPPER_CLOSE          # ignore action[7], stay closed
        else:
            grip_target = float(np.interp(
                action[7], [-1.0, 1.0], [self.GRIPPER_CLOSE, self.GRIPPER_OPEN]
            ))

        scaled = np.concatenate([
            action[0:2] * 0.5,
            [action[2] * scale],
            [action[3] / 4.0 * scale] * 4,
            action[4:7] * scale,
            [grip_target, grip_target],
        ])

        current        = self.robot.get_joint_angles(self.robot.controllable_joints)
        target         = current + scaled
        target[self._IDX_GRIP_L] = grip_target
        target[self._IDX_GRIP_R] = grip_target

        self.robot.control(target)
        m.step_simulation(steps=10, realtime=self.env.render)

        # ── State ─────────────────────────────────────────────────────────────
        ee_pos,  ee_orient = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos, _         = self.object.get_base_pos_orient()
        dist               = float(np.linalg.norm(np.array(ee_pos) - np.array(obj_pos)))
        obj_above          = float(obj_pos[2] - self._obj_init_z)

        ja            = self.robot.get_joint_angles(self.robot.controllable_joints)
        gripper_angle = float((ja[self._IDX_GRIP_L] + ja[self._IDX_GRIP_R]) / 2.0)

        # Filter contacts to gripper finger/fingertip links only.
        # Contacts from the arm, base, or other links are ignored to prevent
        # the policy from exploiting non-gripper body touches.
        all_contacts     = self.robot.get_contact_points(bodyB=self.object)
        gripper_contact  = bool(
            all_contacts and
            any(c['linkA'] in self._GRIPPER_LINK_INDICES for c in all_contacts)
        )

        # ── One-shot lock trigger ─────────────────────────────────────────────
        # Require GRASP_CONFIRM_STEPS consecutive steps with gripper contact,
        # gripper closed enough, and EE close enough.  The counter resets if
        # any condition breaks, preventing momentary-touch false locks.
        was_locked = self._gripper_locked
        if not self._gripper_locked:
            if gripper_contact and gripper_angle < self.GRASP_CLOSE_THRESH and dist < self.GRASP_DIST:
                self._grasp_confirm += 1
                if self._grasp_confirm >= self.GRASP_CONFIRM_STEPS:
                    self._gripper_locked = True
            else:
                self._grasp_confirm = 0

        # ── Reward ────────────────────────────────────────────────────────────
        if not self._gripper_locked:
            # Phase A ── dense reach + orientation shaping, no gripper rewards.

            # Potential-based distance shaping: rewards progress only.
            # Base spinning gives ~0 net reward because it oscillates closer/farther.
            r_reach = float(np.clip((self._prev_dist - dist) * 20.0, -2.0, 2.0))

            # Orientation alignment: EE local x-axis is the gripper's approach
            # direction (arm extension axis, confirmed from URDF).  Reward how
            # well it points toward the object.  Using (dot+1)/2 keeps the full
            # [-1,1] gradient instead of clipping negatives to zero, so the
            # policy gets a learning signal even when pointing the wrong way.
            # Active within ALIGN_DIST to avoid premature shaping far away.
            if dist < self.ALIGN_DIST:
                to_obj = np.array(obj_pos) - np.array(ee_pos)
                to_obj_norm = to_obj / (np.linalg.norm(to_obj) + 1e-6)
                ee_rot = np.array(
                    p.getMatrixFromQuaternion(ee_orient, physicsClientId=self.env.id)
                ).reshape(3, 3)
                ee_approach = ee_rot[:, 0]   # EE local x-axis = arm approach direction
                dot = float(np.dot(ee_approach, to_obj_norm))
                r_orient = ((dot + 1.0) / 2.0) ** 2  # maps [-1,1] → [0,1]
            else:
                r_orient = 0.0

            reward_phase = r_reach + r_orient

        else:
            # Phase B ── sparse grasp bonus + dense lift shaping + sparse success.
            r_reach  = 0.0
            r_orient = 0.0

            # One-time sparse bonus at the moment the grasp is committed.
            r_grasp = 10.0 if not was_locked else 0.0

            # Dense proportional lift: reward every step for raising the object.
            r_lift = float(np.clip(obj_above / self.LIFT_TARGET_M, 0.0, 1.0)) * 5.0

            # One-time sparse bonus when the lift target is first reached.
            if obj_above >= self.LIFT_TARGET_M and not self._success_rewarded:
                r_success = 100.0
                self._success_rewarded = True
            else:
                r_success = 0.0

            reward_phase = r_grasp + r_lift + r_success

        r_step = -0.05
        reward = reward_phase + r_step

        # Update potential for next step.
        self._prev_dist = dist

        # Terminate when success is confirmed; this prevents wasted steps after
        # the task is done and improves credit assignment during training.
        terminated = self._success_rewarded
        truncated  = False

        # ── Info ──────────────────────────────────────────────────────────────
        self._step_info = {
            'dist':              dist,
            'gripper_contact':   1.0 if gripper_contact else 0.0,
            'gripper_angle':     gripper_angle,
            'grasp_confirm':     float(self._grasp_confirm),
            'gripper_locked':    float(self._gripper_locked),
            'obj_lift_m':        obj_above,
            'success':           float(self._success_rewarded),
        }

        return self._get_obs(), reward, terminated, truncated, self._get_info()


gym.register(id='GraspEnv', entry_point=GraspEnv, max_episode_steps=300)
