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
    The gripper starts fully open every episode.  Once it has closed below
    GRASP_CLOSE_THRESH *and* the EE is within GRASP_DIST of the object, the
    gripper is permanently locked closed for the rest of the episode.
    action[7] is ignored while the lock is active.  This prevents oscillation
    and forces the policy to commit to a single grasp attempt.

    Episodes never terminate early (terminated always False).
    The Gymnasium time-limit wrapper ends them at max_episode_steps (300).

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

    Reward (two phases):

    PHASE A  gripper NOT locked yet:
        r_reach   = exp(-5 * dist)              – approach shaping
        r_gripper = +0.5 * open_ratio           – reward staying open when far
                  | +2.0 * progress * closed    – reward closing when near object
        r_contact = 0

    PHASE B  gripper locked:
        r_reach   = 0
        r_gripper = 0
        r_contact = +3.0 if contact else -2.0   – reward holding; penalise drop
        r_lift    = clip(obj_above / LIFT_TARGET, 0, 1) * 30  – proportional height
        r_success = +50 when object reaches LIFT_TARGET (20 cm)

    Always:
        r_step = -0.01
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    # ── Constants ─────────────────────────────────────────────────────────────
    GRIPPER_OPEN        = 0.6    # pybullet joint angle: open
    GRIPPER_CLOSE       = 0.0    # pybullet joint angle: closed
    GRASP_DIST          = 0.15   # metres: EE must be within this to trigger lock
    GRASP_CLOSE_THRESH  = 0.25   # gripper angle below this = "closed enough" to lock
    LIFT_TARGET_M       = 0.20   # metres above table = success (20 cm)

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
        self.table_height: float   = 0.85
        self.object                = None
        self.robot                 = None
        self._gripper_locked: bool = False
        self._step_info: dict      = {}

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

        contact    = self.robot.get_contact_points(bodyB=self.object)
        obj_above  = float(obj_pos[2] - self.table_height)
        in_contact = 1.0 if contact else 0.0

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
        self._gripper_locked = False
        self._step_info      = {}

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
        ee_pos,  _ = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos, _ = self.object.get_base_pos_orient()
        dist       = float(np.linalg.norm(np.array(ee_pos) - np.array(obj_pos)))
        contact    = self.robot.get_contact_points(bodyB=self.object)
        obj_above  = float(obj_pos[2] - self.table_height)

        ja            = self.robot.get_joint_angles(self.robot.controllable_joints)
        gripper_angle = float((ja[self._IDX_GRIP_L] + ja[self._IDX_GRIP_R]) / 2.0)
        open_ratio    = gripper_angle / self.GRIPPER_OPEN   # 1=open, 0=closed
        closed_ratio  = 1.0 - open_ratio

        # ── One-shot lock trigger ─────────────────────────────────────────────
        if not self._gripper_locked:
            if gripper_angle < self.GRASP_CLOSE_THRESH and dist < self.GRASP_DIST:
                self._gripper_locked = True

        # ── Reward ────────────────────────────────────────────────────────────
        if not self._gripper_locked:
            # Phase A: approach + shape gripper
            r_reach = float(np.exp(-5.0 * dist))
            if dist > self.GRASP_DIST:
                r_gripper = 0.5 * open_ratio            # stay open while far
            else:
                progress  = 1.0 - dist / self.GRASP_DIST
                r_gripper = 2.0 * progress * closed_ratio  # close as you get near
            r_contact = 0.0
            r_lift    = 0.0
            r_success = 0.0
        else:
            # Phase B: lifting
            r_reach   = 0.0
            r_gripper = 0.0
            r_contact = 3.0 if contact else -2.0            # penalise dropping
            r_lift    = float(np.clip(obj_above / self.LIFT_TARGET_M, 0.0, 1.0)) * 30.0
            r_success = 50.0 if obj_above >= self.LIFT_TARGET_M else 0.0

        r_step = -0.01
        reward = r_reach + r_gripper + r_contact + r_lift + r_success + r_step

        # Episodes always run to max_episode_steps; never terminate early.
        terminated = False
        truncated  = False

        # ── Info ──────────────────────────────────────────────────────────────
        self._step_info = {
            'dist':            dist,
            'in_contact':      1.0 if contact else 0.0,
            'gripper_angle':   gripper_angle,
            'gripper_locked':  float(self._gripper_locked),
            'obj_above_table': obj_above,
            'success':         float(obj_above >= self.LIFT_TARGET_M and self._gripper_locked),
        }

        return self._get_obs(), reward, terminated, truncated, self._get_info()


gym.register(id='GraspEnv', entry_point=GraspEnv, max_episode_steps=300)
