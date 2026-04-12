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
    after GRASP_CONFIRM_STEPS (2) consecutive simulation steps in which:
      • BOTH the right finger/fingertip (links 26/27) AND the left finger/
        fingertip (links 29/30) are in contact with the object  (bilateral
        contact — ensures the gripper spans the object, not just grazes it),
      • gripper_angle < GRASP_CLOSE_THRESH, and
      • dist(EE, obj) < GRASP_DIST.
    If any condition breaks the counter resets.  Once locked, action[5] is
    ignored and the gripper stays closed for the rest of the episode.

    After locking, a lift-verification window of LIFT_VERIFY_WINDOW (15) steps
    checks whether the object has actually risen by at least LIFT_VERIFY_MIN
    (5 mm) above its position at lock time.  If it has not, the episode
    terminates immediately with a -20 penalty (false-grasp detection).  This
    catches cases where the gripper clips the bottle body without enclosing it.

    Wrist orientation is FIXED to horizontal (wrist pitch and roll are always
    zeroed out).  Only wrist yaw is controllable to allow the arm to face
    the correct direction.  This removes 2 action dims and simplifies learning.

    Episodes terminate early (terminated=True) the step r_success fires.
    The Gymnasium time-limit wrapper also ends them at max_episode_steps (300).

    Observation (16 dims):
        ee_pos_local       (3)  – end-effector in robot base frame
        obj_pos_local      (3)  – object in robot base frame
        diff               (3)  – obj - ee (local)
        joint_obs          (3)  – lift, arm_sum, wrist_yaw
        gripper_angle      (1)  – mean gripper joint (0 = closed, 0.6 = open)
        obj_above_table    (1)  – object z above table surface (signed, metres)
        in_contact         (1)  – 1.0 if robot touches object
        gripper_locked     (1)  – 1.0 once the one-shot close is committed

    Action (6 dims, all in [-1, 1]):
        [0:2]  base wheel velocities (delta)
        [2]    lift (delta)
        [3]    arm extension, shared across 4 prismatic joints (delta)
        [4]    wrist yaw (delta)  — pitch and roll are locked to horizontal
        [5]    gripper open/close target (+1=open, -1=closed) — ignored after lock

    Reward design — dense where possible, sparse for key events:

    PHASE A  gripper NOT locked yet:
        r_reach   = clip((prev_dist - dist) * 20, -2, 2)
                    Potential-based shaping: rewards progress toward the object,
                    penalises moving away.  Zero reward for hovering in place.
                    This suppresses the base-spinning exploit.
        r_contact = +2.0  one-time event bonus the first time gripper fingers
                    touch the object.  Firing only once removes the exploit of
                    rubbing the object every step for free reward.
        r_close   = clip((GRIPPER_OPEN - gripper_angle) / GRIPPER_OPEN, 0, 1) * 2.0
                    Active within GRASP_DIST*2 (0.20 m).  Rewards closing the
                    gripper when already near the object, teaching the policy
                    the "approach → close" sequence before Phase B triggers.

    PHASE B  gripper locked (requires 2 consecutive bilateral-contact steps):
        r_grasp        = +10.0  one-time sparse bonus when the lock first triggers
        r_lift         = clip((obj_above - prev_obj_above) * 25, -2, 2)
                         Differential (potential-based) lift reward: only progress
                         is rewarded.  Hovering at a fixed height gives 0; dropping
                         gives a negative signal.  Prevents "lock → hover" exploit.
        r_fail         = -min((locked_steps - LOCK_GRACE_STEPS) * LOCK_PENALTY_RATE, 2.0)
                         Escalating penalty after LOCK_GRACE_STEPS (30) steps with
                         no successful lift.  Forces the policy to keep lifting
                         rather than stalling after the grasp bonus is collected.
        r_verify_fail  = -20.0  fired once at step LIFT_VERIFY_WINDOW (15) if the
                         object has not risen LIFT_VERIFY_MIN (5 mm) above its
                         position at lock time.  Episode terminates immediately
                         (false-grasp detection for elongated objects).
        r_success      = +100.0  one-time sparse bonus when obj_lift >= LIFT_TARGET
                         Episode terminates immediately after this bonus.

    Always:
        r_step = -0.05  (time pressure)
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    # ── Constants ─────────────────────────────────────────────────────────────
    GRIPPER_OPEN        = 0.6    # pybullet joint angle: open
    GRIPPER_CLOSE       = 0.0    # pybullet joint angle: closed
    GRASP_DIST          = 0.12   # metres: EE must be within this to trigger lock
    GRASP_CLOSE_THRESH  = 0.35   # gripper angle below this = "closed enough" to lock
    GRASP_CONFIRM_STEPS = 2      # consecutive gripper-contact steps required to commit lock
    LIFT_TARGET_M       = 0.20   # metres above initial object z = success (20 cm)
    LOCK_GRACE_STEPS    = 30     # steps after lock before failure penalty starts
    LOCK_PENALTY_RATE   = 0.01   # penalty magnitude added per step beyond the grace window
    LIFT_VERIFY_WINDOW  = 15     # steps after lock within which the object must begin rising
    LIFT_VERIFY_MIN     = 0.005  # metres the object must have risen above lock position to pass

    # PyBullet link indices for the gripper fingers and fingertips only.
    # Contacts from any other robot link (arm, base, …) are ignored for lock/reward.
    #   26 = link_gripper_finger_right   27 = link_gripper_fingertip_right
    #   29 = link_gripper_finger_left    30 = link_gripper_fingertip_left
    _GRIPPER_LINK_INDICES = frozenset([26, 27, 29, 30])
    # Separate left / right sets used for bilateral-contact lock requirement.
    # For elongated objects (mustard bottle) both sides must be in contact to
    # confirm the gripper spans the object rather than swiping one side.
    _GRIPPER_RIGHT_LINKS  = frozenset([26, 27])
    _GRIPPER_LEFT_LINKS   = frozenset([29, 30])

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
            low=-10.0, high=10.0, shape=(16,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.table_height: float     = 0.85
        self.object                  = None
        self.robot                   = None
        self._gripper_locked: bool   = False
        self._grasp_confirm: int     = 0      # consecutive valid-contact steps toward lock
        self._prev_dist: float       = 1.0
        self._obj_init_z: float      = 0.0    # object z at episode start (lift reference)
        self._prev_obj_above: float  = 0.0    # obj_above at the previous step (for r_lift diff)
        self._success_rewarded: bool = False
        self._contact_rewarded: bool = False  # True after the one-time r_contact bonus fires
        self._locked_steps: int      = 0      # steps elapsed since gripper locked
        self._lock_obj_above: float  = 0.0    # obj_above at the moment of locking (for lift verify)
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
            [lift, arm_sum, wrist[0]],   # yaw only; pitch/roll locked to horizontal
            [gripper],
            [obj_above],
            [in_contact],
            [1.0 if self._gripper_locked else 0.0],
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
        self._contact_rewarded = False
        self._locked_steps     = 0
        self._lock_obj_above   = 0.0
        self._prev_dist        = 1.0    # overwritten after scene settles
        self._obj_init_z       = 0.0    # overwritten after scene settles
        self._prev_obj_above   = 0.0    # overwritten after scene settles
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
        self._prev_dist      = float(np.linalg.norm(np.array(ee_pos0) - np.array(obj_pos0)))
        self._obj_init_z     = float(obj_pos0[2])    # lift is measured relative to this
        self._prev_obj_above = 0.0                   # obj_above is 0 by definition at reset

        return self._get_obs(), self._get_info()

    def step(self, action):
        scale  = 0.025
        action = np.asarray(action, dtype=np.float32)

        if self._gripper_locked:
            grip_target = self.GRIPPER_CLOSE          # ignore action[5], stay closed
        else:
            grip_target = float(np.interp(
                action[5], [-1.0, 1.0], [self.GRIPPER_CLOSE, self.GRIPPER_OPEN]
            ))

        scaled = np.concatenate([
            action[0:2] * 0.5,
            [action[2] * scale],
            [action[3] / 4.0 * scale] * 4,
            [action[4] * scale, 0.0, 0.0],   # yaw only; pitch/roll fixed to horizontal
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
        obj_pos, _         = self.object.get_base_pos_orient()
        dist               = float(np.linalg.norm(np.array(ee_pos) - np.array(obj_pos)))
        obj_above          = float(obj_pos[2] - self._obj_init_z)

        ja            = self.robot.get_joint_angles(self.robot.controllable_joints)
        gripper_angle = float((ja[self._IDX_GRIP_L] + ja[self._IDX_GRIP_R]) / 2.0)

        # Filter contacts to gripper finger/fingertip links only.
        # Contacts from the arm, base, or other links are ignored to prevent
        # the policy from exploiting non-gripper body touches.
        all_contacts    = self.robot.get_contact_points(bodyB=self.object)
        # Any gripper contact (either side) — used for r_contact reward and obs.
        gripper_contact = bool(
            all_contacts and
            any(c['linkA'] in self._GRIPPER_LINK_INDICES for c in all_contacts)
        )
        # Bilateral contact: both left AND right finger sets must be in contact.
        # For elongated objects like the mustard bottle this confirms the gripper
        # spans the object body rather than merely grazing one side of the bottle.
        right_contact     = bool(all_contacts and any(c['linkA'] in self._GRIPPER_RIGHT_LINKS for c in all_contacts))
        left_contact      = bool(all_contacts and any(c['linkA'] in self._GRIPPER_LEFT_LINKS  for c in all_contacts))
        bilateral_contact = right_contact and left_contact

        # ── One-shot lock trigger ─────────────────────────────────────────────
        # Require GRASP_CONFIRM_STEPS consecutive steps where:
        #   • both gripper sides touch the object (bilateral contact),
        #   • gripper is closed enough, and
        #   • EE is close enough to the object.
        # The counter resets if any condition breaks.
        was_locked = self._gripper_locked
        if not self._gripper_locked:
            if bilateral_contact and gripper_angle < self.GRASP_CLOSE_THRESH and dist < self.GRASP_DIST:
                self._grasp_confirm += 1
                if self._grasp_confirm >= self.GRASP_CONFIRM_STEPS:
                    self._gripper_locked  = True
                    self._lock_obj_above  = obj_above   # snapshot for lift verification
            else:
                self._grasp_confirm = 0

        # ── Reward ────────────────────────────────────────────────────────────
        false_grasp = False   # set to True in Phase B if lift verification fails

        if not self._gripper_locked:
            # Phase A ── dense reach + contact + close-gripper shaping.

            # Potential-based distance shaping: rewards progress only.
            # Base spinning gives ~0 net reward because it oscillates closer/farther.
            r_reach = float(np.clip((self._prev_dist - dist) * 20.0, -2.0, 2.0))

            # One-time bonus the first time gripper fingers contact the object.
            # Using a single event reward rather than a per-step bonus avoids the
            # exploit of "rub against the object for free points every step".
            if gripper_contact and not self._contact_rewarded:
                r_contact = 2.0
                self._contact_rewarded = True
            else:
                r_contact = 0.0

            # Reward closing the gripper, but ONLY when bilateral contact already
            # holds — i.e. the gripper already spans the object on both sides.
            # Gating on distance alone caused premature closing: the policy would
            # rush within 0.24 m and immediately squeeze, making it harder to
            # subsequently insert the open fingers around the bottle body.
            # Gating on bilateral_contact ensures the "position first, close second"
            # ordering is explicitly rewarded.
            if bilateral_contact:
                r_close = float(np.clip(
                    (self.GRIPPER_OPEN - gripper_angle) / self.GRIPPER_OPEN, 0.0, 1.0
                )) * 2.0
            else:
                r_close = 0.0

            reward_phase = r_reach + r_contact + r_close

        else:
            # Phase B ── sparse grasp bonus + differential lift shaping + failure penalty.
            r_reach = 0.0
            self._locked_steps += 1

            # One-time sparse bonus at the moment the grasp is committed.
            r_grasp = 10.0 if not was_locked else 0.0

            # Potential-based differential lift: only rewards *upward progress*.
            # The policy gets 0 for hovering in place (no free lunch from maintaining
            # height) and a negative signal for dropping, so it is pushed to keep
            # lifting rather than stabilising at a partial height.
            # Scale 25 ≈ 0.05 reward per mm of lift per step; ±2.0 cap absorbs
            # sudden physics impulses without distorting the gradient.
            delta_lift = obj_above - self._prev_obj_above
            r_lift = float(np.clip(delta_lift * 25.0, -2.0, 2.0))

            # Escalating failure penalty after the grace window.
            # Gives the robot LOCK_GRACE_STEPS steps to begin lifting before adding
            # pressure.  Beyond that, the penalty grows linearly, capped at -2.0/step.
            # This prevents the "lock → float in place" exploit.
            steps_past_grace = max(0, self._locked_steps - self.LOCK_GRACE_STEPS)
            r_fail = -float(min(steps_past_grace * self.LOCK_PENALTY_RATE, 2.0))

            # Lift-following verification: at the end of LIFT_VERIFY_WINDOW steps
            # post-lock, check whether the object has actually risen.  If not, the
            # grasp is deemed false (gripper clipped the bottle without spanning it)
            # and the episode ends immediately with a penalty.  This produces a sharp
            # training signal that distinguishes a real grasp from a side-swipe that
            # happened to satisfy the geometric lock conditions.
            if self._locked_steps == self.LIFT_VERIFY_WINDOW:
                rise = obj_above - self._lock_obj_above
                if rise < self.LIFT_VERIFY_MIN:
                    r_verify_fail = -20.0
                    false_grasp   = True
                else:
                    r_verify_fail = 0.0
            else:
                r_verify_fail = 0.0

            # One-time sparse bonus when the lift target is first reached.
            if obj_above >= self.LIFT_TARGET_M and not self._success_rewarded:
                r_success = 100.0
                self._success_rewarded = True
            else:
                r_success = 0.0

            reward_phase = r_grasp + r_lift + r_fail + r_verify_fail + r_success

        r_step = -0.05
        reward = reward_phase + r_step

        # Update potentials for next step.
        self._prev_dist      = dist
        self._prev_obj_above = obj_above

        # Terminate when success is confirmed OR a false-grasp is detected.
        # False-grasp termination (terminated=True, not truncated) gives the
        # value network a clear -20 terminal signal rather than bootstrapping
        # from a hopeless stall.
        terminated = self._success_rewarded or false_grasp
        truncated  = False

        # ── Info ──────────────────────────────────────────────────────────────
        self._step_info = {
            'dist':              dist,
            'gripper_contact':   1.0 if gripper_contact else 0.0,
            'bilateral_contact': 1.0 if bilateral_contact else 0.0,
            'gripper_angle':     gripper_angle,
            'grasp_confirm':     float(self._grasp_confirm),
            'gripper_locked':    float(self._gripper_locked),
            'locked_steps':      float(self._locked_steps),
            'obj_lift_m':        obj_above,
            'false_grasp':       1.0 if false_grasp else 0.0,
            'success':           float(self._success_rewarded),
        }

        return self._get_obs(), reward, terminated, truncated, self._get_info()


gym.register(id='GraspEnv', entry_point=GraspEnv, max_episode_steps=300)
