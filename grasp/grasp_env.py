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

    Observation (33 dims):
        ee_pos_local       (3)  – end-effector in robot base frame
        obj_pos_local      (3)  – object in robot base frame
        diff               (3)  – obj - ee (local)
        pregrasp_err       (3)  – xy / z / yaw error to pregrasp pose
        obj_in_ee          (3)  – object position in EE frame (centering & squeeze signal)
        joint_obs          (3)  – lift, arm_sum, wrist_yaw
        gripper_angle      (1)  – mean gripper joint (0 = closed, 0.6 = open)
        right_contact      (1)  – right finger in contact with object
        left_contact       (1)  – left finger in contact with object
        bilateral_contact  (1)  – both sides in contact simultaneously
        uprightness        (1)  – cos(tilt), 1.0 means upright
        obj_xy_disp        (1)  – table-plane displacement from reset pose
        obj_above_table    (1)  – object z above initial z
        gripper_locked     (1)  – 1.0 once the one-shot close is committed
        grasp_confirm_frac (1)  – lock progress 0→1 (policy sees how close to commit)
        prev_action        (6)  – previous policy action (for smoothness context)

    Action (6 dims, all in [-1, 1]):
        [0:2]  base wheel velocities (delta)
        [2]    lift (delta)
        [3]    arm extension, shared across 4 prismatic joints (delta)
        [4]    wrist yaw (delta)  — pitch and roll are locked to horizontal
        [5]    gripper open/close target (+1=open, -1=closed) — ignored after lock

    Reward design — geometry-aware and stability-aware:

    PHASE A  gripper NOT locked yet:
        r_reach   = clip((prev_pregrasp_metric - pregrasp_metric) * 20, -2, 2)
                    Potential-based shaping to a pregrasp pose, not object center.
        r_contact = +2.0  one-time event bonus the first time gripper fingers
                    touch the object.
        r_close   = clip((GRIPPER_OPEN - gripper_angle) / GRIPPER_OPEN, 0, 1) * 2.0
                    Active only under bilateral contact.

    PHASE B  gripper locked (B1: settle → B2: hold → B3: lift):
        r_grasp        = R_GRASP_BONUS / SETTLE_STEPS per step for first SETTLE_STEPS;
                         same total credit as a one-time bonus but delivered as a
                         smooth ramp — eliminates the single-step gradient spike
        r_hold_quality = continuous score in [-1, +1] via geometric mean of four
                         normalised quality components (uprightness, lateral centering
                         in EE frame, gripper enclosure angle, pregrasp centering);
                         smooth everywhere — no gradient cliffs at threshold edges
        r_lift         = 0 for SETTLE_STEPS, then 0.3× until hold_ok for
                         HOLD_STABLE_STEPS, then full clip(Δobj_z*25, -2, 2)
        r_height_hold  = HEIGHT_HOLD_BONUS * (obj_above / LIFT_TARGET_M) per step
                         when hold is stable and object is above HEIGHT_HOLD_THRESH;
                         rewards *being* at height, complementing the delta-based lift
        r_pose_stable  = pose-change penalty, active only for POSE_STABLE_WINDOW steps
                         then fades to 0 so normal lifting is never penalised
        r_fail         = -min((locked_steps - LOCK_GRACE_STEPS) * LOCK_PENALTY_RATE, 2.0)
        r_verify_fail  = -20.0  fired once at LIFT_VERIFY_WINDOW if rise < LIFT_VERIFY_MIN
        r_success      = +100.0  one-time bonus when obj_lift >= LIFT_TARGET

    Always:
        r_step          = -0.05  (time pressure)
        r_tilt_pen      – penalise object tilt beyond TILT_FREE_RAD
        r_push_pen      – penalise xy displacement; decays to 0 as object is lifted
                          past LIFT_CLEAR_H (so natural lift motion isn't penalised)
        r_action_mag    – penalise large action magnitude
        r_action_delta  – penalise action jump (high-frequency jitter)
        r_action_delta_z – extra z-dimension delta penalty to suppress up/down hunting

    Action scaling:
        Base speed is multiplied by a proximity factor that decreases linearly from
        1.0 at SLOWDOWN_FAR to SLOWDOWN_MIN at SLOWDOWN_NEAR, preventing the policy
        from crashing into the object at full approach speed.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    # ── Constants ─────────────────────────────────────────────────────────────
    GRIPPER_OPEN        = 0.6    # pybullet joint angle: open
    GRIPPER_CLOSE       = 0.0    # pybullet joint angle: closed
    GRASP_DIST          = 0.12   # metres: EE must be within this to trigger lock
    GRASP_CLOSE_THRESH  = 0.35   # gripper angle below this = "closed enough" to lock
    GRASP_CONFIRM_STEPS = 4      # consecutive stable-contact steps required to commit lock
    CLOSE_CONFIRM_STEPS = 3      # consecutive "commanded close" steps before lock can commit
    LOCK_LATERAL_THRESH = 0.05   # metres: object must be centered between fingers
    LOCK_XY_PUSH_THRESH = 0.08   # metres: reject lock if object was pushed too far on table
    UPRIGHT_LOCK_MIN    = 0.88   # cos(tilt) lower bound for lock/hold gating
    GRIP_ENCLOSE_MIN    = 0.08   # gripper angle lower bound: object must be inside, not crushed flat
    OBJ_EE_LAT_THRESH   = 0.07   # metres: obj_in_ee y-component must be small (centered in jaw)
    LIFT_TARGET_M       = 0.20   # metres above initial object z = success (20 cm)
    LIFT_CLEAR_H        = 0.03   # metres above table at which push-penalty starts to decay
    LOCK_GRACE_STEPS    = 30     # steps after lock before failure penalty starts
    LOCK_PENALTY_RATE   = 0.01   # penalty magnitude added per step beyond the grace window
    LIFT_VERIFY_WINDOW  = 20     # steps after lock within which the object must begin rising
    LIFT_VERIFY_MIN     = 0.005  # metres the object must have risen above lock position to pass
    SETTLE_STEPS        = 6      # post-lock quiet window: only hold-quality signal, no lift pressure
    HOLD_STABLE_STEPS   = 4      # consecutive hold-ok steps before full r_lift weight resumes
    POSE_STABLE_WINDOW  = 12     # steps post-lock during which r_pose_stable is active

    PREGRASP_XY_OFFSET  = 0.08   # metres behind object along robot->object direction
    PREGRASP_Z_OFFSET   = 0.00   # metres relative to object COM for pregrasp target
    PREGRASP_YAW_WEIGHT = 0.5    # relative weight for yaw alignment in pregrasp shaping

    TILT_FREE_RAD         = np.deg2rad(10.0)
    TILT_PENALTY_SCALE    = 0.6
    XY_PUSH_PENALTY_SCALE = 1.2
    ACTION_MAG_PENALTY    = 0.01
    ACTION_DELTA_PENALTY  = 0.08
    ACTION_DELTA_Z_PENALTY = 0.12   # extra penalty for z-dimension jitter (hunting up/down)

    # Speed limiting is based on pregrasp metric (same quantity used in r_reach)
    # so the slowdown signal is always consistent with what the reward optimises.
    # When locked the slowdown is bypassed — the robot may need to manoeuvre freely.
    SLOWDOWN_REACH_NEAR = 0.20   # pregrasp metric below which base speed → SLOWDOWN_MIN
    SLOWDOWN_REACH_FAR  = 0.60   # pregrasp metric above which base speed is unrestricted
    SLOWDOWN_MIN        = 0.15   # minimum speed factor

    HEIGHT_HOLD_BONUS  = 0.30   # per-step bonus for stably holding at height (non-potential)
    HEIGHT_HOLD_THRESH = 0.05   # object must be at least this far above table to earn bonus

    ACTION_MAX_DELTA    = np.array([0.25, 0.25, 0.18, 0.18, 0.22, 0.6], dtype=np.float32)
    R_GRASP_BONUS       = 5.0    # one-time lock bonus (was 10, reduced to limit "chase lock" exploit)

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
            low=-10.0, high=10.0, shape=(33,), dtype=np.float32
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
        self._obj_init_xy            = np.zeros(2, dtype=np.float32)
        self._prev_obj_above: float  = 0.0    # obj_above at the previous step (for r_lift diff)
        self._prev_reach_metric: float = 0.0  # previous pregrasp geometry metric
        self._success_rewarded: bool = False
        self._contact_rewarded: bool = False  # True after the one-time r_contact bonus fires
        self._locked_steps: int      = 0      # steps elapsed since gripper locked
        self._lock_obj_above: float  = 0.0    # obj_above at the moment of locking (for lift verify)
        self._close_confirm: int     = 0      # consecutive "close command" steps
        self._hold_stable_steps: int = 0      # consecutive post-lock stable-hold steps
        self._prev_action            = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._prev_rel_obj           = np.zeros(3, dtype=np.float32)
        self._step_info: dict        = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _angle_wrap(x: float) -> float:
        return float(np.arctan2(np.sin(x), np.cos(x)))

    @staticmethod
    def _object_uprightness(obj_quat) -> float:
        mat = np.array(p.getMatrixFromQuaternion(obj_quat), dtype=np.float32).reshape(3, 3)
        # cosine between object local +z and world +z
        return float(np.clip(mat[2, 2], -1.0, 1.0))

    def _pregrasp_errors(self, ee_pos, obj_pos, wrist_yaw: float):
        obj_xy = np.array(obj_pos[:2], dtype=np.float32)
        robot_xy = np.array(self.robot.get_base_pos_orient()[0][:2], dtype=np.float32)
        approach = obj_xy - robot_xy
        norm = float(np.linalg.norm(approach))
        if norm < 1e-6:
            approach = np.array([1.0, 0.0], dtype=np.float32)
        else:
            approach = approach / norm

        pre_xy = obj_xy - approach * self.PREGRASP_XY_OFFSET
        pre_z = float(obj_pos[2] + self.PREGRASP_Z_OFFSET)
        ee_xy = np.array(ee_pos[:2], dtype=np.float32)
        xy_err_vec = ee_xy - pre_xy
        xy_err = float(np.linalg.norm(xy_err_vec))
        z_err = float(ee_pos[2] - pre_z)
        desired_yaw = float(np.arctan2(approach[1], approach[0]))
        yaw_err = self._angle_wrap(wrist_yaw - desired_yaw)
        return xy_err, z_err, yaw_err

    def _get_obs(self) -> np.ndarray:
        ee_pos, ee_quat = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos, obj_quat = self.object.get_base_pos_orient()

        ee_local,  _ = self.robot.global_to_local_coordinate_frame(ee_pos)
        obj_local, _ = self.robot.global_to_local_coordinate_frame(obj_pos)
        diff = obj_local - ee_local

        ja      = self.robot.get_joint_angles(self.robot.controllable_joints)
        lift    = ja[self._IDX_LIFT]
        arm_sum = ja[self._IDX_ARM_START:self._IDX_ARM_END].sum()
        wrist   = ja[self._IDX_WRIST]
        gripper = float((ja[self._IDX_GRIP_L] + ja[self._IDX_GRIP_R]) / 2.0)
        pre_xy_err, pre_z_err, pre_yaw_err = self._pregrasp_errors(
            ee_pos=ee_pos, obj_pos=obj_pos, wrist_yaw=float(wrist[0])
        )

        inv_pos, inv_quat = p.invertTransform(ee_pos, ee_quat)
        obj_in_ee, _ = p.multiplyTransforms(inv_pos, inv_quat, obj_pos, [0, 0, 0, 1])
        obj_in_ee = np.array(obj_in_ee, dtype=np.float32)

        all_contacts = self.robot.get_contact_points(bodyB=self.object)
        right_contact = bool(all_contacts and any(c['linkA'] in self._GRIPPER_RIGHT_LINKS for c in all_contacts))
        left_contact = bool(all_contacts and any(c['linkA'] in self._GRIPPER_LEFT_LINKS for c in all_contacts))
        bilateral_contact = right_contact and left_contact
        obj_above    = float(obj_pos[2] - self._obj_init_z)
        obj_xy_disp = float(np.linalg.norm(np.array(obj_pos[:2], dtype=np.float32) - self._obj_init_xy))
        uprightness = self._object_uprightness(obj_quat)

        grasp_confirm_frac = float(self._grasp_confirm) / float(self.GRASP_CONFIRM_STEPS)

        return np.concatenate([
            ee_local,                                       # 3
            obj_local,                                      # 3
            diff,                                           # 3
            [pre_xy_err, pre_z_err, pre_yaw_err],           # 3
            obj_in_ee,                                      # 3
            [lift, arm_sum, wrist[0]],                      # 3  yaw only
            [gripper],                                      # 1
            [1.0 if right_contact else 0.0],                # 1
            [1.0 if left_contact else 0.0],                 # 1
            [1.0 if bilateral_contact else 0.0],            # 1
            [uprightness],                                  # 1
            [obj_xy_disp],                                  # 1
            [obj_above],                                    # 1
            [1.0 if self._gripper_locked else 0.0],         # 1
            [grasp_confirm_frac],                           # 1  progress toward lock
            self._prev_action,                              # 6
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
        self._close_confirm    = 0
        self._hold_stable_steps = 0
        self._prev_dist        = 1.0    # overwritten after scene settles
        self._obj_init_z       = 0.0    # overwritten after scene settles
        self._obj_init_xy      = np.zeros(2, dtype=np.float32)
        self._prev_obj_above   = 0.0    # overwritten after scene settles
        self._prev_reach_metric = 0.0
        self._prev_action      = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._prev_rel_obj     = np.zeros(3, dtype=np.float32)
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
        self._obj_init_xy    = np.array(obj_pos0[:2], dtype=np.float32)
        self._prev_obj_above = 0.0                   # obj_above is 0 by definition at reset
        self._prev_rel_obj   = np.array(obj_pos0, dtype=np.float32) - np.array(ee_pos0, dtype=np.float32)
        ja0 = self.robot.get_joint_angles(self.robot.controllable_joints)
        wrist_yaw0 = float(ja0[self._IDX_WRIST][0])
        xy0, z0, yaw0 = self._pregrasp_errors(ee_pos=ee_pos0, obj_pos=obj_pos0, wrist_yaw=wrist_yaw0)
        self._prev_reach_metric = xy0 + 0.5 * abs(z0) + self.PREGRASP_YAW_WEIGHT * abs(yaw0)

        return self._get_obs(), self._get_info()

    def step(self, action):
        scale  = 0.025
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_delta = action - self._prev_action
        clipped_delta = np.clip(action_delta, -self.ACTION_MAX_DELTA, self.ACTION_MAX_DELTA)
        action_cmd = np.clip(self._prev_action + clipped_delta, self.action_space.low, self.action_space.high)

        if self._gripper_locked:
            grip_target = self.GRIPPER_CLOSE          # ignore action[5], stay closed
        else:
            grip_target = float(np.interp(
                action_cmd[5], [-1.0, 1.0], [self.GRIPPER_CLOSE, self.GRIPPER_OPEN]
            ))

        # Proximity-based base speed limiting, keyed on the SAME pregrasp metric
        # used in r_reach so that the speed signal and the reward signal are
        # geometrically consistent.  Uses the previous-step snapshot so no extra
        # physics query is needed before the simulation advances.
        # When locked the limit is bypassed: the robot must be free to manoeuvre
        # (e.g. retract arm) while carrying the object.
        if self._gripper_locked:
            speed_factor = 1.0
        else:
            speed_factor = float(np.clip(
                (self._prev_reach_metric - self.SLOWDOWN_REACH_NEAR)
                / (self.SLOWDOWN_REACH_FAR - self.SLOWDOWN_REACH_NEAR),
                self.SLOWDOWN_MIN, 1.0,
            ))

        scaled = np.concatenate([
            action_cmd[0:2] * 0.5 * speed_factor,
            [action_cmd[2] * scale],
            [action_cmd[3] / 4.0 * scale] * 4,
            [action_cmd[4] * scale, 0.0, 0.0],   # yaw only; pitch/roll fixed to horizontal
            [grip_target, grip_target],
        ])

        current        = self.robot.get_joint_angles(self.robot.controllable_joints)
        target         = current + scaled
        target[self._IDX_GRIP_L] = grip_target
        target[self._IDX_GRIP_R] = grip_target

        self.robot.control(target)
        m.step_simulation(steps=10, realtime=self.env.render)

        # ── State ─────────────────────────────────────────────────────────────
        ee_pos, ee_quat = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos, obj_quat = self.object.get_base_pos_orient()
        dist               = float(np.linalg.norm(np.array(ee_pos) - np.array(obj_pos)))
        obj_above          = float(obj_pos[2] - self._obj_init_z)
        obj_xy_disp        = float(np.linalg.norm(np.array(obj_pos[:2], dtype=np.float32) - self._obj_init_xy))
        uprightness        = self._object_uprightness(obj_quat)
        tilt_rad           = float(np.arccos(np.clip(uprightness, -1.0, 1.0)))

        ja            = self.robot.get_joint_angles(self.robot.controllable_joints)
        gripper_angle = float((ja[self._IDX_GRIP_L] + ja[self._IDX_GRIP_R]) / 2.0)
        wrist_yaw = float(ja[self._IDX_WRIST][0])
        pre_xy_err, pre_z_err, pre_yaw_err = self._pregrasp_errors(
            ee_pos=ee_pos, obj_pos=obj_pos, wrist_yaw=wrist_yaw
        )
        reach_now = (
            pre_xy_err
            + 0.5 * abs(pre_z_err)
            + self.PREGRASP_YAW_WEIGHT * abs(pre_yaw_err)
        )

        # Object position in EE frame — needed for hold-quality squeeze detection.
        inv_pos, inv_quat = p.invertTransform(ee_pos, ee_quat)
        obj_in_ee, _ = p.multiplyTransforms(inv_pos, inv_quat, obj_pos, [0, 0, 0, 1])
        obj_in_ee = np.array(obj_in_ee, dtype=np.float32)

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
            close_cmd = action_cmd[5] < -0.2
            if close_cmd:
                self._close_confirm += 1
            else:
                self._close_confirm = 0

            lock_geometry_ok = (
                dist < self.GRASP_DIST
                and pre_xy_err < self.LOCK_LATERAL_THRESH
                and uprightness > self.UPRIGHT_LOCK_MIN
                and obj_xy_disp < self.LOCK_XY_PUSH_THRESH
            )
            if (
                bilateral_contact
                and gripper_angle < self.GRASP_CLOSE_THRESH
                and self._close_confirm >= self.CLOSE_CONFIRM_STEPS
                and lock_geometry_ok
            ):
                self._grasp_confirm += 1
                if self._grasp_confirm >= self.GRASP_CONFIRM_STEPS:
                    self._gripper_locked  = True
                    self._lock_obj_above  = obj_above   # snapshot for lift verification
            else:
                self._grasp_confirm = 0

        # ── Reward ────────────────────────────────────────────────────────────
        false_grasp   = False   # set to True in Phase B if lift verification fails
        quality_score = 0.0     # continuous hold-quality (Phase B only; 0 in Phase A)

        if not self._gripper_locked:
            # Phase A ── dense reach + contact + close-gripper shaping.

            # Pregrasp shaping: reward geometric approach (lateral/height/yaw),
            # not just center distance.
            r_reach = float(np.clip((self._prev_reach_metric - reach_now) * 20.0, -2.0, 2.0))

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
            # Phase B ── settle → hold → lift.
            r_reach = 0.0
            self._locked_steps += 1

            # Lock bonus spread evenly over SETTLE_STEPS rather than fired as a
            # single step spike.  Total credit is identical (R_GRASP_BONUS) but the
            # value surface is much smoother: the critic can fit a gradual ramp
            # instead of a single-step discontinuity.
            r_grasp = (
                self.R_GRASP_BONUS / float(self.SETTLE_STEPS)
                if self._locked_steps <= self.SETTLE_STEPS else 0.0
            )

            # Continuous hold-quality score via geometric mean of four normalised
            # components.  Each is in [0, 1]; the product is 1 only when all
            # dimensions are simultaneously perfect, and drops smoothly toward 0
            # as any single dimension degrades.  This replaces the hard three-tier
            # (+1 / -0.5 / -1) logic that created gradient cliffs at threshold edges.
            obj_ee_lateral = float(abs(obj_in_ee[1]))
            if bilateral_contact:
                upright_q = float(np.clip(
                    (uprightness - self.UPRIGHT_LOCK_MIN) / (1.0 - self.UPRIGHT_LOCK_MIN),
                    0.0, 1.0,
                ))
                lateral_q = float(np.clip(
                    1.0 - obj_ee_lateral / self.OBJ_EE_LAT_THRESH, 0.0, 1.0,
                ))
                # Lower gripper_angle = more closed = better enclosure, up to GRIP_ENCLOSE_MIN.
                # Score is 1.0 when angle == GRIP_ENCLOSE_MIN (tightest valid hold),
                # 0.0 when angle == GRASP_CLOSE_THRESH (barely meets lock threshold).
                enclose_q = float(np.clip(
                    (self.GRASP_CLOSE_THRESH - gripper_angle)
                    / max(self.GRASP_CLOSE_THRESH - self.GRIP_ENCLOSE_MIN, 1e-6),
                    0.0, 1.0,
                ))
                center_q = float(np.clip(
                    1.0 - pre_xy_err / (self.LOCK_LATERAL_THRESH * 1.5), 0.0, 1.0,
                ))
                # Geometric mean: all four components must be high for a good score.
                quality_score = float(
                    np.sqrt(np.sqrt(upright_q * lateral_q * enclose_q * center_q))
                )
                r_hold_quality = 2.0 * quality_score - 1.0   # maps [0,1] → [-1,+1]
            else:
                quality_score  = 0.0
                r_hold_quality = -1.0

            # hold_ok gates r_lift weighting (structural, not reward).
            hold_ok = quality_score > 0.3
            if hold_ok:
                self._hold_stable_steps += 1
            else:
                self._hold_stable_steps = 0

            # r_lift: three-phase.
            #   SETTLE_STEPS:        0     — let grasp settle, no upward pressure.
            #   hold not stable:     0.3×  — weak gradient only.
            #   hold stable:         full  — differential reward.
            delta_lift = obj_above - self._prev_obj_above
            r_lift_raw = float(np.clip(delta_lift * 25.0, -2.0, 2.0))
            if self._locked_steps <= self.SETTLE_STEPS:
                r_lift = 0.0
            elif self._hold_stable_steps >= self.HOLD_STABLE_STEPS:
                r_lift = r_lift_raw
            else:
                r_lift = 0.3 * r_lift_raw

            # Per-step height-hold bonus: rewards *being* at height, complementing
            # the delta-based r_lift.  This gives a stable positive gradient even
            # when hovering at a fixed height, so the policy is not only driven to
            # produce upward jitter.  Scales linearly with obj_above / LIFT_TARGET_M
            # so higher is always strictly better.
            if self._hold_stable_steps >= self.HOLD_STABLE_STEPS and obj_above >= self.HEIGHT_HOLD_THRESH:
                r_height_hold = self.HEIGHT_HOLD_BONUS * float(
                    np.clip(obj_above / self.LIFT_TARGET_M, 0.0, 1.0)
                )
            else:
                r_height_hold = 0.0

            # r_pose_stable fades to zero over POSE_STABLE_WINDOW so normal lifting
            # motion is never penalised.
            rel_obj = np.array(obj_pos, dtype=np.float32) - np.array(ee_pos, dtype=np.float32)
            rel_delta = float(np.linalg.norm(rel_obj - self._prev_rel_obj))
            if self._locked_steps <= self.POSE_STABLE_WINDOW:
                pose_weight = 1.0 - (self._locked_steps - 1) / float(self.POSE_STABLE_WINDOW)
                r_pose_stable = -float(np.clip(rel_delta * 15.0, 0.0, 1.0)) * pose_weight
            else:
                r_pose_stable = 0.0

            # Escalating failure penalty kicks in only after the grace window.
            steps_past_grace = max(0, self._locked_steps - self.LOCK_GRACE_STEPS)
            r_fail = -float(min(steps_past_grace * self.LOCK_PENALTY_RATE, 2.0))

            # False-grasp detection: episode ends with -20 penalty if the object
            # has not risen LIFT_VERIFY_MIN within LIFT_VERIFY_WINDOW steps.
            if self._locked_steps == self.LIFT_VERIFY_WINDOW:
                rise = obj_above - self._lock_obj_above
                if rise < self.LIFT_VERIFY_MIN:
                    r_verify_fail = -20.0
                    false_grasp   = True
                else:
                    r_verify_fail = 0.0
            else:
                r_verify_fail = 0.0

            # One-time sparse success bonus.
            if obj_above >= self.LIFT_TARGET_M and not self._success_rewarded:
                r_success = 100.0
                self._success_rewarded = True
            else:
                r_success = 0.0

            reward_phase = (
                r_grasp + r_hold_quality + r_lift + r_height_hold + r_pose_stable +
                r_fail + r_verify_fail + r_success
            )

        r_step = -0.05

        tilt_excess = max(0.0, tilt_rad - self.TILT_FREE_RAD)
        r_tilt_pen  = -self.TILT_PENALTY_SCALE * tilt_excess

        # Push penalty decays once the object is clearly off the table (lifted phase).
        # Before lock or while near table: full penalty.
        # After lock and rising: penalty fades to zero as obj_above → LIFT_TARGET_M.
        if self._gripper_locked and obj_above > self.LIFT_CLEAR_H:
            push_decay = max(
                0.0,
                1.0 - (obj_above - self.LIFT_CLEAR_H) / (self.LIFT_TARGET_M - self.LIFT_CLEAR_H),
            )
            r_push_pen = -self.XY_PUSH_PENALTY_SCALE * obj_xy_disp * push_decay
        else:
            r_push_pen = -self.XY_PUSH_PENALTY_SCALE * obj_xy_disp

        r_action_mag = -self.ACTION_MAG_PENALTY * float(np.sum(np.square(action_cmd[:5])))
        # General delta penalty for all 5 controllable dims.
        r_action_delta = -self.ACTION_DELTA_PENALTY * float(np.sum(np.square(clipped_delta[:5])))
        # Extra z-axis delta penalty to specifically suppress up/down hunting.
        r_action_delta_z = -self.ACTION_DELTA_Z_PENALTY * float(clipped_delta[2] ** 2)

        reward = (
            reward_phase + r_step
            + r_tilt_pen + r_push_pen
            + r_action_mag + r_action_delta + r_action_delta_z
        )

        # Update potentials for next step.
        self._prev_dist      = dist
        self._prev_obj_above = obj_above
        self._prev_reach_metric = reach_now
        self._prev_action    = action_cmd
        self._prev_rel_obj   = np.array(obj_pos, dtype=np.float32) - np.array(ee_pos, dtype=np.float32)

        # Terminate when success is confirmed OR a false-grasp is detected.
        # False-grasp termination (terminated=True, not truncated) gives the
        # value network a clear -20 terminal signal rather than bootstrapping
        # from a hopeless stall.
        terminated = self._success_rewarded or false_grasp
        truncated  = False

        # ── Info ──────────────────────────────────────────────────────────────
        self._step_info = {
            'dist':                dist,
            'reach_metric':        reach_now,
            'pre_xy_err':          pre_xy_err,
            'pre_z_err':           pre_z_err,
            'pre_yaw_err':         pre_yaw_err,
            'gripper_contact':     1.0 if gripper_contact else 0.0,
            'bilateral_contact':   1.0 if bilateral_contact else 0.0,
            'gripper_angle':       gripper_angle,
            'obj_ee_lateral':      float(abs(obj_in_ee[1])),
            'hold_quality_score':  quality_score if self._gripper_locked else 0.0,
            'grasp_confirm':       float(self._grasp_confirm),
            'close_confirm':       float(self._close_confirm),
            'grasp_confirm_frac':  float(self._grasp_confirm) / float(self.GRASP_CONFIRM_STEPS),
            'gripper_locked':      float(self._gripper_locked),
            'locked_steps':        float(self._locked_steps),
            'hold_stable_steps':   float(self._hold_stable_steps),
            'uprightness':         uprightness,
            'obj_xy_disp_m':       obj_xy_disp,
            'obj_lift_m':          obj_above,
            'speed_factor':        speed_factor,
            'false_grasp':         1.0 if false_grasp else 0.0,
            'success':             float(self._success_rewarded),
        }

        return self._get_obs(), reward, terminated, truncated, self._get_info()


gym.register(id='GraspEnv', entry_point=GraspEnv, max_episode_steps=300)
