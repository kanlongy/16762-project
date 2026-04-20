import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mengine')))

import numpy as np
import pybullet as p
import gymnasium as gym
import mengine as m


class GraspEnv(gym.Env):
    """Demo grasp environment: approach one bottle on a table, auto-grasp, auto-lift.

    The gripper is fully automatic — the policy only moves the robot to a good
    position.  Once the end-effector stays within GRASP_DIST of the object for
    GRASP_CONFIRM steps the gripper closes automatically (Phase A → Phase B).
    After locking, wrist pitch and roll are held at 0 (so the gripper stays
    horizontal and the grasped object cannot tilt or fall out), but base yaw
    and wrist yaw remain free so the policy can reorient the robot toward the
    placement target during transport.  In Phase B the lift joint is driven
    upward at a fixed rate; in Phase C2 it is driven downward for a gentle
    placement.

    Observation (21 dims):
        ee_local        (3)  – EE (fingertip midpoint) position in robot base frame
        diff            (3)  – obj_local - ee_local  (approach vector; 0 when object
                               is centred between the fingers at grasp depth)
        place_diff      (3)  – place_target_local - ee_local  (transport vector)
        joint_obs       (3)  – lift joint, arm_sum, wrist_yaw
        obj_above       (1)  – object height above its initial z (lift progress)
        gripper_locked  (1)  – 0 = Phase A, 1 = Phase B/C
        confirm_frac    (1)  – grasp progress (Phase A); 1.0 when descending (Phase C2)
        gripper_contact (1)  – 1 if any gripper finger touches object
        prev_action     (5)  – previous action (smoothness context)
        ── total: 3+3+3+3+1+1+1+1+5 = 21 ──

    Action (5 dims, all in [-1, 1]):
        [0:2]  base wheel velocities  (right, left)
        [2]    lift joint delta  (auto-lift in Phase B; auto-descend in C2;
                                  free in Phase A / C1)
        [3]    arm extension delta
        [4]    wrist yaw delta    (wrist pitch/roll are hard-locked to 0)

    Phase flow:
        A (approach)  – !gripper_locked
        B (lift)      – gripper_locked AND obj_above < LIFT_TARGET_M
        C (transport) – gripper_locked AND obj_above >= LIFT_TARGET_M

    Triggers:
        A→B: dist(EE, obj) < GRASP_DIST for GRASP_CONFIRM steps
             → gripper closes + rigid pybullet constraint attaches object to EE
        B→C: obj_above >= LIFT_TARGET_M (automatic, no action required)
        C1→C2: lateral dist(EE_xy, place_target_xy) < PLACE_DIST
               → enters controlled-descent sub-phase
        C2→done: obj_above < PLACE_HEIGHT_M while descending
               → constraint released + gripper opens → object rests on table → success

    Reward:
        Phase A:  r_reach    = clip((prev_dist  - dist)       * 10, -2, 2)
        Phase B:  r_lift     = clip(delta_z                   * 20, -2, 2)
                  r_height   = 0.3 * clip(obj_above/LIFT_TARGET, 0, 1)  [above 2 cm]
                  r_lift_done = +10 (one-time, on entering Phase C)
                  r_false    = -10 if rise < LIFT_VERIFY_MIN at LIFT_VERIFY_WIN
        Phase C1: r_transport = clip((prev_place_dist - place_dist) * 10, -2, 2)
        Phase C2: r_descend  = clip((prev_obj_above - obj_above)    * 10, -2, 2)
                  r_place_done = +100 (terminal, on successful placement)
        Always:   r_step = -0.02,  r_action = -0.005 * ||action||^2
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    # ── Constants ─────────────────────────────────────────────────────────────
    GRIPPER_OPEN     = 0.6
    GRIPPER_CLOSE    = 0.0

    # Phase A (approach → grasp)
    # GRASP_DIST is the radius (m) around the fingertip midpoint for the auto-grasp
    # trigger. With the open gripper spanning ~21 cm, the previous 12 cm threshold
    # allowed the trigger to fire even when the object was entirely outside the
    # finger span (e.g. ~10 cm laterally off-centre). We now use 6 cm, which lies
    # inside the half-span of an open gripper (10.5 cm) and ensures the object is
    # between the fingers. The lateral-alignment gate below further guarantees it.
    GRASP_DIST       = 0.06
    # In addition to the 3-D distance check, require the object to lie within this
    # half-width (m) of the line connecting the two fingertips before triggering
    # auto-grasp. Mustard bottle radius ~3.5 cm + 1.5 cm margin = 5 cm. This stops
    # the "right finger on left side of object" failure mode where the policy
    # reduces distance but never centres the object between the fingers.
    GRASP_LATERAL    = 0.05
    GRASP_CONFIRM    = 4       # consecutive close-enough steps to commit

    # Phase B (lift)
    AUTO_LIFT        = 0.8     # lift action override in Phase B
    LIFT_TARGET_M    = 0.15    # metres above initial z → triggers Phase C
    LIFT_VERIFY_WIN  = 20      # steps after lock to start rising
    LIFT_VERIFY_MIN  = 0.005   # minimum rise (m) to pass lift verification
    LIFT_DONE_BONUS  = 10.0    # one-time reward when Phase C begins

    # Phase C1 (lateral transport) → Phase C2 (controlled descent) → release
    PLACE_DIST       = 0.20    # lateral threshold to trigger descent (C1 → C2)
    AUTO_DESCEND     = -0.8    # lift action override during descent (mirrors AUTO_LIFT)
    PLACE_HEIGHT_M   = 0.03    # obj_above below this → release constraint (gentle place)
    PLACE_SUCCESS    = 100.0   # terminal reward on placement
    PLACE_MIN_DIST   = 0.20    # min lateral distance of place_target from pickup
    PLACE_MAX_DIST   = 0.40    # max lateral distance of place_target from pickup

    # Friction coefficients applied at reset (pybullet changeDynamics).
    # Default PyBullet friction (~0.5) is low relative to the object weight.
    OBJ_LATERAL_FRICTION  = 3.0
    OBJ_SPINNING_FRICTION = 0.5
    OBJ_ROLLING_FRICTION  = 0.1
    GRIP_LATERAL_FRICTION = 3.0

    # Gripper finger / fingertip link indices on the Stretch3 body.
    # Stretch3 URDF link layout:
    #   26: link_gripper_finger_right      (revolute joint, closes with gripper)
    #   27: link_gripper_fingertip_right   (fixed to 26, tip of right finger)
    #   29: link_gripper_finger_left       (revolute joint, closes with gripper)
    #   30: link_gripper_fingertip_left    (fixed to 29, tip of left finger)
    #   33: link_grasp_center              (fixed to gripper body, NOT between fingers!)
    # The URDF's `link_grasp_center` is offset ~3.5 cm PAST the fingertip plane in
    # the arm-extension direction. Using it as the EE reference lets the policy
    # satisfy "minimise distance" by driving the object past the fingertips, so
    # fingers close in front of (empty) air. We instead use the midpoint of the
    # two fingertips (links 27 and 30) as the true grasp reference point.
    _GRIPPER_LINKS       = frozenset([26, 27, 29, 30])
    _FINGERTIP_R_LINK    = 27
    _FINGERTIP_L_LINK    = 30

    TILT_FREE_RAD       = np.deg2rad(15.0)
    TILT_PENALTY_SCALE  = 0.3

    ACTION_MAX_DELTA = np.array([0.25, 0.25, 0.18, 0.18, 0.22], dtype=np.float32)

    SLOWDOWN_DIST_NEAR = 0.20   # below this distance base speed is reduced
    SLOWDOWN_DIST_FAR  = 0.60   # above this distance base speed is unrestricted
    SLOWDOWN_MIN       = 0.15

    # controllable_joints = [0, 1, 4, 6, 7, 8, 9, 10, 12, 13, 26, 29]
    # indices:               0  1  2  3  4  5  6   7   8   9  10  11
    _IDX_LIFT       = 2
    _IDX_ARM_START  = 3
    _IDX_ARM_END    = 7
    _IDX_WRIST      = slice(7, 10)
    _IDX_WRIST_YAW  = 7
    _IDX_WRIST_PITCH = 8
    _IDX_WRIST_ROLL  = 9
    _IDX_GRIP_L     = 10
    _IDX_GRIP_R     = 11

    _CAM_EYE    = [0.8, -1.5, 1.6]
    _CAM_TARGET = [-0.6, 0.0, 0.85]
    _CAM_W, _CAM_H = 480, 320

    def __init__(self, render_mode=None, curriculum_stage: int | str = 'full', **kwargs):
        """
        Args:
            curriculum_stage: Controls which sub-task is trained.
                1  – Phase A+B only (approach + grasp + lift).
                     Episode terminates as success when object reaches LIFT_TARGET_M.
                     Use for Stage-1 curriculum pre-training.
                2  – Phase C only (transport + descend + place).
                     Episode starts with robot already holding the lifted object.
                     Use for Stage-2 curriculum pre-training.
                'full' (default) – Full A→B→C pick-and-place task.
        """
        self.render_mode       = render_mode
        self._curriculum_stage = int(curriculum_stage) if curriculum_stage != 'full' else 'full'
        self.env               = m.Env(gravity=[0, 0, -9.81], render=render_mode == 'human')
        # 21 dims: ee_local(3)+diff(3)+place_diff(3)+joint_obs(3)+scalars(4)+prev_action(5)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(21,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.table_height: float   = 0.85
        self.object                = None
        self.robot                 = None
        self._gripper_locked: bool = False
        self._grasp_confirm: int   = 0
        self._locked_steps: int    = 0
        self._lock_obj_above: float = 0.0
        self._obj_init_z: float    = 0.0
        self._prev_dist: float     = 1.0
        self._prev_obj_above: float = 0.0
        self._success_rewarded: bool = False
        self._contact_rewarded: bool = False
        self._grasp_constraint: int  = -1   # pybullet constraint id (-1 = none)
        self._phase_c:         bool = False   # True once lift target reached
        self._place_descend:   bool = False   # True during controlled descent (C2)
        self._lift_done_rewarded: bool = False
        self._place_target: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_place_dist: float   = 1.0
        self._prev_action          = np.zeros(5, dtype=np.float32)
        self._step_info: dict      = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _object_uprightness(obj_quat) -> float:
        mat = np.array(p.getMatrixFromQuaternion(obj_quat), dtype=np.float32).reshape(3, 3)
        return float(np.clip(mat[2, 2], -1.0, 1.0))

    def _gripper_contact(self) -> bool:
        """True if any gripper finger/fingertip is in contact with the object."""
        contacts = self.robot.get_contact_points(bodyB=self.object)
        return bool(contacts and any(c['linkA'] in self._GRIPPER_LINKS for c in contacts))

    def _fingertip_midpoint(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (midpoint_world, lateral_axis_world) of the two gripper fingertips.

        `midpoint_world` is the true grasp point — the geometric centre between
        the right (link 27) and left (link 30) fingertip links. This is what the
        reward and auto-grasp logic use, not the URDF-declared `link_grasp_center`
        which is offset ~3.5 cm past the fingertip plane and causes the policy to
        drive the object past the fingers.

        `lateral_axis_world` is the unit vector from right to left fingertip. The
        component of (obj - midpoint) along this axis measures how much the object
        is shifted sideways out of the gripper span. We gate the auto-grasp on
        this quantity so the trigger only fires with the object squarely between
        the two fingers.
        """
        client = self.env.id
        r_state = p.getLinkState(
            self.robot.body, self._FINGERTIP_R_LINK,
            computeForwardKinematics=True, physicsClientId=client,
        )
        l_state = p.getLinkState(
            self.robot.body, self._FINGERTIP_L_LINK,
            computeForwardKinematics=True, physicsClientId=client,
        )
        r_pos = np.asarray(r_state[4], dtype=np.float32)
        l_pos = np.asarray(l_state[4], dtype=np.float32)
        mid   = (r_pos + l_pos) * 0.5
        axis  = l_pos - r_pos
        norm  = float(np.linalg.norm(axis))
        axis  = axis / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return mid, axis.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        # Use the fingertip midpoint as the EE reference, not link 33 (grasp_center).
        # See `_fingertip_midpoint` for rationale.
        ee_pos_arr, _ = self._fingertip_midpoint()
        ee_pos = ee_pos_arr.tolist()
        obj_pos, _ = self.object.get_base_pos_orient()

        ee_local,  _ = self.robot.global_to_local_coordinate_frame(ee_pos)
        obj_local, _ = self.robot.global_to_local_coordinate_frame(obj_pos)
        diff = obj_local - ee_local

        # Place target direction (always present; policy ignores in Phase A/B)
        pt_local, _ = self.robot.global_to_local_coordinate_frame(
            self._place_target.tolist()
        )
        place_diff = pt_local - ee_local

        ja      = self.robot.get_joint_angles(self.robot.controllable_joints)
        lift    = ja[self._IDX_LIFT]
        arm_sum = ja[self._IDX_ARM_START:self._IDX_ARM_END].sum()
        wrist   = ja[self._IDX_WRIST]

        obj_above = float(obj_pos[2] - self._obj_init_z)

        # confirm_frac: grasp progress in Phase A/B, 1.0 when descending (Phase C2)
        if self._place_descend:
            confirm_frac = 1.0
        else:
            confirm_frac = float(self._grasp_confirm) / float(self.GRASP_CONFIRM)

        contact = self._gripper_contact()

        return np.concatenate([
            ee_local,                                 # 3
            diff,                                     # 3
            place_diff,                               # 3  ← NEW
            [lift, arm_sum, wrist[0]],                # 3
            [obj_above],                              # 1
            [1.0 if self._gripper_locked else 0.0],   # 1
            [confirm_frac],                           # 1
            [1.0 if contact else 0.0],                # 1
            self._prev_action,                        # 5
        ]).astype(np.float32)

    def _init_stage2(self) -> None:
        """Teleport robot to optimal pickup pose, create a grasp constraint, and
        lift the object to LIFT_TARGET_M.  Called at the end of reset() when
        curriculum_stage == 2 so each episode starts directly in Phase C.
        """
        client = self.env.id
        robot  = self.robot
        cj     = robot.controllable_joints

        obj_pos0, obj_quat0 = self.object.get_base_pos_orient()

        # 1. Teleport robot so arm points toward the object in the -x world direction.
        #    Stretch3 arm extends in -y of the robot frame; rotating base by -π/2 makes
        #    the arm extension point in the -x world direction toward the object.
        ARM_EXT  = 0.20   # arm extension target (m)
        ARM_ZERO = 0.415  # approx EE-x offset from base when arm=0
        base_x   = float(obj_pos0[0]) + ARM_ZERO + ARM_EXT
        base_y   = float(obj_pos0[1])
        bq = p.getQuaternionFromEuler([0, 0, -np.pi / 2], physicsClientId=client)
        p.resetBasePositionAndOrientation(
            robot.body, [base_x, base_y, 0], bq, physicsClientId=client
        )

        # 2. Set arm joints to ARM_EXT (four prismatic joints share the extension).
        for ki in range(self._IDX_ARM_START, self._IDX_ARM_END):
            p.resetJointState(robot.body, cj[ki], ARM_EXT / 4, physicsClientId=client)
        m.step_simulation(steps=5, realtime=False)

        # 3. Compute the invariant offset: lz = EE_z - lift_joint_value.
        ee_t, _ = robot.get_link_pos_orient(robot.end_effector)
        ja_t    = robot.get_joint_angles(cj)
        lz      = ee_t[2] - ja_t[self._IDX_LIFT]

        # 4. Set lift joint so EE aligns with the object's initial z (grasp height),
        #    then create the constraint immediately (before any simulation steps that
        #    would let gravity pull the object down).
        tl_grasp = float(np.clip(float(obj_pos0[2]) - lz, 0.05, 1.05))
        p.resetJointState(robot.body, cj[self._IDX_LIFT], tl_grasp, physicsClientId=client)

        # 5. Create constraint at the current EE / object poses (no simulation yet).
        ee_pos, ee_quat   = robot.get_link_pos_orient(robot.end_effector)
        obj_pos_g, oq_g   = self.object.get_base_pos_orient()
        inv_ep, inv_eq    = p.invertTransform(ee_pos, ee_quat)
        rel_pos, rel_quat = p.multiplyTransforms(
            inv_ep, inv_eq, obj_pos_g, oq_g, physicsClientId=client,
        )
        self._grasp_constraint = p.createConstraint(
            robot.body, robot.end_effector,
            self.object.body, -1,
            p.JOINT_FIXED, [0, 0, 0],
            list(rel_pos), [0, 0, 0], list(rel_quat),
            physicsClientId=client,
        )
        # Use a large maxForce so the constraint reliably holds the object during
        # teleportation and subsequent simulation steps. 50 N proved insufficient:
        # the arm's PD controller could outfight the constraint and let the object
        # fall back to the table before episode start, causing spurious
        # "instant success" episodes in Stage 2.
        p.changeConstraint(self._grasp_constraint, maxForce=500, physicsClientId=client)

        # 6. NOW teleport lift joint AND object to the lifted height, then stabilise.
        #    Both are moved together so the constraint can immediately lock the
        #    relative pose without fighting gravity over many simulation steps.
        lifted_z = float(obj_pos0[2]) + self.LIFT_TARGET_M
        tl_lift  = float(np.clip(lifted_z - lz, 0.05, 1.05))
        p.resetJointState(robot.body, cj[self._IDX_LIFT], tl_lift, physicsClientId=client)
        p.resetBasePositionAndOrientation(
            self.object.body,
            [float(obj_pos0[0]), float(obj_pos0[1]), lifted_z],
            list(obj_quat0),
            physicsClientId=client,
        )
        m.step_simulation(steps=40, realtime=False)

        # Verify constraint held: if object slipped to near table height, force it
        # back to the correct lifted position before the episode begins.
        obj_check, _ = self.object.get_base_pos_orient()
        if float(obj_check[2]) < (float(obj_pos0[2]) + self.LIFT_TARGET_M * 0.5):
            p.resetBasePositionAndOrientation(
                self.object.body,
                [float(obj_pos0[0]), float(obj_pos0[1]), lifted_z],
                list(obj_quat0),
                physicsClientId=client,
            )
            m.step_simulation(steps=20, realtime=False)

        # 6. Set all Phase C state variables so step() starts from Phase C.
        # Use the fingertip midpoint as the EE reference (same convention as
        # step()'s `ee_pos`) so `_prev_place_dist` is computed on the same point
        # the reward function will use on the next step.
        ee_final_arr, _ = self._fingertip_midpoint()
        obj_final, _    = self.object.get_base_pos_orient()
        obj_above       = float(obj_final[2] - self._obj_init_z)

        self._gripper_locked     = True
        self._grasp_confirm      = self.GRASP_CONFIRM         # already triggered
        self._locked_steps       = self.LIFT_VERIFY_WIN + 1   # past verification window
        self._lock_obj_above     = obj_above
        self._phase_c            = True
        self._lift_done_rewarded = True
        self._prev_obj_above     = obj_above
        self._prev_dist          = 0.0  # Phase A potential not used in Phase C
        self._prev_place_dist    = float(np.linalg.norm(
            ee_final_arr[:2] - self._place_target[:2]
        ))

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
        self._gripper_locked  = False
        self._grasp_confirm   = 0
        self._locked_steps    = 0
        self._lock_obj_above  = 0.0
        self._success_rewarded     = False
        self._contact_rewarded     = False
        self._grasp_constraint     = -1
        self._phase_c              = False
        self._place_descend        = False
        self._lift_done_rewarded   = False
        self._place_target         = np.zeros(3, dtype=np.float32)
        self._prev_place_dist      = 1.0
        self._prev_dist            = 1.0
        self._obj_init_z      = 0.0
        self._prev_obj_above  = 0.0
        self._prev_action     = np.zeros(5, dtype=np.float32)
        self._step_info       = {}

        _ = m.Ground()
        # Limit table height variation so the lift joint always has enough range
        # to lift 0.15 m above the bottle.
        # Lift max = 1.1 m.  EE_z ≈ lift + 0.111.
        # For bottle at table_height: lift_to_grasp ≈ table_height - 0.111,
        # remaining = 1.1 - lift_to_grasp = 1.211 - table_height ≥ 0.15 m
        # → table_height ≤ 1.061 m, i.e. height_offset ≤ 0.211 - 0.85 ≈ 0.21 m.
        # Use ±0.10 m to have comfortable margin.
        height_offset     = np.random.uniform(-0.10, 0.10)
        self.table_height = 0.85 + height_offset

        _ = m.URDF(
            filename=os.path.join(m.directory, 'table', 'table.urdf'),
            static=True,
            position=[-1.3, 0, height_offset],
            orientation=[0, 0, 0, 1],
        )
        # Object mass: 0.15 kg (previously 0.5 kg).
        # The mustard-bottle mesh is ~23 cm out from the wrist_pitch axis, so
        # every 100 g of object weight translates into ~0.23 N·m of gravitational
        # torque on the wrist. With mengine's soft default PD gains (0.05) and
        # the whole robot arm overridden to 10 g per link in stretch3.py, a
        # 0.5 kg bottle visibly tilts the gripper during lift + transport. 0.15
        # kg keeps the task meaningful (still heavier than the entire wrist +
        # gripper linkage) while staying well inside the PD controller's
        # stable operating range.
        self.object = m.Shape(
            m.Mesh(filename=os.path.join(m.directory, 'ycb', 'mustard.obj'), scale=[1, 1, 1]),
            static=False, mass=0.15,
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
        # Set initial lift so EE starts ~10 cm below table top, giving the
        # policy room to descend to the bottle AND lift it by LIFT_TARGET_M.
        # EE_z ≈ lift + 0.111.  We want EE_z = table_height - 0.10.
        init_lift = float(np.clip(self.table_height - 0.10 - 0.111, 0.05, 1.05))
        self.robot.set_joint_angles(angles=[init_lift], joints=[4])
        self.robot.set_gripper_position(
            [self.GRIPPER_OPEN, self.GRIPPER_OPEN], set_instantly=True
        )

        # Boost friction so the gripper can hold the 0.15 kg bottle without
        # relying entirely on the grasp constraint. PyBullet default (~0.5)
        # produces less finger friction than the bottle's ~1.5 N weight needs
        # when only a finger-object contact is available (edge of constraint
        # cone, low normal force transients, etc.).
        p.changeDynamics(
            self.object.body, -1,
            lateralFriction=self.OBJ_LATERAL_FRICTION,
            spinningFriction=self.OBJ_SPINNING_FRICTION,
            rollingFriction=self.OBJ_ROLLING_FRICTION,
            physicsClientId=self.env.id,
        )
        for link_idx in self._GRIPPER_LINKS:
            p.changeDynamics(
                self.robot.body, link_idx,
                lateralFriction=self.GRIP_LATERAL_FRICTION,
                physicsClientId=self.env.id,
            )

        m.step_simulation(steps=20, realtime=False)

        # Use fingertip midpoint (actual grasp point), not link 33, to match
        # step()'s dist computation.
        ee_pos0_arr, _ = self._fingertip_midpoint()
        ee_pos0        = ee_pos0_arr.tolist()
        obj_pos0, _    = self.object.get_base_pos_orient()
        self._prev_dist     = float(np.linalg.norm(ee_pos0_arr - np.asarray(obj_pos0, dtype=np.float32)))
        self._obj_init_z    = float(obj_pos0[2])
        self._prev_obj_above = 0.0

        # Generate placement target: random point on the table within reach,
        # at least PLACE_MIN_DIST away from the pickup position.
        angle  = np.random.uniform(-np.pi, np.pi)
        radius = np.random.uniform(self.PLACE_MIN_DIST, self.PLACE_MAX_DIST)
        px = float(obj_pos0[0]) + radius * np.cos(angle)
        py = float(obj_pos0[1]) + radius * np.sin(angle)
        # Clamp to table surface area so the target is reachable.
        px = float(np.clip(px, -0.95, -0.50))
        py = float(np.clip(py, -0.30,  0.30))
        self._place_target = np.array([px, py, self.table_height], dtype=np.float32)
        # Initialise transport potential with EE distance to placement target.
        self._prev_place_dist = float(np.linalg.norm(
            np.array(ee_pos0[:2]) - self._place_target[:2]
        ))

        # Stage 2: skip Phase A/B, start directly in Phase C (pre-grasped + lifted).
        if self._curriculum_stage == 2:
            self._init_stage2()

        return self._get_obs(), self._get_info()

    def step(self, action):
        scale = 0.025
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_delta  = action - self._prev_action
        clipped_delta = np.clip(action_delta, -self.ACTION_MAX_DELTA, self.ACTION_MAX_DELTA)
        action_cmd    = np.clip(
            self._prev_action + clipped_delta,
            self.action_space.low, self.action_space.high,
        )

        # Gripper: always open in Phase A, always closed in Phase B.
        grip_target = self.GRIPPER_CLOSE if self._gripper_locked else self.GRIPPER_OPEN

        # Note on gripper uprightness during lift/transport: the two rotational
        # DOFs that could tilt the gripper (wrist pitch, wrist roll) are already
        # hard-locked to 0 below in `scaled` (see the [..., 0.0, 0.0] slots).
        # The remaining rotations accessible to the policy — base yaw (differential
        # wheels) and wrist yaw — are rotations about the vertical axis and
        # therefore cannot tilt the gripper or cause the grasped object to fall
        # over. We intentionally leave those free so the policy can reorient the
        # base toward the placement target during Phase C transport.

        # Phase B:  auto-lift  (override lift action upward)
        # Phase C2: auto-descend (override lift action downward for gentle placement)
        # Phase C1: lift is free (policy maintains height during lateral transport)
        if self._gripper_locked and not self._phase_c:
            action_cmd[2] = self.AUTO_LIFT
        elif self._place_descend:
            action_cmd[2] = self.AUTO_DESCEND

        # Base speed limiting: slow down when close to object (Phase A only).
        if self._gripper_locked:
            speed_factor = 1.0
        else:
            speed_factor = float(np.clip(
                (self._prev_dist - self.SLOWDOWN_DIST_NEAR)
                / (self.SLOWDOWN_DIST_FAR - self.SLOWDOWN_DIST_NEAR),
                self.SLOWDOWN_MIN, 1.0,
            ))

        scaled = np.concatenate([
            action_cmd[0:2] * 0.5 * speed_factor,
            [action_cmd[2] * scale],
            [action_cmd[3] / 4.0 * scale] * 4,
            [action_cmd[4] * scale, 0.0, 0.0],  # wrist yaw only; pitch/roll locked
            [grip_target, grip_target],
        ])

        current = self.robot.get_joint_angles(self.robot.controllable_joints)
        target  = current + scaled
        target[self._IDX_GRIP_L] = grip_target
        target[self._IDX_GRIP_R] = grip_target

        # Absolute-lock wrist pitch and roll to 0 (rather than "current + 0 delta").
        # Using `current + 0` is a *follower* lock: once gravity drags the joint
        # by Δθ, the target becomes current+Δθ and the drift is accepted as the
        # new setpoint, so the gripper tilts progressively under the load of the
        # grasped object. Writing absolute 0 here makes the PD controller push
        # the joint back to horizontal every step, keeping the gripper level
        # regardless of the object weight.
        target[self._IDX_WRIST_PITCH] = 0.0
        target[self._IDX_WRIST_ROLL]  = 0.0

        self.robot.control(target)
        m.step_simulation(steps=10, realtime=self.env.render)

        # ── State ─────────────────────────────────────────────────────────────
        # EE reference is the midpoint between the two fingertips — the actual
        # grasp point. We still use link 33 (grasp_center) as the constraint
        # parent because it is rigidly attached to the gripper body; its frame is
        # fine for multiplyTransforms even though it is not geometrically centred.
        ee_pos_arr, _     = self._fingertip_midpoint()
        ee_pos            = ee_pos_arr.tolist()
        _, ee_quat        = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos, obj_quat = self.object.get_base_pos_orient()
        obj_vec           = np.asarray(obj_pos, dtype=np.float32) - ee_pos_arr
        dist      = float(np.linalg.norm(obj_vec))
        obj_above = float(obj_pos[2] - self._obj_init_z)
        tilt_rad  = float(np.arccos(np.clip(self._object_uprightness(obj_quat), -1.0, 1.0)))
        # Lateral (xy-plane) distance from EE to placement target
        place_dist = float(np.linalg.norm(
            np.array(ee_pos[:2]) - self._place_target[:2]
        ))

        # ── Auto-grasp trigger: distance + lateral-centring ──────────────────
        # Phase A gripper is open (21 cm span) — it physically cannot contact a
        # 7 cm bottle from the side.  We trigger when (a) the object is close to
        # the fingertip midpoint AND (b) the lateral offset along the finger-
        # opening axis is small enough that the object actually sits BETWEEN the
        # fingers. Without (b), the policy satisfies (a) with the object parked
        # next to one finger instead of centred — on closing, the fingers miss.
        _, lateral_axis = self._fingertip_midpoint()
        lateral_offset  = float(abs(np.dot(obj_vec, lateral_axis)))
        gripper_contact = self._gripper_contact()  # used in obs & reward only
        if not self._gripper_locked:
            if dist < self.GRASP_DIST and lateral_offset < self.GRASP_LATERAL:
                self._grasp_confirm += 1
                if self._grasp_confirm >= self.GRASP_CONFIRM:
                    self._gripper_locked = True
                    self._lock_obj_above = obj_above
                    # Attach object to EE with a fixed constraint.
                    ee_pos, ee_quat = self.robot.get_link_pos_orient(
                        self.robot.end_effector
                    )
                    obj_pos_c, obj_quat_c = self.object.get_base_pos_orient()
                    inv_ee_pos, inv_ee_quat = p.invertTransform(ee_pos, ee_quat)
                    rel_pos, rel_quat = p.multiplyTransforms(
                        inv_ee_pos, inv_ee_quat, obj_pos_c, obj_quat_c,
                        physicsClientId=self.env.id,
                    )
                    self._grasp_constraint = p.createConstraint(
                        self.robot.body,          # parentBodyUniqueId
                        self.robot.end_effector,  # parentLinkIndex
                        self.object.body,         # childBodyUniqueId
                        -1,                       # childLinkIndex (base)
                        p.JOINT_FIXED,            # jointType
                        [0, 0, 0],               # jointAxis
                        list(rel_pos),            # parentFramePosition
                        [0, 0, 0],               # childFramePosition
                        list(rel_quat),           # parentFrameOrientation
                        physicsClientId=self.env.id,
                    )
                    # maxForce: must support object weight (~1.5 N for 0.15 kg)
                    # plus transient inertial loads during lift/transport.  50 N
                    # gives a large margin without destabilising the arm's PD.
                    p.changeConstraint(
                        self._grasp_constraint,
                        maxForce=50,
                        physicsClientId=self.env.id,
                    )
            else:
                self._grasp_confirm = 0

        # ── Phase C transition: lift target reached ───────────────────────────
        if self._gripper_locked and not self._phase_c and obj_above >= self.LIFT_TARGET_M:
            self._phase_c = True
            # Stage 1: lifting to target height = success → end episode here.
            if self._curriculum_stage == 1 and not self._success_rewarded:
                self._success_rewarded = True

        # ── Phase C transitions and release ───────────────────────────────────
        placed = False
        if self._phase_c and not self._success_rewarded:
            # C1 → C2: start controlled descent once EE is above the target
            if not self._place_descend and place_dist < self.PLACE_DIST:
                self._place_descend = True

            # C2 release: object is close enough to table → gentle placement
            if self._place_descend and obj_above < self.PLACE_HEIGHT_M:
                if self._grasp_constraint >= 0:
                    p.removeConstraint(
                        self._grasp_constraint, physicsClientId=self.env.id
                    )
                    self._grasp_constraint = -1
                # Brief settle so object rests on the table
                m.step_simulation(steps=15, realtime=self.env.render)
                placed = True
                self._success_rewarded = True

        # ── Reward ────────────────────────────────────────────────────────────
        false_grasp = False

        if not self._gripper_locked:
            # Phase A: dense potential-based approach reward.
            r_reach = float(np.clip((self._prev_dist - dist) * 10.0, -2.0, 2.0))
            reward_phase = r_reach

        elif not self._phase_c:
            # Phase B: lift reward.
            self._locked_steps += 1
            delta_z = obj_above - self._prev_obj_above
            r_lift  = float(np.clip(delta_z * 20.0, -2.0, 2.0))

            r_height = (
                0.3 * float(np.clip(obj_above / self.LIFT_TARGET_M, 0.0, 1.0))
                if obj_above >= 0.02 else 0.0
            )

            if self._locked_steps == self.LIFT_VERIFY_WIN:
                rise = obj_above - self._lock_obj_above
                if rise < self.LIFT_VERIFY_MIN:
                    r_false   = -10.0
                    false_grasp = True
                else:
                    r_false = 0.0
            else:
                r_false = 0.0

            reward_phase = r_lift + r_height + r_false

        else:
            if not self._place_descend:
                # Phase C1: reward lateral approach to placement target.
                r_transport = float(np.clip(
                    (self._prev_place_dist - place_dist) * 10.0, -2.0, 2.0
                ))
                reward_phase = r_transport
            else:
                # Phase C2: reward controlled descent (object lowering).
                r_descend = float(np.clip(
                    (self._prev_obj_above - obj_above) * 10.0, -2.0, 2.0
                ))
                r_place_done = self.PLACE_SUCCESS if placed else 0.0
                reward_phase = r_descend + r_place_done

        # Lift-done bonus fires exactly once at the step Phase C is entered.
        # Previously this block lived inside `elif not self._phase_c:` with a check
        # `if self._phase_c`, which was a logical contradiction (dead code): the outer
        # branch is entered only when _phase_c is False, but the inner check required
        # _phase_c to be True. Because _phase_c is set True *before* the reward block
        # runs (lines above), the elif branch was never entered at the transition step,
        # so neither Stage-1's +100 terminal bonus nor Full's +10 lift bonus were ever
        # awarded. Moving the check here, outside the if-elif-else, fixes the issue.
        if self._phase_c and not self._lift_done_rewarded:
            if self._curriculum_stage == 1:
                r_lift_done = self.PLACE_SUCCESS   # terminal success reward for stage 1
            else:
                r_lift_done = self.LIFT_DONE_BONUS
            self._lift_done_rewarded = True
            reward_phase += r_lift_done

        r_step   = -0.02
        r_action = -0.005 * float(np.sum(np.square(action_cmd)))
        tilt_excess = max(0.0, tilt_rad - self.TILT_FREE_RAD)
        r_tilt = -self.TILT_PENALTY_SCALE * tilt_excess

        reward = reward_phase + r_step + r_action + r_tilt

        # ── Update potentials ─────────────────────────────────────────────────
        self._prev_dist       = dist
        self._prev_obj_above  = obj_above
        self._prev_place_dist = place_dist
        self._prev_action     = action_cmd

        terminated = self._success_rewarded or false_grasp
        truncated  = False

        self._step_info = {
            'dist':           dist,
            'lateral_offset': lateral_offset,
            'place_dist':     place_dist,
            'obj_lift_m':     obj_above,
            'gripper_contact':float(gripper_contact),
            'gripper_locked': float(self._gripper_locked),
            'phase_c':        float(self._phase_c),
            'place_descend':  float(self._place_descend),
            'locked_steps':   float(self._locked_steps),
            'confirm':        float(self._grasp_confirm),
            'speed_factor':   speed_factor,
            'false_grasp':    1.0 if false_grasp else 0.0,
            'placed':         1.0 if placed else 0.0,
            'success':        float(self._success_rewarded),
        }

        return self._get_obs(), reward, terminated, truncated, self._get_info()


gym.register(id='GraspEnv',       entry_point=GraspEnv, max_episode_steps=500)
# Ablation stages: same env class, different curriculum_stage value.
# Stage 1 – Phase A+B only (grasp + lift).  Shorter horizon → fewer max steps.
gym.register(id='GraspEnvStage1', entry_point=GraspEnv,
             kwargs={'curriculum_stage': 1}, max_episode_steps=300)
# Stage 2 – Phase C only (transport + place).  Starts pre-grasped.
gym.register(id='GraspEnvStage2', entry_point=GraspEnv,
             kwargs={'curriculum_stage': 2}, max_episode_steps=400)
