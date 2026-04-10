import os
import sys
sys.path.insert(0, '/home/ye/mengine')

import numpy as np
import gymnasium as gym
import mengine as m


class GraspEnv(gym.Env):
    """Phase 1 of hierarchical pick-and-place: reach the object and grasp it stably.

    The episode succeeds (terminates) when the robot lifts the object at least
    5 cm above the table surface with a stable contact.

    Observation (17 dims):
        ee_pos_local       (3)  – end-effector position in robot base frame
        obj_pos_local      (3)  – object position in robot base frame
        diff               (3)  – obj_pos_local - ee_pos_local
        joint_obs          (5)  – lift, arm_sum, wrist_yaw, wrist_pitch, wrist_roll
        gripper_angle      (1)  – mean of the two gripper finger joints
        obj_above_table    (1)  – object z minus table surface z (signed)
        in_contact         (1)  – 1.0 if robot touches object, else 0.0

    Action (8 dims, all in [-1, 1]):
        action[0:2]  – base wheel velocities (delta)
        action[2]    – lift joint (delta)
        action[3]    – arm extension shared across 4 prismatic joints (delta)
        action[4:7]  – wrist yaw / pitch / roll (delta)
        action[7]    – gripper: +1 = fully open, -1 = fully closed (absolute target)

    Reward components:
        r_reach   – exp(-3 * dist_ee_obj)              dense shaping toward object
        r_contact – +5 on each step with contact        encourage touching
        r_lift    – +20 on each step object is >5 cm above table  goal signal
        r_step    – -0.01                               living cost
    """

    GRIPPER_OPEN  = 0.6   # joint angle for fully open gripper
    GRIPPER_CLOSE = 0.0   # joint angle for fully closed gripper
    LIFT_SUCCESS_M = 0.05 # metres above table to count as a successful lift

    # controllable_joints = [0, 1, 4, 6, 7, 8, 9, 10, 12, 13, 26, 29]
    # indices:               0  1  2  3  4  5  6   7   8   9  10  11
    _IDX_LIFT      = 2
    _IDX_ARM_START = 3   # joints 6,7,8,9 (4 prismatic arm segments)
    _IDX_ARM_END   = 7
    _IDX_WRIST     = slice(7, 10)  # joints 10, 12, 13
    _IDX_GRIP_L    = 10  # joint 26
    _IDX_GRIP_R    = 11  # joint 29

    def __init__(self, render_mode=None, **kwargs):
        self.env = m.Env(gravity=[0, 0, -9.81], render=render_mode == 'human')
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(17,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.table_height: float = 0.85
        self.object = None
        self.robot = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        ee_pos,  _ = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos, _ = self.object.get_base_pos_orient()

        ee_local,  _ = self.robot.global_to_local_coordinate_frame(ee_pos)
        obj_local, _ = self.robot.global_to_local_coordinate_frame(obj_pos)
        diff = obj_local - ee_local

        ja = self.robot.get_joint_angles(self.robot.controllable_joints)
        lift      = ja[self._IDX_LIFT]
        arm_sum   = ja[self._IDX_ARM_START:self._IDX_ARM_END].sum()
        wrist     = ja[self._IDX_WRIST]
        gripper   = (ja[self._IDX_GRIP_L] + ja[self._IDX_GRIP_R]) / 2.0

        contact       = self.robot.get_contact_points(bodyB=self.object)
        obj_above     = float(obj_pos[2] - self.table_height)
        in_contact    = 1.0 if contact else 0.0

        return np.concatenate([
            ee_local,
            obj_local,
            diff,
            [lift, arm_sum, wrist[0], wrist[1], wrist[2]],
            [gripper],
            [obj_above],
            [in_contact],
        ]).astype(np.float32)

    def _get_info(self) -> dict:
        return {}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def seed(self, seed):
        np.random.seed(seed)

    def render(self):
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)
        self.env.reset()

        _ = m.Ground()
        height_offset   = np.random.uniform(-0.2, 0.2)
        self.table_height = 0.85 + height_offset

        _ = m.URDF(
            filename=os.path.join(m.directory, 'table', 'table.urdf'),
            static=True,
            position=[-1.3, 0, height_offset],
            orientation=[0, 0, 0, 1],
        )

        self.object = m.Shape(
            m.Mesh(filename=os.path.join(m.directory, 'ycb', 'mustard.obj'), scale=[1, 1, 1]),
            static=False,
            mass=0.5,
            position=[
                -0.6 - np.random.uniform(0.0, 0.2),
                np.random.uniform(-0.2, 0.2),
                self.table_height,
            ],
            orientation=[0, 0, 0, 1],
            rgba=None,
            visual=True,
            collision=True,
        )

        pos_x = np.random.uniform(-0.1, 0.1)
        pos_y = np.random.uniform(-0.1, 0.1)
        theta = np.random.uniform(-np.pi / 4, np.pi / 4)
        self.robot = m.Robot.Stretch3(position=[pos_x, pos_y, 0], orientation=[0, 0, theta])
        self.robot.set_joint_angles(angles=[0.9], joints=[4])

        # Start with gripper fully open
        self.robot.set_gripper_position(
            [self.GRIPPER_OPEN, self.GRIPPER_OPEN], set_instantly=True
        )

        m.step_simulation(steps=20, realtime=False)
        return self._get_obs(), self._get_info()

    def step(self, action):
        scale  = 0.025
        action = np.asarray(action, dtype=np.float32)

        # Map gripper action [-1, 1] → absolute joint target [closed, open]
        grip_target = float(np.interp(action[7], [-1.0, 1.0], [self.GRIPPER_CLOSE, self.GRIPPER_OPEN]))

        # Build full joint target: delta for everything, absolute for gripper
        scaled = np.concatenate([
            action[0:2] * 0.5,               # base wheels
            [action[2] * scale],             # lift
            [action[3] / 4.0 * scale] * 4,  # 4 arm prismatic joints
            action[4:7] * scale,             # wrist yaw / pitch / roll
            [grip_target, grip_target],      # gripper fingers (absolute)
        ])

        current = self.robot.get_joint_angles(self.robot.controllable_joints)
        target  = current + scaled
        # Override gripper with absolute target (don't add to current)
        target[self._IDX_GRIP_L] = grip_target
        target[self._IDX_GRIP_R] = grip_target

        self.robot.control(target)
        m.step_simulation(steps=10, realtime=self.env.render)

        ee_pos,  _ = self.robot.get_link_pos_orient(self.robot.end_effector)
        obj_pos, _ = self.object.get_base_pos_orient()
        dist       = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        contact    = self.robot.get_contact_points(bodyB=self.object)
        obj_above  = float(obj_pos[2] - self.table_height)

        r_reach   = float(np.exp(-3.0 * dist))
        r_contact = 5.0  if contact   else 0.0
        r_lift    = 20.0 if obj_above > self.LIFT_SUCCESS_M else 0.0
        r_step    = -0.01
        reward    = r_reach + r_contact + r_lift + r_step

        # Terminate on stable lift
        terminated = bool(contact is not None and obj_above > self.LIFT_SUCCESS_M)
        truncated  = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()


gym.register(id='GraspEnv', entry_point=GraspEnv, max_episode_steps=200)
