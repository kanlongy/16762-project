"""deploy_real.py – deploy a trained GraspEnv policy on the physical Stretch3 robot.

This script bridges the gap between the PyBullet simulation and the real robot:
  - Reads joint states from stretch_body and builds the same 17-dim observation
    that the policy was trained on.
  - Applies the policy's 8-dim delta actions through stretch_body commands.
  - Accepts the object position from an external source (e.g. lab3 perception
    pipeline, AprilTag, or manual entry for debugging).

Prerequisites on the robot:
    pip install stretch_body

Usage:
    # Dry-run with manually specified object position (no perception required):
    python deploy_real.py --checkpoint log/GraspPhase1/seed_0/best_policy.pt \\
                          --obj-pos -0.7 0.0 0.9 --dry-run

    # Full run (object position updated live from an external ROS2 topic):
    python deploy_real.py --checkpoint log/GraspPhase1/seed_0/best_policy.pt \\
                          --ros-object-topic /object_detector/goal_pose

Known sim-to-real gaps to address before production deployment:
  1. Gravity: sim uses g = -1 m/s², real is -9.81 m/s²  →  reduce policy action
     scale or retrain with realistic gravity + domain randomisation.
  2. Joint limits: sim limits come from the URDF and may differ slightly from
     real-robot calibration – clamp actions to safe ranges below.
  3. Observation noise: add Gaussian noise during training (domain rand) to
     improve robustness to real sensor readings.
  4. Contact detection: the sim uses pybullet contact points; on the real robot
     approximate with gripper force/current feedback.
  5. EE forward kinematics: the sim provides exact EE pos; on the real robot
     use ikpy or ROS TF (see lab3/ik_ros_utils.py).
"""

import argparse
import sys
import time

import numpy as np
import torch

# ── Joint limits (real robot) ─────────────────────────────────────────────────
# Conservative safe limits to prevent hardware damage.
REAL_JOINT_LIMITS = {
    'lift':        (0.05, 1.10),  # metres
    'arm':         (0.00, 0.52),  # metres (total extension)
    'wrist_yaw':   (-1.75, 4.0),  # radians
    'wrist_pitch': (-1.57, 0.56),
    'wrist_roll':  (-3.14, 3.14),
    'gripper':     (-50,   50),   # stretch_gripper units: +50=open, -50=closed
}

# sim scale factors used in GraspEnv.step() – mirror here for exact correspondence
SIM_SCALE        = 0.025
SIM_BASE_SCALE   = 0.5   # applied to action[0:2]
SIM_ARM_SCALE    = SIM_SCALE / 4.0

# Real-robot control frequency
CONTROL_HZ = 5   # Hz – conservative; increase to 10 once tuned

# ── Helper utilities ──────────────────────────────────────────────────────────

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class PolicyRunner:
    """Loads a trained policy and runs one inference step."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        print(f"[PolicyRunner] loading {checkpoint_path}")
        data   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy = data.policy if hasattr(data, 'policy') else data
        policy.eval()
        self._policy = policy
        self._device = device

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic action for the given 17-dim observation."""
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            from tianshou.data import Batch
            result = self._policy(Batch(obs=obs_t.cpu().numpy(), info={}))
            # For stochastic policies use the mean (deterministic deployment)
            act = result.act
            if hasattr(act, 'numpy'):
                act = act.numpy()
        return np.asarray(act).squeeze()


# ── Observation builder ────────────────────────────────────────────────────────

def build_obs(robot_state: dict, obj_pos_world: np.ndarray, table_height: float) -> np.ndarray:
    """Build the 17-dim observation vector from real-robot state.

    Parameters
    ----------
    robot_state : dict with keys:
        'base_pos'     : np.ndarray (2,) – (x, y) in world frame
        'base_yaw'     : float – heading in radians
        'ee_pos_world' : np.ndarray (3,) – EE xyz in world frame from FK
        'lift'         : float – lift joint position (m)
        'arm'          : float – total arm extension (m) = sum of 4 segments
        'wrist_yaw'    : float
        'wrist_pitch'  : float
        'wrist_roll'   : float
        'gripper'      : float – raw stretch_gripper value [-50, 50]
        'in_contact'   : float – 1.0 if gripper force above threshold, else 0.0
    obj_pos_world : np.ndarray (3,) – object xyz in world frame
    table_height  : float – known table surface height in world frame
    """

    # Convert world coordinates to robot base frame (2-D rotation around Z)
    yaw = robot_state['base_yaw']
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cos_y, -sin_y, 0],
                  [sin_y,  cos_y, 0],
                  [0,      0,     1]])
    base_xy = np.append(robot_state['base_pos'], 0.0)

    def to_local(p_world):
        return R @ (np.array(p_world) - base_xy)

    ee_local  = to_local(robot_state['ee_pos_world'])
    obj_local = to_local(obj_pos_world)
    diff      = obj_local - ee_local

    # Normalise gripper to [0, 1] range matching sim (GRIPPER_CLOSE=0, GRIPPER_OPEN=0.6)
    grip_norm = np.interp(robot_state['gripper'], [-50, 50], [0.0, 0.6])

    obj_above = float(obj_pos_world[2] - table_height)

    obs = np.concatenate([
        ee_local,
        obj_local,
        diff,
        [robot_state['lift'],
         robot_state['arm'],
         robot_state['wrist_yaw'],
         robot_state['wrist_pitch'],
         robot_state['wrist_roll']],
        [grip_norm],
        [obj_above],
        [robot_state['in_contact']],
    ]).astype(np.float32)

    assert obs.shape == (17,), f"obs shape mismatch: {obs.shape}"
    return obs


# ── Real robot controller ──────────────────────────────────────────────────────

class StretchController:
    """Thin wrapper around stretch_body.robot for RL policy execution."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        if not dry_run:
            import stretch_body.robot as sb
            self.robot = sb.Robot()
            if not self.robot.startup():
                raise RuntimeError("stretch_body Robot.startup() failed – is the robot on?")
            print("[StretchController] robot connected")
        else:
            print("[StretchController] DRY-RUN mode (no hardware)")
            # Fake current state for dry-run
            self._fake = {
                'lift': 0.9, 'arm': 0.1,
                'wrist_yaw': 0.0, 'wrist_pitch': 0.0, 'wrist_roll': 0.0,
                'gripper': 50.0,
            }

    def get_state(self) -> dict:
        """Read current joint positions from the robot."""
        if self.dry_run:
            return {**self._fake, 'base_pos': np.zeros(2), 'base_yaw': 0.0,
                    'ee_pos_world': np.array([-0.15, 0.0, self._fake['lift']]),
                    'in_contact': 0.0}
        r = self.robot
        r.pull_status()
        s = {
            'base_pos':     np.array([r.base.status['x'], r.base.status['y']]),
            'base_yaw':     r.base.status['theta'],
            'lift':         r.lift.status['pos'],
            'arm':          r.arm.status['pos'],
            'wrist_yaw':    r.end_of_arm.status['wrist_yaw']['pos'],
            'wrist_pitch':  r.end_of_arm.status['wrist_pitch']['pos'],
            'wrist_roll':   r.end_of_arm.status['wrist_roll']['pos'],
            'gripper':      r.end_of_arm.status['stretch_gripper']['pos'],
            # Proxy for contact: gripper motor effort above threshold
            'in_contact':   1.0 if abs(r.end_of_arm.status['stretch_gripper'].get('effort', 0)) > 5.0
                                else 0.0,
        }
        # Approximate EE world position using lift + arm (ignoring wrist for simplicity)
        # For higher accuracy, use ikpy or ROS TF (lab3/ik_ros_utils.py).
        s['ee_pos_world'] = np.array([
            s['base_pos'][0] - (s['arm'] + 0.15) * np.cos(s['base_yaw']),
            s['base_pos'][1] - (s['arm'] + 0.15) * np.sin(s['base_yaw']),
            s['lift'],
        ])
        return s

    def apply_action(self, action: np.ndarray, current_state: dict):
        """Apply an 8-dim delta action to the robot.

        action[0:2]  base translate / rotate  (scaled by SIM_BASE_SCALE → real metres/rad)
        action[2]    lift delta                (scaled by SIM_SCALE)
        action[3]    arm delta                 (scaled by SIM_ARM_SCALE * 4)
        action[4:7]  wrist yaw/pitch/roll delta (scaled by SIM_SCALE)
        action[7]    gripper absolute target   (mapped to stretch_gripper units)
        """
        # -- base (non-holonomic: translate=forward, rotate=yaw) --
        # action[0] → linear translation along heading (small steps)
        # action[1] → rotation
        translate_m = float(action[0]) * SIM_BASE_SCALE * 0.1   # reduce speed for safety
        rotate_rad  = float(action[1]) * SIM_BASE_SCALE * 0.1

        # -- arm joints (delta, clamped) --
        new_lift  = clamp(current_state['lift']  + float(action[2]) * SIM_SCALE,
                          *REAL_JOINT_LIMITS['lift'])
        new_arm   = clamp(current_state['arm']   + float(action[3]) * SIM_SCALE,
                          *REAL_JOINT_LIMITS['arm'])
        new_wy    = clamp(current_state['wrist_yaw']   + float(action[4]) * SIM_SCALE,
                          *REAL_JOINT_LIMITS['wrist_yaw'])
        new_wp    = clamp(current_state['wrist_pitch'] + float(action[5]) * SIM_SCALE,
                          *REAL_JOINT_LIMITS['wrist_pitch'])
        new_wr    = clamp(current_state['wrist_roll']  + float(action[6]) * SIM_SCALE,
                          *REAL_JOINT_LIMITS['wrist_roll'])

        # -- gripper absolute (action[7] in [-1,1] → stretch units [-50, 50]) --
        new_grip  = clamp(float(np.interp(action[7], [-1.0, 1.0], [-50.0, 50.0])),
                          *REAL_JOINT_LIMITS['gripper'])

        if self.dry_run:
            print(f"  [DRY] translate={translate_m:+.3f}m  rotate={rotate_rad:+.3f}rad  "
                  f"lift={new_lift:.3f}  arm={new_arm:.3f}  "
                  f"wy={new_wy:.3f}  wp={new_wp:.3f}  wr={new_wr:.3f}  "
                  f"grip={new_grip:.1f}")
            # Update fake state
            self._fake.update({'lift': new_lift, 'arm': new_arm,
                               'wrist_yaw': new_wy, 'wrist_pitch': new_wp,
                               'wrist_roll': new_wr, 'gripper': new_grip})
            return

        r = self.robot
        if abs(translate_m) > 1e-4:
            r.base.translate_by(translate_m)
        if abs(rotate_rad) > 1e-4:
            r.base.rotate_by(rotate_rad)
        r.lift.move_to(new_lift)
        r.arm.move_to(new_arm)
        r.end_of_arm.move_to('wrist_yaw',   new_wy)
        r.end_of_arm.move_to('wrist_pitch', new_wp)
        r.end_of_arm.move_to('wrist_roll',  new_wr)
        r.end_of_arm.move_to('stretch_gripper', new_grip)
        r.push_command()

    def stow(self):
        if not self.dry_run:
            self.robot.stow()
            self.robot.push_command()

    def stop(self):
        if not self.dry_run:
            self.robot.stop()


# ── Object position source ────────────────────────────────────────────────────

class StaticObjectSource:
    """Returns a fixed object position (for debugging without perception)."""
    def __init__(self, xyz):
        self._pos = np.array(xyz, dtype=np.float64)

    def get_position(self) -> np.ndarray:
        return self._pos.copy()


class ROS2ObjectSource:
    """Subscribes to a PoseStamped ROS2 topic and returns the latest position.

    This is a lightweight subscriber that doesn't require HelloNode.
    Run your lab3 object detector node first:
        ros2 run <your_pkg> object_detector_node
    """
    def __init__(self, topic: str = '/object_detector/goal_pose'):
        import rclpy
        from geometry_msgs.msg import PoseStamped

        rclpy.init()
        self._node = rclpy.create_node('deploy_grasp_listener')
        self._pos  = None
        self._sub  = self._node.create_subscription(
            PoseStamped, topic, self._cb, 1
        )
        print(f"[ROS2ObjectSource] subscribing to {topic}")

    def _cb(self, msg):
        self._pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def get_position(self) -> np.ndarray | None:
        import rclpy
        rclpy.spin_once(self._node, timeout_sec=0.05)
        return self._pos.copy() if self._pos is not None else None

    def shutdown(self):
        import rclpy
        self._node.destroy_node()
        rclpy.shutdown()


# ── Main control loop ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='Path to trained policy (best_policy.pt)')
    p.add_argument('--obj-pos', nargs=3, type=float, default=None,
                   metavar=('X', 'Y', 'Z'),
                   help='Static object position in world frame (for testing without perception)')
    p.add_argument('--ros-object-topic', default=None,
                   help='ROS2 PoseStamped topic for live object position (e.g. /object_detector/goal_pose)')
    p.add_argument('--table-height', type=float, default=0.85,
                   help='Table surface z in world frame (metres)')
    p.add_argument('--max-steps', type=int, default=200,
                   help='Maximum control steps before giving up')
    p.add_argument('--dry-run', action='store_true',
                   help='Simulate hardware commands without moving the robot')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Policy ──────────────────────────────────────────────────────────────
    runner = PolicyRunner(args.checkpoint, device=args.device)

    # ── Robot ───────────────────────────────────────────────────────────────
    ctrl = StretchController(dry_run=args.dry_run)

    # ── Object position source ───────────────────────────────────────────────
    if args.ros_object_topic:
        obj_src = ROS2ObjectSource(topic=args.ros_object_topic)
    elif args.obj_pos:
        obj_src = StaticObjectSource(args.obj_pos)
    else:
        print("ERROR: provide --obj-pos or --ros-object-topic")
        sys.exit(1)

    # ── Safety: stow first ──────────────────────────────────────────────────
    print("[deploy] Stowing robot before starting...")
    ctrl.stow()
    time.sleep(2.0)

    dt = 1.0 / CONTROL_HZ
    grasped = False

    print(f"[deploy] Starting control loop ({args.max_steps} steps @ {CONTROL_HZ} Hz)")
    for step in range(args.max_steps):
        t_start = time.time()

        # 1. Get object position
        obj_pos = obj_src.get_position()
        if obj_pos is None:
            print(f"[deploy] step {step}: waiting for object position...")
            time.sleep(dt)
            continue

        # 2. Build observation
        state = ctrl.get_state()
        obs   = build_obs(state, obj_pos, args.table_height)

        # 3. Policy inference
        action = runner.act(obs)

        # 4. Apply action
        ctrl.apply_action(action, state)

        # 5. Check termination: object lifted > 5 cm above table
        obj_above = float(obj_pos[2] - args.table_height)
        in_contact = state['in_contact']
        if in_contact and obj_above > 0.05:
            print(f"\n[deploy] SUCCESS at step {step}: object lifted {obj_above*100:.1f} cm above table!")
            grasped = True
            break

        elapsed = time.time() - t_start
        sleep   = max(0.0, dt - elapsed)
        print(f"step {step:3d}  dist_lift={obj_above:+.3f}m  contact={in_contact:.0f}  "
              f"action_norm={np.linalg.norm(action):.3f}  ({elapsed*1000:.0f}ms + sleep {sleep*1000:.0f}ms)",
              end='\r')
        time.sleep(sleep)

    if not grasped:
        print(f"\n[deploy] Episode ended after {args.max_steps} steps without success.")

    print("\n[deploy] Stowing robot.")
    ctrl.stow()
    ctrl.stop()

    if args.ros_object_topic:
        obj_src.shutdown()


if __name__ == '__main__':
    main()
