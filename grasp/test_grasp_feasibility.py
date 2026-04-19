"""test_grasp_feasibility.py — 验证瓶子在仿真环境中是否可以被抓取并提起。

运行方法:
    # 无界面（快速检查）
    conda run -n 16762 python test_grasp_feasibility.py

    # 有界面（可视化观察）
    conda run -n 16762 python test_grasp_feasibility.py --render

测试分三个阶段:
    1. 场景诊断  — 打印 reset 后 EE / 物体 / 距离等关键数值
    2. 强制锁定  — 跳过 Phase A，直接把 gripper 锁定，看 Phase B 能否提起物体
    3. 手动靠近  — 用脚本策略把 EE 移到物体附近，让触发条件自然触发
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mengine')))

import numpy as np
import pybullet as p

# ── rliable mock（和 train_grasp.py 一样的 workaround）────────────────────────
import types
if 'rliable' not in sys.modules:
    _rly = types.ModuleType('rliable')
    _rly.library    = types.ModuleType('rliable.library')
    _rly.plot_utils = types.ModuleType('rliable.plot_utils')
    sys.modules['rliable']            = _rly
    sys.modules['rliable.library']    = _rly.library
    sys.modules['rliable.plot_utils'] = _rly.plot_utils

from grasp_env import GraspEnv  # noqa: F401


def separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── 阶段 1：场景诊断 ──────────────────────────────────────────────────────────
def test_scene_info(render: bool):
    separator("阶段 1 · 场景诊断（reset 后的初始状态）")
    env = GraspEnv(render_mode='human' if render else None)
    obs, _ = env.reset()

    ee_pos, _   = env.robot.get_link_pos_orient(env.robot.end_effector)
    obj_pos, _  = env.object.get_base_pos_orient()
    dist        = float(np.linalg.norm(np.array(ee_pos) - np.array(obj_pos)))
    table_h     = env.table_height

    print(f"  桌面高度        : {table_h:.3f} m")
    print(f"  物体位置 (world): {[round(v, 3) for v in obj_pos]}")
    print(f"  EE   位置 (world): {[round(v, 3) for v in ee_pos]}")
    print(f"  EE → 物体 距离  : {dist:.3f} m")
    print(f"  EE z 与桌面差   : {ee_pos[2] - table_h:+.3f} m  (0 = 桌面齐平)")
    print(f"  触发阈值        : {env.GRASP_DIST:.3f} m")
    print(f"  EE 是否在阈值内 : {'✓ 是' if dist < env.GRASP_DIST else '✗ 否，需要移动'}")

    ja = env.robot.get_joint_angles(env.robot.controllable_joints)
    print(f"\n  关节角 (controllable):")
    print(f"    lift         : {ja[env._IDX_LIFT]:.3f}")
    print(f"    arm sum      : {ja[env._IDX_ARM_START:env._IDX_ARM_END].sum():.3f}")
    print(f"    wrist [0]    : {ja[env._IDX_WRIST][0]:.3f}")
    print(f"    gripper L/R  : {ja[env._IDX_GRIP_L]:.3f} / {ja[env._IDX_GRIP_R]:.3f}")

    env.close()
    return dist


# ── 阶段 2：强制锁定，直接测 Phase B ─────────────────────────────────────────
def test_force_lock(render: bool, n_steps: int = 40):
    separator("阶段 2 · 强制锁定（跳过 Phase A，直接测 Phase B 提升）")
    env = GraspEnv(render_mode='human' if render else None)
    env.reset()

    # 将 EE 大致移到物体高度（模拟已经靠近）再强制锁定
    # ──── 先执行几步让关节稳定 ──────────────────────────────────────────────
    for _ in range(5):
        env.step(np.zeros(5))

    # 强制触发抓取
    env._gripper_locked  = True
    env._lock_obj_above  = 0.0
    env._locked_steps    = 0

    print(f"  已强制锁定 gripper，开始观察 {n_steps} 步 Phase B:")
    print(f"  {'step':>4}  {'obj_above(m)':>12}  {'reward':>8}  {'false_grasp':>11}  {'success':>7}")
    print(f"  {'-'*54}")

    max_rise = 0.0
    for i in range(n_steps):
        obs, rew, term, trunc, info = env.step(np.zeros(5))
        above = info['obj_lift_m']
        max_rise = max(max_rise, above)
        print(f"  {i:4d}  {above:12.4f}  {rew:8.3f}  {info['false_grasp']:11.0f}  {info['success']:7.0f}")
        if term or trunc:
            break

    env.close()
    print(f"\n  ▶ 最大提升高度: {max_rise:.4f} m  (目标: {GraspEnv.LIFT_TARGET_M:.2f} m)")
    if max_rise > GraspEnv.LIFT_VERIFY_MIN:
        print("  ✓ Phase B 物体有上升 —— 夹抓在此位置可以提起物体")
    else:
        print("  ✗ Phase B 物体没有上升 —— 夹爪在 reset 默认位置无法抓到物体")
        print("    可能原因: EE 初始高度不在物体旁边，夹爪闭合后没有包住物体")

    return max_rise


# ── 阶段 3：传送机器人底座到瓶子前方，测 Phase B ──────────────────────────────
def test_teleport_grasp(render: bool, n_steps: int = 60):
    """最关键的测试：直接用 pybullet 把机器人底座传送到瓶子正前方，
    再设好 lift/arm 关节，然后验证 Phase B 能否提起物体。
    """
    separator("阶段 3 · 传送测试（机器人底座传送到瓶前，测 Phase B）")
    import mengine as m

    env = GraspEnv(render_mode='human' if render else None)
    env.reset()

    obj_pos, _ = env.object.get_base_pos_orient()
    robot      = env.robot
    client     = env.env.id
    print(f"  物体位置: {[round(v, 3) for v in obj_pos]}")

    # ── Step 1: 把机器人底座传送到物体旁边 ──────────────────────────────
    # Stretch3 arm 在 yaw=0 时向 -y 延伸（已通过实验验证）
    # → 要让 arm 朝向 -x（物体方向），需要 robot yaw = -π/2
    # 此时 arm 延伸方向 = -x，EE 在 arm=0 时偏移约 (-0.415, +0.021, lift_offset)
    # → 底座放在 obj_x + 0.415 + arm_ext, obj_y 处
    obj_x, obj_y = float(obj_pos[0]), float(obj_pos[1])
    yaw_target = -np.pi / 2        # arm 朝向 -x 方向（指向物体）

    ARM_EXT_TARGET = 0.20          # 目标 arm 延伸量（每段 0.05m，共 4 段）
    ARM_EXT_PER    = ARM_EXT_TARGET / 4.0
    ARM_ZERO_OFFSET = 0.415        # arm=0 时 EE 距底座在 arm 方向的距离（实测）

    # 底座 x = obj_x + (arm=0 偏移) + arm 延伸量
    base_x = obj_x + ARM_ZERO_OFFSET + ARM_EXT_TARGET
    base_y = obj_y   # arm 方向的侧偏约 0.02m，忽略

    base_quat = p.getQuaternionFromEuler([0, 0, yaw_target], physicsClientId=client)
    p.resetBasePositionAndOrientation(
        robot.body, [base_x, base_y, 0], base_quat, physicsClientId=client,
    )

    # ── Step 2: 用 resetJointState 直接强制设置关节（绕过电机收敛问题）──────
    # 先粗略设一个 lift，然后通过 EE 位置反馈微调
    ja = robot.get_joint_angles(robot.controllable_joints)

    # 先设 arm 延伸，步进仿真，读取真实 EE 高度
    for i, jidx in enumerate(robot.controllable_joints):
        val = ja[i]
        if env._IDX_ARM_START <= i < env._IDX_ARM_END:
            val = ARM_EXT_PER
        if i == env._IDX_WRIST.start:
            val = 0.0
        p.resetJointState(robot.body, jidx, val, physicsClientId=client)

    m.step_simulation(steps=5, realtime=False)

    # 读 EE 真实位置，反推 lift 目标
    ee_tmp, _ = robot.get_link_pos_orient(robot.end_effector)
    ja_tmp    = robot.get_joint_angles(robot.controllable_joints)
    lift_now  = ja_tmp[env._IDX_LIFT]
    lift_z_offset = ee_tmp[2] - lift_now          # 真实 offset（含 arm 延伸）
    target_lift   = float(np.clip(obj_pos[2] - lift_z_offset, 0.05, 1.05))

    print(f"  lift_z_offset={lift_z_offset:.3f}  target_lift={target_lift:.3f}")

    # 再次 resetJointState 设置正确 lift
    p.resetJointState(
        robot.body,
        robot.controllable_joints[env._IDX_LIFT],
        target_lift,
        physicsClientId=client,
    )
    m.step_simulation(steps=10, realtime=False)

    # ── Step 3: 检查对齐效果 ─────────────────────────────────────────────
    ee_pos1, _ = robot.get_link_pos_orient(robot.end_effector)
    dist_after = float(np.linalg.norm(np.array(ee_pos1) - np.array(obj_pos)))
    print(f"  传送后机器人底座: ({base_x:.3f}, {base_y:.3f}, 0)")
    print(f"  传送后 EE z={ee_pos1[2]:.3f}  物体 z={obj_pos[2]:.3f}  Δz={ee_pos1[2]-obj_pos[2]:+.3f}")
    print(f"  传送后 EE→物体:  {dist_after:.3f} m  (触发阈值 {env.GRASP_DIST:.3f} m)")

    if dist_after > 0.25:
        print("  ⚠ EE 距离仍较远，arm 延伸量可能需要调整")

    # ── Step 4: 强制闭合夹爪并锁定 ──────────────────────────────────────
    # 直接 reset 夹爪关节到 closed
    p.resetJointState(robot.body, robot.controllable_joints[env._IDX_GRIP_L],
                      GraspEnv.GRIPPER_CLOSE, physicsClientId=client)
    p.resetJointState(robot.body, robot.controllable_joints[env._IDX_GRIP_R],
                      GraspEnv.GRIPPER_CLOSE, physicsClientId=client)
    m.step_simulation(steps=20, realtime=False)

    env._gripper_locked  = True
    env._lock_obj_above  = 0.0
    env._locked_steps    = 0
    env._prev_obj_above  = float(obj_pos[2] - env._obj_init_z)

    print(f"\n  开始 {n_steps} 步 Phase B:")
    print(f"  {'step':>4}  {'obj_z(abs)':>10}  {'obj_above':>10}  {'EE_z':>7}  {'false':>5}  {'ok':>5}")
    print(f"  {'-'*55}")

    max_rise = 0.0
    for i in range(n_steps):
        obs, rew, term, trunc, info = env.step(np.zeros(5))
        ee_now, _ = env.robot.get_link_pos_orient(env.robot.end_effector)
        obj_now, _ = env.object.get_base_pos_orient()
        above = info['obj_lift_m']
        max_rise = max(max_rise, above)
        print(f"  {i:4d}  {obj_now[2]:10.4f}  {above:10.4f}  {ee_now[2]:7.3f}  "
              f"{info['false_grasp']:5.0f}  {info['success']:5.0f}")
        if term:
            break

    env.close()

    print(f"\n  ▶ 最大提升高度: {max_rise:.4f} m  (目标: {GraspEnv.LIFT_TARGET_M:.2f} m)")
    if max_rise >= GraspEnv.LIFT_TARGET_M:
        print("  ✓ 完全成功！物理上瓶子可以被抓取并提升")
    elif max_rise > 0.02:
        print(f"  △ 部分成功（提升了 {max_rise:.3f}m），夹取有效但可能需要微调阈值或位置")
    else:
        print("  ✗ 物体没有上升")
        print("    → 可能是 PyBullet 中夹爪对该形状摩擦不足，或质量/重力比不合适")

    return max_rise


# ── 阶段 4：检查 lift 关节范围 ────────────────────────────────────────────────
def test_joint_range(render: bool):
    separator("阶段 4 · 关节范围检查")
    import pybullet as p

    env = GraspEnv(render_mode='human' if render else None)
    env.reset()

    robot  = env.robot
    joints = robot.controllable_joints
    bid    = robot.body

    print(f"  controllable_joints indices: {joints}")
    print(f"  {'idx':>4}  {'joint_name':30}  {'type':10}  {'lower':>8}  {'upper':>8}  {'current':>8}")
    print(f"  {'-'*72}")

    ja = robot.get_joint_angles(joints)
    for i, jidx in enumerate(joints):
        info_j = p.getJointInfo(bid, jidx, physicsClientId=env.env.id)
        name   = info_j[1].decode()
        jtype  = ['revolute','prismatic','spherical','planar','fixed'][info_j[2]] if info_j[2] < 5 else str(info_j[2])
        lo, hi = info_j[8], info_j[9]
        print(f"  {jidx:4d}  {name:30}  {jtype:10}  {lo:8.3f}  {hi:8.3f}  {ja[i]:8.3f}")

    env.close()


# ── 主入口 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='开启 PyBullet 可视化界面')
    parser.add_argument('--phase', type=int, default=0,
                        help='只运行某一阶段: 1=场景诊断, 2=强制锁定, 3=脚本靠近, 0=全部(默认)')
    args = parser.parse_args()

    if args.phase in (0, 1):
        test_scene_info(args.render)

    if args.phase in (0, 2):
        test_force_lock(args.render)

    if args.phase in (0, 3):
        test_teleport_grasp(args.render)

    if args.phase in (0, 4):
        test_joint_range(args.render)

    print("\n测试完成。")
