# 16762 Robot Autonomy – Course Project

Hierarchical reinforcement learning for robot pick-and-place on the Stretch3 platform.

## Project structure

```
16762-project/
└── grasp/
    ├── grasp_env.py      # Gymnasium environment (Phase 1: reach + grasp)
    ├── train_grasp.py    # PPO training with W&B logging
    ├── eval_grasp.py     # Policy evaluation (sim)
    └── deploy_real.py    # Real robot deployment via stretch_body
```

The project lives alongside the course labs:

```
16762-labs/
├── lab1/ … lab5/
└── project -> /home/ye/16762-project   (symlink)
```

## Environment setup

```bash
conda activate 16762
pip install wandb
```

All other dependencies (`gymnasium`, `tianshou`, `torch`, `mengine`) are already installed in the `16762` conda environment.

## Training (sim)

```bash
cd /home/ye/16762-project/grasp

# Login to W&B once (skip if already logged in)
wandb login

# Start training – logs to W&B project "16762-robot-rl"
conda run -n 16762 python train_grasp.py --project 16762-robot-rl --name GraspPhase1
```

Checkpoints are saved under `log/GraspPhase1/seed_0/`.

## Evaluation (sim)

```bash
conda run -n 16762 python eval_grasp.py                     # latest checkpoint, with GUI
conda run -n 16762 python eval_grasp.py --no-render --n-episodes 100  # headless stats
```

## Deployment (real robot)

```bash
# Dry-run (no hardware required, manual object position)
python deploy_real.py --checkpoint log/GraspPhase1/seed_0/best_policy.pt \
                      --obj-pos -0.7 0.0 0.9 --dry-run

# Full run with live perception from lab3 object detector
python deploy_real.py --checkpoint log/GraspPhase1/seed_0/best_policy.pt \
                      --ros-object-topic /object_detector/goal_pose
```

Make sure `stretch_driver` is running before the full run:

```bash
ros2 launch stretch_core stretch_driver.launch.py
```

## Notes

- The simulation uses `gravity = [0, 0, -1]` (not real gravity) to ease training. For better sim-to-real transfer, retrain with `gravity = [0, 0, -9.81]` and domain randomisation on object mass / friction.
- The deployment script approximates end-effector position from joint angles. For higher accuracy, replace `build_obs()` with a proper FK call via `ikpy` or the ROS TF tree (see `lab3/ik_ros_utils.py`).
