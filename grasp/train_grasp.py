"""train_grasp.py – PPO training for GraspEnv (Phase 1 of hierarchical pick-and-place).

Usage:
    conda run -n 16762 python train_grasp.py [--project MY_PROJECT] [--name MY_RUN]

Dependencies (install once):
    pip install wandb
"""

import argparse
import os
import sys
sys.path.insert(0, '/home/ye/mengine')

import gymnasium as gym

from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.env import EnvFactoryRegistered, VectorEnvType
from tianshou.highlevel.experiment import ExperimentConfig, PPOExperimentBuilder
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.highlevel.params.algorithm_params import PPOParams
from tianshou.highlevel.trainer import (
    EpochStopCallback,
    EpochStopCallbackRewardThreshold,
    TrainingContext,
)

from grasp_env import GraspEnv  # registers 'GraspEnv' with gymnasium  # noqa: F401


# ── Training configuration ────────────────────────────────────────────────────
# Sized for a machine with an RTX 4090 or 5090.
# PyBullet simulation runs on CPU; the GPU handles neural-net forward/backward.
# Levers for GPU utilisation:
#   - hidden_sizes:               larger networks → more compute per forward pass
#   - batch_size:                 larger batches → better GPU memory bandwidth
#   - update_step_num_repetitions more SGD steps per collected batch
#   - num_training_envs:          more parallel CPU envs → faster data collection

TRAINING_CONFIG = OnPolicyTrainingConfig(
    max_epochs=1000,
    # Collect 8192 steps per epoch across 32 envs (~256 steps each before update)
    epoch_num_steps=8192,
    collection_step_num_env_steps=8192,
    num_training_envs=32,   # uses SubprocVectorEnv → 32 independent pybullet servers
    num_test_envs=4,
    test_in_training=True,
    buffer_size=8192,
    batch_size=512,         # larger batch for GPU
    update_step_num_repetitions=15,
)

PPO_PARAMS = PPOParams(
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    eps_clip=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,      # gradient clipping stabilises training
    action_bound_method='clip',
    action_scaling=True,
    advantage_normalization=True,
    recompute_advantage=False,
)

# Larger network than lab5's (64,64) – benefits from GPU for forward/backward
HIDDEN_SIZES = (256, 256)

# Stop early once mean test reward exceeds this threshold.
# Approximate target: agent reliably lifts the object (r_lift ≥ 20 most steps).
STOP_REWARD_THRESHOLD = 180.0


# ── Custom epoch callback: extra metrics to wandb ────────────────────────────

class EpochStopWithLogging(EpochStopCallback):
    """Wraps a reward threshold and prints per-epoch info to stdout."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def should_stop(self, mean_rewards: float, context: TrainingContext) -> bool:
        stop = mean_rewards >= self._threshold
        if stop:
            print(f"\n[EarlyStop] mean_reward={mean_rewards:.2f} >= {self._threshold}")
        return stop


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='16762-robot-rl',
                   help='W&B project name')
    p.add_argument('--name', default='GraspPhase1',
                   help='W&B run / experiment name')
    p.add_argument('--log-dir', default='log',
                   help='Directory for checkpoints and logs')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    experiment = (
        PPOExperimentBuilder(
            EnvFactoryRegistered(
                task='GraspEnv',
                venv_type=VectorEnvType.SUBPROC,
                training_seed=args.seed,
                test_seed=args.seed + 1000,
            ),
            ExperimentConfig(
                persistence_enabled=True,
                persistence_base_dir=args.log_dir,
                watch=False,           # set True to render a live episode periodically
                watch_render=1 / 35,
                watch_num_episodes=5,
            ),
            TRAINING_CONFIG,
        )
        .with_ppo_params(PPO_PARAMS)
        .with_actor_factory_default(hidden_sizes=HIDDEN_SIZES)
        .with_critic_factory_default(hidden_sizes=HIDDEN_SIZES)
        # ── wandb logger (single call replaces TensorBoard) ──────────────────
        .with_logger_factory(
            LoggerFactoryDefault(
                logger_type='wandb',
                wandb_project=args.project,
                save_interval=1,
            )
        )
        # ── name the run so checkpoints go to log/<name>/ ────────────────────
        .with_name(args.name)
        # ── early stopping ───────────────────────────────────────────────────
        .with_epoch_stop_callback(EpochStopWithLogging(STOP_REWARD_THRESHOLD))
        .build()
    )

    print("=" * 60)
    print(f"GraspEnv Phase-1 PPO training")
    print(f"  W&B project : {args.project}")
    print(f"  run name    : {args.name}")
    print(f"  log dir     : {args.log_dir}")
    print(f"  train envs  : {TRAINING_CONFIG.num_training_envs}")
    print(f"  hidden sizes: {HIDDEN_SIZES}")
    print(f"  batch size  : {TRAINING_CONFIG.batch_size}")
    print(f"  max epochs  : {TRAINING_CONFIG.max_epochs}")
    print("=" * 60)

    experiment.run()


if __name__ == '__main__':
    main()
