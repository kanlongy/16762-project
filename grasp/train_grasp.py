"""train_grasp.py – PPO training for GraspEnv (Phase 1 of hierarchical pick-and-place).

Usage:
    conda run -n 16762 python train_grasp.py
    conda run -n 16762 python train_grasp.py --project 16762-robot-rl --name GraspPhase1
    conda run -n 16762 python train_grasp.py --save-every 50 --record-video

Dependencies (install once):
    pip install wandb
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mengine')))

import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.env import EnvFactoryRegistered, VectorEnvType
from tianshou.highlevel.experiment import ExperimentConfig, PPOExperimentBuilder
from tianshou.highlevel.logger import LoggerFactory, TLogger
from tianshou.highlevel.params.algorithm_params import PPOParams
from tianshou.highlevel.trainer import (
    EpochTestCallback,
    EpochTrainCallback,
    TrainingContext,
)
from tianshou.utils import WandbLogger

from grasp_env import GraspEnv  # registers 'GraspEnv' with gymnasium  # noqa: F401


# ── Training configuration ────────────────────────────────────────────────────

EPOCH_STEPS = 8192   # env steps collected per epoch (across all envs)

TRAINING_CONFIG = OnPolicyTrainingConfig(
    max_epochs=1000,
    epoch_num_steps=EPOCH_STEPS,
    collection_step_num_env_steps=EPOCH_STEPS,
    num_training_envs=32,
    num_test_envs=4,
    test_in_training=True,
    buffer_size=EPOCH_STEPS,
    batch_size=512,
    update_step_num_repetitions=15,
)

PPO_PARAMS = PPOParams(
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    eps_clip=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    action_bound_method='clip',
    action_scaling=True,
    advantage_normalization=True,
    recompute_advantage=False,
)

HIDDEN_SIZES = (256, 256)


# ── Custom W&B logger factory ─────────────────────────────────────────────────
# LoggerFactoryDefault only passes save_interval to WandbLogger and leaves
# training_interval=1000, update_interval=1000 (the defaults).  With only 240
# gradient updates per epoch, update_interval=1000 means losses never appear in
# W&B during a short run.  We set both intervals to 1 so every log call goes
# through immediately, giving clean per-epoch curves.

class WandbLoggerFactory(LoggerFactory):
    def __init__(self, project: str) -> None:
        self._project = project

    def create_logger(
        self,
        log_dir: str,
        experiment_name: str,
        run_id: str | None,
        config_dict: dict | None = None,
    ) -> TLogger:
        logger = WandbLogger(
            training_interval=1,   # log every env-step call (effectively once per epoch)
            test_interval=1,
            update_interval=1,     # log every gradient-step call
            project=self._project,
            name=experiment_name.replace(os.path.sep, '__'),
            run_id=run_id,
            config=config_dict,
            log_dir=log_dir,
        )
        # TensorBoard writer must be created AFTER wandb.init() so sync_tensorboard works
        writer = SummaryWriter(log_dir)
        writer.add_text('config', str(config_dict))
        logger.load(writer)
        return logger

    def get_logger_class(self) -> type[TLogger]:
        return WandbLogger


# ── Callbacks ─────────────────────────────────────────────────────────────────


class CheckpointCallback(EpochTrainCallback):
    """Save a dated policy snapshot every N epochs (separate from best_policy.pt)."""

    def __init__(self, every_n: int, save_dir: str) -> None:
        self._every_n  = every_n
        self._save_dir = save_dir

    def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
        if epoch % self._every_n != 0:
            return
        os.makedirs(self._save_dir, exist_ok=True)
        path = os.path.join(self._save_dir, f'epoch_{epoch:04d}.pt')
        torch.save(context.algorithm, path)
        print(f"\n[Checkpoint] epoch {epoch:4d} → {path}")


class VideoLogCallback(EpochTestCallback):
    """Record one evaluation episode and upload to W&B every N epochs.

    Uses pybullet's ER_TINY_RENDERER (CPU, no display/GPU required).
    Skips silently if wandb is not active or recording fails.
    """

    def __init__(self, every_n: int = 10) -> None:
        self._every_n = every_n

    def callback(self, epoch: int, env_step: int | None, context: TrainingContext) -> None:
        if epoch % self._every_n != 0:
            return
        try:
            import wandb
            from tianshou.data import Batch

            env = gym.make('GraspEnv', render_mode='rgb_array')
            obs, _ = env.reset()
            frames: list[np.ndarray] = []

            # Get the underlying policy for inference
            alg    = context.algorithm
            policy = alg.policy if hasattr(alg, 'policy') else alg
            policy.eval()

            for _ in range(200):
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                with torch.no_grad():
                    result = policy(Batch(obs=obs[np.newaxis], info={}))
                    action = np.asarray(result.act[0])
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            env.close()

            if frames and wandb.run is not None:
                # wandb.Video expects (T, C, H, W)
                video = np.stack(frames).transpose(0, 3, 1, 2)
                wandb.log(
                    {'eval/video': wandb.Video(video, fps=15, format='mp4')},
                    step=env_step,
                )
                print(f"\n[VideoLog] epoch {epoch}: uploaded {len(frames)}-frame video to W&B")
        except Exception as e:
            print(f"\n[VideoLog] skipped (epoch {epoch}): {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project',  default='16762-robot-rl', help='W&B project name')
    p.add_argument('--name',     default='GraspPhase1',    help='W&B run / experiment name')
    p.add_argument('--log-dir',  default='log',            help='Base directory for logs & checkpoints')
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--save-every', type=int, default=10,
                   help='Save a dated checkpoint snapshot every N epochs (default: 50)')
    p.add_argument('--record-video', action='store_true',
                   help='Upload an evaluation video to W&B periodically')
    p.add_argument('--video-every', type=int, default=10,
                   help='Upload video every N epochs when --record-video is set (default: 50)')
    return p.parse_args()


def main():
    args   = parse_args()
    ckpt_dir = os.path.join(args.log_dir, args.name, 'snapshots')

    builder = (
        PPOExperimentBuilder(
            EnvFactoryRegistered(
                task='GraspEnv',
                venv_type=VectorEnvType.SUBPROC,
                training_seed=args.seed,
                test_seed=args.seed + 1000,
            ),
            ExperimentConfig(
                persistence_enabled=True,      # saves best_policy.pt automatically
                persistence_base_dir=args.log_dir,
                watch=False,
                watch_render=1 / 35,
                watch_num_episodes=5,
            ),
            TRAINING_CONFIG,
        )
        .with_ppo_params(PPO_PARAMS)
        .with_actor_factory_default(hidden_sizes=HIDDEN_SIZES)
        .with_critic_factory_default(hidden_sizes=HIDDEN_SIZES)
        # ── W&B logger ────────────────────────────────────────────────────────
        # Custom factory: sets training_interval=1 and update_interval=1 so
        # every log call is forwarded to W&B immediately → clean epoch curves.
        .with_logger_factory(WandbLoggerFactory(project=args.project))
        .with_name(args.name)
        # ── No early stopping: always run all max_epochs ──────────────────────
        # ── Periodic snapshot checkpoints ─────────────────────────────────────
        .with_epoch_train_callback(CheckpointCallback(every_n=args.save_every, save_dir=ckpt_dir))
    )

    # ── Optional video recording ──────────────────────────────────────────────
    if args.record_video:
        builder = builder.with_epoch_test_callback(
            VideoLogCallback(every_n=args.video_every)
        )

    experiment = builder.build()

    print("=" * 60)
    print("GraspEnv Phase-1 PPO training")
    print(f"  W&B project  : {args.project}")
    print(f"  run name     : {args.name}")
    print(f"  log dir      : {args.log_dir}")
    print(f"  train envs   : {TRAINING_CONFIG.num_training_envs}")
    print(f"  hidden sizes : {HIDDEN_SIZES}")
    print(f"  batch size   : {TRAINING_CONFIG.batch_size}")
    print(f"  max epochs   : {TRAINING_CONFIG.max_epochs}")
    print(f"  snapshot every {args.save_every} epochs → {ckpt_dir}")
    print(f"  video logging: {'on (every %d epochs)' % args.video_every if args.record_video else 'off'}")
    print("=" * 60)

    experiment.run(raise_error_on_dirname_collision=False)


if __name__ == '__main__':
    main()
