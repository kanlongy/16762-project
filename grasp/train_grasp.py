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
import types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mengine')))

# tianshou imports rliable → arch → statsmodels.deprecate_kwarg at module load
# time, which breaks on some machines due to statsmodels API differences.
# We don't use rliable anywhere in training, so mock it out before importing tianshou.
if 'rliable' not in sys.modules:
    _rly = types.ModuleType('rliable')
    _rly.library    = types.ModuleType('rliable.library')
    _rly.plot_utils = types.ModuleType('rliable.plot_utils')
    sys.modules['rliable']            = _rly
    sys.modules['rliable.library']    = _rly.library
    sys.modules['rliable.plot_utils'] = _rly.plot_utils

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
    EpochStopCallback,
    EpochTestCallback,
    EpochTrainCallback,
    TrainingContext,
)
import wandb as _wandb
from tianshou.utils import WandbLogger
from tianshou.utils.logger.logger_base import VALID_LOG_VALS_TYPE

from grasp_env import GraspEnv  # registers 'GraspEnv' with gymnasium  # noqa: F401


# ── Training configuration ────────────────────────────────────────────────────

# Each epoch collects this many env steps across all parallel envs.
# Doubled from 8192 → 16384 so each update epoch sees ~54 full episodes
# (vs ~27 before), reducing gradient variance substantially.
EPOCH_STEPS = 16384

TRAINING_CONFIG = OnPolicyTrainingConfig(
    # 500 epochs × 16384 steps = ~8.2M total env steps.
    # Contact-rich grasping with a sparse success bonus typically needs 5–10M;
    # the previous 2M budget was too low for the task to converge.
    max_epochs=500,
    epoch_num_steps=EPOCH_STEPS,
    collection_step_num_env_steps=EPOCH_STEPS,
    num_training_envs=32,
    num_test_envs=4,
    test_in_training=True,
    buffer_size=EPOCH_STEPS,
    batch_size=512,
    # Reduced from 15 → 8: 8 repetitions × (16384/512) = 256 gradient updates
    # per epoch, which is already aggressive for PPO.  15 repetitions on the
    # old 8192-step buffer caused 240 updates from nearly identical mini-batches,
    # pushing the policy off the on-policy manifold and hurting stability.
    update_step_num_repetitions=8,
)

PPO_PARAMS = PPOParams(
    lr=3e-4,
    # Increased from 0.99 → 0.995.  With max_episode_steps=300, a reward at
    # the final step is discounted to 0.99^299 ≈ 0.05 at step 0 — the sparse
    # r_success=+100 bonus is nearly invisible at the start of an episode.
    # gamma=0.995 raises that to 0.995^299 ≈ 0.22, making long-horizon credit
    # assignment practical without destabilising value estimation.
    gamma=0.995,
    gae_lambda=0.95,
    eps_clip=0.2,
    # Increased from 0.01 → 0.02.  The bilateral-contact + lift-verify lock
    # conditions create a narrow success funnel in a 6-D action space; extra
    # entropy regularisation helps the policy keep exploring approach directions
    # rather than collapsing to a single sub-optimal routine early in training.
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    action_bound_method='clip',
    action_scaling=True,
    advantage_normalization=True,
    recompute_advantage=False,
)

HIDDEN_SIZES = (256, 256)


# ── Custom W&B logger ─────────────────────────────────────────────────────────
# Root cause of missing metrics: WandbLogger.write() only writes to TensorBoard
# and relies on sync_tensorboard=True patching to forward data to W&B. This
# breaks when wandb's dir and the TensorBoard SummaryWriter dir are the same
# folder: wandb's file-watcher finds the tfevents file, copies it into
# <log_dir>/wandb/run-xxx/files/, finds the copy, copies it again, ad infinitum.
# The recursive file loop saturates wandb's internal pipeline and no metrics
# ever reach the dashboard (confirmed by an empty wandb-summary.json after 2h).
#
# Fix A: point wandb's own dir one level above log_dir so the wandb/ subfolder
#         is not inside the TensorBoard watch tree.
# Fix B: subclass WandbLogger and override write() to ALSO call wandb.log()
#         directly, bypassing sync_tensorboard entirely for metric delivery.

class _DirectWandbLogger(WandbLogger):
    """WandbLogger that calls wandb.log() directly in write(), bypassing sync_tensorboard."""

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        super().write(step_type, step, data)
        if _wandb.run is not None:
            scope = step_type.split("/")[0]
            _wandb.log({f"{scope}/{k}": v for k, v in data.items()}, step=step)


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
        # Fix A: wandb stores its own files one level above log_dir so the
        # wandb/ subfolder is never inside the TensorBoard directory, preventing
        # the infinite recursive file-copy loop.
        wandb_dir = os.path.dirname(os.path.abspath(log_dir))
        os.makedirs(wandb_dir, exist_ok=True)
        logger = _DirectWandbLogger(
            training_interval=1,
            test_interval=1,
            update_interval=1,
            project=self._project,
            name=experiment_name.replace(os.path.sep, '__'),
            run_id=run_id,
            config=config_dict,
            log_dir=wandb_dir,
        )
        # TensorBoard writer must be created AFTER wandb.init()
        writer = SummaryWriter(log_dir)
        writer.add_text('config', str(config_dict))
        logger.load(writer)
        return logger

    def get_logger_class(self) -> type[TLogger]:
        return _DirectWandbLogger


# ── Callbacks ─────────────────────────────────────────────────────────────────


class BestRewardCallback(EpochStopCallback):
    """Tracks best mean reward and saves policy when a new best is reached.

    Satisfies the test_in_training=True requirement (never triggers early stop)
    while also maintaining a best_reward_policy.pt that always reflects the
    highest mean-reward checkpoint seen so far.
    """

    def __init__(self, save_dir: str) -> None:
        self._save_dir        = save_dir
        self._best_reward     = float('-inf')

    def should_stop(self, mean_rewards: float, context: TrainingContext) -> bool:
        if mean_rewards > self._best_reward:
            self._best_reward = mean_rewards
            os.makedirs(self._save_dir, exist_ok=True)
            path = os.path.join(self._save_dir, 'best_reward_policy.pt')
            torch.save(context.algorithm, path)
            print(f"\n[BestReward] new best {mean_rewards:.2f} → {path}")
        return False


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
                    act = result.act[0]
                    if hasattr(act, 'cpu'):
                        act = act.cpu()
                    action = np.asarray(act)
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
        # ── Stop callback: never stops, but saves best_reward_policy.pt ────────
        .with_epoch_stop_callback(BestRewardCallback(save_dir=ckpt_dir))
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
    print(f"  epoch steps  : {EPOCH_STEPS}  (total ≈ {EPOCH_STEPS * TRAINING_CONFIG.max_epochs // 1_000_000:.1f}M)")
    print(f"  hidden sizes : {HIDDEN_SIZES}")
    print(f"  batch size   : {TRAINING_CONFIG.batch_size}")
    print(f"  max epochs   : {TRAINING_CONFIG.max_epochs}")
    print(f"  gamma        : {PPO_PARAMS.gamma}  ent_coef: {PPO_PARAMS.ent_coef}")
    print(f"  snapshot every {args.save_every} epochs → {ckpt_dir}")
    print(f"  video logging: {'on (every %d epochs)' % args.video_every if args.record_video else 'off'}")
    print("=" * 60)

    experiment.run(raise_error_on_dirname_collision=False)


if __name__ == '__main__':
    main()
