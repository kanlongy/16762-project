"""train_grasp.py – PPO training for GraspEnv (pick-and-place, ablation-ready).

Usage:
    # Full task (Phase A+B+C) – default
    conda run -n 16762 python train_grasp.py --stage full

    # Stage-1 ablation: Phase A+B only (approach + grasp + lift)
    conda run -n 16762 python train_grasp.py --stage 1

    # Stage-2 ablation: Phase C only (transport + descend + place, pre-grasped start)
    conda run -n 16762 python train_grasp.py --stage 2

    # Custom run name / W&B project
    conda run -n 16762 python train_grasp.py --stage full --project 16762-robot-rl --name MyRun

    # Extra options
    conda run -n 16762 python train_grasp.py --stage 1 --save-every 20 --record-video

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
    max_epochs=100,
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
    ent_coef=0.03,
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

    Satisfies the test_in_training=True requirement (never triggers early stop).
    Saves two files on every improvement:
      • best_reward_policy.pt        – full algorithm object (for resume/eval)
      • best_reward_policy_sd.pt     – policy state dict only (lightweight eval)

    The final policy loaded at evaluation time should use best_reward_policy.pt,
    not the last checkpoint, to guarantee the best-seen performance.
    """

    def __init__(self, save_dir: str) -> None:
        self._save_dir    = save_dir
        self._best_reward = float('-inf')

    def should_stop(self, mean_rewards: float, context: TrainingContext) -> bool:
        if mean_rewards > self._best_reward:
            self._best_reward = mean_rewards
            os.makedirs(self._save_dir, exist_ok=True)

            # Full algorithm snapshot — use this to resume or evaluate.
            path_alg = os.path.join(self._save_dir, 'best_reward_policy.pt')
            torch.save(context.algorithm, path_alg)

            # Lightweight state-dict — easy to load for inference only.
            policy = (
                context.algorithm.policy
                if hasattr(context.algorithm, 'policy')
                else context.algorithm
            )
            path_sd = os.path.join(self._save_dir, 'best_reward_policy_sd.pt')
            torch.save(policy.state_dict(), path_sd)

            print(
                f"\n[BestReward] ★ new best {mean_rewards:.2f}"
                f" → {path_alg}  (also saved state dict)"
            )
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


class LoadActorCallback(EpochTrainCallback):
    """Transfer ACTOR weights from a Stage-1 checkpoint into the current policy.

    Fires once at the end of epoch 1, overwriting the freshly-initialized actor
    with the pre-trained weights.  The critic is intentionally NOT loaded because
    its value estimates are calibrated to Stage-1 rewards and would mislead PPO
    in the new reward landscape of Stage 2 / Full.

    Timing note: tianshou fires EpochTrainCallback at the END of each epoch
    (after gradient updates for that epoch).  Therefore epoch 1 will use random
    weights, and from epoch 2 onward the actor uses the loaded weights.  This
    one-epoch warm-up is negligible over a 100+ epoch run.

    Args:
        checkpoint_path: Path to a .pt file saved by BestRewardCallback.
            Accepts either the full algorithm object (best_reward_policy.pt)
            or the policy state dict (best_reward_policy_sd.pt).
        actor_only: If True (default), only the actor network is transferred.
            Set to False to also transfer the critic (rarely useful).
    """

    def __init__(self, checkpoint_path: str, actor_only: bool = True) -> None:
        self._path       = checkpoint_path
        self._actor_only = actor_only
        self._loaded     = False

    def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
        if self._loaded:
            return
        self._loaded = True

        raw = torch.load(self._path, map_location='cpu', weights_only=False)

        # Accept either a full algorithm object or a bare policy / state-dict.
        if isinstance(raw, dict):
            # It's a state dict (best_reward_policy_sd.pt)
            src_sd = raw
            if self._actor_only:
                src_sd = {k: v for k, v in src_sd.items() if k.startswith('actor.')}
            dst_alg    = context.algorithm
            dst_policy = dst_alg.policy if hasattr(dst_alg, 'policy') else dst_alg
            dst_policy.load_state_dict(src_sd, strict=False)
        else:
            # It's a full algorithm object (best_reward_policy.pt)
            src_policy = raw.policy if hasattr(raw, 'policy') else raw
            dst_alg    = context.algorithm
            dst_policy = dst_alg.policy if hasattr(dst_alg, 'policy') else dst_alg

            if self._actor_only:
                src_actor = src_policy.actor if hasattr(src_policy, 'actor') else src_policy
                dst_actor = dst_policy.actor if hasattr(dst_policy, 'actor') else dst_policy
                dst_actor.load_state_dict(src_actor.state_dict(), strict=False)
                what = 'actor'
            else:
                dst_policy.load_state_dict(src_policy.state_dict(), strict=False)
                what = 'actor+critic'

            print(
                f"\n[LoadActor] ★ epoch {epoch}: transferred {what} weights"
                f" from {self._path}"
            )
            return

        print(
            f"\n[LoadActor] ★ epoch {epoch}: transferred actor weights (state-dict)"
            f" from {self._path}"
        )


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
    p = argparse.ArgumentParser(
        description='PPO training for GraspEnv pick-and-place.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--project',  default='16762-robot-rl', help='W&B project name')
    p.add_argument('--name',     default=None,
                   help='W&B run / experiment name (default: auto from --stage)')
    p.add_argument('--log-dir',  default='log',  help='Base directory for logs & checkpoints')
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--save-every', type=int, default=10,
                   help='Save a dated checkpoint snapshot every N epochs (default: 10)')
    p.add_argument('--record-video', action='store_true',
                   help='Upload an evaluation video to W&B periodically')
    p.add_argument('--video-every', type=int, default=10,
                   help='Upload video every N epochs when --record-video is set (default: 10)')
    p.add_argument(
        '--stage',
        choices=['1', '2', 'full'],
        default='full',
        help=(
            'Curriculum stage to train (ablation control):\n'
            '  1    – Phase A+B only: approach + grasp + lift to LIFT_TARGET_M.\n'
            '         Episode ends as success when object is lifted high enough.\n'
            '         Env id: GraspEnvStage1  (max_episode_steps=300)\n'
            '  2    – Phase C only: transport + descend + place.\n'
            '         Episode starts pre-grasped at LIFT_TARGET_M height.\n'
            '         Env id: GraspEnvStage2  (max_episode_steps=400)\n'
            '  full – Complete A→B→C pick-and-place task (default).\n'
            '         Env id: GraspEnv         (max_episode_steps=500)'
        ),
    )
    p.add_argument('--max-epochs', type=int, default=None,
                   help='Override max training epochs (default: 100 / 120 / 200 for stage 1/2/full)')
    p.add_argument(
        '--load-from', default=None, metavar='CKPT',
        help=(
            'Path to a Stage-1 checkpoint to warm-start the actor.\n'
            'Accepts best_reward_policy.pt (full algorithm) or\n'
            '        best_reward_policy_sd.pt (state dict).\n'
            'Only the ACTOR network is transferred; the critic re-trains\n'
            'from scratch because Stage-2 reward distributions differ.\n'
            'Typical use: train Stage 1, then pass its best checkpoint\n'
            'when launching Stage 2 for curriculum-style training.'
        ),
    )
    p.add_argument(
        '--load-actor-critic', action='store_true',
        help='When --load-from is set, also transfer critic weights (not recommended).',
    )
    return p.parse_args()


def _make_train_callback(args, ckpt_dir: str) -> EpochTrainCallback:
    """Build the EpochTrainCallback to pass to the experiment builder.

    When --load-from is NOT set  →  CheckpointCallback only.
    When --load-from IS set      →  LoadActorCallback fires once at epoch 1,
                                    then CheckpointCallback fires every N epochs.
    Both are combined into a tiny composite so the builder only sees one object.
    """
    ckpt_cb = CheckpointCallback(every_n=args.save_every, save_dir=ckpt_dir)

    if not args.load_from:
        return ckpt_cb

    if not os.path.isfile(args.load_from):
        raise FileNotFoundError(f"--load-from: checkpoint not found: {args.load_from}")

    load_cb = LoadActorCallback(
        checkpoint_path=args.load_from,
        actor_only=not args.load_actor_critic,
    )

    class _Combined(EpochTrainCallback):
        def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
            load_cb.callback(epoch, env_step, context)
            ckpt_cb.callback(epoch, env_step, context)

    return _Combined()


def main():
    args = parse_args()

    # ── Stage-specific configuration ─────────────────────────────────────────
    _STAGE_CFG = {
        '1':    dict(task='GraspEnvStage1', default_name='GraspAblation_S1',
                     default_epochs=100, label='Stage 1  (Phase A+B: approach+grasp+lift)'),
        '2':    dict(task='GraspEnvStage2', default_name='GraspAblation_S2',
                     default_epochs=120, label='Stage 2  (Phase C: transport+place)'),
        'full': dict(task='GraspEnv',       default_name='GraspAblation_Full',
                     default_epochs=200, label='Full     (Phase A+B+C: complete pick-and-place)'),
    }
    cfg = _STAGE_CFG[args.stage]

    run_name   = args.name or cfg['default_name']
    max_epochs = args.max_epochs or cfg['default_epochs']
    ckpt_dir   = os.path.join(args.log_dir, run_name, 'snapshots')

    # Build a training config with the stage-appropriate epoch count.
    training_cfg = OnPolicyTrainingConfig(
        max_epochs=max_epochs,
        epoch_num_steps=EPOCH_STEPS,
        collection_step_num_env_steps=EPOCH_STEPS,
        num_training_envs=TRAINING_CONFIG.num_training_envs,
        num_test_envs=TRAINING_CONFIG.num_test_envs,
        test_in_training=TRAINING_CONFIG.test_in_training,
        buffer_size=TRAINING_CONFIG.buffer_size,
        batch_size=TRAINING_CONFIG.batch_size,
        update_step_num_repetitions=TRAINING_CONFIG.update_step_num_repetitions,
    )

    builder = (
        PPOExperimentBuilder(
            EnvFactoryRegistered(
                task=cfg['task'],
                venv_type=VectorEnvType.SUBPROC,
                training_seed=args.seed,
                test_seed=args.seed + 1000,
            ),
            ExperimentConfig(
                persistence_enabled=True,
                persistence_base_dir=args.log_dir,
                watch=False,
                watch_render=1 / 35,
                watch_num_episodes=5,
            ),
            training_cfg,
        )
        .with_ppo_params(PPO_PARAMS)
        .with_actor_factory_default(hidden_sizes=HIDDEN_SIZES)
        .with_critic_factory_default(hidden_sizes=HIDDEN_SIZES)
        .with_logger_factory(WandbLoggerFactory(project=args.project))
        .with_name(run_name)
        .with_epoch_stop_callback(BestRewardCallback(save_dir=ckpt_dir))
        .with_epoch_train_callback(_make_train_callback(args, ckpt_dir))
    )

    if args.record_video:
        builder = builder.with_epoch_test_callback(
            VideoLogCallback(every_n=args.video_every)
        )

    experiment = builder.build()

    print("=" * 64)
    print(f"GraspEnv PPO training  –  {cfg['label']}")
    print(f"  --stage        : {args.stage}  (env: {cfg['task']})")
    print(f"  W&B project    : {args.project}")
    print(f"  run name       : {run_name}")
    print(f"  log dir        : {args.log_dir}")
    print(f"  train envs     : {training_cfg.num_training_envs}")
    print(f"  epoch steps    : {EPOCH_STEPS}"
          f"  (total ≈ {EPOCH_STEPS * max_epochs / 1_000_000:.1f}M)")
    print(f"  hidden sizes   : {HIDDEN_SIZES}")
    print(f"  batch size     : {training_cfg.batch_size}")
    print(f"  max epochs     : {max_epochs}")
    print(f"  gamma          : {PPO_PARAMS.gamma}  ent_coef: {PPO_PARAMS.ent_coef}")
    print(f"  snapshot every {args.save_every} epochs → {ckpt_dir}")
    print(f"  video logging  : {'on (every %d epochs)' % args.video_every if args.record_video else 'off'}")
    if args.load_from:
        what = 'actor+critic' if args.load_actor_critic else 'actor only'
        print(f"  warm-start from: {args.load_from}  [{what}]")
    print("=" * 64)

    experiment.run(raise_error_on_dirname_collision=False)


if __name__ == '__main__':
    main()
