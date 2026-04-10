"""eval_grasp.py – evaluate a trained GraspEnv policy (visual + metrics).

Usage:
    # Evaluate the latest checkpoint from a training run:
    conda run -n 16762 python eval_grasp.py

    # Evaluate a specific checkpoint:
    conda run -n 16762 python eval_grasp.py --checkpoint log/GraspPhase1/seed_0/policy.pt

    # Run headless (no GUI) for bulk stats:
    conda run -n 16762 python eval_grasp.py --no-render --n-episodes 100
"""

import argparse
import os
import sys
sys.path.insert(0, '/home/ye/mengine')

import numpy as np
import torch
import gymnasium as gym

from tianshou.data import Collector, CollectStats
from tianshou.env import DummyVectorEnv

from grasp_env import GraspEnv  # noqa: F401 – registers 'GraspEnv'


def find_latest_checkpoint(log_dir: str, run_name: str) -> str:
    """Return the path to policy.pt in the most recently modified run folder."""
    base = os.path.join(log_dir, run_name)
    if not os.path.isdir(base):
        # Fall back: search any subfolder of log_dir
        all_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir)
                    if os.path.isdir(os.path.join(log_dir, d))]
        base = max(all_dirs, key=os.path.getmtime)
    # Find the seed subfolder
    seed_dirs = [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
    seed_dir = seed_dirs[0] if seed_dirs else ''
    return os.path.join(base, seed_dir, 'policy.pt')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None,
                   help='Path to policy.pt (auto-detected if not given)')
    p.add_argument('--log-dir', default='log')
    p.add_argument('--run-name', default='GraspPhase1')
    p.add_argument('--n-episodes', type=int, default=10)
    p.add_argument('--no-render', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint(args.log_dir, args.run_name)
    print(f"Loading checkpoint: {ckpt}")

    render_mode = None if args.no_render else 'human'
    env  = gym.make('GraspEnv', render_mode=render_mode)
    venv = DummyVectorEnv([lambda: env])

    data   = torch.load(ckpt, weights_only=False)
    policy = data.policy if hasattr(data, 'policy') else data
    policy.eval()

    collector = Collector[CollectStats](policy=policy, env=venv)
    result    = collector.collect(
        n_episode=args.n_episodes,
        render=0.0,
        reset_before_collect=True,
    )

    print(f"\n{'─'*40}")
    print(f"Episodes  : {args.n_episodes}")
    print(f"Mean reward: {result.returns.mean():.2f} ± {result.returns.std():.2f}")
    print(f"Mean length: {result.lens.mean():.1f} steps")
    success = (result.returns > 0).mean() * 100  # proxy: positive return → lifted
    print(f"{'─'*40}")

    venv.close()


if __name__ == '__main__':
    main()
