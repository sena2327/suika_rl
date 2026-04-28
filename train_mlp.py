"""
PPO training with MLP input from internal board features.

Input pipeline:
  - Use obs["board_top50_exyir"] with shape (50, 5): [exist, x, y, id, r]
  - Normalize x,y,id,r (x,y already in [0,1], id/11, r/256)
  - One-hot current_hand and next_hand (11 dims each), concatenate
  - Feed concatenated vector to MLP extractor
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

import suika_env  # noqa: F401
import suika_env_node  # noqa: F401
from policy_gif_callback import PolicyGifCallback
from train import (
    ActionStatsLoggingCallback,
    BrowserRestartCallback,
    EpisodeLengthMaxLoggingCallback,
    FinalScoreLoggingCallback,
    PolicyStdLoggingCallback,
    resolve_device,
    restore_terminal_cursor,
)


class SuikaMLPObsWrapper(gym.ObservationWrapper):
    """Keep only board_top50_exyir and current/next hand type for MLP policy."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "board_top50_exyir": spaces.Box(
                    low=0.0,
                    high=256.0,
                    shape=(50, 5),
                    dtype=np.float32,
                ),
                "current_fruit_type": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "next_fruit_type": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            }
        )

    def observation(self, observation):
        return {
            "board_top50_exyir": observation["board_top50_exyir"].astype(np.float32, copy=False),
            "current_fruit_type": observation["current_fruit_type"].astype(np.float32, copy=False),
            "next_fruit_type": observation["next_fruit_type"].astype(np.float32, copy=False),
        }


class SuikaMLPExtractor(BaseFeaturesExtractor):
    """MLP extractor over flattened normalized board features + hand one-hots."""

    def __init__(self, observation_space: spaces.Dict):
        # board: 50 * 5 = 250, hand one-hot: 11 + 11 = 22, total 272
        super().__init__(observation_space, features_dim=256)
        self.n_hand = 11
        in_dim = (50 * 5) + (self.n_hand * 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self._features_dim = 256

    def forward(self, observations):
        exyir = observations["board_top50_exyir"].float()  # (B,50,5)
        exist = exyir[:, :, 0:1]  # 0/1
        x = exyir[:, :, 1:2]  # already 0..1
        y = exyir[:, :, 2:3]  # already 0..1
        fruit_id = exyir[:, :, 3:4] / 11.0  # 0..11 -> 0..1
        radius = exyir[:, :, 4:5] / 256.0  # pixel radius -> ~0..1
        board_feat = th.cat([exist, x, y, fruit_id, radius], dim=2).reshape(exyir.shape[0], -1)

        cur_idx = observations["current_fruit_type"].long().squeeze(1).clamp(0, self.n_hand - 1)
        nxt_idx = observations["next_fruit_type"].long().squeeze(1).clamp(0, self.n_hand - 1)
        cur_onehot = F.one_hot(cur_idx, num_classes=self.n_hand).float()
        nxt_onehot = F.one_hot(nxt_idx, num_classes=self.n_hand).float()

        feat = th.cat([board_feat, cur_onehot, nxt_onehot], dim=1)
        return self.mlp(feat)


def make_env_mlp(
    rank: int,
    seed: int,
    headless: bool,
    port_base: int,
    env_id: str,
    reward_norm_gamma: float,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env_kwargs = dict(
            headless=headless,
            delay_before_img_capture=0.0,
            mute_sound=True,
            wait_for_ready_on_step=True,
            ready_poll_interval=0.02,
            ready_timeout=2.0,
            enable_image_observation=False,
            bitmap_size=128,
            img_width=128,
            img_height=128,
        )
        if env_id == "SuikaEnv-v0":
            env_kwargs["port"] = port_base + rank
        env = gym.make(env_id, **env_kwargs)
        env = SuikaMLPObsWrapper(env)
        env = gym.wrappers.NormalizeReward(env, gamma=reward_norm_gamma)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--rollout-steps-total", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward-norm-gamma", type=float, default=0.99)
    p.add_argument(
        "--env-id",
        type=str,
        default="SuikaEnvNode-v0",
        choices=["SuikaEnv-v0", "SuikaEnvNode-v0"],
    )
    p.add_argument("--port-base", type=int, default=8923)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("models/mlp/ppo_suika_mlp"))
    p.add_argument("--wandb-project", type=str, default="suika-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--save-every-steps", type=int, default=20_000)
    p.add_argument("--restart-browser-every-steps", type=int, default=3_000)
    p.add_argument("--gif-eval-every-steps", type=int, default=10_000)
    p.add_argument("--gif-eval-steps", type=int, default=10000)
    p.add_argument("--gif-fps", type=int, default=20)
    p.add_argument("--gif-dir", type=Path, default=Path("gifs/mlp"))
    p.add_argument("--device", type=str, default="cuda", help="auto|cpu|cuda|mps")
    p.add_argument("--gpu-id", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    actual_device = resolve_device(args.device)
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    effective_n_steps = args.n_steps
    if args.rollout_steps_total > 0:
        effective_n_steps = max(1, args.rollout_steps_total // args.n_envs)
    effective_rollout_total = effective_n_steps * args.n_envs
    print(
        f"[train_mlp] n_envs={args.n_envs}, n_steps={effective_n_steps}, "
        f"rollout_total={effective_rollout_total}"
    )

    wandb_enabled = bool(args.wandb_run_name)
    run_name = args.wandb_run_name or f"ppo-suika-mlp-seed{args.seed}"
    tb_dir = Path("runs/tb") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env_mlp(
            i,
            args.seed,
            args.headless,
            args.port_base,
            args.env_id,
            args.reward_norm_gamma,
        )
        for i in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns) if args.n_envs == 1 else SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    run = None
    if wandb_enabled:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "algo": "PPO",
                "extractor": "MLP",
                "obs": "board_top50_exyir + hand_onehot",
                "env_id": args.env_id,
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "reward_norm_gamma": args.reward_norm_gamma,
                "learning_rate": 3e-4,
                "n_steps": effective_n_steps,
                "rollout_steps_total": effective_rollout_total,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.05,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "device": actual_device,
                "gpu_id": args.gpu_id,
                "save_every_steps": args.save_every_steps,
                "restart_browser_every_steps": args.restart_browser_every_steps,
                "gif_eval_every_steps": args.gif_eval_every_steps,
                "gif_eval_steps": args.gif_eval_steps,
                "gif_fps": args.gif_fps,
            },
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
        )

    interrupted = False
    try:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=dict(features_extractor_class=SuikaMLPExtractor),
            learning_rate=3e-4,
            n_steps=effective_n_steps,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(tb_dir),
            verbose=1,
            seed=args.seed,
            device=actual_device,
        )

        callbacks = []
        if wandb_enabled:
            save_freq = args.save_every_steps if args.save_every_steps > 0 else 0
            callbacks.append(
                WandbCallback(
                    gradient_save_freq=0,
                    model_save_path=str(args.save_path.parent / "wandb_checkpoints"),
                    model_save_freq=save_freq,
                    verbose=2,
                )
            )
        callbacks.append(FinalScoreLoggingCallback(verbose=0))
        callbacks.append(EpisodeLengthMaxLoggingCallback(verbose=0))
        callbacks.append(ActionStatsLoggingCallback(verbose=0))
        callbacks.append(PolicyStdLoggingCallback(verbose=0))
        if args.gif_eval_every_steps > 0:
            callbacks.append(
                PolicyGifCallback(
                    every_steps=args.gif_eval_every_steps,
                    max_steps_per_episode=args.gif_eval_steps,
                    fps=args.gif_fps,
                    out_dir=args.gif_dir,
                    seed=args.seed,
                    headless=args.headless,
                    port_base=args.port_base,
                    env_id=args.env_id,
                    total_timesteps=args.total_timesteps,
                    verbose=1,
                )
            )
        if args.env_id == "SuikaEnv-v0" and args.restart_browser_every_steps > 0:
            callbacks.append(
                BrowserRestartCallback(
                    every_steps=args.restart_browser_every_steps,
                    verbose=1,
                )
            )

        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=True,
            log_interval=1,
            callback=CallbackList(callbacks),
        )
        model.save(str(args.save_path))
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Finishing cleanup...")
        if "model" in locals():
            interrupted_path = args.save_path.parent / f"{args.save_path.name}_interrupted"
            model.save(str(interrupted_path))
            print(f"Saved interrupted model to: {interrupted_path}.zip")
        restore_terminal_cursor()
    finally:
        vec_env.close()
        if run is not None:
            run.finish()
        restore_terminal_cursor()

    if not interrupted:
        print(f"Saved model to: {args.save_path}.zip")


if __name__ == "__main__":
    main()
