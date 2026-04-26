"""
PPO training with Transformer extractor over board-fruit tokens.

Token spec per fruit:
  [x, y, radius, mass, fruit_type_one_hot]

Notes:
  - Max board tokens = 30 (padded with mask)
  - CLS token output is fed to PPO policy/value heads
  - Transformer uses residual connections internally
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
from policy_gif_callback import PolicyGifCallback
from train import (
    ActionStatsLoggingCallback,
    BrowserRestartCallback,
    FinalScoreLoggingCallback,
    PolicyStdLoggingCallback,
    restore_terminal_cursor,
)


class SuikaTransformerObsWrapper(gym.ObservationWrapper):
    """Keep only board-token features needed by transformer extractor."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "current_fruit_type": spaces.Box(
                    low=env.observation_space["current_fruit_type"].low.astype(np.float32),
                    high=env.observation_space["current_fruit_type"].high.astype(np.float32),
                    shape=env.observation_space["current_fruit_type"].shape,
                    dtype=np.float32,
                ),
                "next_fruit_type": spaces.Box(
                    low=env.observation_space["next_fruit_type"].low.astype(np.float32),
                    high=env.observation_space["next_fruit_type"].high.astype(np.float32),
                    shape=env.observation_space["next_fruit_type"].shape,
                    dtype=np.float32,
                ),
                "current_fruit_x": spaces.Box(
                    low=env.observation_space["current_fruit_x"].low.astype(np.float32),
                    high=env.observation_space["current_fruit_x"].high.astype(np.float32),
                    shape=env.observation_space["current_fruit_x"].shape,
                    dtype=np.float32,
                ),
                "board_fruit_xy": spaces.Box(
                    low=env.observation_space["board_fruit_xy"].low.astype(np.float32),
                    high=env.observation_space["board_fruit_xy"].high.astype(np.float32),
                    shape=env.observation_space["board_fruit_xy"].shape,
                    dtype=np.float32,
                ),
                "board_fruit_radius": spaces.Box(
                    low=env.observation_space["board_fruit_radius"].low.astype(np.float32),
                    high=env.observation_space["board_fruit_radius"].high.astype(np.float32),
                    shape=env.observation_space["board_fruit_radius"].shape,
                    dtype=np.float32,
                ),
                "board_fruit_mass": spaces.Box(
                    low=env.observation_space["board_fruit_mass"].low.astype(np.float32),
                    high=env.observation_space["board_fruit_mass"].high.astype(np.float32),
                    shape=env.observation_space["board_fruit_mass"].shape,
                    dtype=np.float32,
                ),
                "board_fruit_type": spaces.Box(
                    low=env.observation_space["board_fruit_type"].low.astype(np.float32),
                    high=env.observation_space["board_fruit_type"].high.astype(np.float32),
                    shape=env.observation_space["board_fruit_type"].shape,
                    dtype=np.float32,
                ),
                "board_fruit_mask": spaces.Box(
                    low=env.observation_space["board_fruit_mask"].low.astype(np.float32),
                    high=env.observation_space["board_fruit_mask"].high.astype(np.float32),
                    shape=env.observation_space["board_fruit_mask"].shape,
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation):
        return {
            "current_fruit_type": observation["current_fruit_type"].astype(np.float32, copy=False),
            "next_fruit_type": observation["next_fruit_type"].astype(np.float32, copy=False),
            "current_fruit_x": observation["current_fruit_x"].astype(np.float32, copy=False),
            "board_fruit_xy": observation["board_fruit_xy"].astype(np.float32, copy=False),
            "board_fruit_radius": observation["board_fruit_radius"].astype(np.float32, copy=False),
            "board_fruit_mass": observation["board_fruit_mass"].astype(np.float32, copy=False),
            "board_fruit_type": observation["board_fruit_type"].astype(np.float32, copy=False),
            "board_fruit_mask": observation["board_fruit_mask"].astype(np.float32, copy=False),
        }


class SuikaTransformerExtractor(BaseFeaturesExtractor):
    """
    Transformer extractor:
      - d_model=64
      - nhead=4
      - ff_dim=128
      - max tokens=30 + CLS
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=64)
        self.max_board_tokens = 30
        self.d_model = 64
        self.num_fruit_types = int(observation_space["board_fruit_type"].high.max()) + 1

        # [x, y, radius, mass] + one-hot(type) + one-hot(source: current/next/board)
        self.token_input_dim = 4 + self.num_fruit_types + 3
        self.token_proj = nn.Linear(self.token_input_dim, self.d_model)

        self.cls_token = nn.Parameter(th.zeros(1, 1, self.d_model))
        self.total_tokens = 2 + self.max_board_tokens  # current + next + board(30)
        self.pos_embed = nn.Parameter(th.zeros(1, self.total_tokens + 1, self.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        # Residual MLP block on CLS output.
        self.cls_mlp = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, self.d_model),
        )
        self._features_dim = self.d_model

    def forward(self, observations):
        xy = observations["board_fruit_xy"].float().view(-1, self.max_board_tokens, 2)
        radius = observations["board_fruit_radius"].float().view(-1, self.max_board_tokens, 1)
        mass = observations["board_fruit_mass"].float().view(-1, self.max_board_tokens, 1)
        mask = observations["board_fruit_mask"].float().view(-1, self.max_board_tokens)
        t_idx = observations["board_fruit_type"].long().clamp(0, self.num_fruit_types - 1)
        current_t_idx = observations["current_fruit_type"].long().squeeze(1).clamp(0, self.num_fruit_types - 1)
        next_t_idx = observations["next_fruit_type"].long().squeeze(1).clamp(0, self.num_fruit_types - 1)
        current_x = observations["current_fruit_x"].float()

        # Board tokens
        board_t_onehot = F.one_hot(t_idx, num_classes=self.num_fruit_types).float()
        board_t_onehot = board_t_onehot * mask.unsqueeze(-1)
        board_source = th.tensor([0.0, 0.0, 1.0], dtype=xy.dtype, device=xy.device).view(1, 1, 3)
        board_source = board_source.expand(xy.shape[0], self.max_board_tokens, 3)
        board_feat = th.cat([xy, radius, mass, board_t_onehot, board_source], dim=-1)
        board_emb = self.token_proj(board_feat)
        board_emb = board_emb * mask.unsqueeze(-1)

        # Current-fruit token: source=[1,0,0], uses current x and type.
        cur_xy = th.cat([current_x, th.zeros_like(current_x)], dim=1).unsqueeze(1)
        cur_r = th.zeros((xy.shape[0], 1, 1), dtype=xy.dtype, device=xy.device)
        cur_m = th.zeros((xy.shape[0], 1, 1), dtype=xy.dtype, device=xy.device)
        cur_t = F.one_hot(current_t_idx, num_classes=self.num_fruit_types).float().unsqueeze(1)
        cur_source = th.tensor([1.0, 0.0, 0.0], dtype=xy.dtype, device=xy.device).view(1, 1, 3)
        cur_source = cur_source.expand(xy.shape[0], 1, 3)
        cur_feat = th.cat([cur_xy, cur_r, cur_m, cur_t, cur_source], dim=-1)
        cur_emb = self.token_proj(cur_feat)

        # Next-fruit token: source=[0,1,0], type only (x/r/m set to 0).
        nxt_xy = th.zeros((xy.shape[0], 1, 2), dtype=xy.dtype, device=xy.device)
        nxt_r = th.zeros((xy.shape[0], 1, 1), dtype=xy.dtype, device=xy.device)
        nxt_m = th.zeros((xy.shape[0], 1, 1), dtype=xy.dtype, device=xy.device)
        nxt_t = F.one_hot(next_t_idx, num_classes=self.num_fruit_types).float().unsqueeze(1)
        nxt_source = th.tensor([0.0, 1.0, 0.0], dtype=xy.dtype, device=xy.device).view(1, 1, 3)
        nxt_source = nxt_source.expand(xy.shape[0], 1, 3)
        nxt_feat = th.cat([nxt_xy, nxt_r, nxt_m, nxt_t, nxt_source], dim=-1)
        nxt_emb = self.token_proj(nxt_feat)

        token_emb = th.cat([cur_emb, nxt_emb, board_emb], dim=1)
        batch = token_emb.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        seq = th.cat([cls, token_emb], dim=1)
        seq = seq + self.pos_embed[:, : self.total_tokens + 1, :]

        fixed_valid = th.ones((batch, 2), dtype=th.bool, device=seq.device)
        pad_mask = th.cat(
            [
                th.zeros(batch, 1, dtype=th.bool, device=seq.device),
                ~fixed_valid,
                (mask < 0.5),
            ],
            dim=1,
        )

        enc = self.encoder(seq, src_key_padding_mask=pad_mask)
        cls_out = enc[:, 0, :]
        cls_out = cls_out + self.cls_mlp(cls_out)
        return cls_out


def make_env(
    rank: int,
    seed: int,
    headless: bool,
    port_base: int,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(
            "SuikaEnv-v0",
            headless=headless,
            delay_before_img_capture=0.0,
            port=port_base + rank,
            mute_sound=True,
            wait_for_ready_on_step=True,
            ready_poll_interval=0.02,
            ready_timeout=2.0,
            enable_image_observation=False,
        )
        env = SuikaTransformerObsWrapper(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--rollout-steps-total", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port-base", type=int, default=8923)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("models/ppo_suika_transformer"))
    p.add_argument("--wandb-project", type=str, default="suika-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--save-every-steps", type=int, default=20_000)
    p.add_argument("--restart-browser-every-steps", type=int, default=3_000)
    p.add_argument(
        "--gif-eval-every-steps",
        type=int,
        default=10_000,
        help="Export policy gameplay GIF every N timesteps (<=0 disables).",
    )
    p.add_argument(
        "--gif-eval-steps",
        type=int,
        default=0,
        help="Safety cap for policy steps per GIF episode (0 disables cap).",
    )
    p.add_argument("--gif-fps", type=int, default=20, help="GIF playback FPS.")
    p.add_argument("--gif-dir", type=Path, default=Path("gifs"))
    p.add_argument("--device", type=str, default="cuda", help="auto|cpu|cuda")
    p.add_argument("--gpu-id", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    effective_n_steps = args.n_steps
    if args.rollout_steps_total > 0:
        effective_n_steps = max(1, args.rollout_steps_total // args.n_envs)
    effective_rollout_total = effective_n_steps * args.n_envs
    print(
        f"[train_transformer] n_envs={args.n_envs}, n_steps={effective_n_steps}, "
        f"rollout_total={effective_rollout_total}"
    )

    run_name = args.wandb_run_name or f"ppo-suika-transformer-seed{args.seed}"
    tb_dir = Path("runs/tb") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(i, args.seed, args.headless, args.port_base) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns) if args.n_envs == 1 else SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "algo": "PPO",
            "extractor": "Transformer",
            "env_id": "SuikaEnv-v0",
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "headless": args.headless,
            "learning_rate": 3e-4,
            "n_steps": effective_n_steps,
            "rollout_steps_total": effective_rollout_total,
            "batch_size": args.batch_size,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "device": args.device,
            "gpu_id": args.gpu_id,
            "save_every_steps": args.save_every_steps,
            "restart_browser_every_steps": args.restart_browser_every_steps,
            "gif_eval_every_steps": args.gif_eval_every_steps,
            "gif_eval_steps": args.gif_eval_steps,
            "gif_fps": args.gif_fps,
            "d_model": 64,
            "n_heads": 4,
            "ff_dim": 128,
            "max_tokens": 30,
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
            policy_kwargs=dict(
                features_extractor_class=SuikaTransformerExtractor,
                share_features_extractor=False,
            ),
            learning_rate=3e-4,
            n_steps=effective_n_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.10,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(tb_dir),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

        save_freq = args.save_every_steps if args.save_every_steps > 0 else 0
        wandb_callback = WandbCallback(
            gradient_save_freq=0,
            model_save_path=str(args.save_path.parent / "wandb_checkpoints"),
            model_save_freq=save_freq,
            verbose=2,
        )
        callbacks = [wandb_callback]
        callbacks.append(FinalScoreLoggingCallback(verbose=0))
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
                    total_timesteps=args.total_timesteps,
                    verbose=1,
                )
            )
        if args.restart_browser_every_steps > 0:
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
        run.finish()
        restore_terminal_cursor()

    if not interrupted:
        print(f"Saved model to: {args.save_path}.zip")


if __name__ == "__main__":
    main()
