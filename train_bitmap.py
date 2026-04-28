"""
PPO training (Bitmap 3-frame + CoordConv) for SuikaEnv-v0 / SuikaEnvNode-v0.

Input design:
  - bitmap: 3 stacked bitmap frames, nearest-neighbor resized to 64x64
  - hand_onehot: concat(one-hot(current_fruit_type), one-hot(next_fruit_type))
  - extractor appends CoordConv (x,y), so CNN input becomes [64,64,5] when frame_stack=3
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
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


class SuikaBitmapFrameStackWrapper(gym.Wrapper):
    """Build bitmap+hand observation for PPO from env raw observation."""

    def __init__(self, env: gym.Env, n_frames: int = 3, target_hw: tuple[int, int] = (64, 64)):
        super().__init__(env)
        self.n_frames = max(1, int(n_frames))
        self.target_h, self.target_w = int(target_hw[0]), int(target_hw[1])
        self._frames = deque(maxlen=self.n_frames)
        self._n_fruit_types = 11
        self.observation_space = spaces.Dict(
            {
                "bitmap": spaces.Box(
                    low=0,
                    high=self._n_fruit_types,
                    shape=(self.target_h, self.target_w, self.n_frames),
                    dtype=np.uint8,
                ),
                "hand_onehot": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._n_fruit_types * 2,),
                    dtype=np.float32,
                ),
            }
        )

    def _resize_nearest_u8(self, img_hw: np.ndarray) -> np.ndarray:
        src_h, src_w = img_hw.shape
        ys = (np.arange(self.target_h) * src_h / self.target_h).astype(np.int32)
        xs = (np.arange(self.target_w) * src_w / self.target_w).astype(np.int32)
        ys = np.clip(ys, 0, src_h - 1)
        xs = np.clip(xs, 0, src_w - 1)
        return img_hw[ys[:, None], xs[None, :]]

    def _build_hand_onehot(self, obs: dict) -> np.ndarray:
        cur = int(np.clip(float(np.asarray(obs["current_fruit_type"]).reshape(-1)[0]), 0, self._n_fruit_types - 1))
        nxt = int(np.clip(float(np.asarray(obs["next_fruit_type"]).reshape(-1)[0]), 0, self._n_fruit_types - 1))
        one = np.zeros((self._n_fruit_types * 2,), dtype=np.float32)
        one[cur] = 1.0
        one[self._n_fruit_types + nxt] = 1.0
        return one

    def _get_frame(self, obs: dict) -> np.ndarray:
        bitmap = np.asarray(obs["bitmap"], dtype=np.uint8)
        if bitmap.ndim != 2:
            raise ValueError(f'Expected obs["bitmap"] shape=(H,W), got {bitmap.shape}')
        return self._resize_nearest_u8(bitmap)

    def _pack_obs(self, obs: dict) -> dict:
        if len(self._frames) != self.n_frames:
            raise RuntimeError("frame buffer is not initialized")
        stacked = np.stack(list(self._frames), axis=2)  # (H, W, n_frames)
        return {
            "bitmap": stacked,
            "hand_onehot": self._build_hand_onehot(obs),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._get_frame(obs)
        self._frames.clear()
        for _ in range(self.n_frames):
            self._frames.append(frame.copy())
        return self._pack_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(self._get_frame(obs).copy())
        return self._pack_obs(obs), reward, terminated, truncated, info


class SuikaBitmapCoordConvExtractor(BaseFeaturesExtractor):
    """Extractor for bitmap stack + hand one-hot.

    CNN path:
      bitmap (64,64,3) -> normalize by /11.0 -> append CoordConv(x,y) -> (64,64,5)
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        bitmap_space = observation_space["bitmap"]
        hand_space = observation_space["hand_onehot"]
        if len(bitmap_space.shape) != 3:
            raise ValueError(f"Expected bitmap shape=(H,W,C), got {bitmap_space.shape}")
        h, w, c = [int(v) for v in bitmap_space.shape]
        self._h = h
        self._w = w
        self._bitmap_c = c
        self._coord_c = 2
        self._hand_dim = int(np.prod(hand_space.shape))
        self._bitmap_norm = 11.0

        self.cnn = nn.Sequential(
            nn.Conv2d(self._bitmap_c + self._coord_c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(bitmap_space.sample()[None]).float()  # (B,H,W,C)
            sample = sample.permute(0, 3, 1, 2).contiguous() / self._bitmap_norm
            sample = self._append_coord_channels(sample)
            n_flatten = int(self.cnn(sample).shape[1])
        self.bitmap_head = nn.Sequential(nn.Linear(n_flatten, 224), nn.ReLU())
        self.hand_head = nn.Sequential(nn.Linear(self._hand_dim, 32), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(224 + 32, 256), nn.ReLU())
        self._features_dim = 256

    def _append_coord_channels(self, bitmap: th.Tensor) -> th.Tensor:
        b, _, h, w = bitmap.shape
        y_lin = th.linspace(-1.0, 1.0, h, device=bitmap.device, dtype=bitmap.dtype)
        x_lin = th.linspace(-1.0, 1.0, w, device=bitmap.device, dtype=bitmap.dtype)
        y_map = y_lin.view(1, 1, h, 1).expand(b, 1, h, w)
        x_map = x_lin.view(1, 1, 1, w).expand(b, 1, h, w)
        return th.cat([bitmap, x_map, y_map], dim=1)

    def forward(self, observations):
        bitmap = observations["bitmap"].float() / self._bitmap_norm  # (B,H,W,C)
        bitmap = bitmap.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        bitmap = self._append_coord_channels(bitmap)  # (B,C+2,H,W) -> [64,64,5] when C=3
        bitmap_feat = self.bitmap_head(self.cnn(bitmap))
        hand_feat = self.hand_head(observations["hand_onehot"].float())
        return self.fuse(th.cat([bitmap_feat, hand_feat], dim=1))


def make_env_bitmap(
    rank: int,
    seed: int,
    headless: bool,
    port_base: int,
    env_id: str,
    frame_stack: int,
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
        env = SuikaBitmapFrameStackWrapper(env, n_frames=frame_stack, target_hw=(64, 64))
        return env

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--rollout-steps-total", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bitmap-frame-stack", type=int, default=3)
    p.add_argument(
        "--env-id",
        type=str,
        default="SuikaEnvNode-v0",
        choices=["SuikaEnv-v0", "SuikaEnvNode-v0"],
    )
    p.add_argument("--port-base", type=int, default=8923)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("models/bitmap/ppo_suika_bitmap"))
    p.add_argument("--wandb-project", type=str, default="suika-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--save-every-steps", type=int, default=20_000)
    p.add_argument("--restart-browser-every-steps", type=int, default=3_000)
    p.add_argument("--gif-eval-every-steps", type=int, default=10_000)
    p.add_argument("--gif-eval-steps", type=int, default=10000)
    p.add_argument("--gif-fps", type=int, default=20)
    p.add_argument("--gif-dir", type=Path, default=Path("gifs/bitmap"))
    p.add_argument("--device", type=str, default="cuda", help="auto|cpu|cuda|mps")
    p.add_argument("--gpu-id", type=int, default=None)
    p.add_argument("--check", type=lambda x: str(x).lower() == "true", default=False)
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
        f"[train_bitmap] n_envs={args.n_envs}, n_steps={effective_n_steps}, "
        f"rollout_total={effective_rollout_total}"
    )

    wandb_enabled = bool(args.wandb_run_name)
    run_name = args.wandb_run_name or f"ppo-suika-bitmap-seed{args.seed}"
    tb_dir = Path("runs/tb") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env_bitmap(
            i,
            args.seed,
            args.headless,
            args.port_base,
            args.env_id,
            args.bitmap_frame_stack,
        )
        for i in range(args.n_envs)
    ]
    if args.check:
        env = env_fns[0]()
        try:
            obs, _ = env.reset(seed=args.seed)
            print("[check] model input preview (train_bitmap.py)")
            for k, v in obs.items():
                arr = np.asarray(v)
                print(
                    f"- {k}: shape={arr.shape}, dtype={arr.dtype}, "
                    f"min={float(np.min(arr)):.6f}, max={float(np.max(arr)):.6f}"
                )
        finally:
            env.close()
        return
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
                "extractor": "BitmapCoordConv",
                "env_id": args.env_id,
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "bitmap_frame_stack": args.bitmap_frame_stack,
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
            policy_kwargs=dict(features_extractor_class=SuikaBitmapCoordConvExtractor),
            learning_rate=1e-4,
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
