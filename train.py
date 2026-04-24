"""
PPO training draft for SuikaEnv-v0.

Requirements:
  uv pip install stable-baselines3 torch wandb

Run:
  uv run python train.py --total-timesteps 200000 --n-envs 1
  uv run python train.py --device cuda --gpu-id 0
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
import wandb
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

import suika_env  # noqa: F401  # Registers "SuikaEnv-v0"


class SuikaObsWrapper(gym.ObservationWrapper):
    """Convert image HWC->CHW for Torch CNN and keep score as float32."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        img_space = env.observation_space["image"]
        score_space = env.observation_space["score"]
        h, w, c = img_space.shape
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8),
                "score": spaces.Box(
                    low=score_space.low.astype(np.float32),
                    high=score_space.high.astype(np.float32),
                    shape=score_space.shape,
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation):
        image = np.transpose(observation["image"], (2, 0, 1)).copy()
        score = observation["score"].astype(np.float32, copy=False)
        return {"image": image, "score": score}


class SuikaCombinedExtractor(BaseFeaturesExtractor):
    """Small CNN for image + tiny MLP for score, then concatenate."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        n_input_channels = observation_space["image"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space["image"].sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.image_head = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
        self.score_head = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self._features_dim = 256 + 16

    def forward(self, observations):
        img = observations["image"].float() / 255.0
        score = observations["score"].float()
        img_feat = self.image_head(self.cnn(img))
        score_feat = self.score_head(score)
        return th.cat([img_feat, score_feat], dim=1)


def make_env(rank: int, seed: int, headless: bool, delay: float) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(
            "SuikaEnv-v0",
            headless=headless,
            delay_before_img_capture=0.0,
            port=8923,
            mute_sound=True,
            wait_for_ready_on_step=True,
            ready_poll_interval=0.005,
            ready_timeout=2.0,
        )
        env = SuikaObsWrapper(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay-before-img-capture", type=float, default=0.1)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("models/ppo_suika"))
    p.add_argument("--wandb-project", type=str, default="suika-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", help="SB3 device, e.g. auto|cpu|cuda")
    p.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Physical GPU index to expose via CUDA_VISIBLE_DEVICES (e.g. 0).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_run_name or f"ppo-suika-seed{args.seed}"
    tb_dir = Path("runs/tb") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(i, args.seed, args.headless, args.delay_before_img_capture)
        for i in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns) if args.n_envs == 1 else SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "algo": "PPO",
            "env_id": "SuikaEnv-v0",
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "delay_before_img_capture": args.delay_before_img_capture,
            "headless": args.headless,
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "device": args.device,
            "gpu_id": args.gpu_id,
        },
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    interrupted = False
    try:
        policy_kwargs = dict(features_extractor_class=SuikaCombinedExtractor)
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(tb_dir),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

        callback = WandbCallback(
            gradient_save_freq=0,
            model_save_path=str(args.save_path.parent / "wandb_checkpoints"),
            model_save_freq=20_000,
            verbose=2,
        )
        model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback)
        model.save(str(args.save_path))
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Finishing cleanup...")
        if "model" in locals():
            interrupted_path = args.save_path.parent / f"{args.save_path.name}_interrupted"
            model.save(str(interrupted_path))
            print(f"Saved interrupted model to: {interrupted_path}.zip")
    finally:
        vec_env.close()
        run.finish()
    if not interrupted:
        print(f"Saved model to: {args.save_path}.zip")


if __name__ == "__main__":
    main()
