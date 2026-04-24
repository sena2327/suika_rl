"""
PPO training draft for SuikaEnv-v0.

Requirements:
  uv pip install stable-baselines3 torch wandb

Run:
  uv run python train.py --total-timesteps 200000 --n-envs 1
  uv run python train.py --device cuda --gpu-id 0
  uv run python train.py --save-every-steps 10000
  uv run python train.py --n-envs 16 --rollout-steps-total 1024
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

import suika_env  # noqa: F401  # Registers "SuikaEnv-v0"


class SuikaObsWrapper(gym.ObservationWrapper):
    """Convert image HWC->CHW and keep aux features for policy input."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        img_space = env.observation_space["image"]
        fruit_type_space = env.observation_space["current_fruit_type"]
        fruit_x_space = env.observation_space["current_fruit_x"]
        h, w, c = img_space.shape
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8),
                "current_fruit_type": spaces.Box(
                    low=fruit_type_space.low.astype(np.float32),
                    high=fruit_type_space.high.astype(np.float32),
                    shape=fruit_type_space.shape,
                    dtype=np.float32,
                ),
                "current_fruit_x": spaces.Box(
                    low=fruit_x_space.low.astype(np.float32),
                    high=fruit_x_space.high.astype(np.float32),
                    shape=fruit_x_space.shape,
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation):
        image = np.transpose(observation["image"], (2, 0, 1)).copy()
        fruit_type = observation["current_fruit_type"].astype(np.float32, copy=False)
        fruit_x = observation["current_fruit_x"].astype(np.float32, copy=False)
        return {"image": image, "current_fruit_type": fruit_type, "current_fruit_x": fruit_x}


class SuikaCombinedExtractor(BaseFeaturesExtractor):
    """Small CNN for image + (fruit type embedding + position), then concatenate."""

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

        num_fruit_types = int(observation_space["current_fruit_type"].high[0]) + 1
        self.fruit_type_embedding = nn.Embedding(num_fruit_types, 8)
        self.image_head = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
        self.aux_head = nn.Sequential(nn.Linear(9, 16), nn.ReLU())
        self._features_dim = 256 + 16

    def forward(self, observations):
        img = observations["image"].float() / 255.0
        fruit_x = observations["current_fruit_x"].float()
        fruit_type_idx = observations["current_fruit_type"].long().squeeze(1)
        fruit_type_emb = self.fruit_type_embedding(fruit_type_idx)
        aux = th.cat([fruit_type_emb, fruit_x], dim=1)
        img_feat = self.image_head(self.cnn(img))
        aux_feat = self.aux_head(aux)
        return th.cat([img_feat, aux_feat], dim=1)


class SuikaFrameStackWrapper(gym.Wrapper):
    """Stack last k image observations on channel axis (C*k, H, W)."""

    def __init__(self, env: gym.Env, k: int = 3):
        super().__init__(env)
        self.k = max(1, int(k))
        self._frames = deque(maxlen=self.k)
        img_space = env.observation_space["image"]
        fruit_type_space = env.observation_space["current_fruit_type"]
        fruit_x_space = env.observation_space["current_fruit_x"]
        c, h, w = img_space.shape
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(c * self.k, h, w),
                    dtype=np.uint8,
                ),
                "current_fruit_type": spaces.Box(
                    low=fruit_type_space.low.astype(np.float32),
                    high=fruit_type_space.high.astype(np.float32),
                    shape=fruit_type_space.shape,
                    dtype=np.float32,
                ),
                "current_fruit_x": spaces.Box(
                    low=fruit_x_space.low.astype(np.float32),
                    high=fruit_x_space.high.astype(np.float32),
                    shape=fruit_x_space.shape,
                    dtype=np.float32,
                ),
            }
        )

    def _build_obs(self, obs):
        stacked = np.concatenate(list(self._frames), axis=0)
        return {
            "image": stacked,
            "current_fruit_type": obs["current_fruit_type"],
            "current_fruit_x": obs["current_fruit_x"],
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(obs["image"].copy())
        return self._build_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs["image"].copy())
        return self._build_obs(obs), reward, terminated, truncated, info


class BrowserRestartCallback(BaseCallback):
    """Queue browser restarts and apply them only on episode boundaries."""

    def __init__(self, every_steps: int = 3000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.every_steps = max(1, int(every_steps))
        self._last_restart_at = 0
        self._pending_env_indices = set()

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_restart_at) >= self.every_steps and not self._pending_env_indices:
            self._pending_env_indices = set(range(self.training_env.num_envs))
            if self.verbose > 0:
                print(
                    "[BrowserRestartCallback] Restart queued for all envs "
                    f"at step={self.num_timesteps}"
                )

        if self._pending_env_indices:
            dones = self.locals.get("dones", None)
            if dones is not None:
                done_indices = set(np.nonzero(np.asarray(dones))[0].tolist())
                ready_to_restart = sorted(done_indices.intersection(self._pending_env_indices))
                if ready_to_restart:
                    if self.verbose > 0:
                        print(
                            "[BrowserRestartCallback] Restarting browsers on episode boundary "
                            f"at step={self.num_timesteps}, envs={ready_to_restart}"
                        )
                    self.training_env.env_method("restart_browser", indices=ready_to_restart)
                    self.training_env.env_method("reset", indices=ready_to_restart)
                    self._pending_env_indices.difference_update(ready_to_restart)
                    if not self._pending_env_indices:
                        self._last_restart_at = self.num_timesteps
                        if self.verbose > 0:
                            print(
                                "[BrowserRestartCallback] Restart cycle completed "
                                f"at step={self.num_timesteps}"
                            )
        return True


class FinalScoreLoggingCallback(BaseCallback):
    """Log episode final score from info['score'] to W&B on done steps."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if dones is None or infos is None:
            return True

        final_scores = []
        for i, done in enumerate(np.asarray(dones).tolist()):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            score = info.get("score", None)
            if score is None:
                continue
            final_scores.append(float(score))
            wandb.log({"rollout/final_score": float(score)}, step=self.num_timesteps)

        if final_scores:
            mean_score = float(np.mean(final_scores))
            self.logger.record("rollout/final_score_mean", mean_score)
            if self.verbose > 0:
                print(
                    f"[FinalScoreLoggingCallback] step={self.num_timesteps} "
                    f"final_score_mean={mean_score:.2f} n={len(final_scores)}"
                )
        return True


def make_env(
    rank: int,
    seed: int,
    headless: bool,
    delay: float,
    port_base: int,
    frame_stack: int,
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
        )
        env = SuikaObsWrapper(env)
        env = SuikaFrameStackWrapper(env, k=frame_stack)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument(
        "--n-steps",
        type=int,
        default=128,
        help="PPO rollout length per env before each update (smaller => more frequent logs).",
    )
    p.add_argument(
        "--rollout-steps-total",
        type=int,
        default=0,
        help=(
            "Target total rollout steps per PPO update. "
            "If > 0, n_steps is set to max(1, rollout_steps_total // n_envs)."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-stack", type=int, default=1, help="Number of stacked image frames.")
    p.add_argument("--port-base", type=int, default=8923)
    p.add_argument("--delay-before-img-capture", type=float, default=0.1)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("models/ppo_suika"))
    p.add_argument("--wandb-project", type=str, default="suika-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=20_000,
        help="Save checkpoint every N env steps (<=0 disables periodic checkpointing).",
    )
    p.add_argument(
        "--restart-browser-every-steps",
        type=int,
        default=3_000,
        help="Restart all browser envs every N global train timesteps (<=0 disables).",
    )
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

    effective_n_steps = args.n_steps
    if args.rollout_steps_total > 0:
        effective_n_steps = max(1, args.rollout_steps_total // args.n_envs)
    effective_rollout_total = effective_n_steps * args.n_envs
    print(
        f"[train] n_envs={args.n_envs}, n_steps={effective_n_steps}, "
        f"rollout_total={effective_rollout_total}"
    )

    run_name = args.wandb_run_name or f"ppo-suika-seed{args.seed}"
    tb_dir = Path("runs/tb") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(
            i,
            args.seed,
            args.headless,
            args.delay_before_img_capture,
            args.port_base,
            args.frame_stack,
        )
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
            "frame_stack": args.frame_stack,
            "delay_before_img_capture": args.delay_before_img_capture,
            "headless": args.headless,
            "learning_rate": 3e-4,
            "n_steps": effective_n_steps,
            "rollout_steps_total": effective_rollout_total,
            "batch_size": 256,
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
    finally:
        vec_env.close()
        run.finish()
    if not interrupted:
        print(f"Saved model to: {args.save_path}.zip")


if __name__ == "__main__":
    main()
