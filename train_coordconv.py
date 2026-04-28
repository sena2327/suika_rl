"""
PPO training (CNN + CoordConv image input) for SuikaEnv-v0 / SuikaEnvNode-v0.

Run:
  uv run python train_coordconv.py --env-id SuikaEnvNode-v0 --n-envs 16 --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    make_env,
    resolve_device,
    restore_terminal_cursor,
)


class SuikaCoordConvExtractor(BaseFeaturesExtractor):
    """CNN extractor with CoordConv channels (x, y)."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        image_space = observation_space["image"]
        if len(image_space.shape) != 3:
            raise ValueError(f"Expected 3D image space, got shape={image_space.shape}")

        shp = tuple(int(v) for v in image_space.shape)
        self._channels_first = shp[0] <= 16 and shp[1] > 16 and shp[2] > 16
        if self._channels_first:
            c, h, w = int(shp[0]), int(shp[1]), int(shp[2])
        else:
            h, w, c = int(shp[0]), int(shp[1]), int(shp[2])
        self._h = h
        self._w = w
        self._base_c = c
        self._coord_c = 2

        self.cnn = nn.Sequential(
            nn.Conv2d(self._base_c + self._coord_c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(image_space.sample()[None]).float()
            if not self._channels_first:
                sample = sample.permute(0, 3, 1, 2).contiguous()
            sample = sample / 255.0
            sample = self._append_coord_channels(sample)
            n_flatten = int(self.cnn(sample).shape[1])

        self.linear = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
        self._features_dim = 256

    def _append_coord_channels(self, image: th.Tensor) -> th.Tensor:
        b, _, h, w = image.shape
        y_lin = th.linspace(-1.0, 1.0, h, device=image.device, dtype=image.dtype)
        x_lin = th.linspace(-1.0, 1.0, w, device=image.device, dtype=image.dtype)
        y_map = y_lin.view(1, 1, h, 1).expand(b, 1, h, w)
        x_map = x_lin.view(1, 1, 1, w).expand(b, 1, h, w)
        return th.cat([image, x_map, y_map], dim=1)

    def forward(self, observations):
        image = observations["image"].float() / 255.0
        if not self._channels_first:
            image = image.permute(0, 3, 1, 2).contiguous()
        image = self._append_coord_channels(image)
        return self.linear(self.cnn(image))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--rollout-steps-total", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-frame-stack", type=int, default=3)
    p.add_argument(
        "--env-id",
        type=str,
        default="SuikaEnvNode-v0",
        choices=["SuikaEnv-v0", "SuikaEnvNode-v0"],
    )
    p.add_argument("--port-base", type=int, default=8923)
    p.add_argument("--delay-before-img-capture", type=float, default=0.1)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("models/coordconv/ppo_suika_coordconv"))
    p.add_argument("--wandb-project", type=str, default="suika-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--save-every-steps", type=int, default=20_000)
    p.add_argument("--restart-browser-every-steps", type=int, default=3_000)
    p.add_argument("--gif-eval-every-steps", type=int, default=10_000)
    p.add_argument("--gif-eval-steps", type=int, default=10000)
    p.add_argument("--gif-fps", type=int, default=20)
    p.add_argument("--gif-dir", type=Path, default=Path("gifs/coordconv"))
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
        f"[train_coordconv] n_envs={args.n_envs}, n_steps={effective_n_steps}, "
        f"rollout_total={effective_rollout_total}"
    )

    wandb_enabled = bool(args.wandb_run_name)
    run_name = args.wandb_run_name or f"ppo-suika-coordconv-seed{args.seed}"
    tb_dir = Path("runs/tb") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(
            i,
            args.seed,
            args.headless,
            args.delay_before_img_capture,
            args.port_base,
            args.env_id,
            args.image_frame_stack,
        )
        for i in range(args.n_envs)
    ]
    if args.check:
        env = env_fns[0]()
        try:
            obs, _ = env.reset(seed=args.seed)
            print("[check] model input preview (train_coordconv.py)")
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
                "extractor": "CoordConvCNN",
                "env_id": args.env_id,
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "image_frame_stack": args.image_frame_stack,
                "delay_before_img_capture": args.delay_before_img_capture,
                "headless": args.headless,
                "learning_rate": 3e-4,
                "n_steps": effective_n_steps,
                "rollout_steps_total": effective_rollout_total,
                "batch_size": 1024,
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
        policy_kwargs = dict(features_extractor_class=SuikaCoordConvExtractor)
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=effective_n_steps,
            batch_size=1024,
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
