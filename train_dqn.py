"""
DQN training (CNN image input) for SuikaEnvNode-v0.

Run:
  uv run python train_dqn.py --n-envs 16 --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections import deque
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import wandb
from gymnasium import spaces
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

import suika_env_node  # noqa: F401  # Registers "SuikaEnvNode-v0"
from policy_gif_callback import PolicyGifCallback


def restore_terminal_cursor():
    for cmd in (["stty", "sane"], ["tput", "cnorm"]):
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass


def resolve_device(requested: str) -> str:
    dev = str(requested).lower().strip()
    if dev == "mps":
        mps_ok = bool(getattr(th.backends, "mps", None)) and th.backends.mps.is_available()
        if not mps_ok:
            print("[train_dqn] MPS requested but unavailable. Falling back to cpu.")
            return "cpu"
        return "mps"
    if dev == "cuda" and not th.cuda.is_available():
        print("[train_dqn] CUDA requested but unavailable. Falling back to cpu.")
        return "cpu"
    return dev


class SuikaImageObsWrapper(gym.ObservationWrapper):
    """Keep image-only observation (grayscale 64x64)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._target_hw = (64, 64)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._target_hw[0], self._target_hw[1], 1),
                    dtype=np.uint8,
                )
            }
        )

    def observation(self, observation):
        img = observation["image"]
        if img.shape[0] != self._target_hw[0] or img.shape[1] != self._target_hw[1]:
            pil = Image.fromarray(img)
            pil = pil.resize((self._target_hw[1], self._target_hw[0]), Image.Resampling.LANCZOS)
        else:
            pil = Image.fromarray(img)
        gray = np.asarray(pil.convert("L"), dtype=np.uint8)
        gray = np.expand_dims(gray, axis=2)
        return {"image": gray}


class SuikaImageFrameStackWrapper(gym.Wrapper):
    """Stack latest N image frames along channel axis (H, W, C*N)."""

    def __init__(self, env: gym.Env, n_frames: int = 3):
        super().__init__(env)
        self.n_frames = max(1, int(n_frames))
        img_space = env.observation_space["image"]
        h, w, c = img_space.shape
        self._frames = deque(maxlen=self.n_frames)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(h, w, c * self.n_frames),
                    dtype=np.uint8,
                )
            }
        )

    def _stacked_obs(self):
        return {"image": np.concatenate(list(self._frames), axis=2)}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        first = obs["image"]
        for _ in range(self.n_frames):
            self._frames.append(first.copy())
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs["image"].copy())
        return self._stacked_obs(), reward, terminated, truncated, info


class DiscreteActionWrapper(gym.Wrapper):
    """Convert discrete action index -> centered continuous x in [-0.5, 0.5]."""

    def __init__(self, env: gym.Env, n_bins: int = 51):
        super().__init__(env)
        self.n_bins = max(2, int(n_bins))
        self.action_space = spaces.Discrete(self.n_bins)

    def _idx_to_x(self, idx: int) -> float:
        t = float(np.clip(idx, 0, self.n_bins - 1)) / float(self.n_bins - 1)
        return float(t - 0.5)

    def step(self, action):
        idx = int(np.asarray(action).reshape(-1)[0])
        x = self._idx_to_x(idx)
        obs, reward, terminated, truncated, info = self.env.step(np.array([x], dtype=np.float32))
        info = dict(info)
        info["action_x"] = x
        info["action_idx"] = idx
        return obs, reward, terminated, truncated, info


class SuikaImageCnnExtractor(BaseFeaturesExtractor):
    """CNN extractor for image-only dict observation."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        image_space = observation_space["image"]
        if len(image_space.shape) != 3:
            raise ValueError(f"Expected 3D image space, got shape={image_space.shape}")

        shp = tuple(int(v) for v in image_space.shape)
        self._channels_first = shp[0] <= 64 and shp[1] > 16 and shp[2] > 16
        c = int(shp[0] if self._channels_first else shp[2])

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
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
            n_flatten = int(self.cnn(sample).shape[1])
        self.linear = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
        self._features_dim = 256

    def forward(self, observations):
        image = observations["image"].float() / 255.0
        if not self._channels_first:
            image = image.permute(0, 3, 1, 2).contiguous()
        return self.linear(self.cnn(image))


class FinalScoreLoggingCallback(BaseCallback):
    """Log episode final score from info['score'] on done steps."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._last_final_score_mean: float | None = None

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
            term_info = info.get("final_info", info)
            score = term_info.get("score", info.get("score", None))
            if score is None:
                continue
            score_f = float(score)
            if not np.isfinite(score_f):
                continue
            final_scores.append(score_f)
            self.logger.record_mean("rollout/final_score", score_f)

        if final_scores:
            mean_score = float(np.mean(final_scores))
            self._last_final_score_mean = mean_score
            self.logger.record("rollout/final_score_mean", mean_score)
        elif self._last_final_score_mean is not None:
            self.logger.record("rollout/final_score_mean", self._last_final_score_mean)
        return True


class ActionStatsLoggingCallback(BaseCallback):
    """Log mapped action-x mean/variance from info."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        xs = []
        for info in infos:
            x = info.get("action_x", None)
            if x is None:
                continue
            xs.append(float(x))
        if xs:
            arr = np.asarray(xs, dtype=np.float32)
            self.logger.record_mean("rollout/action_x_mean", float(np.mean(arr)))
            self.logger.record_mean("rollout/action_x_var", float(np.var(arr)))
        return True


def make_env(
    rank: int,
    seed: int,
    headless: bool,
    port_base: int,
    image_frame_stack: int,
    reward_norm_gamma: float,
    dqn_action_bins: int,
    node_bin: str,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(
            "SuikaEnvNode-v0",
            headless=headless,
            delay_before_img_capture=0.0,
            port=port_base + rank,
            mute_sound=True,
            wait_for_ready_on_step=True,
            ready_poll_interval=0.02,
            ready_timeout=2.0,
            enable_image_observation=True,
            img_width=64,
            img_height=64,
            node_bin=node_bin,
        )
        env = SuikaImageObsWrapper(env)
        env = SuikaImageFrameStackWrapper(env, n_frames=image_frame_stack)
        env = DiscreteActionWrapper(env, n_bins=dqn_action_bins)
        env = gym.wrappers.NormalizeReward(env, gamma=reward_norm_gamma)
        return env

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port-base", type=int, default=8923)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("dqn/models/suika_dqn_node"))
    p.add_argument("--wandb-project", type=str, default="suika-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default="dqn")
    p.add_argument("--save-every-steps", type=int, default=20_000)
    p.add_argument(
        "--gif-eval-every-steps",
        type=int,
        default=10_000,
        help="Export policy gameplay GIF every N timesteps (<=0 disables).",
    )
    p.add_argument(
        "--gif-eval-steps",
        type=int,
        default=10_000,
        help="Safety cap for policy steps per GIF episode (0 disables cap).",
    )
    p.add_argument("--gif-fps", type=int, default=20)
    p.add_argument("--gif-dir", type=Path, default=Path("gifs/dqn"))
    p.add_argument("--device", type=str, default="cuda", help="auto|cpu|cuda|mps")
    p.add_argument("--gpu-id", type=int, default=None)
    p.add_argument("--image-frame-stack", type=int, default=3)
    p.add_argument("--reward-norm-gamma", type=float, default=0.99)
    p.add_argument("--dqn-action-bins", type=int, default=51)
    p.add_argument("--node-bin", type=str, default="node")

    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--target-update-interval", type=int, default=1000)
    p.add_argument("--train-freq", type=int, default=64)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--exploration-fraction", type=float, default=0.01)
    p.add_argument("--exploration-initial-eps", type=float, default=1.0)
    p.add_argument("--exploration-final-eps", type=float, default=0.02)
    return p.parse_args()


def main():
    args = parse_args()
    actual_device = resolve_device(args.device)
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = args.wandb_run_name or f"dqn-suika-node-seed{args.seed}"
    tb_dir = Path("dqn/runs/tb") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(
            rank=i,
            seed=args.seed,
            headless=args.headless,
            port_base=args.port_base,
            image_frame_stack=args.image_frame_stack,
            reward_norm_gamma=args.reward_norm_gamma,
            dqn_action_bins=args.dqn_action_bins,
            node_bin=args.node_bin,
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
            "algo": "DQN",
            "env_id": "SuikaEnvNode-v0",
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "image_frame_stack": args.image_frame_stack,
            "reward_norm_gamma": args.reward_norm_gamma,
            "dqn_action_bins": args.dqn_action_bins,
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "target_update_interval": args.target_update_interval,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "exploration_fraction": args.exploration_fraction,
            "exploration_initial_eps": args.exploration_initial_eps,
            "exploration_final_eps": args.exploration_final_eps,
            "device": actual_device,
            "gpu_id": args.gpu_id,
            "save_every_steps": args.save_every_steps,
            "gif_eval_every_steps": args.gif_eval_every_steps,
            "gif_eval_steps": args.gif_eval_steps,
            "gif_fps": args.gif_fps,
            "node_bin": args.node_bin,
        },
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    interrupted = False
    try:
        policy_kwargs = dict(features_extractor_class=SuikaImageCnnExtractor)
        model = DQN(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=(args.train_freq, "step"),
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            tensorboard_log=str(tb_dir),
            verbose=1,
            seed=args.seed,
            device=actual_device,
        )

        save_freq = args.save_every_steps if args.save_every_steps > 0 else 0
        wandb_callback = WandbCallback(
            gradient_save_freq=0,
            model_save_path=str(Path("dqn/checkpoints/wandb")),
            model_save_freq=save_freq,
            verbose=2,
        )

        callbacks = [wandb_callback, FinalScoreLoggingCallback(verbose=0), ActionStatsLoggingCallback(verbose=0)]

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
                    env_id="SuikaEnvNode-v0",
                    algo="dqn",
                    total_timesteps=args.total_timesteps,
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
