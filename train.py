"""
PPO training (CNN image input) for SuikaEnv-v0 / SuikaEnvNode-v0.

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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

import suika_env  # noqa: F401  # Registers "SuikaEnv-v0"
import suika_env_node  # noqa: F401  # Registers "SuikaEnvNode-v0"
from policy_gif_callback import PolicyGifCallback


def restore_terminal_cursor():
    # Ensure cursor/TTY state is restored after rich/tqdm interruption.
    for cmd in (["stty", "sane"], ["tput", "cnorm"]):
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass


class SuikaImageObsWrapper(gym.ObservationWrapper):
    """Use image-only observation for CNN policy input."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        image_space = env.observation_space["image"]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=image_space.low.astype(np.uint8),
                    high=image_space.high.astype(np.uint8),
                    shape=image_space.shape,
                    dtype=np.uint8,
                )
            }
        )

    def observation(self, observation):
        return {"image": observation["image"]}


class SuikaImageFrameStackWrapper(gym.Wrapper):
    """Stack the latest N image frames along channel axis (H, W, C*N)."""

    def __init__(self, env: gym.Env, n_frames: int = 3):
        super().__init__(env)
        self.n_frames = max(1, int(n_frames))
        base_img_space = env.observation_space["image"]
        h, w, c = base_img_space.shape
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
        stacked = np.concatenate(list(self._frames), axis=2)
        return {"image": stacked}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        img = obs["image"]
        self._frames.clear()
        for _ in range(self.n_frames):
            self._frames.append(img.copy())
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs["image"].copy())
        return self._stacked_obs(), reward, terminated, truncated, info


class SuikaImageCnnExtractor(BaseFeaturesExtractor):
    """CNN extractor for image-only dict observation."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        image_space = observation_space["image"]
        if len(image_space.shape) != 3:
            raise ValueError(f"Expected 3D image space, got shape={image_space.shape}")
        # Support both HWC and CHW inputs.
        shp = tuple(int(v) for v in image_space.shape)
        self._channels_first = shp[0] <= 16 and shp[1] > 16 and shp[2] > 16
        if self._channels_first:
            c = int(shp[0])
        else:
            c = int(shp[2])
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
        self._last_final_score_mean: float | None = None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if dones is None or infos is None:
            return True

        final_scores = []
        # SubprocVecEnv may provide terminal-only info under "final_info".
        # Prefer terminal score when available.
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
            # Aggregate done-episode scores within current SB3 log window.
            self.logger.record_mean("rollout/final_score", score_f)

        if final_scores:
            mean_score = float(np.mean(final_scores))
            self._last_final_score_mean = mean_score
            self.logger.record("rollout/final_score_mean", mean_score)
        else:
            # Keep the latest valid value when no episode ended in this step window.
            if self._last_final_score_mean is not None:
                self.logger.record("rollout/final_score_mean", self._last_final_score_mean)
        if final_scores and self.verbose > 0:
            mean_score = float(np.mean(final_scores))
            print(
                f"[FinalScoreLoggingCallback] step={self.num_timesteps} "
                f"final_score_mean={mean_score:.2f} n={len(final_scores)}"
            )
        return True


class ActionStatsLoggingCallback(BaseCallback):
    """Log clipped action-x mean/variance during rollout collection."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        actions = self.locals.get("clipped_actions", None)
        if actions is None:
            actions = self.locals.get("actions", None)
        if actions is None:
            return True

        arr = np.asarray(actions)
        if arr.size == 0:
            return True
        if arr.ndim == 1:
            x = arr.astype(np.float32)
        else:
            x = arr[:, 0].astype(np.float32)
        # For centered action-space envs, clamp for interpretable rollout stats.
        x = np.clip(x, -0.5, 0.5)

        x_mean = float(np.mean(x))
        x_var = float(np.var(x))
        # Mean over callback calls within rollout/log window.
        self.logger.record_mean("rollout/action_x_mean", x_mean)
        self.logger.record_mean("rollout/action_x_var", x_var)
        return True


class LoseHeightDebugLoggingCallback(BaseCallback):
    """Log LOSE-height trigger diagnostics from env info."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None:
            return True

        lose_hits = []
        collision_pairs = []
        triggered = []
        ay_values = []
        by_values = []

        dones_arr = np.asarray(dones).tolist() if dones is not None else [False] * len(infos)
        for i, info in enumerate(infos):
            step_pairs = info.get("collision_pairs_step", None)
            step_hits = info.get("lose_height_hits_step", None)
            step_trig = info.get("lose_height_triggered", None)
            if step_pairs is not None:
                collision_pairs.append(float(step_pairs))
            if step_hits is not None:
                lose_hits.append(float(step_hits))
            if step_trig is not None:
                triggered.append(float(step_trig))

            if i < len(dones_arr) and dones_arr[i]:
                term_info = info.get("final_info", info)
                lose_event = term_info.get("lose_event", None)
                if isinstance(lose_event, dict):
                    if "aY" in lose_event:
                        ay_values.append(float(lose_event["aY"]))
                    if "bY" in lose_event:
                        by_values.append(float(lose_event["bY"]))

        if collision_pairs:
            self.logger.record_mean("rollout/collision_pairs_step", float(np.mean(collision_pairs)))
        if lose_hits:
            self.logger.record_mean("rollout/lose_height_hits_step", float(np.mean(lose_hits)))
        if triggered:
            self.logger.record_mean("rollout/lose_height_trigger_rate", float(np.mean(triggered)))
        if ay_values:
            self.logger.record_mean("rollout/lose_event_aY", float(np.mean(ay_values)))
        if by_values:
            self.logger.record_mean("rollout/lose_event_bY", float(np.mean(by_values)))
        return True


class PolicyStdLoggingCallback(BaseCallback):
    """Log policy std (and log_std) for continuous-action PPO policies."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        policy = self.model.policy
        if not hasattr(policy, "log_std"):
            return
        log_std_t = policy.log_std.detach()
        std_t = th.exp(log_std_t)
        log_std_mean = float(log_std_t.mean().cpu().item())
        std_mean = float(std_t.mean().cpu().item())
        std_var = float(std_t.var(unbiased=False).cpu().item())
        self.logger.record("rollout/policy_log_std_mean", log_std_mean)
        self.logger.record("rollout/policy_std_mean", std_mean)
        self.logger.record("rollout/policy_std_var", std_var)


def make_env(
    rank: int,
    seed: int,
    headless: bool,
    delay: float,
    port_base: int,
    env_id: str,
    image_frame_stack: int,
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
            enable_image_observation=True,
        )
        if env_id == "SuikaEnv-v0":
            env_kwargs["port"] = port_base + rank
        env = gym.make(env_id, **env_kwargs)
        env = SuikaImageObsWrapper(env)
        env = SuikaImageFrameStackWrapper(env, n_frames=image_frame_stack)
        env = gym.wrappers.NormalizeReward(env, gamma=reward_norm_gamma)
        return env

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=16)
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
    p.add_argument(
        "--image-frame-stack",
        type=int,
        default=3,
        help="Number of stacked image frames for CNN input.",
    )
    p.add_argument(
        "--reward-norm-gamma",
        type=float,
        default=0.99,
        help="Gamma for reward normalization wrapper.",
    )
    p.add_argument(
        "--env-id",
        type=str,
        default="SuikaEnvNode-v0",
        choices=["SuikaEnv-v0", "SuikaEnvNode-v0"],
        help="Environment id for training backend.",
    )
    p.add_argument("--port-base", type=int, default=8923)
    p.add_argument("--delay-before-img-capture", type=float, default=0.1)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--save-path", type=Path, default=Path("models/cnn/ppo_suika"))
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
    p.add_argument(
        "--gif-eval-every-steps",
        type=int,
        default=10_000,
        help="Export policy gameplay GIF every N timesteps (<=0 disables).",
    )
    p.add_argument(
        "--gif-eval-steps",
        type=int,
        default=10000,
        help="Safety cap for policy steps per GIF episode (0 disables cap).",
    )
    p.add_argument(
        "--gif-fps",
        type=int,
        default=20,
        help="GIF playback FPS (higher is faster).",
    )
    p.add_argument(
        "--gif-dir",
        type=Path,
        default=Path("gifs/cnn"),
        help="Directory for periodic policy GIFs.",
    )
    p.add_argument("--device", type=str, default="cuda", help="SB3 device, e.g. auto|cpu|cuda")
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
            args.env_id,
            args.image_frame_stack,
            args.reward_norm_gamma,
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
            "env_id": args.env_id,
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "image_frame_stack": args.image_frame_stack,
            "reward_norm_gamma": args.reward_norm_gamma,
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
        policy_kwargs = dict(features_extractor_class=SuikaImageCnnExtractor)
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
        run.finish()
        restore_terminal_cursor()
    if not interrupted:
        print(f"Saved model to: {args.save_path}.zip")


if __name__ == "__main__":
    main()
