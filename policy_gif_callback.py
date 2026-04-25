from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import imageio.v2 as imageio
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

import suika_env  # noqa: F401


def _obs_to_hwc(raw_env):
    # Use raw canvas capture from underlying browser env through wrappers.
    base_env = getattr(raw_env, "unwrapped", raw_env)
    if not hasattr(base_env, "capture_canvas_raw_rgba"):
        raise AttributeError(f"{type(raw_env).__name__} has no capture_canvas_raw_rgba")
    return base_env.capture_canvas_raw_rgba()


def _build_model_obs(obs: dict, model: PPO) -> dict:
    # Pass only keys the trained policy expects (supports train.py and train_transformer.py).
    keys = model.observation_space.spaces.keys()
    return {k: np.expand_dims(obs[k], axis=0) for k in keys}


def _get_sigma(model: PPO, model_obs: dict) -> float:
    with np.errstate(all="ignore"):
        obs_tensor = obs_as_tensor(model_obs, model.device)
        dist = model.policy.get_distribution(obs_tensor).distribution
        if hasattr(dist, "stddev"):
            std = dist.stddev.detach().cpu().numpy().reshape(-1)
            if std.size > 0:
                return float(std[0])
        if hasattr(model.policy, "log_std"):
            log_std = model.policy.log_std.detach().cpu().numpy().reshape(-1)
            if log_std.size > 0:
                return float(np.exp(log_std[0]))
    return float("nan")


def _generate_policy_gif_worker(
    model_path: str,
    max_steps_per_episode: int,
    fps: int,
    gif_path: str,
    export_idx: int,
    num_timesteps: int,
    verbose: int,
    seed: int,
    headless: bool,
    port_base: int,
):
    try:
        model = PPO.load(model_path, device="cpu")
        action_log_path = str(Path(gif_path).with_name("latest_policy_action.txt"))

        def make_eval_env(rank: int):
            port = port_base + 1000 + rank
            env = gym.make(
                "SuikaEnv-v0",
                headless=headless,
                delay_before_img_capture=0.0,
                port=port,
                mute_sound=True,
                wait_for_ready_on_step=True,
                ready_poll_interval=0.02,
                ready_timeout=2.0,
            )
            return env

        raw_envs = [make_eval_env(0)]
        envs = raw_envs
        frames = []
        action_logs = []
        obs_list = []
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=seed + 1000 * export_idx + i)
            obs_list.append(obs)

        frames.append(_obs_to_hwc(raw_envs[0]))

        step_count = 0
        while True:
            next_obs_list = []
            done = False
            for i, env in enumerate(envs):
                obs = obs_list[i]
                model_obs = _build_model_obs(obs, model)
                sigma = _get_sigma(model, model_obs)
                action, _ = model.predict(model_obs, deterministic=True)
                action = np.asarray(action).reshape(-1)
                obs2, reward, terminated, truncated, _ = env.step(action)
                done = done or bool(terminated or truncated)
                next_obs_list.append(obs2)
                x = float(action[0]) if action.size > 0 else float("nan")
                action_logs.append(
                    f"{step_count + 1}\t{float(reward):.6f}\t{x:.6f}\t{action.tolist()}\t{sigma:.6f}"
                )
            obs_list = next_obs_list
            frames.append(_obs_to_hwc(raw_envs[0]))
            step_count += 1
            if done:
                break
            if max_steps_per_episode > 0 and step_count >= max_steps_per_episode:
                if verbose > 0:
                    print(
                        "[PolicyGifCallback] Reached GIF max steps without episode end: "
                        f"max_steps={max_steps_per_episode}"
                    )
                break

        imageio.mimsave(gif_path, frames, fps=fps)
        with open(action_log_path, "w", encoding="utf-8") as f:
            f.write("step\treward\tx\taction\tsigma\n")
            for line in action_logs:
                f.write(line + "\n")
        if verbose > 0:
            print(f"[PolicyGifCallback] Saved {gif_path} at step={num_timesteps}")
    except Exception as exc:
        err_path = str(Path(gif_path).with_suffix(".error.txt"))
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"GIF worker failed at step={num_timesteps}\n{type(exc).__name__}: {exc}\n")
        raise
    finally:
        envs = locals().get("envs", [])
        raw_envs = locals().get("raw_envs", [])
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        for env in raw_envs:
            try:
                env.close()
            except Exception:
                pass


class PolicyGifCallback(BaseCallback):
    """Export current-policy gameplay GIF periodically in a separate process."""

    def __init__(
        self,
        every_steps: int,
        max_steps_per_episode: int,
        fps: int,
        out_dir: Path,
        seed: int,
        headless: bool,
        port_base: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.every_steps = max(1, int(every_steps))
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.fps = max(1, int(fps))
        self.out_dir = out_dir
        self.seed = int(seed)
        self.headless = bool(headless)
        self.port_base = int(port_base)
        self._last_export_at = 0
        self._export_idx = 0
        self._proc: mp.Process | None = None

    def _start_policy_gif_process(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Keep only one GIF path (overwrite).
        gif_path = self.out_dir / "latest_policy.gif"
        model_path = self.out_dir / "latest_policy_model.zip"
        self.model.save(str(model_path))

        # Use spawn for CUDA safety.
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_generate_policy_gif_worker,
            args=(
                str(model_path),
                self.max_steps_per_episode,
                self.fps,
                str(gif_path),
                self._export_idx,
                self.num_timesteps,
                self.verbose,
                self.seed,
                self.headless,
                self.port_base,
            ),
            daemon=False,
        )
        proc.start()
        self._proc = proc
        if self.verbose > 0:
            print(
                f"[PolicyGifCallback] Started GIF worker pid={proc.pid} "
                f"at step={self.num_timesteps}"
            )

    def _on_step(self) -> bool:
        if self._proc is not None and not self._proc.is_alive():
            self._proc = None

        if (self.num_timesteps - self._last_export_at) < self.every_steps:
            return True

        # If previous export is still running, skip this trigger to avoid queueing.
        if self._proc is not None and self._proc.is_alive():
            if self.verbose > 0:
                print(
                    f"[PolicyGifCallback] Skip export at step={self.num_timesteps} "
                    "(previous GIF worker still running)"
                )
            self._last_export_at = self.num_timesteps
            return True

        self._start_policy_gif_process()
        self._last_export_at = self.num_timesteps
        self._export_idx += 1
        return True

    def _on_training_end(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            # Give GIF worker enough time to finish writing output.
            self._proc.join(timeout=300.0)
            if self._proc.is_alive():
                self._proc.terminate()
