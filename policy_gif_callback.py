from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import imageio.v2 as imageio
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import suika_env  # noqa: F401


def _obs_to_hwc(obs_dict):
    # obs["image"]: (C*k, H, W). Use latest RGBA frame (last 4 channels).
    chw = obs_dict["image"]
    if chw.shape[0] >= 4:
        latest = chw[-4:, :, :]
    else:
        latest = chw
    return np.transpose(latest, (1, 2, 0))


def _make_grid(images):
    top = np.concatenate(images[:2], axis=1)
    bottom = np.concatenate(images[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _generate_policy_gif_worker(
    model_path: str,
    steps_per_gif: int,
    fps: int,
    gif_path: str,
    export_idx: int,
    num_timesteps: int,
    verbose: int,
    seed: int,
    headless: bool,
    frame_stack: int,
    port_base: int,
):
    try:
        # Import wrappers lazily to avoid import cycle at module import time.
        from train import SuikaFrameStackWrapper, SuikaObsWrapper

        model = PPO.load(model_path, device="cpu")

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
            env = SuikaObsWrapper(env)
            env = SuikaFrameStackWrapper(env, k=frame_stack)
            return env

        envs = [make_eval_env(i) for i in range(4)]
        frames = []
        obs_list = []
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=seed + 1000 * export_idx + i)
            obs_list.append(obs)

        frames.append(_make_grid([_obs_to_hwc(o) for o in obs_list]))

        for _ in range(steps_per_gif):
            next_obs_list = []
            for i, env in enumerate(envs):
                obs = obs_list[i]
                model_obs = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
                action, _ = model.predict(model_obs, deterministic=True)
                action = np.asarray(action).reshape(-1)
                obs2, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs2, _ = env.reset()
                next_obs_list.append(obs2)
            obs_list = next_obs_list
            frames.append(_make_grid([_obs_to_hwc(o) for o in obs_list]))

        imageio.mimsave(gif_path, frames, fps=fps)
        if verbose > 0:
            print(f"[PolicyGifCallback] Saved {gif_path} at step={num_timesteps}")
    except Exception as exc:
        err_path = str(Path(gif_path).with_suffix(".error.txt"))
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"GIF worker failed at step={num_timesteps}\n{type(exc).__name__}: {exc}\n")
        raise
    finally:
        envs = locals().get("envs", [])
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


class PolicyGifCallback(BaseCallback):
    """Export current-policy gameplay GIF periodically in a separate process."""

    def __init__(
        self,
        every_steps: int,
        steps_per_gif: int,
        fps: int,
        out_dir: Path,
        seed: int,
        headless: bool,
        frame_stack: int,
        port_base: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.every_steps = max(1, int(every_steps))
        self.steps_per_gif = max(1, int(steps_per_gif))
        self.fps = max(1, int(fps))
        self.out_dir = out_dir
        self.seed = int(seed)
        self.headless = bool(headless)
        self.frame_stack = max(1, int(frame_stack))
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
                self.steps_per_gif,
                self.fps,
                str(gif_path),
                self._export_idx,
                self.num_timesteps,
                self.verbose,
                self.seed,
                self.headless,
                self.frame_stack,
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
