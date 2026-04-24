from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
    model,
    make_eval_env: Callable[[int, int], object],
    steps_per_gif: int,
    fps: int,
    gif_path: str,
    export_idx: int,
    num_timesteps: int,
    verbose: int,
):
    envs = [make_eval_env(i, export_idx) for i in range(4)]
    frames = []
    try:
        obs_list = []
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=1000 * export_idx + i)
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
    finally:
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
        make_eval_env: Callable[[int, int], object],
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.every_steps = max(1, int(every_steps))
        self.steps_per_gif = max(1, int(steps_per_gif))
        self.fps = max(1, int(fps))
        self.out_dir = out_dir
        self.make_eval_env = make_eval_env
        self._last_export_at = 0
        self._export_idx = 0
        self._proc: mp.Process | None = None

    def _start_policy_gif_process(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Keep only one GIF path (overwrite).
        gif_path = self.out_dir / "latest_policy.gif"

        # Use fork so child can use current model object without pickling/reloading.
        ctx = mp.get_context("fork")
        proc = ctx.Process(
            target=_generate_policy_gif_worker,
            args=(
                self.model,
                self.make_eval_env,
                self.steps_per_gif,
                self.fps,
                str(gif_path),
                self._export_idx,
                self.num_timesteps,
                self.verbose,
            ),
            daemon=True,
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
            # Do not block for long at training end.
            self._proc.join(timeout=1.0)
