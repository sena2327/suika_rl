from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class PolicyGifCallback(BaseCallback):
    """Export current-policy gameplay GIF periodically in 2x2 grid style."""

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

    def _obs_to_hwc(self, obs_dict):
        # obs["image"]: (C*k, H, W). Use latest RGBA frame (last 4 channels).
        chw = obs_dict["image"]
        if chw.shape[0] >= 4:
            latest = chw[-4:, :, :]
        else:
            latest = chw
        return np.transpose(latest, (1, 2, 0))

    def _make_grid(self, images):
        top = np.concatenate(images[:2], axis=1)
        bottom = np.concatenate(images[2:], axis=1)
        return np.concatenate([top, bottom], axis=0)

    def _record_policy_gif(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = self.out_dir / f"policy_step_{self.num_timesteps:08d}.gif"
        envs = [self.make_eval_env(i, self._export_idx) for i in range(4)]
        frames = []

        try:
            obs_list = []
            for i, env in enumerate(envs):
                obs, _ = env.reset(seed=1000 * self._export_idx + i)
                obs_list.append(obs)

            frames.append(self._make_grid([self._obs_to_hwc(o) for o in obs_list]))

            for _ in range(self.steps_per_gif):
                next_obs_list = []
                for i, env in enumerate(envs):
                    obs = obs_list[i]
                    model_obs = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
                    action, _ = self.model.predict(model_obs, deterministic=True)
                    action = np.asarray(action).reshape(-1)
                    obs2, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        obs2, _ = env.reset()
                    next_obs_list.append(obs2)
                obs_list = next_obs_list
                frames.append(self._make_grid([self._obs_to_hwc(o) for o in obs_list]))

            imageio.mimsave(gif_path, frames, fps=self.fps)
            wandb.log(
                {"eval/policy_gif": wandb.Video(str(gif_path), fps=self.fps, format="gif")},
                step=self.num_timesteps,
            )
            if self.verbose > 0:
                print(f"[PolicyGifCallback] Saved {gif_path}")
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_export_at) < self.every_steps:
            return True
        try:
            self._record_policy_gif()
        except Exception as exc:
            print(f"[PolicyGifCallback] GIF export failed at step={self.num_timesteps}: {exc}")
        self._last_export_at = self.num_timesteps
        self._export_idx += 1
        return True

