from __future__ import annotations

import multiprocessing as mp
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

import suika_env  # noqa: F401
import suika_env_node  # noqa: F401


def _obs_to_hwc(raw_env):
    # Use raw canvas capture from underlying browser env through wrappers.
    base_env = getattr(raw_env, "unwrapped", raw_env)
    if not hasattr(base_env, "capture_canvas_raw_rgba"):
        raise AttributeError(f"{type(raw_env).__name__} has no capture_canvas_raw_rgba")
    return base_env.capture_canvas_raw_rgba()


class _ImageObsAdapter:
    """Adapt raw env image obs to saved model image shape (stack/transposed)."""

    def __init__(self, model: PPO):
        img_space = model.observation_space.spaces.get("image", None)
        self.enabled = img_space is not None
        self.frames = None
        self.n_frames = 1
        self.channels_first = False
        self.expected_shape = None
        if not self.enabled:
            return

        shp = tuple(int(v) for v in img_space.shape)
        if len(shp) != 3:
            return
        self.expected_shape = shp
        # CHW if first dim looks like channels.
        self.channels_first = shp[0] <= 64 and shp[1] > 16 and shp[2] > 16

    def _init_if_needed(self, image_hwc: np.ndarray):
        if not self.enabled or self.frames is not None:
            return
        c_cur = int(image_hwc.shape[2])
        if self.channels_first:
            c_exp = int(self.expected_shape[0])
        else:
            c_exp = int(self.expected_shape[2])
        self.n_frames = max(1, c_exp // max(1, c_cur))
        self.frames = deque(maxlen=self.n_frames)
        for _ in range(self.n_frames):
            self.frames.append(image_hwc.copy())

    def reset(self, image_hwc: np.ndarray):
        if not self.enabled:
            return
        self.frames = None
        self._init_if_needed(image_hwc)

    def update(self, image_hwc: np.ndarray):
        if not self.enabled:
            return
        self._init_if_needed(image_hwc)
        self.frames.append(image_hwc.copy())

    def transform(self, image_hwc: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image_hwc
        self._init_if_needed(image_hwc)
        if self.n_frames > 1:
            out = np.concatenate(list(self.frames), axis=2)
        else:
            out = image_hwc
        if self.channels_first:
            out = np.transpose(out, (2, 0, 1))
        return out


def _build_model_obs(obs: dict, model: PPO, image_adapter: _ImageObsAdapter | None = None) -> dict:
    # Pass only keys the trained policy expects (supports train.py and train_transformer.py).
    keys = model.observation_space.spaces.keys()
    model_obs = {}
    for k in keys:
        v = obs[k]
        if k == "image" and image_adapter is not None:
            v = image_adapter.transform(v)
        model_obs[k] = np.expand_dims(v, axis=0)
    return model_obs


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
    env_id: str,
):
    try:
        model = PPO.load(model_path, device="cpu")
        image_adapter = _ImageObsAdapter(model)
        gif_file = Path(gif_path)
        action_log_path = str(gif_file.with_name(f"{gif_file.stem}_action.txt"))

        def make_eval_env(rank: int):
            port = port_base + 1000 + rank
            env_kwargs = dict(
                headless=headless,
                delay_before_img_capture=0.0,
                port=port,
                mute_sound=True,
                wait_for_ready_on_step=True,
                ready_poll_interval=0.02,
                ready_timeout=2.0,
            )
            # For node env, enable GUI only during GIF generation.
            if env_id == "SuikaEnvNode-v0":
                env_kwargs["gui"] = True
                env_kwargs["gui_fps"] = float(max(1, fps))
            env = gym.make(
                env_id,
                **env_kwargs,
            )
            return env

        raw_envs = [make_eval_env(0)]
        envs = raw_envs
        frames = []
        action_logs = []
        obs_list = []
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=seed + 1000 * export_idx + i)
            if "image" in obs:
                image_adapter.reset(obs["image"])
            obs_list.append(obs)

        frames.append(_obs_to_hwc(raw_envs[0]))

        step_count = 0
        while True:
            next_obs_list = []
            done = False
            for i, env in enumerate(envs):
                obs = obs_list[i]
                model_obs = _build_model_obs(obs, model, image_adapter=image_adapter)
                sigma = _get_sigma(model, model_obs)
                action, _ = model.predict(model_obs, deterministic=False)
                action = np.asarray(action).reshape(-1)
                obs2, reward, terminated, truncated, info = env.step(action)
                if "image" in obs2:
                    image_adapter.update(obs2["image"])
                done = done or bool(terminated or truncated)
                next_obs_list.append(obs2)
                x = float(action[0]) if action.size > 0 else float("nan")
                score = float(info.get("score", float("nan")))
                action_logs.append(
                    f"{step_count + 1}\t{float(reward):.6f}\t{x:.6f}\t{action.tolist()}\t"
                    f"{sigma:.6f}\t{score:.6f}\t{int(bool(terminated))}\t{int(bool(truncated))}"
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
            f.write("step\treward\tx\taction\tsigma\tscore\tterminated\ttruncated\n")
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
        env_id: str = "SuikaEnv-v0",
        total_timesteps: int | None = None,
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
        self.env_id = str(env_id)
        self.total_timesteps_target = int(total_timesteps) if total_timesteps is not None else None
        self._last_export_at = 0
        self._export_idx = 0
        self._proc: mp.Process | None = None
        self._pending_milestones: list[tuple[int, int]] = []
        self._done_milestone_pcts: set[int] = set()
        if self.total_timesteps_target is not None and self.total_timesteps_target > 0:
            for pct in range(10, 101, 10):
                step = max(1, int(self.total_timesteps_target * pct / 100))
                self._pending_milestones.append((pct, step))

    def _start_policy_gif_process(self, gif_name: str):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = self.out_dir / gif_name
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
                self.env_id,
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

        if self._proc is not None and self._proc.is_alive():
            return True

        # 10%-milestone exports (policy_10pct.gif ... policy_100pct.gif).
        if self._pending_milestones:
            ready = [(pct, step) for (pct, step) in self._pending_milestones if self.num_timesteps >= step]
            if ready:
                pct, step = ready[0]
                self._pending_milestones = [
                    (p, s) for (p, s) in self._pending_milestones if p != pct
                ]
                if pct not in self._done_milestone_pcts:
                    gif_name = f"policy_{pct}pct.gif"
                    self._start_policy_gif_process(gif_name=gif_name)
                    self._done_milestone_pcts.add(pct)
                    if self.verbose > 0:
                        print(
                            f"[PolicyGifCallback] Milestone GIF queued: {gif_name} "
                            f"at step={self.num_timesteps} (target={step})"
                        )
                    self._export_idx += 1
                    return True

        # Periodic latest GIF export.
        if (self.num_timesteps - self._last_export_at) >= self.every_steps:
            self._start_policy_gif_process(gif_name="latest_policy.gif")
            self._last_export_at = self.num_timesteps
            self._export_idx += 1
        return True

    def _on_training_end(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            # Give GIF worker enough time to finish writing output.
            self._proc.join(timeout=300.0)
            if self._proc.is_alive():
                self._proc.terminate()
