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
import subprocess
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
from policy_gif_callback import PolicyGifCallback


def restore_terminal_cursor():
    # Ensure cursor/TTY state is restored after rich/tqdm interruption.
    for cmd in (["stty", "sane"], ["tput", "cnorm"]):
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass


class SuikaObsWrapper(gym.ObservationWrapper):
    """Use feature-only observation for policy input (no image)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        fruit_type_space = env.observation_space["current_fruit_type"]
        next_fruit_type_space = env.observation_space["next_fruit_type"]
        fruit_x_space = env.observation_space["current_fruit_x"]
        stage_top10_xy_space = env.observation_space["stage_top10_xy"]
        top10_fruit_types_space = env.observation_space["top10_fruit_types"]
        top10_mask_space = env.observation_space["top10_mask"]
        max_height_space = env.observation_space["max_height"]
        danger_count_space = env.observation_space["danger_count"]
        largest_fruit_type_space = env.observation_space["largest_fruit_type"]
        fruit_count_space = env.observation_space["fruit_count"]
        self.observation_space = spaces.Dict(
            {
                "current_fruit_type": spaces.Box(
                    low=fruit_type_space.low.astype(np.float32),
                    high=fruit_type_space.high.astype(np.float32),
                    shape=fruit_type_space.shape,
                    dtype=np.float32,
                ),
                "next_fruit_type": spaces.Box(
                    low=next_fruit_type_space.low.astype(np.float32),
                    high=next_fruit_type_space.high.astype(np.float32),
                    shape=next_fruit_type_space.shape,
                    dtype=np.float32,
                ),
                "current_fruit_x": spaces.Box(
                    low=fruit_x_space.low.astype(np.float32),
                    high=fruit_x_space.high.astype(np.float32),
                    shape=fruit_x_space.shape,
                    dtype=np.float32,
                ),
                "stage_top10_xy": spaces.Box(
                    low=stage_top10_xy_space.low.astype(np.float32),
                    high=stage_top10_xy_space.high.astype(np.float32),
                    shape=stage_top10_xy_space.shape,
                    dtype=np.float32,
                ),
                "top10_fruit_types": spaces.Box(
                    low=top10_fruit_types_space.low.astype(np.float32),
                    high=top10_fruit_types_space.high.astype(np.float32),
                    shape=top10_fruit_types_space.shape,
                    dtype=np.float32,
                ),
                "top10_mask": spaces.Box(
                    low=top10_mask_space.low.astype(np.float32),
                    high=top10_mask_space.high.astype(np.float32),
                    shape=top10_mask_space.shape,
                    dtype=np.float32,
                ),
                "max_height": spaces.Box(
                    low=max_height_space.low.astype(np.float32),
                    high=max_height_space.high.astype(np.float32),
                    shape=max_height_space.shape,
                    dtype=np.float32,
                ),
                "danger_count": spaces.Box(
                    low=danger_count_space.low.astype(np.float32),
                    high=danger_count_space.high.astype(np.float32),
                    shape=danger_count_space.shape,
                    dtype=np.float32,
                ),
                "largest_fruit_type": spaces.Box(
                    low=largest_fruit_type_space.low.astype(np.float32),
                    high=largest_fruit_type_space.high.astype(np.float32),
                    shape=largest_fruit_type_space.shape,
                    dtype=np.float32,
                ),
                "fruit_count": spaces.Box(
                    low=fruit_count_space.low.astype(np.float32),
                    high=fruit_count_space.high.astype(np.float32),
                    shape=fruit_count_space.shape,
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation):
        fruit_type = observation["current_fruit_type"].astype(np.float32, copy=False)
        next_fruit_type = observation["next_fruit_type"].astype(np.float32, copy=False)
        fruit_x = observation["current_fruit_x"].astype(np.float32, copy=False)
        stage_top10_xy = observation["stage_top10_xy"].astype(np.float32, copy=False)
        top10_fruit_types = observation["top10_fruit_types"].astype(np.float32, copy=False)
        top10_mask = observation["top10_mask"].astype(np.float32, copy=False)
        max_height = observation["max_height"].astype(np.float32, copy=False)
        danger_count = observation["danger_count"].astype(np.float32, copy=False)
        largest_fruit_type = observation["largest_fruit_type"].astype(np.float32, copy=False)
        fruit_count = observation["fruit_count"].astype(np.float32, copy=False)
        return {
            "current_fruit_type": fruit_type,
            "next_fruit_type": next_fruit_type,
            "current_fruit_x": fruit_x,
            "stage_top10_xy": stage_top10_xy,
            "top10_fruit_types": top10_fruit_types,
            "top10_mask": top10_mask,
            "max_height": max_height,
            "danger_count": danger_count,
            "largest_fruit_type": largest_fruit_type,
            "fruit_count": fruit_count,
        }


class SuikaCombinedExtractor(BaseFeaturesExtractor):
    """Feature-only extractor (no image)."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        num_fruit_types = int(observation_space["current_fruit_type"].high[0]) + 1
        self.current_fruit_embedding = nn.Embedding(num_fruit_types, 8)
        self.next_fruit_embedding = nn.Embedding(num_fruit_types, 8)
        self.largest_fruit_embedding = nn.Embedding(num_fruit_types, 4)
        self.top10_fruit_embedding = nn.Embedding(num_fruit_types, 4)
        self.num_fruit_types = num_fruit_types
        # 8(current type emb) + 8(next type emb) + 4(largest type emb)
        # + 40(top10 type emb with mask) + 1(current x) + 20(top10 xy) + 10(top10 mask)
        # + 1(max height) + 1(danger count) + 1(fruit count) = 94
        self.aux_head = nn.Sequential(nn.Linear(94, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self._features_dim = 64

    def forward(self, observations):
        fruit_x = observations["current_fruit_x"].float()
        stage_top10_xy = observations["stage_top10_xy"].float()
        top10_fruit_types = observations["top10_fruit_types"].long().clamp(0, self.num_fruit_types - 1)
        top10_mask = observations["top10_mask"].float()
        max_height = observations["max_height"].float()
        danger_count = observations["danger_count"].float() / 10.0
        fruit_count = observations["fruit_count"].float() / 50.0
        fruit_type_idx = observations["current_fruit_type"].long().squeeze(1)
        next_fruit_type_idx = observations["next_fruit_type"].long().squeeze(1)
        largest_fruit_type_idx = observations["largest_fruit_type"].long().squeeze(1)
        fruit_type_emb = self.current_fruit_embedding(fruit_type_idx)
        next_fruit_emb = self.next_fruit_embedding(next_fruit_type_idx)
        largest_fruit_emb = self.largest_fruit_embedding(largest_fruit_type_idx)
        top10_type_emb = self.top10_fruit_embedding(top10_fruit_types)
        top10_type_emb = top10_type_emb * top10_mask.unsqueeze(-1)
        top10_type_emb = top10_type_emb.flatten(start_dim=1)
        aux = th.cat(
            [
                fruit_type_emb,
                next_fruit_emb,
                largest_fruit_emb,
                top10_type_emb,
                fruit_x,
                stage_top10_xy,
                top10_mask,
                max_height,
                danger_count,
                fruit_count,
            ],
            dim=1,
        )
        return self.aux_head(aux)


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
            # Continuous 1D action space: first column is x in [0, 1].
            x = arr[:, 0].astype(np.float32)

        x_mean = float(np.mean(x))
        x_var = float(np.var(x))
        self.logger.record("rollout/action_x_mean", x_mean)
        self.logger.record("rollout/action_x_var", x_var)
        wandb.log(
            {
                "rollout/action_x_mean": x_mean,
                "rollout/action_x_var": x_var,
            },
            step=self.num_timesteps,
        )
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
        wandb.log(
            {
                "rollout/policy_log_std_mean": log_std_mean,
                "rollout/policy_std_mean": std_mean,
                "rollout/policy_std_var": std_var,
            },
            step=self.num_timesteps,
        )


def make_env(
    rank: int,
    seed: int,
    headless: bool,
    delay: float,
    port_base: int,
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
    p.add_argument(
        "--gif-eval-every-steps",
        type=int,
        default=10_000,
        help="Export policy gameplay GIF every N timesteps (<=0 disables).",
    )
    p.add_argument(
        "--gif-eval-steps",
        type=int,
        default=400,
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
        default=Path("gifs"),
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
                    verbose=1,
                )
            )
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
        restore_terminal_cursor()
    finally:
        vec_env.close()
        run.finish()
        restore_terminal_cursor()
    if not interrupted:
        print(f"Saved model to: {args.save_path}.zip")


if __name__ == "__main__":
    main()
