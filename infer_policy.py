"""
Run a trained PPO policy on SuikaEnv with live visualization.

Features:
  - Shows game screen (left)
  - Shows per-step table of x and sigma (right)
  - Runs the agent directly in real environment
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

import suika_env  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True, help="Path to model zip")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--port", type=int, default=8923)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--table-rows", type=int, default=20)
    p.add_argument("--mute-sound", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true", default=False, help="Disable per-step stdout logs.")
    return p.parse_args()


def build_model_obs(obs: dict, model: PPO) -> dict:
    keys = model.observation_space.spaces.keys()
    return {k: np.expand_dims(obs[k], axis=0) for k in keys}


def get_sigma(model: PPO, model_obs: dict) -> float:
    with th.no_grad():
        obs_tensor = obs_as_tensor(model_obs, model.device)
        dist = model.policy.get_distribution(obs_tensor).distribution
        if hasattr(dist, "stddev"):
            std = dist.stddev.detach().cpu().numpy().reshape(-1)
            return float(std[0])
        if hasattr(model.policy, "log_std"):
            log_std = model.policy.log_std.detach().cpu().numpy().reshape(-1)
            return float(np.exp(log_std[0]))
    return float("nan")


def get_frame(env) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    if hasattr(base, "capture_canvas_raw_rgba"):
        return base.capture_canvas_raw_rgba()
    img = base.render() if hasattr(base, "render") else None
    if img is None:
        raise RuntimeError("Could not capture frame from environment.")
    return img


def get_raw_js_score(env) -> float:
    base = getattr(env, "unwrapped", env)
    driver = getattr(base, "driver", None)
    if driver is None:
        return float("nan")
    try:
        v = driver.execute_script("return Number.isFinite(window.Game?.score) ? window.Game.score : NaN;")
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def main():
    args = parse_args()

    model = PPO.load(str(args.model_path), device=args.device)
    env = gym.make(
        "SuikaEnv-v0",
        headless=args.headless,
        delay_before_img_capture=0.0,
        port=args.port,
        mute_sound=args.mute_sound,
        wait_for_ready_on_step=True,
        ready_poll_interval=0.02,
        ready_timeout=2.0,
    )
    obs, _ = env.reset(seed=args.seed)

    rows = deque(maxlen=max(5, args.table_rows))
    delay = 1.0 / max(1e-6, args.fps)
    ep_return = 0.0

    fig, (ax_img, ax_tbl) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    fig.suptitle("Suika Policy Inference")

    frame = get_frame(env)
    im = ax_img.imshow(frame)
    ax_img.set_title("Environment")
    ax_img.axis("off")

    ax_tbl.axis("off")
    txt = ax_tbl.text(
        0.0,
        1.0,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    try:
        for step in range(1, args.steps + 1):
            model_obs = build_model_obs(obs, model)
            sigma = get_sigma(model, model_obs)
            action, _ = model.predict(model_obs, deterministic=args.deterministic)
            x = float(np.asarray(action).reshape(-1)[0])

            obs, reward, terminated, truncated, info = env.step(np.asarray(action).reshape(-1))
            score = float(info.get("score", 0.0))
            ep_return += float(reward)

            raw_js_score = get_raw_js_score(env)
            rows.append((step, x, sigma, float(reward), score, raw_js_score, ep_return))
            lines = ["step      x        sigma      reward   info_score   js_score   ep_return", "-" * 78]
            for s, xv, sv, rv, sc, jsc, er in rows:
                lines.append(
                    f"{s:>4d}  {xv:>7.3f}  {sv:>8.4f}  {rv:>10.3f}  {sc:>10.1f}  {jsc:>9.1f}  {er:>10.3f}"
                )
            txt.set_text("\n".join(lines))
            if not args.quiet:
                print(
                    f"step={step:>4d} x={x:+.4f} sigma={sigma:.5f} "
                    f"reward={float(reward):+.5f} score={score:.1f} js_score={raw_js_score:.1f} "
                    f"ep_return={ep_return:+.5f}"
                )

            frame = get_frame(env)
            im.set_data(frame)
            ax_img.set_xlabel(
                f"step={step}  action_x={x:.3f}  sigma={sigma:.4f}  "
                f"reward={float(reward):.1f}  score={score:.1f}"
            )
            plt.pause(delay)

            if terminated or truncated:
                if not args.quiet:
                    print(
                        f"[episode_end] step={step} terminated={terminated} "
                        f"truncated={truncated} final_score={score:.1f} js_score={raw_js_score:.1f} "
                        f"ep_return={ep_return:+.5f}"
                    )
                obs, _ = env.reset()
                ep_return = 0.0

    finally:
        env.close()
        plt.close(fig)


if __name__ == "__main__":
    main()
