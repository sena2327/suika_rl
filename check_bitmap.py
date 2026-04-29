"""
Show current environment screen and single-frame bitmap observation.

Behavior:
  - Take one action every N seconds (default: 10s)
  - Left: real game screen
  - Right: obs["bitmap"] (single frame, no stacking)

Example:
  uv run python suika_rl/check_bitmap.py --env-id SuikaEnvNode-v0 --no-headless
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import suika_env  # noqa: F401
import suika_env_node  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env-id",
        type=str,
        default="SuikaEnvNode-v0",
        choices=["SuikaEnv-v0", "SuikaEnvNode-v0"],
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action-interval-sec", type=float, default=10.0)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--port", type=int, default=8923)
    p.add_argument("--node-bin", type=str, default="node")
    p.add_argument("--deterministic-x", type=float, default=0.0, help="Centered action in [-1, 1].")
    p.add_argument(
        "--grid-size",
        type=int,
        default=32,
        help="Display bitmap as NxN numeric grid (nearest-neighbor).",
    )
    return p.parse_args()


def get_canvas_frame(env) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    if hasattr(base, "capture_canvas_full_rgba"):
        return base.capture_canvas_full_rgba()
    if hasattr(base, "capture_canvas_raw_rgba"):
        return base.capture_canvas_raw_rgba()
    img = base.render() if hasattr(base, "render") else None
    if img is None:
        raise RuntimeError("No frame source available on env.")
    return img


def resize_nearest_2d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    src_h, src_w = arr.shape
    ys = (np.arange(h) * src_h / h).astype(np.int32)
    xs = (np.arange(w) * src_w / w).astype(np.int32)
    ys = np.clip(ys, 0, src_h - 1)
    xs = np.clip(xs, 0, src_w - 1)
    return arr[ys[:, None], xs[None, :]]


def bitmap_to_text_grid(bitmap: np.ndarray, grid_size: int) -> str:
    n = max(4, int(grid_size))
    small = resize_nearest_2d(bitmap, n, n)
    lines = []
    for r in range(n):
        row = " ".join(f"{int(v):2d}" for v in small[r])
        lines.append(row)
    return "\n".join(lines)


def main():
    args = parse_args()
    kwargs = dict(
        headless=args.headless,
        mute_sound=True,
        wait_for_ready_on_step=True,
        ready_poll_interval=0.02,
        ready_timeout=2.0,
        enable_image_observation=False,
        bitmap_size=128,
        img_width=128,
        img_height=128,
    )
    if args.env_id == "SuikaEnv-v0":
        kwargs["port"] = args.port
        kwargs["delay_before_img_capture"] = 0.0
    else:
        kwargs["node_bin"] = args.node_bin

    env = gym.make(args.env_id, **kwargs)
    obs, _ = env.reset(seed=args.seed)

    fig, (ax_img, ax_bitmap) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.7, 1.3]}
    )
    fig.suptitle(f"check_bitmap: {args.env_id}")
    ax_img.axis("off")
    ax_bitmap.axis("off")

    frame = get_canvas_frame(env)
    bitmap = np.asarray(obs["bitmap"], dtype=np.int8)

    im = ax_img.imshow(frame)
    ax_bitmap.set_title(f'obs["bitmap"] shape={bitmap.shape} -> grid {args.grid_size}x{args.grid_size}')
    txt = ax_bitmap.text(
        0.0,
        1.0,
        bitmap_to_text_grid(bitmap, args.grid_size),
        va="top",
        ha="left",
        family="monospace",
        fontsize=6,
    )

    action_value = float(np.clip(args.deterministic_x, -1.0, 1.0))
    action = np.array([action_value], dtype=np.float32)

    try:
        for step in range(1, args.steps + 1):
            obs, reward, terminated, truncated, info = env.step(action)

            frame = get_canvas_frame(env)
            bitmap = np.asarray(obs["bitmap"], dtype=np.int8)
            im.set_data(frame)
            txt.set_text(bitmap_to_text_grid(bitmap, args.grid_size))
            ax_img.set_xlabel(
                f"step={step} action_x={action_value:+.3f} "
                f"reward={float(reward):+.4f} score={float(info.get('score', 0.0)):.1f}"
            )
            plt.pause(0.001)

            if terminated or truncated:
                obs, _ = env.reset()
                time.sleep(0.2)

            time.sleep(max(0.0, args.action_interval_sec))
    finally:
        env.close()
        plt.close(fig)


if __name__ == "__main__":
    main()
