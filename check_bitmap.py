"""
Advance environment for N steps and dump obs["bitmap"] to a plain text file.

Default behavior:
  - steps: 10
  - bitmap size: 96x96
  - output: check_bitmap/bitmap_step10.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
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
    p.add_argument("--steps", type=int, default=10, help="Number of env steps before dumping bitmap.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--port", type=int, default=8923)
    p.add_argument("--node-bin", type=str, default="node")
    p.add_argument("--action-x", type=float, default=0.0, help="Fixed action in [-1, 1].")
    p.add_argument("--out", type=Path, default=Path("check_bitmap/bitmap_step10.txt"))
    return p.parse_args()


def main():
    args = parse_args()
    kwargs = dict(
        headless=args.headless,
        mute_sound=True,
        wait_for_ready_on_step=True,
        ready_poll_interval=0.02,
        ready_timeout=2.0,
        enable_image_observation=False,
        bitmap_size=96,
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
    action = np.array([float(np.clip(args.action_x, -1.0, 1.0))], dtype=np.float32)

    try:
        for _ in range(max(0, int(args.steps))):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()

    bitmap = np.asarray(obs["bitmap"], dtype=np.int8)
    if bitmap.shape != (96, 96):
        raise ValueError(f'Expected bitmap shape (96,96), got {bitmap.shape}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in bitmap:
            f.write(" ".join(str(int(v)) for v in row))
            f.write("\n")

    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

