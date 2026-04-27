"""
Show current agent input with a live GUI.

Behavior:
  - Take one action every N seconds (default: 10s)
  - Show game screen on the left
  - Show current observation values on the right

Example:
  uv run python check_input.py --env-id SuikaEnvNode-v0 --steps 30 --action-interval-sec 10
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import suika_env  # noqa: F401
import suika_env_node  # noqa: F401
from train import SuikaImageFrameStackWrapper, SuikaImageObsWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env-id",
        type=str,
        default="SuikaEnvNode-v0",
        choices=["SuikaEnv-v0", "SuikaEnvNode-v0"],
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action-interval-sec", type=float, default=10.0)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--port", type=int, default=8923)
    p.add_argument("--node-bin", type=str, default="node")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--deterministic-x", type=float, default=0.0, help="Centered action in [-1, 1].")
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


def to_cnn_input_vis(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.clip(arr[:, :, 0], 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Show exactly what goes into CNN (H, W, C) after wrappers.
        return np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
    return np.zeros((64, 64), dtype=np.uint8)


def main():
    args = parse_args()

    kwargs = dict(
        headless=args.headless,
        mute_sound=True,
        wait_for_ready_on_step=True,
        ready_poll_interval=0.02,
        ready_timeout=2.0,
        enable_image_observation=True,
        img_width=args.img_size,
        img_height=args.img_size,
    )
    if args.env_id == "SuikaEnv-v0":
        kwargs["port"] = args.port
        kwargs["delay_before_img_capture"] = 0.0
    else:
        kwargs["node_bin"] = args.node_bin

    env = gym.make(args.env_id, **kwargs)
    env = SuikaImageObsWrapper(env)
    env = SuikaImageFrameStackWrapper(env, n_frames=args.frame_stack)
    obs, _ = env.reset(seed=args.seed)

    fig, (ax_img, ax_obs) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.7, 1.3]}
    )
    fig.suptitle(f"check_input: {args.env_id}")
    ax_img.axis("off")
    ax_obs.axis("off")

    frame = get_canvas_frame(env)
    obs_img = to_cnn_input_vis(obs["image"])
    im = ax_img.imshow(frame)
    if obs_img.ndim == 2:
        im_obs = ax_obs.imshow(obs_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        im_obs = ax_obs.imshow(obs_img, interpolation="nearest")
    ax_obs.set_title(f'CNN input obs["image"] shape={obs["image"].shape}')

    action_value = float(np.clip(args.deterministic_x, -1.0, 1.0))
    action = np.array([action_value], dtype=np.float32)

    try:
        for step in range(1, args.steps + 1):
            obs, reward, terminated, truncated, info = env.step(action)

            frame = get_canvas_frame(env)
            obs_img = to_cnn_input_vis(obs["image"])
            im.set_data(frame)
            im_obs.set_data(obs_img)
            ax_img.set_xlabel(f"step={step}  action_x={action_value:+.3f}")
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
