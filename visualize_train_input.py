"""
Visualize the exact image observation used by train.py (after SuikaObsWrapper).

Examples:
  uv run python visualize_train_input.py --steps 200 --show
  uv run python visualize_train_input.py --steps 200 --gif-path train_input.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np

import suika_env  # noqa: F401  # Registers "SuikaEnv-v0"
from train import SuikaFrameStackWrapper, SuikaObsWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--port", type=int, default=8923)
    p.add_argument("--img-size", type=int, default=160, help="Observation image size (square).")
    p.add_argument("--frame-stack", type=int, default=1, help="Number of stacked frames (same as train.py).")
    p.add_argument("--show", action="store_true", help="Show live preview with matplotlib.")
    p.add_argument("--gif-path", type=Path, default=None, help="Optional GIF output path.")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def resize_nearest_rgba(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w, _ = image.shape
    ys = (np.arange(target_h) * src_h / target_h).astype(np.int32)
    xs = (np.arange(target_w) * src_w / target_w).astype(np.int32)
    ys = np.clip(ys, 0, src_h - 1)
    xs = np.clip(xs, 0, src_w - 1)
    return image[ys[:, None], xs[None, :], :]


def compose_panel(raw_rgba: np.ndarray, stacked_frames_rgba: list[np.ndarray]) -> np.ndarray:
    # Scale raw image to frame height for visual comparison.
    target_h = stacked_frames_rgba[0].shape[0]
    target_w = max(1, int(round(raw_rgba.shape[1] * target_h / raw_rgba.shape[0])))
    raw_small = resize_nearest_rgba(raw_rgba, target_h, target_w)
    sep = np.full((target_h, 4, 4), 255, dtype=np.uint8)
    parts = [raw_small]
    for frame in stacked_frames_rgba:
        parts.extend([sep, frame])
    return np.concatenate(parts, axis=1)


def main():
    args = parse_args()
    env = gym.make(
        "SuikaEnv-v0",
        headless=args.headless,
        delay_before_img_capture=0.0,
        port=args.port,
        mute_sound=True,
        wait_for_ready_on_step=True,
        ready_poll_interval=0.02,
        ready_timeout=2.0,
        img_width=args.img_size,
        img_height=args.img_size,
    )
    env = SuikaObsWrapper(env)
    env = SuikaFrameStackWrapper(env, k=args.frame_stack)
    base_env = env.unwrapped
    obs, _ = env.reset(seed=args.seed)

    frames = []
    viewer = None
    viewer_ax = None
    viewer_im = None

    if args.show:
        try:
            import matplotlib.pyplot as plt

            viewer = plt
            viewer_ax = plt.gca()
            viewer_ax.set_title("raw (left) + stacked train inputs (right)")
            viewer_ax.axis("off")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] --show requested but matplotlib is unavailable: {exc}")

    try:
        for step in range(args.steps):
            # Wrapper output is CHW uint8 with stacked channels: (C*k, H, W).
            stacked_chw = obs["image"]
            channels_total, h, w = stacked_chw.shape
            if channels_total % args.frame_stack != 0:
                raise RuntimeError(
                    f"invalid stacked image channels: {channels_total} not divisible by frame_stack={args.frame_stack}"
                )
            channels_per_frame = channels_total // args.frame_stack
            stacked_frames_hwc = []
            for i in range(args.frame_stack):
                chunk = stacked_chw[i * channels_per_frame : (i + 1) * channels_per_frame, :, :]
                stacked_frames_hwc.append(np.transpose(chunk, (1, 2, 0)))
            raw_hwc = base_env.capture_canvas_raw_rgba()
            panel = compose_panel(raw_hwc, stacked_frames_hwc)
            frames.append(panel)

            if viewer is not None:
                if viewer_im is None:
                    viewer_im = viewer_ax.imshow(panel)
                else:
                    viewer_im.set_data(panel)
                viewer_ax.set_xlabel(
                    f"step={step} fruit_type={float(obs['current_fruit_type'][0]):.0f} "
                    f"x={float(obs['current_fruit_x'][0]):.3f}"
                )
                viewer.pause(0.001)

            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
        if viewer is not None:
            viewer.close()

    if args.gif_path is not None:
        args.gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif_path, frames, fps=args.fps)
        print(f"Saved GIF to: {args.gif_path}")

    print(
        f"Done. collected_frames={len(frames)}, "
        f"image_shape={frames[0].shape if frames else 'N/A'}, "
        f"dtype={frames[0].dtype if frames else 'N/A'}"
    )


if __name__ == "__main__":
    main()
