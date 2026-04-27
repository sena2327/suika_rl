from __future__ import annotations

import argparse
import time

import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

import suika_env_node  # noqa: F401


FRUIT_RADII = [24, 32, 40, 56, 64, 72, 84, 96, 128, 160, 192]
COLORS = [
    "#f94144",
    "#f3722c",
    "#f8961e",
    "#f9844a",
    "#f9c74f",
    "#90be6d",
    "#43aa8b",
    "#4d908e",
    "#577590",
    "#277da1",
    "#9b5de5",
]


def parse_args():
    p = argparse.ArgumentParser(description="GUI demo for SuikaEnvNode-v0 with random actions")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=1)
    return p.parse_args()


def draw(ax, obs: dict, step: int, episode: int, reward: float, score: float):
    ax.clear()
    ax.set_xlim(0, 640)
    ax.set_ylim(960, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), 640, 960, facecolor="#ffdcae", edgecolor="none"))
    ax.plot([0, 640], [84, 84], color="red", linewidth=2, alpha=0.8)

    board_xy = obs["board_fruit_xy"].reshape(-1, 2)
    board_r = obs["board_fruit_radius"].reshape(-1)
    board_t = obs["board_fruit_type"].reshape(-1).astype(int)
    board_m = obs["board_fruit_mask"].reshape(-1)

    for i in range(board_m.shape[0]):
        if board_m[i] < 0.5:
            continue
        x = float(board_xy[i, 0]) * 640.0
        y = float(board_xy[i, 1]) * 960.0
        r = float(board_r[i]) * 100.0
        t = int(np.clip(board_t[i], 0, 10))
        ax.add_patch(Circle((x, y), r, facecolor=COLORS[t], edgecolor="black", linewidth=1.0, alpha=0.95))

    cur_x = float(obs["current_fruit_x"][0]) * 640.0
    cur_t = int(np.clip(obs["current_fruit_type"][0], 0, 10))
    cur_r = FRUIT_RADII[cur_t]
    ax.add_patch(Circle((cur_x, 18 + cur_r), cur_r, facecolor=COLORS[cur_t], edgecolor="black", linewidth=1.0, alpha=0.6))

    ax.set_title(
        f"SuikaEnvNode-v0 Random Play | ep={episode} step={step} score={score:.0f} reward={reward:+.1f}",
        fontsize=10,
    )


def main():
    args = parse_args()
    delay = 1.0 / max(args.fps, 1e-6)
    rng = np.random.default_rng(args.seed)

    env = gym.make("SuikaEnvNode-v0", enable_image_observation=False)
    fig, ax = plt.subplots(figsize=(6, 9))
    plt.tight_layout()

    obs, _ = env.reset(seed=args.seed)
    episode = 1
    score = 0.0

    try:
        for step in range(1, args.steps + 1):
            action = np.array([rng.uniform(-0.5, 0.5)], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            score = float(info.get("score", score))

            draw(ax, obs, step=step, episode=episode, reward=float(reward), score=score)
            plt.pause(0.001)
            time.sleep(delay)

            if terminated or truncated:
                episode += 1
                if episode > args.episodes:
                    break
                obs, _ = env.reset(seed=args.seed + episode)
                score = 0.0

    finally:
        env.close()
        plt.show(block=True)


if __name__ == "__main__":
    main()

