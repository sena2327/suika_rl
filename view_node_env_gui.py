from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np

import suika_env_node  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description="View SuikaEnvNode-v0 GUI with random actions.")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gui-fps", type=float, default=20.0)
    p.add_argument("--sleep", type=float, default=0.02, help="Sleep per step for readability.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env = gym.make(
        "SuikaEnvNode-v0",
        gui=True,
        gui_fps=args.gui_fps,
        enable_image_observation=True,
    )

    obs, _ = env.reset(seed=args.seed)
    _ = obs
    ep = 1
    ep_return = 0.0
    try:
        for step in range(1, args.steps + 1):
            action = np.array([rng.uniform(-0.5, 0.5)], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            score = float(info.get("score", 0.0))
            print(
                f"ep={ep:02d} step={step:04d} action={float(action[0]):+.3f} "
                f"reward={float(reward):+.3f} score={score:.1f} term={terminated} trunc={truncated}"
            )
            if terminated or truncated:
                print(f"[episode_end] ep={ep:02d} return={ep_return:+.3f} final_score={score:.1f}")
                ep += 1
                ep_return = 0.0
                obs, _ = env.reset(seed=args.seed + ep)
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    main()

