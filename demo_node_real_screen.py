from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By


def parse_args():
    p = argparse.ArgumentParser(description="Show real suika-game screen and play random actions.")
    p.add_argument("--port", type=int, default=8923)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--drop-interval", type=float, default=0.55, help="Seconds between random drops.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    game_dir = Path(__file__).resolve().parent / "suika_env_node" / "suika-game"
    server = subprocess.Popen(
        ["python3", "-m", "http.server", str(args.port)],
        cwd=str(game_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    driver = None
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--width=1200")
        options.add_argument("--height=900")
        if args.headless:
            options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=options)
        driver.get(f"http://localhost:{args.port}/")

        driver.find_element(By.ID, "start-game-button").click()
        time.sleep(1.0)

        for step in range(1, args.max_steps + 1):
            x = random.randint(0, 640)
            inp = driver.find_element(By.ID, "fruit-position")
            inp.clear()
            inp.send_keys(str(x))
            driver.find_element(By.ID, "drop-fruit-button").click()

            time.sleep(args.drop_interval)

            state, score = driver.execute_script(
                "return [window.Game?.stateIndex ?? 0, window.Game?.score ?? 0];"
            )
            print(f"step={step:4d} x={x:4d} state={state} score={score}")

            if int(state) == 3:
                print("Game over reached (stateIndex=3).")
                break

        print("Done.")
        if not args.headless:
            print("Press Ctrl+C to close.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if driver is not None:
            driver.quit()
        if server.poll() is None:
            if os.name == "nt":
                server.terminate()
            else:
                server.terminate()


if __name__ == "__main__":
    main()

