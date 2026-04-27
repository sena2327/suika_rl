from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SuikaNodeEnv(gymnasium.Env):
    def __init__(
        self,
        img_width: int = 128,
        img_height: int = 128,
        enable_image_observation: bool = False,
        gui: bool = False,
        gui_fps: float = 20.0,
        node_bin: str = "node",
        worker_path: str | None = None,
        node_max_old_space_mb: int = 8192,
        worker_recycle_steps: int = 3000,
        **_: Any,
    ) -> None:
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.enable_image_observation = bool(enable_image_observation)
        self.gui = bool(gui)
        self.gui_fps = float(max(gui_fps, 1e-6))
        self._gui_last_ts = 0.0
        self._fig = None
        self._ax = None
        self._im = None
        self._plt = None
        self._render_width = 640
        self._render_height = 960
        self._render_crop_width = 520
        self._fruit_radii = [24, 32, 40, 56, 64, 72, 84, 96, 128, 160, 192]
        assets_dir = Path(__file__).with_name("suika-game") / "assets" / "img"
        self._circle_sprites: dict[int, Image.Image] = {}
        for i in range(11):
            p = assets_dir / f"circle{i}.png"
            if p.exists():
                self._circle_sprites[i] = Image.open(p).convert("RGBA")
        self._blank_image = np.zeros((self.img_height, self.img_width, 4), dtype=np.uint8)
        self._last_frame_full = np.zeros((self._render_height, self._render_width, 4), dtype=np.uint8)
        self._last_frame_raw = np.zeros((self._render_height, self._render_crop_width, 4), dtype=np.uint8)
        self.score = 0.0
        self._score_font = self._load_score_font()
        self._node_max_old_space_mb = int(max(256, node_max_old_space_mb))
        self._worker_recycle_steps = int(max(100, worker_recycle_steps))
        self._worker_steps = 0
        self._recycle_on_next_reset = False

        if worker_path is None:
            worker_path = str(Path(__file__).with_name("suika_node_worker.cjs"))
        self.worker_path = worker_path
        self.node_bin = node_bin
        self._proc = self._start_worker()

        self.observation_space = gymnasium.spaces.Dict(
            {
                "image": gymnasium.spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 4), dtype="uint8"),
                "current_fruit_type": gymnasium.spaces.Box(low=0, high=10, shape=(1,), dtype="float32"),
                "next_fruit_type": gymnasium.spaces.Box(low=0, high=10, shape=(1,), dtype="float32"),
                "current_fruit_x": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype="float32"),
                "stage_top10_xy": gymnasium.spaces.Box(low=0, high=1, shape=(20,), dtype="float32"),
                "top10_fruit_types": gymnasium.spaces.Box(low=0, high=10, shape=(10,), dtype="float32"),
                "top10_mask": gymnasium.spaces.Box(low=0, high=1, shape=(10,), dtype="float32"),
                "max_height": gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype="float32"),
                "danger_count": gymnasium.spaces.Box(low=0, high=500, shape=(1,), dtype="float32"),
                "largest_fruit_type": gymnasium.spaces.Box(low=0, high=10, shape=(1,), dtype="float32"),
                "fruit_count": gymnasium.spaces.Box(low=0, high=500, shape=(1,), dtype="float32"),
                "board_fruit_xy": gymnasium.spaces.Box(low=0, high=1, shape=(80,), dtype="float32"),
                "board_fruit_radius": gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
                "board_fruit_mass": gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
                "board_fruit_type": gymnasium.spaces.Box(low=0, high=10, shape=(40,), dtype="float32"),
                "board_fruit_mask": gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
            }
        )
        self.action_space = gymnasium.spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

    def _start_worker(self) -> subprocess.Popen:
        return subprocess.Popen(
            [self.node_bin, f"--max-old-space-size={self._node_max_old_space_mb}", self.worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _stop_worker(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            if self._proc.stdout is not None:
                self._proc.stdout.close()
        except Exception:
            pass
        try:
            if self._proc.stderr is not None:
                self._proc.stderr.close()
        except Exception:
            pass
        try:
            if self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def _restart_worker_process(self) -> None:
        self._stop_worker()
        self._proc = self._start_worker()
        self._worker_steps = 0
        self._recycle_on_next_reset = False

    def _rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Node worker is not available.")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = ""
            if self._proc.stderr is not None:
                err = self._proc.stderr.read()
            raise RuntimeError(f"Node worker exited. {err}")
        out = json.loads(line)
        if not out.get("ok", False):
            raise RuntimeError(f"Node worker RPC error: {out.get('error', 'unknown')}")
        return out

    def _paste_fruit_sprite(self, canvas: Image.Image, fruit_type: int, x: float, y: float, radius: float, alpha: int = 255):
        fruit_type = int(np.clip(fruit_type, 0, 10))
        sprite = self._circle_sprites.get(fruit_type, None)
        diameter = max(1, int(round(2.0 * float(radius))))
        if sprite is None:
            return
        patch = sprite.resize((diameter, diameter), Image.Resampling.LANCZOS)
        if alpha < 255:
            a = patch.split()[-1].point(lambda v: int(v * alpha / 255))
            patch.putalpha(a)
        left = int(round(float(x) - diameter / 2.0))
        top = int(round(float(y) - diameter / 2.0))
        canvas.alpha_composite(patch, dest=(left, top))

    def _load_score_font(self):
        # Browser UI uses 84px score text in index.html.
        candidates = [
            "Arial Bold.ttf",
            "Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, 84)
            except Exception:
                continue
        return ImageFont.load_default()

    def _build_frame_raw(self, snap: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        # Match browser canvas look from suika-game: 640x960 then crop left 520px.
        canvas = Image.new("RGBA", (self._render_width, self._render_height), (255, 220, 174, 255))
        draw = ImageDraw.Draw(canvas, "RGBA")

        # Bottom static wall tint (matches gameStatics bottom wall visibility).
        floor_top = self._render_height - 48
        floor = Image.new("RGBA", (self._render_width, self._render_height - floor_top), (255, 238, 219, 255))
        canvas.alpha_composite(floor, dest=(0, floor_top))
        # Game over line (loseHeight = 84)
        draw.line((0, 84, self._render_width, 84), fill=(214, 48, 49, 255), width=3)

        board_xy = np.asarray(snap["board_fruit_xy"], dtype=np.float32).reshape(-1, 2)
        board_r = np.asarray(snap["board_fruit_radius"], dtype=np.float32).reshape(-1)
        board_t = np.asarray(snap["board_fruit_type"], dtype=np.int32).reshape(-1)
        board_m = np.asarray(snap["board_fruit_mask"], dtype=np.float32).reshape(-1)

        for i in range(board_m.shape[0]):
            if board_m[i] < 0.5:
                continue
            x = float(board_xy[i, 0]) * float(self._render_width)
            y = float(board_xy[i, 1]) * float(self._render_height)
            t = int(np.clip(board_t[i], 0, 10))
            # board_fruit_radius is normalized and clipped in observation space,
            # so it saturates for large fruits. Use fruit type for faithful GUI size.
            r = float(self._fruit_radii[t])
            self._paste_fruit_sprite(canvas, t, x, y, r, alpha=255)

        # Current preview ball is centered at y=0 in the original JS.
        cur_x = float(np.clip(snap["current_fruit_x"], 0.0, 1.0)) * float(self._render_width)
        cur_t = int(np.clip(snap["current_fruit_type"], 0, 10))
        cur_r = self._fruit_radii[cur_t]
        self._paste_fruit_sprite(canvas, cur_t, cur_x, 0.0, float(cur_r), alpha=180)

        # HUD text similar to browser UI (#game-score: 84px)
        score = float(snap.get("score", 0.0))
        draw.text((12, 8), f"{score:.0f}", font=self._score_font, fill=(24, 24, 24, 255))

        # suika_env screenshot crops canvas from x:[0, 520).
        full = np.asarray(canvas, dtype=np.uint8)
        cropped = canvas.crop((0, 0, self._render_crop_width, self._render_height))
        return full, np.asarray(cropped, dtype=np.uint8)

    def _ensure_gui(self):
        if not self.gui or self._fig is not None:
            return
        import matplotlib.pyplot as plt  # lazy import to avoid headless cost when gui=False

        self._plt = plt
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(5.2, 9.6))
        self._ax.set_title("SuikaEnvNode-v0 GUI")
        self._ax.axis("off")
        self._im = self._ax.imshow(self._last_frame_full)
        self._fig.tight_layout()

    def _maybe_show_gui(self):
        if not self.gui:
            return
        now = time.time()
        if (now - self._gui_last_ts) < (1.0 / self.gui_fps):
            return
        self._gui_last_ts = now
        try:
            self._ensure_gui()
            if self._im is not None and self._plt is not None:
                self._im.set_data(self._last_frame_full)
                self._plt.pause(0.001)
        except Exception:
            # Disable GUI if backend/display is unavailable.
            self.gui = False

    def _obs(self, snap: dict[str, Any]) -> dict[str, np.ndarray]:
        self._last_frame_full, self._last_frame_raw = self._build_frame_raw(snap)
        if self.enable_image_observation:
            image = np.asarray(
                Image.fromarray(self._last_frame_raw, mode="RGBA").resize(
                    (self.img_width, self.img_height), Image.Resampling.LANCZOS
                ),
                dtype=np.uint8,
            )
        else:
            image = self._blank_image
        self._maybe_show_gui()
        return {
            "image": image,
            "current_fruit_type": np.array([snap["current_fruit_type"]], dtype=np.float32),
            "next_fruit_type": np.array([snap["next_fruit_type"]], dtype=np.float32),
            "current_fruit_x": np.array([snap["current_fruit_x"]], dtype=np.float32),
            "stage_top10_xy": np.array(snap["stage_top10_xy"], dtype=np.float32),
            "top10_fruit_types": np.array(snap["top10_fruit_types"], dtype=np.float32),
            "top10_mask": np.array(snap["top10_mask"], dtype=np.float32),
            "max_height": np.array([snap["max_height"]], dtype=np.float32),
            "danger_count": np.array([snap["danger_count"]], dtype=np.float32),
            "largest_fruit_type": np.array([snap["largest_fruit_type"]], dtype=np.float32),
            "fruit_count": np.array([snap["fruit_count"]], dtype=np.float32),
            "board_fruit_xy": np.array(snap["board_fruit_xy"], dtype=np.float32),
            "board_fruit_radius": np.array(snap["board_fruit_radius"], dtype=np.float32),
            "board_fruit_mass": np.array(snap["board_fruit_mass"], dtype=np.float32),
            "board_fruit_type": np.array(snap["board_fruit_type"], dtype=np.float32),
            "board_fruit_mask": np.array(snap["board_fruit_mask"], dtype=np.float32),
        }

    def restart_browser(self):
        self._rpc({"cmd": "restart"})

    def capture_canvas_raw_rgba(self):
        return self._last_frame_raw

    def capture_canvas_full_rgba(self):
        return self._last_frame_full

    def reset(self, *, seed=None, options=None):
        if self._recycle_on_next_reset:
            self._restart_worker_process()
        try:
            out = self._rpc({"cmd": "reset", "seed": seed})
        except RuntimeError:
            # Worker may have crashed (e.g. OOM). Recreate and retry once.
            self._restart_worker_process()
            out = self._rpc({"cmd": "reset", "seed": seed})
        self.score = float(out.get("score", 0.0))
        return self._obs(out), {}

    def step(self, action):
        x = float(np.asarray(action).reshape(-1)[0])
        try:
            out = self._rpc({"cmd": "step", "action": x})
        except RuntimeError as exc:
            # Keep vec worker alive even if node process died.
            self._restart_worker_process()
            obs, _ = self.reset()
            info = {
                "score": float(self.score),
                "worker_error": str(exc),
                "worker_recovered": True,
            }
            return obs, -200.0, True, True, info

        self._worker_steps += 1
        obs = self._obs(out)
        reward = float(out.get("reward", 0.0))
        terminated = bool(out.get("terminated", False))
        truncated = bool(out.get("truncated", False))
        score = float(out.get("score", self.score))
        self.score = score
        info = dict(out.get("info", {}))
        info["score"] = score

        # Proactively recycle worker before long-run heap growth causes OOM.
        if self._worker_steps >= self._worker_recycle_steps:
            self._recycle_on_next_reset = True
            if not terminated:
                truncated = True
                info["worker_recycle_due"] = True

        return obs, reward, terminated, truncated, info

    def close(self):
        super().close()
        if self._plt is not None and self._fig is not None:
            try:
                self._plt.close(self._fig)
            except Exception:
                pass
        self._fig = None
        self._ax = None
        self._im = None
        self._plt = None
        if self._proc is None:
            return
        try:
            self._rpc({"cmd": "close"})
        except Exception:
            pass
        self._stop_worker()
