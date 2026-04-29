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
        bitmap_size: int = 64,
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
        # Bitmap observation is standardized to 96x96.
        self.bitmap_size = 96
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
        self._render_crop_width = self._render_width
        self._fruit_radii = [24, 32, 40, 56, 64, 72, 84, 96, 128, 160, 192]
        self._graph_max_nodes = 50
        self._graph_node_dim = 14  # [x, y, radius_norm, one-hot(11)]
        self._graph_knn_k = 8
        self._graph_dist_threshold = 0.35
        self._graph_touch_margin_px = 2.0
        self._graph_max_edges = (self._graph_max_nodes * (self._graph_max_nodes - 1)) // 2
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
                "bitmap": gymnasium.spaces.Box(low=-1, high=11, shape=(self.bitmap_size, self.bitmap_size), dtype="int8"),
                "board_top50_exyir": gymnasium.spaces.Box(low=0, high=256, shape=(50, 5), dtype="float32"),
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
                "global_feature": gymnasium.spaces.Box(low=0, high=1, shape=(4,), dtype="float32"),
                "largest_type_onehot": gymnasium.spaces.Box(low=0, high=1, shape=(11,), dtype="float32"),
                "board_fruit_xy": gymnasium.spaces.Box(low=0, high=1, shape=(80,), dtype="float32"),
                "board_fruit_radius": gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
                "board_fruit_mass": gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
                "board_fruit_type": gymnasium.spaces.Box(low=0, high=10, shape=(40,), dtype="float32"),
                "board_fruit_mask": gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
                "node": gymnasium.spaces.Box(low=-1, high=1, shape=(self._graph_max_nodes, self._graph_node_dim), dtype="float32"),
                "node_mask": gymnasium.spaces.Box(low=0, high=1, shape=(self._graph_max_nodes,), dtype="float32"),
                "edge": gymnasium.spaces.Box(low=-2, high=2, shape=(self._graph_max_edges, 6), dtype="float32"),
                "edge_index": gymnasium.spaces.Box(low=0, high=self._graph_max_nodes - 1, shape=(self._graph_max_edges, 2), dtype="int32"),
                "edge_mask": gymnasium.spaces.Box(low=0, high=1, shape=(self._graph_max_edges,), dtype="float32"),
            }
        )
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

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

        # Keep full canvas for observation to avoid information loss.
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
        bitmap = self._build_bitmap(snap)
        node_feat, node_mask, edge_feat, edge_index, edge_mask = self._build_graph_from_top50(snap["board_top50_exyir"])
        global_feature, largest_type_onehot = self._build_global_feature(snap)
        self._maybe_show_gui()
        return {
            "image": image,
            "bitmap": bitmap,
            "board_top50_exyir": np.array(snap["board_top50_exyir"], dtype=np.float32),
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
            "global_feature": global_feature,
            "largest_type_onehot": largest_type_onehot,
            "board_fruit_xy": np.array(snap["board_fruit_xy"], dtype=np.float32),
            "board_fruit_radius": np.array(snap["board_fruit_radius"], dtype=np.float32),
            "board_fruit_mass": np.array(snap["board_fruit_mass"], dtype=np.float32),
            "board_fruit_type": np.array(snap["board_fruit_type"], dtype=np.float32),
            "board_fruit_mask": np.array(snap["board_fruit_mask"], dtype=np.float32),
            "node": node_feat,
            "node_mask": node_mask,
            "edge": edge_feat,
            "edge_index": edge_index,
            "edge_mask": edge_mask,
        }

    def _build_global_feature(self, snap: dict[str, Any]) -> np.ndarray:
        board_xy = np.asarray(snap["board_fruit_xy"], dtype=np.float32).reshape(-1, 2)
        board_mask = np.asarray(snap["board_fruit_mask"], dtype=np.float32).reshape(-1)
        board_type = np.asarray(snap["board_fruit_type"], dtype=np.float32).reshape(-1)
        valid = board_mask > 0.5
        if np.any(valid):
            x_vals = board_xy[valid, 0]
            x_mean = float(np.clip(np.mean(x_vals), 0.0, 1.0))
            x_var = float(np.clip(np.var(x_vals), 0.0, 1.0))
            largest_idx = int(np.clip(np.max(board_type[valid]), 0.0, 10.0))
        else:
            x_mean = 0.5
            x_var = 0.0
            largest_idx = 0
        fruit_count = float(np.clip(float(snap.get("fruit_count", 0.0)) / 50.0, 0.0, 1.0))
        max_height = float(np.clip(float(snap.get("max_height", 0.0)), 0.0, 1.0))
        largest_onehot = np.zeros((11,), dtype=np.float32)
        largest_onehot[largest_idx] = 1.0
        return np.array([fruit_count, max_height, x_mean, x_var], dtype=np.float32), largest_onehot

    def _build_graph_from_top50(self, top50: Any):
        arr = np.asarray(top50, dtype=np.float32).reshape(self._graph_max_nodes, 5)
        # Reorder nodes by y (top to bottom), then pad with zeros.
        valid = arr[:, 0] > 0.5
        valid_arr = arr[valid]
        if valid_arr.shape[0] > 0:
            order = np.argsort(valid_arr[:, 2], kind="stable")
            valid_arr = valid_arr[order]
        arr_sorted = np.zeros_like(arr)
        n_valid = min(valid_arr.shape[0], self._graph_max_nodes)
        if n_valid > 0:
            arr_sorted[:n_valid] = valid_arr[:n_valid]
        arr = arr_sorted
        exist = arr[:, 0] > 0.5
        x = arr[:, 1]
        y = arr[:, 2]
        fruit_id = np.clip(arr[:, 3].astype(np.int32), 0, 11)  # 0 padding, 1..11 real
        radius_px = np.clip(arr[:, 4], 0.0, 256.0)

        node = np.zeros((self._graph_max_nodes, self._graph_node_dim), dtype=np.float32)
        node[:, 0] = x
        node[:, 1] = y
        node[:, 2] = np.clip(radius_px / 256.0, 0.0, 1.0)
        for i in range(self._graph_max_nodes):
            if not exist[i]:
                continue
            fid0 = int(np.clip(fruit_id[i] - 1, 0, 10))
            node[i, 3 + fid0] = 1.0
        node_mask = exist.astype(np.float32)

        edge = np.zeros((self._graph_max_edges, 6), dtype=np.float32)
        edge_index = np.zeros((self._graph_max_edges, 2), dtype=np.int32)
        edge_mask = np.zeros((self._graph_max_edges,), dtype=np.float32)
        valid_idx = np.flatnonzero(exist).astype(np.int32)
        keep_pairs = set()
        if valid_idx.size >= 2:
            for i in valid_idx.tolist():
                dx_all = x[valid_idx] - x[i]
                dy_all = y[valid_idx] - y[i]
                d2_all = dx_all * dx_all + dy_all * dy_all
                d_all = np.sqrt(d2_all)
                order = np.argsort(d2_all)
                chosen = 0
                for oi in order:
                    j = int(valid_idx[oi])
                    if j == i:
                        continue
                    if float(d_all[oi]) >= self._graph_dist_threshold:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    keep_pairs.add((a, b))
                    chosen += 1
                    if chosen >= self._graph_knn_k:
                        break
        k = 0
        for i in range(self._graph_max_nodes):
            for j in range(i + 1, self._graph_max_nodes):
                edge_index[k, 0] = i
                edge_index[k, 1] = j
                if (i, j) in keep_pairs:
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = float(np.sqrt(dx * dx + dy * dy))
                    same_type = 1.0 if fruit_id[i] == fruit_id[j] else 0.0
                    xi, yi = x[i] * 640.0, y[i] * 960.0
                    xj, yj = x[j] * 640.0, y[j] * 960.0
                    dpx = float(np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2))
                    touching = 1.0 if dpx <= (radius_px[i] + radius_px[j] + self._graph_touch_margin_px) else 0.0
                    overlap_margin = float((radius_px[i] + radius_px[j]) - dpx)
                    overlap_margin_norm = float(np.clip(overlap_margin / 256.0, -2.0, 2.0))
                    edge[k] = np.array([dx, dy, dist, same_type, touching, overlap_margin_norm], dtype=np.float32)
                    edge_mask[k] = 1.0
                k += 1
        return node, node_mask, edge, edge_index, edge_mask

    def _build_bitmap(self, snap: dict[str, Any]) -> np.ndarray:
        # 1) Build semantic map on 640x800 (W x H).
        map_w, map_h = 640, 800
        full = np.zeros((map_h, map_w), dtype=np.int8)
        # Borders: left/right/top/bottom as -1.
        full[0, :] = -1
        full[-1, :] = -1
        full[:, 0] = -1
        full[:, -1] = -1
        # Preserve LOSE line visibility: original y=84 in 960-space -> 800-space, with y±1 set to -1.
        lose_y = int(round((84.0 / 960.0) * (map_h - 1)))
        y0 = max(0, lose_y - 1)
        y1 = min(map_h - 1, lose_y + 1)
        full[y0 : y1 + 1, :] = -1
        board_xy = np.asarray(snap["board_fruit_xy"], dtype=np.float32).reshape(-1, 2)
        board_t = np.asarray(snap["board_fruit_type"], dtype=np.int32).reshape(-1)
        board_m = np.asarray(snap["board_fruit_mask"], dtype=np.float32).reshape(-1)
        for i in range(board_m.shape[0]):
            if board_m[i] < 0.5:
                continue
            t = int(np.clip(board_t[i], 0, 10))
            val = np.int8(t + 1)
            x = float(np.clip(board_xy[i, 0], 0.0, 1.0)) * (map_w - 1)
            y = float(np.clip(board_xy[i, 1], 0.0, 1.0)) * (map_h - 1)
            r = float(self._fruit_radii[t])
            rx = max(1.0, r)
            ry = max(1.0, r * (map_h / 960.0))
            x0 = max(0, int(np.floor(x - rx)))
            x1 = min(map_w - 1, int(np.ceil(x + rx)))
            y0 = max(0, int(np.floor(y - ry)))
            y1 = min(map_h - 1, int(np.ceil(y + ry)))
            if x1 < x0 or y1 < y0:
                continue
            yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
            mask = (((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2) <= 1.0
            patch = full[y0 : y1 + 1, x0 : x1 + 1]
            patch[mask] = np.maximum(patch[mask], val)
        # 2) Keep aspect ratio and resize to 77x96 (W x H) with nearest.
        resized = Image.fromarray(full.astype(np.int32), mode="I").resize((77, 96), Image.Resampling.NEAREST)
        arr = np.asarray(resized, dtype=np.int16)
        arr = np.clip(arr, -1, 11).astype(np.int8)
        # 3) Pad left/right with 0 to reach 96x96.
        pad_total = 96 - arr.shape[1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        arr = np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)
        return arr

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
                "discard_episode": True,
                "discard_reason": "node_worker_recovery",
            }
            return obs, 0.0, True, True, info

        self._worker_steps += 1
        obs = self._obs(out)
        reward = float(out.get("reward", 0.0))
        terminated = bool(out.get("terminated", False))
        truncated = bool(out.get("truncated", False))
        score = float(out.get("score", self.score))
        self.score = score
        info = dict(out.get("info", {}))
        info["score"] = score
        info["final_score_valid"] = bool(terminated) and (not bool(info.get("discard_episode", False)))

        # Proactively recycle worker before long-run heap growth causes OOM.
        if self._worker_steps >= self._worker_recycle_steps:
            self._recycle_on_next_reset = True
            if not terminated:
                truncated = True
                info["worker_recycle_due"] = True
                info["final_score_valid"] = False

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
