from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib3.exceptions import ReadTimeoutError
import time
import gymnasium
import ipdb
import io
import numpy as np    
from PIL import Image
import imageio
import subprocess
import socket
import os

# Pillow 10+ removed Image.ANTIALIAS.
if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = Image.ANTIALIAS

class SuikaBrowserEnv(gymnasium.Env):
    def __init__(
        self,
        headless=True,
        port=8923,
        delay_before_img_capture=0.5,
        mute_sound=False,
        wait_for_ready_on_step=False,
        ready_poll_interval=0.01,
        ready_timeout=2.0,
        stable_velocity_threshold=0.05,
        stable_velocity_polls=5,
        stable_position_threshold=0.5,
        img_width=128,
        img_height=128,
        bitmap_size=128,
        enable_image_observation=True,
    ) -> None:
        self.game_url = f"http://localhost:{port}/"
        # Check if port is already in use
        self.server = None
        if not self.is_port_in_use(port):
            # Get the absolute path of the current script
            script_dir = os.path.dirname(os.path.realpath(__file__))
            # Construct the absolute path of the suika-game directory
            suika_game_dir = os.path.join(script_dir, 'suika-game')
            self.server = subprocess.Popen(["python", "-m", "http.server", str(port)], cwd=suika_game_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        self._chrome_options = webdriver.ChromeOptions()
        self._chrome_options.add_argument("--width=1024")
        self._chrome_options.add_argument("--height=768")
        # Stability flags for headless/server environments.
        self._chrome_options.add_argument("--disable-dev-shm-usage")
        self._chrome_options.add_argument("--no-sandbox")
        self._chrome_options.add_argument("--disable-gpu")
        self._chrome_options.add_argument("--disable-background-timer-throttling")
        self._chrome_options.add_argument("--disable-renderer-backgrounding")
        self._chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        self.headless = headless
        if headless:
            self._chrome_options.add_argument("--headless=new")
        self.delay_before_img_capture = delay_before_img_capture
        self.mute_sound = mute_sound
        self.wait_for_ready_on_step = wait_for_ready_on_step
        self.ready_poll_interval = ready_poll_interval
        self.ready_timeout = ready_timeout
        self.stable_velocity_threshold = float(stable_velocity_threshold)
        self.stable_velocity_polls = int(max(1, stable_velocity_polls))
        self.stable_position_threshold = float(stable_position_threshold)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.bitmap_size = int(bitmap_size)
        self.enable_image_observation = bool(enable_image_observation)
        self._blank_image = np.zeros((self.img_height, self.img_width, 4), dtype=np.uint8)
        self._fruit_radii = np.array([24, 32, 40, 56, 64, 72, 84, 96, 128, 160, 192], dtype=np.float32)
        self._prev_merged_counts = np.zeros(11, dtype=np.float32)
        self.driver = self._create_driver()
        _obs_dict = {
            'image': gymnasium.spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 4),  dtype="uint8"),
            'bitmap': gymnasium.spaces.Box(low=0, high=11, shape=(self.bitmap_size, self.bitmap_size),  dtype="uint8"),
            'board_top50_exyir': gymnasium.spaces.Box(low=0, high=256, shape=(50, 5), dtype="float32"),
            'current_fruit_type': gymnasium.spaces.Box(low=0, high=10, shape=(1,), dtype="float32"),
            'next_fruit_type': gymnasium.spaces.Box(low=0, high=10, shape=(1,), dtype="float32"),
            'current_fruit_x': gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype="float32"),
            'stage_top10_xy': gymnasium.spaces.Box(low=0, high=1, shape=(20,), dtype="float32"),
            'top10_fruit_types': gymnasium.spaces.Box(low=0, high=10, shape=(10,), dtype="float32"),
            'top10_mask': gymnasium.spaces.Box(low=0, high=1, shape=(10,), dtype="float32"),
            'max_height': gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype="float32"),
            'danger_count': gymnasium.spaces.Box(low=0, high=500, shape=(1,), dtype="float32"),
            'largest_fruit_type': gymnasium.spaces.Box(low=0, high=10, shape=(1,), dtype="float32"),
            'fruit_count': gymnasium.spaces.Box(low=0, high=500, shape=(1,), dtype="float32"),
            'board_fruit_xy': gymnasium.spaces.Box(low=0, high=1, shape=(80,), dtype="float32"),
            'board_fruit_radius': gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
            'board_fruit_mass': gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
            'board_fruit_type': gymnasium.spaces.Box(low=0, high=10, shape=(40,), dtype="float32"),
            'board_fruit_mask': gymnasium.spaces.Box(low=0, high=1, shape=(40,), dtype="float32"),
        }
        self.observation_space = gymnasium.spaces.Dict(_obs_dict)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _create_driver(self):
        driver = webdriver.Chrome(options=self._chrome_options)
        # Keep selenium command timeouts shorter so worker failures recover quickly.
        driver.set_script_timeout(20)
        return driver

    def _restart_driver(self):
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
        self.driver = self._create_driver()

    def restart_browser(self):
        """Public hook for vec_env.env_method()."""
        self._restart_driver()

    def reset(self,seed=None, options=None):
        self._reload()
        info = {}
        self.score = 0
        self._prev_merged_counts = np.zeros(11, dtype=np.float32)
        obs, status, _, snapshot = self._get_obs_and_status()
        merged = snapshot.get("merged_counts", None)
        if merged is not None:
            self._prev_merged_counts = np.asarray(merged, dtype=np.float32)
        return obs, info

    def _reload(self):
        # open the game.
        self.driver.get(self.game_url)
        if self.mute_sound:
            self._mute_audio()
        # Ensure initial controllable x position is centered (0.5 -> 320/640).
        self.driver.execute_script("""
            const input = document.getElementById('fruit-position');
            if (input) input.value = '320';
        """)
        # click start game button with id "start-game-button"
        self.driver.find_element(By.ID, 'start-game-button').click()
        time.sleep(1)

    def _mute_audio(self):
        # Disable audio playback in browser for training speed/stability.
        self.driver.execute_script("""
            (() => {
                const mute = (a) => {
                    if (!a) return;
                    a.muted = true;
                    a.volume = 0.0;
                    a.play = () => Promise.resolve();
                };
                if (window.Game && window.Game.sounds) {
                    Object.values(window.Game.sounds).forEach(mute);
                }
                document.querySelectorAll("audio").forEach(mute);
            })();
        """)
    
    def _get_obs_and_status(self):
        if self.enable_image_observation:
            img = self._capture_canvas()
        else:
            img = self._blank_image
        snapshot = self._query_game_snapshot()
        bitmap = self._build_bitmap(snapshot)
        return dict(
            image=img,
            bitmap=bitmap,
            board_top50_exyir=np.array(snapshot["board_top50_exyir"], dtype=np.float32),
            current_fruit_type=np.array([snapshot["current_fruit_type"]], dtype=np.float32),
            next_fruit_type=np.array([snapshot["next_fruit_type"]], dtype=np.float32),
            current_fruit_x=np.array([snapshot["current_fruit_x"]], dtype=np.float32),
            stage_top10_xy=np.array(snapshot["stage_top10_xy"], dtype=np.float32),
            top10_fruit_types=np.array(snapshot["top10_fruit_types"], dtype=np.float32),
            top10_mask=np.array(snapshot["top10_mask"], dtype=np.float32),
            max_height=np.array([snapshot["max_height"]], dtype=np.float32),
            danger_count=np.array([snapshot["danger_count"]], dtype=np.float32),
            largest_fruit_type=np.array([snapshot["largest_fruit_type"]], dtype=np.float32),
            fruit_count=np.array([snapshot["fruit_count"]], dtype=np.float32),
            board_fruit_xy=np.array(snapshot["board_fruit_xy"], dtype=np.float32),
            board_fruit_radius=np.array(snapshot["board_fruit_radius"], dtype=np.float32),
            board_fruit_mass=np.array(snapshot["board_fruit_mass"], dtype=np.float32),
            board_fruit_type=np.array(snapshot["board_fruit_type"], dtype=np.float32),
            board_fruit_mask=np.array(snapshot["board_fruit_mask"], dtype=np.float32),
        ), snapshot["status"], float(snapshot["score"]), snapshot

    def _build_bitmap(self, snapshot):
        n = self.bitmap_size
        bitmap = np.zeros((n, n), dtype=np.uint8)
        board_xy = np.asarray(snapshot["board_fruit_xy"], dtype=np.float32).reshape(-1, 2)
        board_t = np.asarray(snapshot["board_fruit_type"], dtype=np.int32).reshape(-1)
        board_m = np.asarray(snapshot["board_fruit_mask"], dtype=np.float32).reshape(-1)
        for i in range(board_m.shape[0]):
            if board_m[i] < 0.5:
                continue
            t = int(np.clip(board_t[i], 0, 10))
            val = np.uint8(t + 1)
            x = float(np.clip(board_xy[i, 0], 0.0, 1.0)) * (n - 1)
            y = float(np.clip(board_xy[i, 1], 0.0, 1.0)) * (n - 1)
            r = float(self._fruit_radii[t])
            rx = max(1.0, r / 640.0 * (n - 1))
            ry = max(1.0, r / 960.0 * (n - 1))
            x0 = max(0, int(np.floor(x - rx)))
            x1 = min(n - 1, int(np.ceil(x + rx)))
            y0 = max(0, int(np.floor(y - ry)))
            y1 = min(n - 1, int(np.ceil(y + ry)))
            if x1 < x0 or y1 < y0:
                continue
            yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
            mask = (((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2) <= 1.0
            patch = bitmap[y0 : y1 + 1, x0 : x1 + 1]
            patch[mask] = np.maximum(patch[mask], val)
        return bitmap

    def _query_game_snapshot(self):
        # Game object can be transiently unavailable right after reload/start.
        # Retry briefly and fall back to safe defaults instead of crashing workers.
        deadline = time.time() + 2.0
        while True:
            out = self.driver.execute_script("""
            (() => {
                const game = window.Game || {};
                const engine = window.__suikaEngine;
                const status = Number.isFinite(game.stateIndex) ? game.stateIndex : 0;
                const score = Number.isFinite(game.score) ? game.score : 0;
                const fruitType = Number.isFinite(game.currentFruitSize) ? game.currentFruitSize : 0;
                const nextFruitType = Number.isFinite(game.nextFruitSize) ? game.nextFruitSize : 0;
                let x = 0.5;
                const previewX = game?.elements?.previewBall?.position?.x;
                if (Number.isFinite(previewX)) {
                    x = previewX / 640.0;
                } else {
                    const input = document.getElementById("fruit-position");
                    const parsed = Number(input?.value);
                    if (Number.isFinite(parsed)) x = parsed / 640.0;
                }
                x = Math.max(0.0, Math.min(1.0, x));
                const top10 = [];
                const top10Types = [];
                const top10Mask = [];
                const boardXY = [];
                const boardRadius = [];
                const boardMass = [];
                const boardType = [];
                const boardMask = [];
                const top50 = [];
                const bodies = engine?.world?.bodies || [];
                const fruits = bodies
                  .filter((b) => !b.isStatic && Number.isFinite(b?.position?.x) && Number.isFinite(b?.position?.y) && Number.isFinite(b?.sizeIndex))
                const sortedByY = [...fruits].sort((a, b) => a.position.y - b.position.y);
                const topFruits = sortedByY.slice(0, 10);
                for (const b of topFruits) {
                  let nx = b.position.x / 640.0;
                  let ny = b.position.y / 960.0;
                  nx = Math.max(0.0, Math.min(1.0, nx));
                  ny = Math.max(0.0, Math.min(1.0, ny));
                  let t = Number.isFinite(b.sizeIndex) ? Math.floor(b.sizeIndex) : 0;
                  t = Math.max(0, Math.min(10, t));
                  top10.push(nx, ny);
                  top10Types.push(t);
                  top10Mask.push(1.0);
                }
                const boardFruits = sortedByY.slice(0, 40);
                for (const b of boardFruits) {
                  let nx = b.position.x / 640.0;
                  let ny = b.position.y / 960.0;
                  nx = Math.max(0.0, Math.min(1.0, nx));
                  ny = Math.max(0.0, Math.min(1.0, ny));
                  let nr = Number.isFinite(b.circleRadius) ? (b.circleRadius / 100.0) : 0.0;
                  let nm = Number.isFinite(b.mass) ? (b.mass / 1000.0) : 0.0;
                  nr = Math.max(0.0, Math.min(1.0, nr));
                  nm = Math.max(0.0, Math.min(1.0, nm));
                  let t = Number.isFinite(b.sizeIndex) ? Math.floor(b.sizeIndex) : 0;
                  t = Math.max(0, Math.min(10, t));
                  boardXY.push(nx, ny);
                  boardRadius.push(nr);
                  boardMass.push(nm);
                  boardType.push(t);
                  boardMask.push(1.0);
                }
                const topFruitsByTop = [...fruits]
                  .sort((a, b) => {
                    const at = (a.position.y + (Number.isFinite(a.circleRadius) ? a.circleRadius : 0));
                    const bt = (b.position.y + (Number.isFinite(b.circleRadius) ? b.circleRadius : 0));
                    return at - bt;
                  })
                  .slice(0, 50);
                for (const b of topFruitsByTop) {
                  let nx = b.position.x / 640.0;
                  let ny = b.position.y / 960.0;
                  nx = Math.max(0.0, Math.min(1.0, nx));
                  ny = Math.max(0.0, Math.min(1.0, ny));
                  let t = Number.isFinite(b.sizeIndex) ? Math.floor(b.sizeIndex) + 1 : 0;
                  t = Math.max(0, Math.min(11, t));
                  let r = Number.isFinite(b.circleRadius) ? b.circleRadius : 0.0;
                  r = Math.max(0.0, Math.min(256.0, r));
                  top50.push([1.0, nx, ny, t, r]);
                }
                while (top50.length < 50) {
                  top50.push([0.0, 0.0, 0.0, 0.0, 0.0]);
                }
                while (top10Types.length < 10) top10Types.push(0.0);
                while (top10Mask.length < 10) top10Mask.push(0.0);
                while (top10.length < 20) top10.push(0.0);
                while (boardXY.length < 80) boardXY.push(0.0);
                while (boardRadius.length < 40) boardRadius.push(0.0);
                while (boardMass.length < 40) boardMass.push(0.0);
                while (boardType.length < 40) boardType.push(0.0);
                while (boardMask.length < 40) boardMask.push(0.0);
                const fruitCount = fruits.length;
                const largestFruitType = fruits.reduce((m, b) => Math.max(m, Number.isFinite(b.sizeIndex) ? b.sizeIndex : 0), 0);
                const minY = fruits.length > 0 ? Math.min(...fruits.map((b) => b.position.y)) : 960;
                const maxHeight = Math.max(0.0, Math.min(1.0, (960 - minY) / 960.0));
                const dangerCount = fruits.filter((b) => b.position.y <= 84).length;
                const merged = Array.isArray(game.fruitsMerged) ? game.fruitsMerged : [];
                const mergedCounts = [];
                for (let i = 0; i < 11; i++) {
                  const c = Number(merged[i]);
                  mergedCounts.push(Number.isFinite(c) ? c : 0);
                }
                return {
                  status,
                  score,
                  current_fruit_type: fruitType,
                  next_fruit_type: nextFruitType,
                  current_fruit_x: x,
                  stage_top10_xy: top10,
                  top10_fruit_types: top10Types,
                  top10_mask: top10Mask,
                  max_height: maxHeight,
                  danger_count: dangerCount,
                  largest_fruit_type: largestFruitType,
                  fruit_count: fruitCount,
                  board_fruit_xy: boardXY,
                  board_fruit_radius: boardRadius,
                  board_fruit_mass: boardMass,
                  board_fruit_type: boardType,
                  board_fruit_mask: boardMask,
                  board_top50_exyir: top50,
                  merged_counts: mergedCounts,
                };
            })();
        """)
            if isinstance(out, dict) and "status" in out:
                return out
            if time.time() >= deadline:
                return {
                    "status": 0,
                    "score": float(self.score),
                    "current_fruit_type": 0,
                    "next_fruit_type": 0,
                    "current_fruit_x": 0.5,
                    "stage_top10_xy": [0.0] * 20,
                    "top10_fruit_types": [0.0] * 10,
                    "top10_mask": [0.0] * 10,
                    "max_height": 0.0,
                    "danger_count": 0.0,
                    "largest_fruit_type": 0.0,
                    "fruit_count": 0.0,
                    "board_fruit_xy": [0.0] * 80,
                    "board_fruit_radius": [0.0] * 40,
                    "board_fruit_mass": [0.0] * 40,
                    "board_fruit_type": [0.0] * 40,
                    "board_fruit_mask": [0.0] * 40,
                    "board_top50_exyir": [[0.0, 0.0, 0.0, 0.0, 0.0]] * 50,
                    "merged_counts": [0.0] * 11,
                }
            time.sleep(0.02)

    def capture_canvas_raw_rgba(self):
        """Debug helper: return full canvas before resize."""
        return self._capture_canvas_raw()

    def capture_canvas_full_rgba(self):
        """Debug helper: return full canvas."""
        canvas = self.driver.find_element(By.ID, 'game-canvas')
        image_string = canvas.screenshot_as_png
        img = Image.open(io.BytesIO(image_string)).convert("RGBA")
        return np.asarray(img)
    
    def _capture_canvas(self):
        img = Image.fromarray(self._capture_canvas_raw(), mode="RGBA")
        imgResized = img.resize((self.img_width, self.img_height), RESAMPLE_LANCZOS)
        arr = np.asarray(imgResized)
        return arr

    def _capture_canvas_raw(self):
        # screenshots the game canvas with id "game-canvas" and stores it in a numpy array
        canvas = self.driver.find_element(By.ID, 'game-canvas')
        image_string = canvas.screenshot_as_png
        img = Image.open(io.BytesIO(image_string)).convert("RGBA")
        # Keep full game canvas for observation to avoid losing state information.
        return np.asarray(img)

    
    def step(self, action):
        driver = self.driver
        action = float(action[0])
        info = {}
        try:
            # Centered action in [-1, 1] -> normalized position in [0, 1].
            x_centered = float(np.clip(action, -1.0, 1.0))
            x_norm = (x_centered + 1.0) * 0.5
            action = str(int(x_norm * 640))
            # clear the input box with id "fruit-position"
            driver.find_element(By.ID, 'fruit-position').clear()
            # enter in the number into the input box with id "fruit-position"
            driver.find_element(By.ID, 'fruit-position').send_keys(action)
            # click the button with id "drop-fruit-button"
            driver.find_element(By.ID, 'drop-fruit-button').click()
            if self.wait_for_ready_on_step:
                self._wait_until_step_stable()
            elif self.delay_before_img_capture > 0:
                time.sleep(self.delay_before_img_capture)

            obs, status, score, snapshot = self._get_obs_and_status()
            js_score = self.driver.execute_script(
                "return Number.isFinite(window.Game?.score) ? window.Game.score : null;"
            )
            if js_score is not None:
                score = float(js_score)
            # Robust game-over detection: stateIndex LOSE or visible end UI.
            js_state, end_visible = self.driver.execute_script("""
                return [
                    Number.isFinite(window.Game?.stateIndex) ? window.Game.stateIndex : null,
                    getComputedStyle(document.getElementById('game-end-container')).display !== 'none'
                ];
            """)
            # check if game is over.
            terminal = (status == 3) or (js_state == 3) or bool(end_visible)
            truncated = False 
            info['score'] = score
            info["final_score_valid"] = bool(terminal)

            # Reward design:
            #   merge reward: cherry=+0.1, strawberry=+0.2, ..., melon=+1.0
            #   fruit-count penalty: 0.001 * fruit_count
            #   height penalty: -x/100 where x is board height percent (max_height in [0,1] => 0.1*max_height)
            #   gameover penalty: -2.0
            merged_now = np.asarray(snapshot.get("merged_counts", [0.0] * 11), dtype=np.float32)
            merged_prev = getattr(self, "_prev_merged_counts", np.zeros(11, dtype=np.float32))
            merged_delta = np.maximum(merged_now - merged_prev, 0.0)
            merge_reward_weights = np.minimum((np.arange(11, dtype=np.float32) + 1.0) * 0.1, 1.0)
            merge_reward = float(np.sum(merged_delta * merge_reward_weights))
            self._prev_merged_counts = merged_now

            fruit_count = float(obs.get("fruit_count", np.array([0.0], dtype=np.float32))[0])
            max_height = float(obs.get("max_height", np.array([0.0], dtype=np.float32))[0])
            fruit_penalty = 0.001 * fruit_count
            height_penalty = 0.1 * max_height
            terminal_penalty = 2.0 if terminal else 0.0
            reward = merge_reward - fruit_penalty - height_penalty - terminal_penalty
            self.score = score

            return obs, reward, terminal, truncated, info
        except (TimeoutException, WebDriverException, ReadTimeoutError, TimeoutError, OSError) as exc:
            # Recover from browser crashes/timeouts instead of killing the worker process.
            prev_score = float(self.score)
            info["browser_error"] = str(exc)
            info["recovered"] = True
            info["discard_episode"] = True
            info["discard_reason"] = "browser_recovery"
            info["score"] = prev_score
            info["final_score_valid"] = False
            self._restart_driver()
            obs, _ = self.reset()
            return obs, 0.0, True, True, info

    def _wait_until_step_stable(self):
        # In the JS game, DROP state is 2.
        # Waiting only for state transition can be too early for score update,
        # so we additionally wait until score stays unchanged and
        # (speed is low OR body positions are stable).
        start = time.time()
        stable_polls = 0
        last_score = None
        last_pos = None
        while True:
            state, score, max_speed, pos_list = self.driver.execute_script("""
                return [
                    Number.isFinite(window.Game?.stateIndex) ? window.Game.stateIndex : 0,
                    Number.isFinite(window.Game?.score) ? window.Game.score : 0,
                    (() => {
                        const engine = window.__suikaEngine;
                        const bodies = engine?.world?.bodies || [];
                        let m = 0.0;
                        for (const b of bodies) {
                            if (!b || b.isStatic) continue;
                            const vx = Number.isFinite(b?.velocity?.x) ? b.velocity.x : 0.0;
                            const vy = Number.isFinite(b?.velocity?.y) ? b.velocity.y : 0.0;
                            const s = Math.hypot(vx, vy);
                            if (s > m) m = s;
                        }
                        return m;
                    })(),
                    (() => {
                        const engine = window.__suikaEngine;
                        const bodies = engine?.world?.bodies || [];
                        const out = [];
                        for (const b of bodies) {
                            if (!b || b.isStatic) continue;
                            if (!Number.isFinite(b?.id) || !Number.isFinite(b?.position?.x) || !Number.isFinite(b?.position?.y)) continue;
                            out.push([b.id, b.position.x, b.position.y]);
                        }
                        return out;
                    })()
                ];
            """)

            speed_ok = float(max_speed) <= self.stable_velocity_threshold
            cur_pos = {}
            if isinstance(pos_list, list):
                for item in pos_list:
                    if isinstance(item, list) and len(item) >= 3:
                        try:
                            cur_pos[int(item[0])] = (float(item[1]), float(item[2]))
                        except Exception:
                            pass
            position_ok = False
            if last_pos is not None and cur_pos:
                common = set(last_pos.keys()).intersection(cur_pos.keys())
                if common:
                    max_delta = 0.0
                    for bid in common:
                        x0, y0 = last_pos[bid]
                        x1, y1 = cur_pos[bid]
                        d = float(np.hypot(x1 - x0, y1 - y0))
                        if d > max_delta:
                            max_delta = d
                    position_ok = max_delta <= self.stable_position_threshold
            last_pos = cur_pos

            if state != 2 and (speed_ok or position_ok):
                if last_score is not None and score == last_score:
                    stable_polls += 1
                else:
                    stable_polls = 0
                last_score = score
                if stable_polls >= self.stable_velocity_polls:
                    return
            else:
                stable_polls = 0
                last_score = score

            if (time.time() - start) >= self.ready_timeout:
                return
            time.sleep(self.ready_poll_interval)


    def is_port_in_use(self, port):
        """Check if a given port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def close(self):
        super().close()
        if self.driver is not None:
            self.driver.quit()
        # Stop the server
        if self.server is not None:
            self.server.terminate()

if __name__ == "__main__":
    env = SuikaBrowserEnv(headless=False, delay_before_img_capture=0.5)
    try:
        video = []
        obs, info = env.reset()
        # video.append(obs['image'])
        # import imageio
        terminated = False
        while not terminated:
            action = [0]
            obs, rew, terminated, truncated, info = env.step(action)
            # video.append(obs['image'])
            if terminated:
                break
    finally:
        env.close()
