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
        img_width=128,
        img_height=128,
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
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.driver = self._create_driver()
        _obs_dict = {
            'image': gymnasium.spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 4),  dtype="uint8"),
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
            'board_fruit_xy': gymnasium.spaces.Box(low=0, high=1, shape=(60,), dtype="float32"),
            'board_fruit_radius': gymnasium.spaces.Box(low=0, high=1, shape=(30,), dtype="float32"),
            'board_fruit_mass': gymnasium.spaces.Box(low=0, high=1, shape=(30,), dtype="float32"),
            'board_fruit_type': gymnasium.spaces.Box(low=0, high=10, shape=(30,), dtype="float32"),
            'board_fruit_mask': gymnasium.spaces.Box(low=0, high=1, shape=(30,), dtype="float32"),
        }
        self.observation_space = gymnasium.spaces.Dict(_obs_dict)
        self.action_space = gymnasium.spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

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
        obs, status, _ = self._get_obs_and_status()
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
        img = self._capture_canvas()
        snapshot = self._query_game_snapshot()
        return dict(
            image=img,
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
        ), snapshot["status"], float(snapshot["score"])

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
                const boardFruits = sortedByY.slice(0, 30);
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
                while (top10Types.length < 10) top10Types.push(0.0);
                while (top10Mask.length < 10) top10Mask.push(0.0);
                while (top10.length < 20) top10.push(0.0);
                while (boardXY.length < 60) boardXY.push(0.0);
                while (boardRadius.length < 30) boardRadius.push(0.0);
                while (boardMass.length < 30) boardMass.push(0.0);
                while (boardType.length < 30) boardType.push(0.0);
                while (boardMask.length < 30) boardMask.push(0.0);
                const fruitCount = fruits.length;
                const largestFruitType = fruits.reduce((m, b) => Math.max(m, Number.isFinite(b.sizeIndex) ? b.sizeIndex : 0), 0);
                const minY = fruits.length > 0 ? Math.min(...fruits.map((b) => b.position.y)) : 960;
                const maxHeight = Math.max(0.0, Math.min(1.0, (960 - minY) / 960.0));
                const dangerCount = fruits.filter((b) => b.position.y <= 84).length;
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
                    "board_fruit_xy": [0.0] * 60,
                    "board_fruit_radius": [0.0] * 30,
                    "board_fruit_mass": [0.0] * 30,
                    "board_fruit_type": [0.0] * 30,
                    "board_fruit_mask": [0.0] * 30,
                }
            time.sleep(0.02)

    def capture_canvas_raw_rgba(self):
        """Debug helper: return cropped canvas before resize."""
        return self._capture_canvas_raw()
    
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
        # first crop out right hand side and lower bar.
        img = img.crop((0, 0, 520, img.height))
        return np.asarray(img)

    
    def step(self, action):
        driver = self.driver
        action = float(action[0])
        info = {}
        try:
            # Centered action in [-0.5, 0.5] -> normalized position in [0, 1].
            x_centered = float(np.clip(action, -0.5, 0.5))
            x_norm = x_centered + 0.5
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

            obs, status, score = self._get_obs_and_status()
            js_score = self.driver.execute_script(
                "return Number.isFinite(window.Game?.score) ? window.Game.score : null;"
            )
            if js_score is not None:
                score = float(js_score)
            # check if game is over.
            terminal = status == 3
            truncated = False 
            info['score'] = score
            reward = score - self.score
            if terminal:
                reward = -500.0
            self.score = score

            return obs, reward, terminal, truncated, info
        except (TimeoutException, WebDriverException, ReadTimeoutError, TimeoutError, OSError) as exc:
            # Recover from browser crashes/timeouts instead of killing the worker process.
            info["browser_error"] = str(exc)
            info["recovered"] = True
            self._restart_driver()
            obs, _ = self.reset()
            return obs, 0.0, True, True, info

    def _wait_until_step_stable(self):
        # In the JS game, DROP state is 2.
        # Waiting only for state transition can be too early for score update,
        # so we additionally wait until score stays unchanged for a short window.
        start = time.time()
        stable_polls = 0
        last_score = None
        while True:
            state, score = self.driver.execute_script("""
                return [
                    Number.isFinite(window.Game?.stateIndex) ? window.Game.stateIndex : 0,
                    Number.isFinite(window.Game?.score) ? window.Game.score : 0
                ];
            """)

            if state != 2:
                if last_score is not None and score == last_score:
                    stable_polls += 1
                else:
                    stable_polls = 0
                last_score = score
                if stable_polls >= 5:
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
