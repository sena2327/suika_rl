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
            'current_fruit_x': gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype="float32"),
        }
        self.observation_space = gymnasium.spaces.Dict(_obs_dict)
        self.action_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,))

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
        status, score, fruit_type, fruit_x = self._query_game_snapshot()
        return dict(
            image=img,
            current_fruit_type=np.array([fruit_type], dtype=np.float32),
            current_fruit_x=np.array([fruit_x], dtype=np.float32),
        ), status, float(score)

    def _query_game_snapshot(self):
        # Game object can be transiently unavailable right after reload/start.
        # Retry briefly and fall back to safe defaults instead of crashing workers.
        deadline = time.time() + 2.0
        while True:
            out = self.driver.execute_script("""
            (() => {
                const game = window.Game || {};
                const status = Number.isFinite(game.stateIndex) ? game.stateIndex : 0;
                const score = Number.isFinite(game.score) ? game.score : 0;
                const fruitType = Number.isFinite(game.currentFruitSize) ? game.currentFruitSize : 0;
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
                return [status, score, fruitType, x];
            })();
        """)
            if isinstance(out, (list, tuple)) and len(out) == 4:
                return out
            if time.time() >= deadline:
                return [0, float(self.score), 0, 0.5]
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
        action = action[0]
        info = {}
        try:
            # action is a float from 0 to 1. need to convert to int from 0 to 640 and then string.
            action = str(int(action * 640))
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
            reward = 0
            # check if game is over.
            terminal = status == 3
            truncated = False 
            info['score'] = score
            reward += score - self.score
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
        # In the JS game, DROP state is 2. We poll until it changes or timeout.
        start = time.time()
        while True:
            state = self.driver.execute_script("return window.Game.stateIndex;")
            if state != 2:
                return
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
