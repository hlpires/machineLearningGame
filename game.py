# %%

# MSS used for Screen cap
from mss import mss

# Sending commands
import pydirectinput

# frame processing
import cv2

# transformation framework
import numpy as np

# OCR for game over extraction
import pytesseract

# Visualize captured frames
from matplotlib import pyplot as plt

# bring in time for pauses
import time

# enviroment components
from gym import Env
from gym.spaces import Box, Discrete

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\Higor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)


class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=0, high=255, shape=(1, 83, 100), dtype=np.uint8
        )
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {"top": 300, "left": 0, "width": 600, "height": 500}
        self.done_location = {"top": 375, "left": 630, "width": 660, "height": 70}
        pass

    def step(self, action):
        action_map = {0: "space", 1: "down", 2: "no_op"}
        if action != 2:
            pydirectinput.press(action_map[action])
            done = env.get_done()
            new_observation = self.get_observation()
            reward = 1
            info = {}
            print(done, new_observation)
            return done, new_observation

    def render(self):
        cv2.imshow("Game", np.array(self.cap.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()

    def close(self):
        cv2.destroyAllWindows()
        pass

    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press("space")
        return self.get_observation()

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (100, 83))
        channel = np.reshape(resized, (1, 83, 100))
        return channel

    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]
        done_strings = ["GAME", "GAHE", "GARN", "GANM"]
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done


env = WebGame()
for episode in range(10):
    obs = env.reset()
    done = env.get_done()
    total_reward = 0

    while not done:
        reward = 1
        [done, new_observation] = env.step(env.action_space.sample())
        total_reward = reward + total_reward
        print(f"Total reward for episode {episode} is {total_reward}")

# %%
