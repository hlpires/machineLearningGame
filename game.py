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

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Higor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

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
        pass

    def render(self):
        pass

    def reset(self):
        pass

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (100, 83))
        channel = np.reshape(resized, (1, 83, 100))
        return channel

    def get_done(self):
        done_cap =  np.array(self.cap.grab(self.done_location))[:, :, :3]
        done_strings = ["GAME", "GAHE","GARN"] 
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done_cap,done, res
    

env = WebGame()
done_cap,done,res = env.get_done()
plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_BGR2RGB))
# done, done_cap = env.get_done()
plt.imshow(done_cap)
print(res)

# %%
