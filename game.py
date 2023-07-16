#%%

#MSS used for Screen cap
from mss import mss
#Sending commands
import pydirectinput 
#frame processing
import cv2
# transformation framework
import numpy as np
#OCR for game over extraction
import pytesseract
#Visualize captured frames
from matplotlib import pyplot as plt
#bring in time for pauses
import time 
#enviroment components
from gym import Env
from gym.spaces import Box,Discrete

class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low = 0, high = 255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {"top":300,"left":0,"width":600,"height":500}
        self.done_location = {"top":405,"left":630,"width":660,"height":500}
        pass
    def step(self,action):
        pass
    def render(self):
        pass
    def reset(self):
        pass
    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,3].astype(np.uint8)
        return raw
        pass
    def get_done(self):
        pass
env = WebGame()
print(np.array(env.get_observation()))
env.action_space.sample()
plt.imshow(env.observation_space.sample()[0])
   
# %%
