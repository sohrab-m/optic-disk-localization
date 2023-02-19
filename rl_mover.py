# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 18:20:09 2023
@author: amire
---x
|
y
"""

import gym 
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os
import cv2


class optic_disc(gym.Env):
    
    def __init__(self):
        super(optic_disc, self).__init__()
        self.resolution= (154,154)
        self.x=154
        self.y=154
        # lets define the action set up, down, left, right
        self.action_set=            [ 0,     1,    2,    3]
        self.action_space=Discrete(4)
        self.done=False
        self.disc_loc=[1000,1000]
        
        self.observation=Box(low=0, high=255,
                                shape=(self.resolution[0], self.resolution[0], 3), dtype=np.uint8)
        self.create_world()
    
    def step(self, action):
        self.action=action
        
        # if self.action==0:
        #     self.move=[0,-154]
        self.move=[self.action//2,self.action%2]
        
        if self.action//2:
            self.x+=(2*(self.action%2)-1) * self.resolution[0]
        else:
            self.y+=(2*(self.action%2)-1) * self.resolution[1]


        print('x', self.x, 'y', self.y)
        
        
        
        ### TODO:load the reward database in the __ini__ and use it here
            
            
        if self.x-self.resolution[0]<self.disc_loc[0]<self.x+self.resolution[0]\
        and self.y-self.resolution[1]<self.disc_loc[1]<self.y+self.resolution[1]:
            self.reward=100
            self.done=True
            
        else:
            self.reward=-1
        
        observation=self.observation
        done=self.done
        info={}
        reward=self.reward
        

        return observation, reward, done, info
    
    def reset(self):
        
    ### TODO: make sure to load all the frames in the daataset
        return
    
    def close(self):
        return
        
    def create_world(self):
        # load the frame 
        img_path = os.path.join(r"C:\Users\ssohr\OneDrive\Documents\optic-disk-localization\dataset\Base11\Base11", "20051019_38557_0100_PP.tif")
        self.world = cv2.imread("valid.jpg")
        
    

        
        
mover=optic_disc()

        
        # self.location[0]=self.location[0]+

if __name__ == "__main__":
    env = optic_disc()
    world = env.world
    cv2.imwrite("test.jpg", world)

