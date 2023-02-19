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
        
        self.create_world()
        
        self.resolution= (154,154)
        self.x=1000
        self.y=1000
        self.optic_x = 800
        self.optic_y = 667
        self.optic_rad = 100
        self.create_mask()
        self.reward_map()
        # lets define the action set up, down, left, right
        self.action_set=            [ 0,     1,    2,    3]
        self.action_space=Discrete(4)
        self.done=False
        self.disc_loc=[1000,1000]
        # initial location in the form of (x,y)
        self.inital_loc=[1000,1000]
  
        # resultion can  be an even number only
        self.x_bounds=[self.resolution[0]//2 , self.world.shape[1]-self.resolution[0]//2]
        self.y_bounds=[self.resolution[0]//2 , self.world.shape[0]-self.resolution[0]//2]

        self.observation=Box(low=0, high=255,
                                shape=(self.resolution[0], self.resolution[0], 3), dtype=np.uint8)

    def make_valid(self):
        self.x = optic_disc.cut_off(self.x, self.x_bounds[0], self.x_bounds[1])
        self.y = optic_disc.cut_off(self.y, self.y_bounds[0], self.y_bounds[1])

    @staticmethod
    def cut_off(x, m, M):
        if x < m:
            return m
        if x > M:
            return M
        return x
    
    def step(self, action):
        self.action=action
        
        # if self.action==0:
        #     self.move=[0,-154]
        self.move=[self.action//2,self.action%2]
        
        
        if self.action//2:
            self.x+=(2*(self.action%2)-1) * self.resolution[0]
        else:
            self.y+=(2*(self.action%2)-1) * self.resolution[1]

        # forcing the x and y values to be inside the acceptable range
        self.make_valid()

        print('x', self.x, 'y', self.y)
        
        
        
        ### TODO:load the reward database in the __ini__ and use it here
            
            
        if self.x-self.resolution[0]<self.disc_loc[0]<self.x+self.resolution[0]\
        and self.y-self.resolution[1]<self.disc_loc[1]<self.y+self.resolution[1]:
            self.reward=100
            self.done=True
            
        else:
            self.reward=-1
        
        
        
        observation=self.get_frame()
        done=self.done
        info={}
        reward=self.reward
        

        return observation, reward, done, info
    
    def reset(self):
        
    ### TODO: make sure to load all the frames in the daataset
        return

    def create_mask(self):
        self.patch_rad = int(self.resolution[0]/2)
        self.patch_mask = np.asarray([[1  if (x-self.patch_rad)**2 + (y-self.patch_rad)**2 < self.patch_rad ** 2 else 0 \
         for x in range(self.patch_rad*2)] for y in range(self.patch_rad*2)])
        self.patch_mask = np.expand_dims(self.patch_mask, -1) 
        
        
    def get_frame(self):
        # self.patch = self.world[self.x-self.patch_rad:self.x+self.patch_rad, self.y-self.patch_rad:self.y+self.patch_rad, :]
        # print(self.world.shape, self.patch.shape)
        # self.patch=self.world[1000:1154,1000:1154,:]

        self.patch = self.world[ self.y-self.patch_rad:self.y+self.patch_rad,self.x-self.patch_rad:self.x+self.patch_rad, :] * self.patch_mask

        return self.patch

    def reward_map(self):
        self.rewards = np.asarray([[1  if (x-self.optic_x)**2 + (y-self.optic_y)**2 < self.optic_rad ** 2 else 0 \
         for x in range(self.world.shape[1])] for y in range(self.world.shape[0])])

        
        
    def close(self):
        return
        
    def create_world(self):
        # load the frame 
        # img_path = os.path.join(r"C:\Users\ssohr\OneDrive\Documents\optic-disk-localization\dataset\Base11\Base11", "20051019_38557_0100_PP.tif")
        self.world = cv2.imread("sample_img.tif")
        return self.world
        
        
# mover=optic_disc()

        
        # self.location[0]=self.location[0]+

if __name__ == "__main__":
    env = optic_disc()
    # world = env.world
    img= env.create_world()
    # print('world shape', img.shape)

    for i in range(5):
        env.step(0)
        obs1=env.get_frame()
        cv2.imwrite(str(i)+".jpg", np.array(obs1, dtype=np.uint8))
        img=cv2.putText(img, str(i), (env.x, env.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        

    for i in range(5):
        env.step(3)
        obs1=env.get_frame()
        cv2.imwrite(str(i+5)+".jpg", np.array(obs1, dtype=np.uint8))
        img=cv2.putText(img, str(i+5), (env.x, env.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    # cv2.imwrite(str(i+5)+".jpg", np.array(obs1, dtype=np.uint8))
    
    cv2.imwrite("world.jpg", np.array(img, dtype=np.uint8))
    cv2.imwrite("rewards.jpg", np.array(env.rewards*255, dtype=np.uint8))
    cv2.imwrite("rewards_world.jpg", np.array(np.expand_dims(env.rewards, -1)*img, dtype=np.uint8))




    
    

