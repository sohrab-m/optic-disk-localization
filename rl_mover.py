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
from utils import angle_between
import scipy.io as sio
import csv
import random
import torch
import ast

class optic_disc(gym.Env):
    
    def __init__(self, path_to_annotations, random_start=True, N=100, device=None):
        super(optic_disc, self).__init__()

        self.device= device
        self.resolution= (154,154)
        self.optic_rad = 150
        self.N = N
        # self.load_annots()
        self.load_annots_path_xy(path_to_annotations)
        self.load_images()
        self.create_world()

        self.initial_loc=[np.random.randint(low=self.resolution[1], high=self.world.shape[1]-self.resolution[1]) ,
                          np.random.randint(low=self.resolution[0], high=self.world.shape[0]-self.resolution[0])] if random_start else [700,700]
        # if random_start:
        #     self.initial_loc=[np.random.randint(low=self.resolution[1], high=self.world.shape[1]-self.resolution[1]) ,np.random.randint(low=self.resolution[0], high=self.world.shape[0]-self.resolution[0])]
        # else:
        #     self.inital_loc=[700,700]
        self.x=self.initial_loc[0]
        self.y=self.initial_loc[1]
        
        self.create_mask()
        # self.reward_map()
        self.reward=0
        # lets define the action set up, down, left, right
        self.action_set=            [ 0,     1,    2,    3]
        self.action_space=Discrete(4)
        self.done=False
        self.previous_x = None
        self.previous_y = None
        self.r_coeff = 1
        self.invalid_coeff = 100

        # initial location in the form of (x,y)
        self.step_count = 0
  

        self.observation_space=Box(low=0, high=255,
                                shape=(self.resolution[0], self.resolution[1], 3), dtype=np.uint8)
    
    def load_images(self):
        self.worlds = []
        self.N = len(self.filepaths)
        for idx in range(self.N):
            self.worlds.append(cv2.imread(self.filepaths[idx]))

    def load_annots_path_xy(self, path_to_annotations):
        self.filepaths=[]
        self.targets=[]
        print('loading annotations from', path_to_annotations)
        with open(path_to_annotations, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.filepaths.append(row[0])
                coords=ast.literal_eval(row[1])
                self.targets.append(coords)

        
        print('len optic disc imgs', len(self.filepaths))
        print('len of optic disc locations', len(self.targets))
        


    def load_annots(self):
        # Create empty lists to store the data
        self.filenames = []
        self.x_values = []
        self.y_values = []
        self.filepaths = []
        self.lefts = []
        self.ups = []
        self.rights = []
        self.downs = []

        # Open the CSV file and read the data into the lists
        with open("merged_annots.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.filenames.append(row[0])
                self.x_values.append(float(row[1]))
                self.y_values.append(float(row[2]))
                self.filepaths.append(row[3])
                self.lefts.append(float(row[4]))   
                self.ups.append(float(row[5]))   
                self.rights.append(float(row[6]))   
                self.downs.append(float(row[7]))               

    def make_valid(self):
        self.x, re1 = optic_disc.cut_off(self.x, self.x_bounds[0], self.x_bounds[1])
        self.y, re2 = optic_disc.cut_off(self.y, self.y_bounds[0], self.y_bounds[1])
        if re1 == -1 or re2 == -1:
            return -self.r_coeff * self.invalid_coeff
        else:
            return 0

    @staticmethod
    def cut_off(x, m, M):
        if x < m:
            return [m, -1]
        if x > M:
            return [M, -1]
        return [x, 0]
    
    def step(self, action):
        self.action=action
        
        # if self.action==0:
        #     self.move=[0,-154]
        self.move=[self.action//2,self.action%2]
        self.previous_x=self.x
        self.previous_y=self.y
        
        if self.action//2:
            self.x+=(2*(self.action%2)-1) * self.resolution[0] // 2
        else:
            self.y+=(2*(self.action%2)-1) * self.resolution[1] // 2

        # forcing the x and y values to be inside the acceptable range
        step_cost = self.make_valid()

        # print('x', self.x, 'y', self.y)
        
               
        observation=self.get_frame()
        # reward=np.float(np.sum(self.reward_of_patch))
        dummy = self.calculate_reward_direction()  #just finds the distances at all times  
        # self.reward = -2*self.curr_dist/self.world.shape[0]
        self.calculate_reward_uniform()

        reward=self.reward * self.r_coeff 
        self.done= (self.curr_dist < self.optic_rad) or self.step_count > 100
        done=self.done
        info={"x": self.x, "y": self.y, "reward": reward, 'done': done, 'action': self.action, 'prev_and_now': (self.prev_dist, self.curr_dist)}
        # info={"x": self.x, "y": self.y, "reward": reward, 'done': done, 'action': self.action}        
        self.step_count += 1
        # if self.step_count>99:
        #     print('100th STEP', reward )
        return observation, reward, done, info

    def calculate_reward_direction(self):
        self.prev_dist= np.sqrt((self.previous_x-self.optic_x)**2 + (self.previous_y-self.optic_y)**2)
        self.curr_dist=  np.sqrt((self.x-self.optic_x)**2 + (self.y-self.optic_y)**2)
        return (self.prev_dist - self.curr_dist)*self.r_coeff
    
    def calculate_reward_distance(self):
        self.prev_dist= np.sqrt((self.previous_x-self.optic_x)**2 + (self.previous_y-self.optic_y)**2)
        self.curr_dist=  np.sqrt((self.x-self.optic_x)**2 + (self.y-self.optic_y)**2)  
        if self.curr_dist < self.resolution[0]:
            return self.r_coeff ** 2
        return self.curr_distance/self.world.shape[0] 

    def calculate_reward_uniform(self): #reward is -1 for each step that does not find the disc and 1 otherwise
        if self.curr_dist < self.optic_rad:
            print('found the disc')
            self.reward = 1
        else:
            self.reward = -1
        


    
    def reset(self):
        self.create_world()
        # self.reward_map()
        self.initial_loc=[np.random.randint(low=self.resolution[1], high=self.world.shape[1]-self.resolution[1]) ,np.random.randint(low=self.resolution[0], high=self.world.shape[0]-self.resolution[0])]
        self.x=self.initial_loc[0]
        self.y=self.initial_loc[1]
        self.done=False
        observation=self.get_frame()
        self.step_count = 0
        return observation

        
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

        self.patch = self.world[self.y-self.patch_rad:self.y+self.patch_rad,self.x-self.patch_rad:self.x+self.patch_rad, :] * self.patch_mask
        # self.reward_of_patch = self.rewards[self.y-self.patch_rad:self.y+self.patch_rad, self.x-self.patch_rad:self.x+self.patch_rad] * self.patch_mask[:,:,0]
        # print('reward of patch: ', np.sum(self.reward_of_patch))
        # cv2.imwrite('rPatch.jpg', self.reward_of_patch*255)
        self.patch = np.uint8(self.patch)
        
        return self.patch

    def reward_map(self):
        # self.rewards = np.asarray([[1  if (x-self.optic_x)**2 + (y-self.optic_y)**2 < self.optic_rad ** 2 else 0 \
        #  for x in range(self.world.shape[1])] for y in range(self.world.shape[0])])
        self.rewards = np.asarray([[ 255/(np.sqrt((x-self.optic_x)**2 + (y-self.optic_y)**2)+1) \
         for x in range(self.world.shape[1])] for y in range(self.world.shape[0])])
        self.total_reward=np.sum(self.rewards)
        # print('total rewards', self.total_reward)
        

        
        
    def close(self):
        return
    
    def create_world(self):
        # load the frame 
        idx = random.randint(0, self.N-1)
        self.idx = idx
        self.world = self.worlds[idx]
        self.optic_x =self.targets[idx][0]
        self.optic_y =self.targets[idx][1]

        # self.left = self.lefts[idx]
        # self.up = self.ups[idx]
        # self.optic_x = self.x_values[idx] - self.left
        # self.optic_y = self.y_values[idx] - self.up

        # resolution can be an even number only
        self.x_bounds=[self.resolution[0]//2+1 , self.world.shape[1]-self.resolution[0]//2-1]
        self.y_bounds=[self.resolution[0]//2+1 , self.world.shape[0]-self.resolution[0]//2-1]
        self.world = torch.from_numpy(self.world)
        self.world.to(self.device)
        return self.world
        
        
# mover=optic_disc()

        
        # self.location[0]=self.location[0]+

if __name__ == "__main__":
    path_xy_annots = "data/sample_optic_disc/path_xy_annots_.csv"
    env = optic_disc(path_xy_annots)

    img = env.create_world()
    img=np.array(img, dtype=np.uint8)
    
    

    obs1=env.get_frame()

    for i in range(3):
        observation, reward, done, info = env.step(1)
        # obs1=env.get_frame()
        print('reward from obs1', observation.shape, reward, done, info)
        cv2.imwrite(str(i)+".jpg", np.array(observation, dtype=np.uint8))
        
        img=cv2.putText(img, str(i), (env.x, env.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        

    for i in range(6):
        observation, reward, done, info = env.step(3)
        print('reward from obs1',observation.shape, reward, done, info)
        cv2.imwrite(str(i+5)+".jpg", np.array(observation, dtype=np.uint8))
        img=cv2.putText(img, str(i+5), (env.x, env.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    # cv2.imwrite(str(i+5)+".jpg", np.array(obs1, dtype=np.uint8))
    # cv2.imwrite("world.jpg", np.array(img, dtype=np.uint8))
    # cv2.imwrite("rewards.jpg", np.array(env.rewards, dtype=np.uint8))
    # cv2.imwrite("rewards_world.jpg", np.array(np.expand_dims(env.rewards, -1)*img, dtype=np.uint8))
    cv2.imwrite("world_after_step.jpg", img)
    print(img.shape)




    
    
