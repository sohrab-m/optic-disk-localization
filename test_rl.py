import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from rl_mover import optic_disc
from stable_baselines3 import PPO, A2C
import os
import time

# Create environment
env = optic_disc()

# Load the trained agent
model_path = f"Training/Models/1676854481/110000"
model=A2C.load(model_path, env=env)

episodes=1

for ep in range(episodes):
    obs=env.reset()
    done=False
    while not done:
        action, _=model.predict(obs)
        obs, reward, done, info = env.step(action)
        x=env.x
        y=env.y
        print('(x,y):',(x,y) , 'reward:',  reward)

