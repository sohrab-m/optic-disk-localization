# import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from rl_mover import optic_disc
from stable_baselines3 import PPO, A2C
from stable_baselines3.dqn import DQN
import os
import time

# Create environment
env = optic_disc(N=100)

# Load the trained agent
model_path = f"Training/Models/half_step_04-17-2023-16-44-17/1300000"
# model=PPO.load(model_path, env=env)
model=DQN.load(model_path, env=env)

episodes = 100
idxs = []

for ep in range(episodes):
    obs=env.reset()
    idxs.append(env.idx)
    done=False
    xs = []
    ys = []
    while not done:
        action, _= model.predict(obs)
        obs, reward, done, info = env.step(action)
        x=env.x
        y=env.y
        world = env.world
        xs.append(x)
        ys.append(y)
        
        print(info)

    with open(f"Testing/moves_{ep}.txt", "w") as file:
        # Write the first list to the file
        for item in xs:
            file.write(str(item) + " ")
        file.write("\n")
        
        # Write the second list to the file
        for item in ys:
            file.write(str(item) + " ")
        file.write("\n")

with open('./Testing/idx.txt', 'w') as f:
    for num in idxs:
        f.write(str(num) + '\n')