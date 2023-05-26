from stable_baselines3.common.env_checker import check_env
from rl_mover import optic_disc
from stable_baselines3 import PPO, A2C
from stable_baselines3.dqn import DQN
import os
import time
from datetime import datetime
import torch
import scipy.io as sio


env = optic_disc()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Training on CPU")

# load model
models_dir = "./Training/Models/04-10-2023-19-42-48"
model = DQN.load(f"{models_dir}/2000000.zip", env=env, device=device)
print(f"Loaded model {model} successfully.")




# Set hyperparameters
buffer_size = 20
exploration_fraction = 0.95
learning_starts = 2500
gamma = 1

TIMESTEPS = 100000
iters = 0

while iters<50:
    iters += 1
    print('iteration: ', iters)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

