from stable_baselines3.common.env_checker import check_env
from rl_mover import optic_disc
from stable_baselines3 import PPO, A2C
from stable_baselines3.dqn import DQN
import os
import time
from datetime import datetime
import torch
import scipy.io as sio

from torch.nn.parallel import DataParallel

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Training on GPUs: {torch.cuda.device_count()} {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_name(1)}")
else:
    device = torch.device("cpu")
    print("Training on CPU")
env = optic_disc(device=device)


# load model
models_dir = "./Training/Models/500img_04-17-2023-18-30-35"
model = DQN.load(f"{models_dir}/100000.zip", env=env, device=device, batch_size=64)

print(model)
# Wrap the model in DataParallel
model = DataParallel(model)


TIMESTEPS = 100000
iters = 0

while iters<50:
    iters += 1
    print('iteration: ', iters)
    model.module.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.module.save(f"{models_dir}/{TIMESTEPS*iters}")
