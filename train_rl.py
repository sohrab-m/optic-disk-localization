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

now = datetime.now()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Training on GPUs: {torch.cuda.device_count()} {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_name(1)}")
else:
    device = torch.device("cpu")
    print("Training on CPU")
env = optic_disc(device=device, N=500)


# It will check your custom environment and output additional warnings if needed
# check_env(env)
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
print(date_time)
models_dir = f"Training/Models/A2C_{date_time}/"
logdir = f"Training/Logs/A2C_{date_time}/"


if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env.reset()
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=logdir, device=device, learning_rate=0.007, ent_coef=0.1) # , buffer_size=400000,learning_starts=20000 , exploration_fraction=0.95,

model = DataParallel(model)

TIMESTEPS = 100000
iters = 0

while iters<50:
    iters += 1
    print('iteration: ', iters)
    model.module.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.module.save(f"{models_dir}/{TIMESTEPS*iters}")