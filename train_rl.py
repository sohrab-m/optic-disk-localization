from stable_baselines3.common.env_checker import check_env
from rl_mover import optic_disc
from stable_baselines3 import PPO, A2C
from stable_baselines3.dqn import DQN
from stable_baselines3.common.policies import CnnPolicy
import os
import time
from datetime import datetime
import torch

env = optic_disc()
now = datetime.now()

# It will check your custom environment and output additional warnings if needed
# check_env(env)
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
print(date_time)
models_dir = f"Training/Models/{date_time}/"
logdir = f"Training/Logs/{date_time}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env.reset()

# Set hyperparameters
buffer_size = 20
exploration_fraction = 0.95
learning_starts = 2500
gamma = 1

pretrained_network = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
policy = CnnPolicy(env.observation_space, env.action_space, feature_extractor=pretrained_network)

# Create model with specified hyperparameters
model = DQN(policy, env, verbose=1, tensorboard_log=logdir, buffer_size=buffer_size, exploration_fraction=exploration_fraction, learning_starts=learning_starts, gamma=gamma)

TIMESTEPS = 50000
iters = 0

while iters<10:
    iters += 1
    print('iteration: ', iters)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")