from stable_baselines3.common.env_checker import check_env
from rl_mover import optic_disc
from stable_baselines3 import PPO, A2C
from stable_baselines3.dqn import DQN
import os
import time
from datetime import datetime

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

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logdir)
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir, buffer_size=1000, exploration_fraction=0.9)

TIMESTEPS = 50000
iters = 0

while iters<10:
    iters += 1
    print('iteration: ', iters)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
