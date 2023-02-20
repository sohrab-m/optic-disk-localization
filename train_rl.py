from stable_baselines3.common.env_checker import check_env
from rl_mover import optic_disc
from stable_baselines3 import PPO, A2C
import os
import time

env = optic_disc()

# It will check your custom environment and output additional warnings if needed
check_env(env)

models_dir = f"Training/Models/{int(time.time())}/"
logdir = f"Training/Logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env.reset()

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000000
iters = 0

while True:
    iters += 1
    print('iteration: ', iters)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
