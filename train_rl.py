from stable_baselines3.common.env_checker import check_env
from rl_mover import optic_disc

env = optic_disc()

# It will check your custom environment and output additional warnings if needed
check_env(env)