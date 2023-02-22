import pickle
from copy import deepcopy
from pickle import PicklingError

import gymnasium as gym


def my_deepcopy(initial_env: gym.Env, curr_env: gym.Env):
    try:
        new_env = pickle.loads(pickle.dumps(initial_env))

    except PicklingError:
        new_env = deepcopy(initial_env)

    new_env.action_space = curr_env.action_space
    return new_env
