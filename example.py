from copy import copy

import gym

from action_selection_functions import ucb1
from mcts import Mcts

real_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
observation, info = real_env.reset(seed=42, return_info=True)
sim_env = copy(real_env)
done = False

while not done:
    agent = Mcts(2, 0.7, 200, observation, sim_env, ucb1, 15)
    action = agent.fit()
    observation, reward, done, info = real_env.step(action)
    real_env.render()

real_env.close()