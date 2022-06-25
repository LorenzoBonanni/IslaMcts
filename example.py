import time
from copy import copy

import gym

from action_selection_functions import ucb1
from mcts import Mcts

real_env = gym.make("FrozenLake-v1")
observation = real_env.reset()
real_env.render()
sim_env = copy(real_env)
done = False
last_state = None

while not done:
    last_state = real_env.s
    agent = Mcts(2, 2000, observation, sim_env, ucb1, 500, 0.3)
    action = agent.fit()
    observation, reward, done, _ = real_env.step(action)
    print(f"S: {last_state} A: {action}, S': {real_env.s}, R: {reward}")
    time.sleep(0.5)
    real_env.render()

time.sleep(2)
real_env.close()
