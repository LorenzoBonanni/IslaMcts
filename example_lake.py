import time
from copy import copy

import gym
import numpy as np

from action_selection_functions import ucb1, grid_policy, discrete_default_policy
from mcts import Mcts

action_string = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}


# key = current_row * nrows + current_col
# distance =  manhattan distance
# np.abs(target_c-c)+np.abs(target_r-r)
def next_state_distance(r, c ,a):
    target_c = 3
    target_r = 3
    new_c = 0
    new_r = 0

    # 0: LEFT
    if a == 0:
        new_c = c - 1
        new_r = r
    # 1: DOWN
    elif a == 1:
        new_c = c
        new_r = r+1
    # 2: RIGHT
    elif a == 2:
        new_c = c+1
        new_r = r
    # 3: UP
    elif a == 3:
        new_c = c
        new_r = r-1

    if new_c < 0 or new_c>3 or new_r<0 or new_r>3:
        new_c = c
        new_r = r

    # return new_r*4+new_c
    return np.abs(target_c-new_c)+np.abs(target_r-new_r)


n_actions = 4
# heuristic dictionary
distances = {
    r*4+c: [
        next_state_distance(r, c, a)
        for a in range(n_actions)
    ]
    for c in range(4)
    for r in range(4)
}

rollout_fn = grid_policy(distances, n_actions)

real_env = gym.make("FrozenLake-v1", is_slippery=False)
observation = real_env.reset()
done = False
last_state = None
real_env.s = real_env.unwrapped.s
real_env.render()


while not done:
    last_state = real_env.unwrapped.s
    agent = Mcts(
        C=90,
        n_sim=1000,
        root_data=observation,
        env=copy(real_env.unwrapped),
        action_selection_fn=ucb1,
        max_depth=10000,
        gamma=0.2,
        rollout_selection_fn=rollout_fn
    )
    action = agent.fit()
    # agent.visualize()
    print(agent.q_values)
    observation, reward, done, _ = real_env.step(action)
    real_env.s = real_env.unwrapped.s
    print(f"S: {last_state} A: {action_string[action]}, S': {real_env.unwrapped.s}, R: {reward}")
    print()
    real_env.render()
    # break
# real_env.close()
