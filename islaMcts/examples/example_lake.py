import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from islaMcts.utils.action_selection_functions import ucb1, continuous_default_policy, grid_policy
from islaMcts.agents.mcts import Mcts
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
from islaMcts.utils.mcts_utils import my_deepcopy

action_string = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}


# key = current_row * nrows + current_col
# distance =  manhattan distance
# np.abs(target_c-c)+np.abs(target_r-r)
def next_state_distance(r, c, a):
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
        new_r = r + 1
    # 2: RIGHT
    elif a == 2:
        new_c = c + 1
        new_r = r
    # 3: UP
    elif a == 3:
        new_c = c
        new_r = r - 1

    if new_c < 0 or new_c > 3 or new_r < 0 or new_r > 3:
        new_c = c
        new_r = r

    # return new_r*4+new_c
    return np.abs(target_c - new_c) + np.abs(target_r - new_r)


n_actions = 4
# heuristic dictionary
distances = {
    r * 4 + c: [
        next_state_distance(r, c, a)
        for a in range(n_actions)
    ]
    for c in range(4)
    for r in range(4)
}

rollout_fn = grid_policy(distances, n_actions)
# rollout_fn = continuous_default_policy

times = []
rewards = []

for _ in tqdm(range(100)):
    real_env = gym.make("FrozenLake-v1", is_slippery=False).unwrapped
    observation = real_env.reset()
    parameters = MctsParameters(
                C=0.5,
                n_sim=100,
                root_data=observation,
                env=real_env,
                action_selection_fn=ucb1,
                max_depth=1000,
                gamma=1,
                rollout_selection_fn=rollout_fn,
                n_actions=real_env.action_space.n,
                x_values=[],
                y_values=[]
            )
    done = False
    start_time = time.time()
    while not done:
        parameters.env = real_env
        agent = Mcts(
            param=parameters
        )
        action = agent.fit()
        observation, reward, done, _, _ = real_env.step(action)
        rewards.append(reward)
        parameters.root_data = observation
    times.append(time.time() - start_time)
print(f"TIME: {np.mean(times)}")
print(f"N TERMINAL {np.sum(rewards)}")
# real_env.close()
