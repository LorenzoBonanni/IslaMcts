import time
from copy import copy, deepcopy

import gym
from action_selection_functions import ucb1
from mcts import Mcts

action_string = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}
real_env = gym.make("FrozenLake-v1", is_slippery=False)
# real_env = gym.make("Taxi-v3")
observation = real_env.reset()
done = False
last_state = None
real_env.s = real_env.unwrapped.s
real_env.render()

# while not done:
# TODO check state evolution
while not done:
    last_state = real_env.unwrapped.s
    agent = Mcts(
        C=90,
        n_sim=100,
        root_data=observation,
        env=copy(real_env.unwrapped),
        action_selection_fn=ucb1,
        max_depth=1000,
        gamma=0.2
    )
    action = agent.fit()
    print(agent.q_values)
    observation, reward, done, _ = real_env.step(action)
    real_env.s = real_env.unwrapped.s
    print(f"S: {last_state} A: {action_string[action]}, S': {real_env.unwrapped.s}, R: {reward}")
    print()
    real_env.render()
    time.sleep(0.5)
# real_env.close()
