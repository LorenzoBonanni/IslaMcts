import collections
from copy import copy

import gym

from action_selection_functions import ucb1, discrete_default_policy
from mcts import Mcts
from mcts_hashstate import MctsHashState

real_env = env = gym.make(
    "LunarLander-v2"
)
observation = real_env.reset()
print(observation)
done = False
last_state = None
real_env.render()
rollout_fn = discrete_default_policy(4)
hashable = True

# check if obs is hashable
if not isinstance(observation, collections.Hashable):
    hashable = False

while not done:
    last_state = real_env.unwrapped.state
    if hashable:
        agent = Mcts(
            C=2,
            n_sim=50,
            root_data=observation,
            env=copy(real_env.unwrapped),
            action_selection_fn=ucb1,
            max_depth=100,
            gamma=0.9,
            rollout_selection_fn=rollout_fn,
            state_variable="s",
        )
    else:
        agent = MctsHashState(
            C=2,
            n_sim=1,
            root_data=observation,
            env=copy(real_env.unwrapped),
            action_selection_fn=ucb1,
            max_depth=100,
            gamma=0.6,
            rollout_selection_fn=rollout_fn,
            state_variable="s",
        )
    action = agent.fit()
    # agent.visualize()
    observation, reward, done, _ = real_env.step(action)
    real_env.state = real_env.unwrapped.state
    print(f"S: {last_state} A: {action}, S': {real_env.unwrapped.state}, R: {reward}")
    print()
    real_env.render()
    # break
real_env.close()
