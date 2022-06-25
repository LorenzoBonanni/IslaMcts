from copy import copy

import numpy as np
from gym import Env

from action_selection_functions import ucb1


class Mcts:
    def __init__(self, C: float, gamma: float, n_sim: int, root_data, env: Env, action_selection_fn, max_depth: int):
        self.C = C
        self.gamma = gamma
        self.n_sim = n_sim
        self.max_depth = max_depth
        self.env = env
        self.action_selection_fn = action_selection_fn
        self.root = StateNode(root_data, env, C, gamma)

    def fit(self):
        action = 0
        current = self.root
        depth = 0
        child = 1
        for s in range(self.n_sim):
            curr_env = copy(self.env)
            # selection
            while child:
                action = self.action_selection_fn(current.total, self.C, current.visit_actions, current.n)
                child = current.actions.get(action, None)
            observation, _, _, _ = curr_env.step(action)
            state = StateNode(observation, curr_env, self.C, self.gamma)
            # expansion
            current.actions[action] = {state}
            # rollout
            reward = state.rollout(self.max_depth)
            current.total += reward
            current.n += 1
            current.visit_actions[action] += 1
            # TODO backpropagate
            # reset child to a value
            child = 1


class StateNode:
    def __init__(self, data, env, C, gamma):
        self.data = data
        self.env = env
        self.total = 0
        self.n = 0
        self.visit_actions = np.zeros(env.action_space.n)
        self.actions = {}
        self.C = C
        self.gamma = gamma

    def rollout(self, max_depth) -> float:
        environment = copy(self.env)
        done = False
        reward = 0
        depth = 0
        while not done or depth < max_depth:
            _, reward, done, _ = environment.step(environment.action_space.sample())
            depth += 1
        return reward

    def expand_action(self):
        pass

    def tree_traversal(self):
        action = ucb1(self.total, self.C, self.visit_actions, self.n)
        observation, reward, done, info = self.env.step(action)
