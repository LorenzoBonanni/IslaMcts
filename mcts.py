from copy import copy

import numpy as np
from gym import Env


class Mcts:
    def __init__(self, C: float, n_sim: int, root_data, env: Env, action_selection_fn, max_depth: int):
        self.C = C
        self.n_sim = n_sim
        self.max_depth = max_depth
        self.env = copy(env)
        self.action_selection_fn = action_selection_fn
        self.root = StateNode(root_data, env, C, self.action_selection_fn)

    def fit(self):
        for s in range(self.n_sim):
            self.root.build_tree(self.max_depth)

        return self.action_selection_fn(self.root.total, self.C, self.root.visit_actions, self.root.n)


class StateNode:
    def __init__(self, data, env, C, action_selection_fn):
        self.data = data
        self.env = copy(env)
        self.total = 0
        self.n = 0
        self.visit_actions = np.zeros(env.action_space.n)
        self.actions = {}
        self.C = C
        self.action_selection_fn = action_selection_fn

    def rollout(self, max_depth) -> float:
        curr_env = copy(self.env)
        done = False
        reward = 0
        depth = 0
        while not done and depth < max_depth:
            # print(f"State: {curr_env.s}")
            sampled_action = curr_env.action_space.sample()
            obs, reward, done, _ = curr_env.step(sampled_action)
            # print(f"action:{sampled_action}, s':{curr_env.s}")
            depth += 1
        return reward

    def build_tree(self, max_depth):
        curr_env = copy(self.env)
        # selection
        action = self.action_selection_fn(self.total, self.C, self.visit_actions, self.n)
        child = self.actions.get(action, None)
        if child is None:
            child = ActionNode(action, curr_env, self.C, self.action_selection_fn)
            self.actions[action] = child
        # rollout + backpropagation
        reward = child.build_tree(max_depth)
        self.n += 1
        self.visit_actions[action] += 1
        return reward


class ActionNode:
    def __init__(self, data, env, C, action_selection_fn):
        self.data = data
        self.env = copy(env)
        self.total = 0
        self.n = 0
        self.C = C
        self.children = {}
        self.action_selection_fn = action_selection_fn

    def build_tree(self, max_depth) -> float:
        observation, reward, terminal, _ = self.env.step(self.data)
        state = self.children.get(observation, None)
        if state is None:
            state = StateNode(observation, self.env, self.C, self.action_selection_fn)
            self.children[observation] = state
            # rollout
            reward = state.rollout(max_depth)
        else:
            # go deeper the tree
            if terminal:
                self.total += reward
                self.n += 1
                return reward
            else:
                reward = state.build_tree(max_depth)

        # backpropagation
        self.total += reward
        self.n += 1
        state.n += 1
        state.total += reward
        return reward
