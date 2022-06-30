import gc
from collections import OrderedDict
from copy import copy, deepcopy

import numpy as np
import random
from gym import Env


class Mcts:
    def __init__(self, C: float, n_sim: int, root_data, env: Env, action_selection_fn, max_depth: int, gamma: float):
        """
        :param C: exploration-exploitation factor
        :param n_sim: number of simulations from root node
        :param root_data: data of the root node
        :param env: simulation environment
        :param action_selection_fn: the function to select actions
        :param max_depth: max depth during rollout
        :param gamma: discount factor
        """
        self.q_values = None
        self.C = C
        self.n_sim = n_sim
        self.max_depth = max_depth
        self.env = env
        self.action_selection_fn = action_selection_fn
        self.root_data = root_data
        self.root = StateNode(root_data, env, C, self.action_selection_fn, gamma)

    def fit(self) -> int:
        """
        Starting method, builds the tree and then gives back the best action
        :return: the best action
        """
        for s in range(self.n_sim):
            self.env.s = self.root_data
            self.root.build_tree(self.max_depth)

        # SELECT USING Q-VALUE
        # Q-VAL total / n_visit
        self.root.actions = OrderedDict(sorted(self.root.actions.items()))

        vals = np.array([node.total for node in self.root.actions.values()])
        n_visit = np.array([node.na for node in self.root.actions.values()])
        q_val = vals / n_visit
        self.q_values = q_val
        return np.random.choice(np.flatnonzero(q_val == q_val.max()))


class StateNode:
    def __init__(self, data, env: Env, C: float, action_selection_fn, gamma: float):
        """
        :param C: exploration-exploitation factor
        :param data: data of the node
        :param env: simulation environment
        :param action_selection_fn: the function to select actions
        :param gamma: discount factor
        """
        self.gamma = gamma
        self.data = data
        self.env = env
        # total reward
        self.total = 0
        # number of visits
        self.ns = 0
        # number of visits for each child action
        self.visit_actions = np.zeros(env.action_space.n)
        # dictionary containing mapping between action number and corresponding action node
        self.actions = {}
        self.C = C
        self.action_selection_fn = action_selection_fn

    def rollout(self, max_depth) -> float:
        """
        Random play out until max depth or a terminal state is reached
        :param max_depth: max depth of simulation
        :return: reward obtained from the state
        """
        # curr_env = copy(self.env)
        curr_env = self.env
        done = False
        reward = 0
        depth = 0
        while not done:
            # sampled_action = curr_env.action_space.sample()
            # TODO make prettier
            # random action
            indices = list(range(curr_env.action_space.n))
            probs = [1 / len(indices)] * len(indices)
            sampled_action = random.choices(population=indices, weights=probs)[0]

            # execute action
            obs, reward, done, _ = curr_env.step(sampled_action)
            depth += 1
        return reward

    def build_tree(self, max_depth):
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        # curr_env = copy(self.env)
        # selection
        # avoid bias
        if 0 in self.visit_actions:
            indices = np.where(self.visit_actions == 0)[0]
            # TODO may introduce prior knowledge here
            probs = [1 / len(indices)] * len(indices)
            action = random.choices(population=indices, weights=probs)[0]
        else:
            action = self.action_selection_fn(self.total, self.C, self.visit_actions, self.ns)

        child = self.actions.get(action, None)
        if child is None:
            child = ActionNode(action, self.env, self.C, self.action_selection_fn, self.gamma)
            self.actions[action] = child
        # rollout + backpropagation
        reward = child.build_tree(max_depth)
        self.ns += 1
        self.visit_actions[action] += 1
        self.total += reward
        return reward


class ActionNode:
    def __init__(self, data, env, C, action_selection_fn, gamma):
        """
        :param C: exploration-exploitation factor
        :param data: data of the node
        :param env: simulation environment
        :param action_selection_fn: the function to select actions
        :param gamma: discount factor
        """
        self.gamma = gamma
        self.data = data
        self.env = env
        self.total = 0
        self.na = 0
        self.C = C
        self.children = {}
        self.action_selection_fn = action_selection_fn

    def build_tree(self, max_depth) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        observation, instant_reward, terminal, _ = self.env.step(self.data)

        state = self.children.get(observation, None)

        if terminal:
            self.total += instant_reward
            self.na += 1
            return instant_reward
        else:
            if state is None:
                state = StateNode(observation, self.env, self.C, self.action_selection_fn, self.gamma)
                self.children[observation] = state
                # rollout
                delayed_reward = self.gamma * state.rollout(max_depth)

                # backpropagation
                self.total += (instant_reward + delayed_reward)
                self.na += 1
                state.ns += 1
                state.total += (instant_reward + delayed_reward)
                return instant_reward + delayed_reward
            else:
                # go deeper the tree
                reward = state.build_tree(max_depth)
                return reward
