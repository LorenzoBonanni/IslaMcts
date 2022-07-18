import math
import random
from collections import OrderedDict

import numpy as np
from gym import Env

from src.agents.mcts_hash import MctsHash, StateNodeHash, ActionNodeHash


class MctsDoubleProgressiveWideningHash(MctsHash):
    def __init__(self, C: float, n_sim: int, root_data, env: Env, action_selection_fn, max_depth: int, gamma: float,
                 rollout_selection_fn, state_variable, alpha1: float, k1: float, alpha2: float, k2: float):
        """
        :param C: exploration-exploitation factor
        :param n_sim: number of simulations from root node
        :param root_data: data of the root node
        :param env: simulation environment
        :param action_selection_fn: the function to select actions
        :param max_depth: max depth during rollout
        :param gamma: discount factor
        :param state_variable: the name of the variable containing state information inside the environment
        """
        super().__init__(
            root_data=root_data,
            env=env,
            n_sim=n_sim,
            C=C,
            action_selection_fn=action_selection_fn,
            gamma=gamma,
            rollout_selection_fn=rollout_selection_fn,
            state_variable=state_variable,
            max_depth=max_depth
        )

        self.root = StateNodeDoubleProgressiveWideningHash(root_data, env, C, self.action_selection_fn, gamma,
                                                           rollout_selection_fn, state_variable, alpha1, k1, alpha2, k2)

        # constants for Action Progressive Widening
        self.alpha1 = alpha1
        self.k1 = k1
        # constants for State Progressive Widening
        self.alpha2 = alpha2
        self.k2 = k2

    def fit(self) -> np.ndarray:
        """
        Starting method, builds the tree and then gives back the best action

        :return: the best action
        """
        for s in range(self.n_sim):
            self.env.__dict__[self.state_variable] = self.env.unwrapped.__dict__[self.state_variable] = self.root_data
            self.root.build_tree(self.max_depth)

        # order actions dictionary so that action indices correspond to the action number
        self.root.actions = OrderedDict(sorted(self.root.actions.items()))

        # compute q_values
        vals = np.array([node.total for node in self.root.actions.values()])
        n_visit = np.array([node.na for node in self.root.actions.values()])
        q_val = vals / n_visit
        self.q_values = q_val

        # to avoid biases choose random between the actions with the highest q_value
        index = np.random.choice(np.flatnonzero(q_val == q_val.max()))
        a = list(self.root.actions.keys())[index]
        return np.frombuffer(a, dtype=float)


class StateNodeDoubleProgressiveWideningHash(StateNodeHash):
    def __init__(self, data, env: Env, C: float, action_selection_fn, gamma: float, rollout_selection_fn,
                 state_variable, alpha1: float, k1: float, alpha2: float, k2: float):
        """
        :param C: exploration-exploitation factor
        :param data: data of the node
        :param env: simulation environment
        :param action_selection_fn: the function to select actions
        :param gamma: discount factor
        """
        super().__init__(data, env, C, action_selection_fn, gamma, rollout_selection_fn, state_variable)
        # constants for Action Progressive Widening
        self.alpha1 = alpha1
        self.k1 = k1
        # constants for State Progressive Widening
        self.alpha2 = alpha2
        self.k2 = k2
        self.visit_actions = {}

    def build_tree(self, max_depth):
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        # SELECTION
        # if the number of child actions is less than k1*ns^alpha1 then add a new child
        if len(self.actions) == 0 or len(self.actions) < math.ceil(self.k1 * (self.ns ** self.alpha1)):
            action = self.env.action_space.sample()
            action_bytes = action.tobytes()
            child = ActionNodeDoubleProgressiveWideningHash(action, self.env, self.C, self.action_selection_fn,
                                                            self.gamma,
                                                            self.rollout_selection_fn, self.state_variable, self.alpha1,
                                                            self.k1, self.alpha2, self.k2)
            self.visit_actions[action_bytes] = 0
            self.actions[action_bytes] = child
        else:
            action_index = self.action_selection_fn(self.total, self.C, list(self.visit_actions.values()), self.ns)
            action_bytes = list(self.visit_actions.keys())[action_index]
            child = self.actions[action_bytes]

        # in order to get instant_reward set the state into the environment to the current state
        self.env.__dict__[self.state_variable] = self.env.unwrapped.__dict__[self.state_variable] = self.data
        # ROLLOUT + BACKPROPAGATION
        reward = child.build_tree(max_depth)
        self.ns += 1
        self.visit_actions[action_bytes] += 1
        self.total += reward
        return reward

    def rollout(self, max_depth) -> float:
        """
        Random play out until max depth or a terminal state is reached
        :param max_depth: max depth of simulation
        :return: reward obtained from the state
        """
        curr_env = self.env
        done = False
        reward = 0
        depth = 0
        while not done and depth < max_depth:
            sampled_action = curr_env.action_space.sample()

            # execute action
            obs, reward, done, _ = curr_env.step(sampled_action)
            depth += 1
        return reward


class ActionNodeDoubleProgressiveWideningHash(ActionNodeHash):
    def __init__(self, data, env, C, action_selection_fn, gamma, rollout_selection_fn, state_variable, alpha1: float,
                 k1: float, alpha2: float, k2: float):
        """
        :param C: exploration-exploitation factor
        :param data: data of the node
        :param env: simulation environment
        :param action_selection_fn: the function to select actions
        :param gamma: discount factor
        """
        super().__init__(data, env, C, action_selection_fn, gamma, rollout_selection_fn, state_variable)
        # constants for Action Progressive Widening
        self.alpha1 = alpha1
        self.k1 = k1
        # constants for State Progressive Widening
        self.alpha2 = alpha2
        self.k2 = k2

    def build_tree(self, max_depth) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        observation, instant_reward, terminal, _ = self.env.step(self.data)
        obs_bytes = observation.tobytes()
        if len(self.children) == 0 or len(self.children) <= self.k2 * (self.na ** self.alpha2):
            # EXPAND
            # if the node is terminal back-propagate instant reward
            if terminal:
                # add terminal states for visualization
                # add child node
                state = StateNodeDoubleProgressiveWideningHash(observation, self.env, self.C,
                                                               self.action_selection_fn,
                                                               self.gamma,
                                                               self.rollout_selection_fn, self.state_variable,
                                                               self.alpha1,
                                                               self.k1, self.alpha2, self.k2)
                state.terminal = True
                self.children[obs_bytes] = state

                self.total += instant_reward
                self.na += 1
                state.ns += 1
                return instant_reward
            else:
                # add child node
                state = StateNodeDoubleProgressiveWideningHash(observation, self.env, self.C,
                                                               self.action_selection_fn,
                                                               self.gamma, self.rollout_selection_fn,
                                                               self.state_variable,
                                                               self.alpha1,
                                                               self.k1, self.alpha2, self.k2)
                self.children[obs_bytes] = state
                # ROLLOUT
                delayed_reward = self.gamma * state.rollout(max_depth)

                # BACK-PROPAGATION
                self.na += 1
                state.ns += 1
                self.total += (instant_reward + delayed_reward)
                state.total += (instant_reward + delayed_reward)
                return instant_reward + delayed_reward
        else:
            # SAMPLE FROM VISITED STATES
            key = random.choices(
                population=list(self.children.keys()),
                weights=self.na / np.array([c.ns for c in list(self.children.values())])
            )[0]
            state = self.children[key]
            self.env.__dict__[self.state_variable] = self.env.unwrapped.__dict__[self.state_variable] = state.data
            # go deeper the tree
            delayed_reward = self.gamma * state.build_tree(max_depth)
            return instant_reward + delayed_reward
