import math
from collections import OrderedDict
from typing import Any

import numpy as np

from src.agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode
from src.agents.pw_parameters import PwParameters


class MctsActionProgressiveWideningHash(AbstractMcts):
    def __init__(self, param: PwParameters):
        super().__init__(param)
        self.root = StateNodeProgressiveWideningHash(
            data=param.root_data,
            param=param
        )

    def fit(self) -> int:
        """
        Starting method, builds the tree and then gives back the best action

        :return: the best action
        """
        for s in range(self.param.n_sim):
            self.param.env.__dict__[self.param.state_variable] = self.param.env.unwrapped.__dict__[
                self.param.state_variable] = self.param.root_data
            self.root.build_tree(self.param.max_depth)

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


class StateNodeProgressiveWideningHash(AbstractStateNode):
    def __init__(self, data: Any, param: PwParameters):
        super().__init__(data, param)
        self.visit_actions = {}

    def build_tree(self, max_depth):
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        # SELECTION
        if len(self.actions) == 0 or len(self.actions) <= math.ceil(self.param.k * (self.ns ** self.param.alpha)):
            action = self.param.env.action_space.sample()
            action_bytes = action.tobytes()
            child = ActionNodeProgressiveWideningHash(data=action, param=self.param)
            self.visit_actions[action_bytes] = 0
            self.actions[action_bytes] = child
        else:
            action_index = self.param.action_selection_fn(self)
            action_bytes = list(self.visit_actions.keys())[action_index]
            child = self.actions[action_bytes]

        # in order to get instant_reward set the state into the environment to the current state
        self.param.env.__dict__[self.param.state_variable] = self.param.env.unwrapped.__dict__[self.param.state_variable] = self.data
        # ROLLOUT + BACKPROPAGATION
        reward = child.build_tree(self.param.max_depth)
        self.ns += 1
        self.visit_actions[action_bytes] += 1
        self.total += reward
        return reward


class ActionNodeProgressiveWideningHash(AbstractActionNode):

    def build_tree(self, max_depth) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        observation, instant_reward, terminal, _ = self.param.env.step(self.data)
        obs_bytes = observation.tobytes()

        # if the node is terminal back-propagate instant reward
        if terminal:
            state = self.children.get(obs_bytes, None)
            # add terminal states for visualization
            if state is None:
                # add child node
                state = StateNodeProgressiveWideningHash(data=observation, param=self.param)
                state.terminal = True
                self.children[obs_bytes] = state
            # ORIGINAL
            self.total += instant_reward
            self.na += 1
            # MODIFIED
            state.ns += 1
            return instant_reward
        else:
            # check if the node has been already visited
            state = self.children.get(obs_bytes, None)
            if state is None:
                # add child node
                state = StateNodeProgressiveWideningHash(data=observation, param=self.param)
                self.children[obs_bytes] = state
                # ROLLOUT
                delayed_reward = self.param.gamma * state.rollout(max_depth)

                # BACK-PROPAGATION
                self.na += 1
                state.ns += 1
                self.total += (instant_reward + delayed_reward)
                state.total += (instant_reward + delayed_reward)
                return instant_reward + delayed_reward
            else:
                # go deeper the tree
                delayed_reward = self.param.gamma * state.build_tree(max_depth)

                # # BACK-PROPAGATION
                self.total += (instant_reward + delayed_reward)
                self.na += 1
                return instant_reward + delayed_reward
