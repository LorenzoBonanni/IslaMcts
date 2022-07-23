import random
from collections import OrderedDict
from typing import Any

import numpy as np

from islaMcts.agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
from islaMcts.agents.parameters.pw_parameters import PwParameters


class MctsStateProgressiveWideningHash(AbstractMcts):
    def __init__(self, param: PwParameters):
        super().__init__(param)
        self.root = StateNodeProgressiveWideningHash(data=self.param.root_data, param=param)

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
        return np.random.choice(np.flatnonzero(q_val == q_val.max()))


class StateNodeProgressiveWideningHash(AbstractStateNode):

    def __init__(self, data: Any, param: MctsParameters):
        super().__init__(data, param)
        self.visit_actions = np.zeros(param.n_actions)

    def build_tree(self, max_depth):
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        # SELECTION
        # to avoid biases if there are unvisited actions we sample randomly from them
        if 0 in self.visit_actions:
            # random action
            action = np.random.choice(np.flatnonzero(self.visit_actions == 0))
            child = ActionNodeProgressiveWideningHash(data=action, param=self.param)
            self.actions[action] = child
        else:
            action = self.param.action_selection_fn(self)
            child = self.actions.get(action)
        reward = child.build_tree(max_depth)
        self.ns += 1
        self.visit_actions[action] += 1
        self.total += self.param.gamma * reward
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
            # add terminal states for visualization
            # add child node
            state = StateNodeProgressiveWideningHash(
                data=observation,
                param=self.param
            )
            state.terminal = True
            self.children[obs_bytes] = state

            self.total += instant_reward
            self.na += 1
            state.ns += 1
            return instant_reward

        if len(self.children) == 0 or len(self.children) <= self.param.k * (self.na ** self.param.alpha):
            # EXPAND
            # add child node
            state = StateNodeProgressiveWideningHash(
                data=observation,
                param=self.param
            )
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
            # SAMPLE FROM VISITED STATES
            # filter out terminal States
            children_visits = []
            children_keys = []
            for k, c in self.children.items():
                if not c.terminal:
                    children_visits.append(c.ns)
                    children_keys.append(k)

                key = random.choices(
                    population=list(children_keys),
                    weights=self.na / np.array(children_visits)
                )[0]
                state = self.children[key]
                self.param.env.__dict__[self.param.state_variable] = self.param.env.unwrapped.__dict__[self.param.state_variable] = state.data
                # go deeper the tree
                delayed_reward = self.param.gamma * state.build_tree(max_depth)
                return instant_reward + delayed_reward
