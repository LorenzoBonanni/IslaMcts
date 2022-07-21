from collections import OrderedDict
from typing import Any

import numpy as np

from src.agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode
from src.agents.mcts_parameters import MctsParameters


class Mcts(AbstractMcts):
    def __init__(self, param: MctsParameters):
        super().__init__(param)
        self.root = StateNode(
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
        return np.random.choice(np.flatnonzero(q_val == q_val.max()))


class StateNode(AbstractStateNode):
    def __init__(self, data: Any, param: MctsParameters):
        super().__init__(data, param)
        self.visit_actions = np.zeros(param.n_actions)

    def rollout(self, max_depth: int) -> float:
        """
        Play out until max depth or a terminal state is reached

        :param max_depth: max depth of simulation
        :return: reward obtained from the state
        """
        curr_env = self.param.env
        done = False
        reward = 0
        depth = 0
        while not done and depth < max_depth:
            sampled_action = self.param.rollout_selection_fn(state=curr_env.__dict__[self.param.state_variable])

            # execute action
            obs, reward, done, _ = curr_env.step(sampled_action)
            depth += 1
        return reward

    def build_tree(self, max_depth: int):
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
            child = ActionNode(data=action, param=self.param)
            self.actions[action] = child
        else:
            action = self.param.action_selection_fn(self)
            child = self.actions.get(action)
        reward = child.build_tree(max_depth)
        self.ns += 1
        self.visit_actions[action] += 1
        self.total += self.param.gamma * reward
        return reward


class ActionNode(AbstractActionNode):

    def build_tree(self, max_depth: int) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that

        :param max_depth:  max depth of simulation
        :return:
        """
        observation, instant_reward, terminal, _ = self.param.env.step(self.data)

        # if the node is terminal back-propagate instant reward
        if terminal:
            state = self.children.get(observation, None)
            # add terminal states for visualization
            if state is None:
                # add child node
                state = StateNode(data=observation, param=self.param)
                state.terminal = True
                self.children[observation] = state
            self.total += instant_reward
            self.na += 1
            state.ns += 1
            return instant_reward
        else:
            # check if the node has been already visited
            state = self.children.get(observation, None)
            if state is None:
                # add child node
                state = StateNode(data=observation, param=self.param)
                self.children[observation] = state
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
