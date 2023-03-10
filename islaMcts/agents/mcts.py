from typing import Any

import numpy as np

from islaMcts.agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
import islaMcts.utils.mcts_utils as utils


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
        initial_env = utils.my_deepcopy(self.param.env, self.param.env)
        for s in range(self.param.n_sim):
            self.param.env = utils.my_deepcopy(initial_env, self.param.env)
            self.root.build_tree_state(self.param.max_depth)

        # compute q_values
        self.q_values = np.array([node.q_value for node in self.root.actions.values()])

        # return the action with maximum q_value
        max_q = max(self.q_values)

        # get the children which has the maximum q_value
        max_children = list(filter(lambda c: c.q_value == max_q, list(self.root.actions.values())))
        policy: ActionNode = np.random.choice(max_children)
        return policy.data


class StateNode(AbstractStateNode):
    def __init__(self, data: Any, param: MctsParameters):
        super().__init__(data, param)
        self.visit_actions = np.zeros(param.n_actions)

    def build_tree_state(self, max_depth: int):
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
        reward = child.build_tree_action(max_depth)
        self.ns += 1
        self.visit_actions[action] += 1
        self.total += reward
        return reward


class ActionNode(AbstractActionNode):

    def build_tree_action(self, max_depth: int) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that

        :param max_depth:  max depth of simulation
        :return:
        """
        vals = self.param.env.step(self.data)
        observation = vals[0]
        instant_reward = vals[1]
        terminal = vals[2]

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
                delayed_reward = self.param.gamma * state.build_tree_state(max_depth)

                # BACK-PROPAGATION
                self.total += (instant_reward + delayed_reward)
                self.na += 1
                return instant_reward + delayed_reward
