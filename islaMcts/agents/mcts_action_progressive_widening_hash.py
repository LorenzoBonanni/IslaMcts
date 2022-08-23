import logging
import math
from collections import OrderedDict
from typing import Any

import numpy as np

from islaMcts import utils
from islaMcts.agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode
from islaMcts.agents.parameters.pw_parameters import PwParameters


# logger = logging.getLogger(__name__)


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
        initial_env = utils.my_deepcopy(self.param.env)
        for s in range(self.param.n_sim):
            # logger.debug(f"SIM {s}")
            self.param.env = utils.my_deepcopy(initial_env)
            self.root.build_tree(self.param.max_depth)

        # compute q_values
        self.q_values = np.array([node.q_value for node in self.root.actions.values()])

        # return the action with maximum q_value
        max_q = max(self.q_values)
        # get the children which has the maximum q_value
        max_children = list(filter(lambda c: c.q_value == max_q, list(self.root.actions.values())))
        policy: ActionNodeProgressiveWideningHash = np.random.choice(max_children)
        return policy.data


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
            action = self.param.action_expansion_function(self)
            # logger.debug(f"{action}\t-\tRandom")
            action_bytes = action.tobytes()
            child = self.actions.get(action_bytes, None)
            if child is None:
                child = ActionNodeProgressiveWideningHash(data=action, param=self.param)
                self.visit_actions[action_bytes] = 0
                self.actions[action_bytes] = child
        else:
            action = self.param.action_selection_fn(self)
            action_bytes = action.tobytes()
            child = self.actions[action_bytes]

            # ucb_value = (child.total / child.na) + self.param.C* np.sqrt(np.log(self.ns) / child.na)
            # logger.debug(f"{child.data}\t{ucb_value}\tUCB")

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
