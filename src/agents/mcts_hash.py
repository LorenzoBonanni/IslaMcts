import numpy as np
from gym import Env

from mcts import ActionNode, StateNode, Mcts


class MctsHash(Mcts):
    def __init__(self, C: float, n_sim: int, root_data, env: Env, action_selection_fn, max_depth: int, gamma: float,
                 rollout_selection_fn, state_variable):
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
        super().__init__(C, n_sim, root_data, env, action_selection_fn, max_depth, gamma, rollout_selection_fn,
                         state_variable)

        self.root = StateNodeHash(root_data, env, C, self.action_selection_fn, gamma,
                                  rollout_selection_fn, state_variable)


class StateNodeHash(StateNode):
    def __init__(self, data, env: Env, C: float, action_selection_fn, gamma: float, rollout_selection_fn,
                 state_variable):
        """
        :param C: exploration-exploitation factor
        :param data: data of the node
        :param env: simulation environment
        :param action_selection_fn: the function to select actions
        :param gamma: discount factor
        """
        super().__init__(data, env, C, action_selection_fn, gamma, rollout_selection_fn, state_variable)

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
            child = ActionNodeHash(action, self.env, self.C, self.action_selection_fn, self.gamma,
                                   self.rollout_selection_fn, self.state_variable)
            self.actions[action] = child
        else:
            action = self.action_selection_fn(self.total, self.C, self.visit_actions, self.ns)
            child = self.actions.get(action)
        reward = child.build_tree(max_depth)
        self.ns += 1
        self.visit_actions[action] += 1
        self.total += self.gamma * reward
        return reward


class ActionNodeHash(ActionNode):
    def __init__(self, data, env, C, action_selection_fn, gamma, rollout_selection_fn, state_variable):
        """
        :param C: exploration-exploitation factor
        :param data: data of the node
        :param env: simulation environment
        :param action_selection_fn: the function to select actions
        :param gamma: discount factor
        """
        super().__init__(data, env, C, action_selection_fn, gamma, rollout_selection_fn, state_variable)

    def build_tree(self, max_depth) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        observation, instant_reward, terminal, _ = self.env.step(self.data)

        # if the node is terminal back-propagate instant reward
        if terminal:
            state = self.children.get(observation.tobytes(), None)
            # add terminal states for visualization
            if state is None:
                # add child node
                state = StateNodeHash(observation, self.env, self.C, self.action_selection_fn, self.gamma,
                                      self.rollout_selection_fn, self.state_variable)
                state.terminal = True
                self.children[observation.tobytes()] = state
            # ORIGINAL
            self.total += instant_reward
            self.na += 1
            # MODIFIED
            state.ns += 1
            return instant_reward
        else:
            # check if the node has been already visited
            state = self.children.get(observation.tobytes(), None)
            if state is None:
                # add child node
                state = StateNode(observation, self.env, self.C, self.action_selection_fn, self.gamma,
                                  self.rollout_selection_fn, self.state_variable)
                self.children[observation.tobytes()] = state
                # ROLLOUT
                delayed_reward = self.gamma * state.rollout(max_depth)

                # BACK-PROPAGATION
                self.na += 1
                state.ns += 1
                self.total += (instant_reward + delayed_reward)
                state.total += (instant_reward + delayed_reward)
                return instant_reward + delayed_reward
            else:
                # go deeper the tree
                delayed_reward = self.gamma * state.build_tree(max_depth)

                # # BACK-PROPAGATION
                self.total += (instant_reward + delayed_reward)
                self.na += 1
                return instant_reward + delayed_reward
