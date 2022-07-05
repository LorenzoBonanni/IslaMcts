import subprocess
from collections import OrderedDict

import graphviz
import numpy as np
from gym import Env


class Mcts:
    def __init__(self, C: float, n_sim: int, root_data, env: Env, action_selection_fn, max_depth: int, gamma: float,
                 rollout_selection_fn):
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
        self.rollout_selection_fn = rollout_selection_fn
        self.root_data = root_data
        self.root = StateNode(root_data, env, C, self.action_selection_fn, gamma, rollout_selection_fn)

    def fit(self) -> int:
        """
        Starting method, builds the tree and then gives back the best action
        :return: the best action
        """
        for s in range(self.n_sim):
            self.env.s = self.root_data
            self.root.build_tree(self.max_depth)

        # order actions dictionary so that action indices correspond to the action number
        self.root.actions = OrderedDict(sorted(self.root.actions.items()))

        # compute q_values
        vals = np.array([node.total for node in self.root.actions.values()])
        n_visit = np.array([node.na for node in self.root.actions.values()])
        q_val = vals / n_visit
        self.q_values = q_val

        # to avoid biases choose random between the actions with the highest q_value
        return np.random.choice(np.flatnonzero(q_val == q_val.max()))

    def visualize(self, extension: str = '0'):
        """
        creates a visualization of the tree
        :param extension: extension to the file name
        :return:
        """
        filename = f'mcts_{extension}'
        g = graphviz.Digraph('g', filename=f'{filename}.gv', directory='output')
        n = 0
        self.root.visualize(n, None, g)

        # save gv file
        g.save()
        # render gv file to an svg
        with open(f'output/{filename}.svg', 'w') as f:
            subprocess.Popen(['dot', '-Tsvg', f'output/{filename}.gv'], stdout=f)


class StateNode:
    def __init__(self, data, env: Env, C: float, action_selection_fn, gamma: float, rollout_selection_fn):
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
        # number of visits of the node
        self.ns = 0
        # number of visits for each child action
        self.visit_actions = np.zeros(env.action_space.n)
        # dictionary containing mapping between action number and corresponding action node
        self.actions = {}
        self.C = C
        self.action_selection_fn = action_selection_fn
        self.rollout_selection_fn = rollout_selection_fn

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
            sampled_action = self.rollout_selection_fn(state=curr_env.s)

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
        # SELECTION
        # to avoid biases if there are unvisited actions we sample randomly from them
        if 0 in self.visit_actions:
            # random action
            action = np.random.choice(np.flatnonzero(self.visit_actions == 0))
        else:
            action = self.action_selection_fn(self.total, self.C, self.visit_actions, self.ns)

        child = self.actions.get(action, None)
        # if child is None create a new ActionNode
        if child is None:
            child = ActionNode(action, self.env, self.C, self.action_selection_fn, self.gamma,
                               self.rollout_selection_fn)
            self.actions[action] = child
        # ROLLOUT + BACKPROPAGATION
        reward = child.build_tree(max_depth)
        self.ns += 1
        self.visit_actions[action] += 1
        self.total += reward
        return reward

    def visualize(self, n, father, g):
        """
        add the current node to the graph and recursively adds child nodes to the graph
        :param n: the last node number
        :param father: the father node name
        :param g: the graph
        :return: updated n
        """
        # add the node its self
        g.attr('node', shape='circle')
        name = f"node{n}"
        g.node(name, f"{self.data}\nn={self.ns}\nV={(self.total/self.ns):.3f}")
        # for root node father is None
        if father is not None:
            g.edge(father, name)
        n += 1
        # add its child nodes
        for action_node in self.actions.values():
            father = name
            n = action_node.visualize(n, father, g)
        # to avoid losing the updated n value every time the function end returns the most updated n value
        return n


class ActionNode:
    def __init__(self, data, env, C, action_selection_fn, gamma, rollout_selection_fn):
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
        self.rollout_selection_fn = rollout_selection_fn

    def build_tree(self, max_depth) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that
        :param max_depth:  max depth of simulation
        :return:
        """
        observation, instant_reward, terminal, _ = self.env.step(self.data)

        # if the node is terminal back-propagate instant reward
        if terminal:
            self.total += instant_reward
            self.na += 1
            return instant_reward
        else:
            # check if the node has been already visited
            state = self.children.get(observation, None)
            if state is None:
                # add child node
                state = StateNode(observation, self.env, self.C, self.action_selection_fn, self.gamma,
                                  self.rollout_selection_fn)
                self.children[observation] = state
                # ROLLOUT
                delayed_reward = self.gamma * state.rollout(max_depth)

                # BACK-PROPAGATION
                self.total += (instant_reward + delayed_reward)
                self.na += 1
                state.ns += 1
                state.total += (instant_reward + delayed_reward)
                return instant_reward + delayed_reward
            else:
                # go deeper the tree
                reward = state.build_tree(max_depth)
                return reward

    def visualize(self, n, father, g):
        """
        add the current node to the graph and recursively adds child nodes to the graph
        :param n: the last node number
        :param father: the father node name
        :param g: the graph
        :return: updated n
        """
        # add the node its self
        g.attr('node', shape='box')
        name = f"node{n}"
        g.node(name, f"{self.data}\nn={self.na}\nQ={(self.total/self.na):.3f}")
        # connect to father node
        g.edge(father, name)
        n += 1
        # add its child nodes
        for state_node in self.children.values():
            father = name
            n = state_node.visualize(n, father, g)
        # to avoid losing the updated n value every time the function end returns the most updated n value
        return n
