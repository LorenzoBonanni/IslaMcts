import subprocess
from abc import ABC, abstractmethod
from typing import Any

import graphviz
import numpy as np

from islaMcts.agents.parameters.mcts_parameters import MctsParameters


class AbstractMcts(ABC):
    __slots__ = "param", "root", "data", "q_values", "trajectories_x", "trajectories_y"

    def __init__(self, param: MctsParameters):
        self.param: MctsParameters = param
        self.root: AbstractStateNode | None = None
        self.data: Any = None
        self.q_values = None
        self.trajectories_x = []
        self.trajectories_y = []

    @abstractmethod
    def fit(self) -> int | np.ndarray:
        """
        Starting method, builds the tree and then gives back the best action

        :return: the best action
        """

    def visualize(self, extension: str = '0') -> None:
        """
        creates a visualization of the tree

        :param extension: extension to the file name
        :return:
        """
        np.set_printoptions(precision=2)
        filename = f'mcts_{extension}'
        g = graphviz.Digraph('g', filename=f'{filename}.gv', directory='output/tree')
        n = 0
        self.root.visualize(n, None, g)

        # save gv file
        g.save()
        # render gv file to an svg
        with open(f'output/tree/{filename}.svg', 'w') as f:
            subprocess.Popen(['dot', '-Tsvg', f'output/tree/{filename}.gv'], stdout=f)


class AbstractStateNode(ABC):
    __slots__ = "data", "param", "terminal", "total", "ns", "actions", "terminal_reward"

    def __init__(self, data: Any, param: MctsParameters):
        self.data: Any = data
        self.param: MctsParameters = param
        self.terminal: bool = False
        # total reward
        self.total: float = 0
        # number of visits of the node
        self.ns: int = 0
        # dictionary containing mapping between action number and corresponding action node
        self.actions: dict[Any, AbstractActionNode] = {}
        # if the state is terminal save it's terminal reward
        self.terminal_reward: float | None = None

    @abstractmethod
    def build_tree_state(self, max_depth: int):
        raise NotImplementedError

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
        x_values = []
        y_values = []
        while not done and depth < max_depth:
            sampled_action = self.param.rollout_selection_fn(env=curr_env, node=self)

            # execute action
            obs, reward, done, _ = curr_env.step(sampled_action)
            x_values.append(obs[0])
            y_values.append(obs[1])
            depth += 1

        self.param.x_values.extend(x_values)
        self.param.y_values.extend(y_values)
        return reward

    def visualize(self, n: int, father: str, g: graphviz.Digraph):
        """
        add the current node to the graph and recursively adds child nodes to the graph

        :param n: the last node number
        :param father: the father node name
        :param g: the graph
        :return: updated n
        """
        # add the node its self
        g.attr('node', shape='circle')
        if self.terminal:
            g.attr('node', fillcolor='green', style='filled')
        name = f"node{n}"
        g.node(name, f"{self.data}\nn={self.ns}\nV={(self.total / self.ns):.3f}")
        g.attr('node', fillcolor='white', style='filled')
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


class AbstractActionNode(ABC):
    __slots__ = "data", "total", "na", "children", "param"

    def __init__(self, data: Any, param: MctsParameters):
        self.data: Any = data
        self.total: float = 0
        self.na: int = 0
        self.children: dict[Any, AbstractStateNode] = {}
        self.param: MctsParameters = param

    @property
    def q_value(self):
        return self.total / self.na

    @abstractmethod
    def build_tree_action(self, max_depth: int) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that

        :param max_depth:  max depth of simulation
        :return:
        """

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
        g.node(name, f"{self.data}\nn={self.na}\nQ={(self.total / self.na):.3f}")
        # connect to father node
        g.edge(father, name)
        n += 1
        # add its child nodes
        for state_node in self.children.values():
            father = name
            n = state_node.visualize(n, father, g)
        # to avoid losing the updated n value every time the function end returns the most updated n value
        return n
