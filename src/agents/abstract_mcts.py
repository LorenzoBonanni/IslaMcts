import subprocess
from abc import ABC, abstractmethod
from ctypes import Union
from typing import Any

import graphviz
import numpy as np
from graphviz import Digraph

from src.agents.mcts_parameters import MctsParameters


class AbstractMcts(ABC):

    def __init__(self, param: MctsParameters):
        self.param = param
        self.root = None
        self.data = None
        self.q_values = None

    def fit(self) -> int:
        """
        Starting method, builds the tree and then gives back the best action

        :return: the best action
        """
        pass

    def visualize(self, extension: str = '0') -> None:
        """
        creates a visualization of the tree

        :param extension: extension to the file name
        :return:
        """
        np.set_printoptions(precision=2)
        filename = f'mcts_{extension}'
        g = graphviz.Digraph('g', filename=f'{filename}.gv', directory='output')
        n = 0
        self.root.visualize(n, None, g)

        # save gv file
        g.save()
        # render gv file to an svg
        with open(f'output/{filename}.svg', 'w') as f:
            subprocess.Popen(['dot', '-Tsvg', f'output/{filename}.gv'], stdout=f)


class AbstractStateNode(ABC):

    def __init__(self, data: Any, param: MctsParameters):
        self.data = data
        self.param = param
        self.terminal = False
        # total reward
        self.total = 0
        # number of visits of the node
        self.ns = 0
        # dictionary containing mapping between action number and corresponding action node
        self.actions = {}

    @abstractmethod
    def build_tree(self, max_depth: int):
        pass

    @abstractmethod
    def rollout(self, max_depth: int) -> float:
        """
        Random play out until max depth or a terminal state is reached

        :param max_depth: max depth of simulation
        :return: reward obtained from the state
        """
        pass

    def visualize(self, n: int, father: str, g: Digraph):
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
    def __init__(self, data: Any, param: MctsParameters):
        self.data = data
        self.total = 0
        self.na = 0
        self.children = {}
        self.param = param

    @abstractmethod
    def build_tree(self, max_depth: int) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that

        :param max_depth:  max depth of simulation
        :return:
        """
        pass

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
