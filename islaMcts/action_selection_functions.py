import logging
import random

import gym
import numpy as np

from islaMcts.agents.abstract_mcts import AbstractStateNode


def ucb1(node: AbstractStateNode):
    """
    computes the best action based on the ucb1 value
    """
    n_visits = node.ns
    visit_child = []
    node_values = []
    children = list(node.actions.values())
    for c in children:
        visit_child.append(c.na)
        node_values.append(c.total / c.na)

    ucb_score = np.array(node_values) + node.param.C * np.sqrt(np.log(n_visits) / np.array(visit_child))

    # to avoid biases we randomly choose between the actions which maximize the ucb1 score
    a = np.random.choice(np.flatnonzero(ucb_score == ucb_score.max()))
    return a


def discrete_default_policy(n_actions: int):
    """
    random policy
    :type n_actions: the number of available actions
    :return:
    """
    n_actions = n_actions

    def policy(*args, **kwargs):
        # choose an action uniformly random
        indices = list(range(n_actions))
        probs = [1 / len(indices)] * len(indices)
        sampled_action = random.choices(population=indices, weights=probs)[0]
        return sampled_action

    return policy


def continuous_default_policy(env, *args, **kwargs):
    return env.action_space.sample()


def grid_policy(prior_knowledge, n_actions):
    """
    a rollout policy based on
    :param prior_knowledge: a dictionary where the key is the state and the value is a vector
    representing the value of an action based on the knowledge (lower the value the better the action)
    :param n_actions: the number of available actions
    :return:
    """
    knowledge = prior_knowledge
    n_actions = n_actions

    def policy(env: gym.Env, node: AbstractStateNode, *args, **kwargs):
        """
        computes the best action based on the heuristic
        :return:
        """
        state = env.__dict__[node.param.state_variable]
        ks = np.array(knowledge[state])

        # to avoid biases if two or more actions have the same value we choose randomly between them
        sampled_action = np.random.choice(np.flatnonzero(ks == ks.min()))

        return sampled_action

    return policy
