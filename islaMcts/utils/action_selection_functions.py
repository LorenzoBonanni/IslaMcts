import logging
import random

import gym
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Polygon
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
    index = np.random.choice(np.flatnonzero(ucb_score == ucb_score.max()))
    return children[index].data


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


def continuous_default_policy(node: AbstractStateNode, *args, **kwargs):
    return node.param.env.action_space.sample()


def genetic_policy(epsilon, default_policy, n_samples):
    epsilon = epsilon
    n_samples = n_samples
    default_policy = default_policy

    def policy(node: AbstractStateNode):
        action_space = node.param.env.action_space
        if node.ns == 0:
            high = action_space.high
            low = action_space.low
            center = np.zeros(high.shape)
            for i in range(len(high)):
                center[i] = np.median([high[i], low[i]])
            return center
        elif node.ns == 1:
            return random.choices([action_space.high, action_space.low])[0]
        elif node.ns == 2:
            if action_space.high.tobytes() in node.actions:
                return action_space.low
            else:
                return action_space.high
        else:
            p = np.random.random()
            if p < epsilon:
                actions = [node for node in node.actions.values()]
                actions.sort(key=lambda n: n.q_value)
                best_action = actions[0].data
                second_best = actions[1].data
                # compute euclidian distance with respect to the low point of the action space
                mht_best_action = distance.cityblock(action_space.low, best_action)
                mht_second_best_action = distance.cityblock(action_space.low, second_best)
                # the point nearer to the low is the low point of the new action space
                if mht_best_action >= mht_second_best_action:
                    high = best_action
                    low = second_best
                else:
                    high = second_best
                    low = best_action
                # hybrid action between two best actions
                # sample random between the two best actions and then choose the one nearer to the best
                best_dist = np.inf
                best_found = None
                for _ in range(n_samples):
                    sample = np.random.uniform(low=low, high=high, size=high.shape)
                    dist = distance.euclidean(best_action, sample)
                    if dist < best_dist:
                        best_dist = dist
                        best_found = sample
                return best_found
            else:
                return default_policy(node)

    return policy


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
