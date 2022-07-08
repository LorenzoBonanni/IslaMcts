import random

import numpy as np


def ucb1(total: float, C: float, visit_child: list, n_visits: int):
    """
    computes the best action based on the ucb1 value
    :param total: the total reward obtained by the node so far
    :param C: exploration constant
    :param visit_child: an array containing the number of visits of each action
    :param n_visits: the number of visits of the node
    :return:
    """
    avg_reward = np.array(total) / len(visit_child)
    ucb_score = avg_reward + C * np.sqrt(np.log(n_visits) / np.array(visit_child))

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

    def policy(**kwargs):
        # choose an action uniformly random
        indices = list(range(n_actions))
        probs = [1 / len(indices)] * len(indices)
        sampled_action = random.choices(population=indices, weights=probs)[0]
        return sampled_action

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

    def policy(state, **kwargs):
        """
        computes the best action based on the heuristic
        :param state: the state in which the agent is in
        :return:
        """
        indices = list(range(n_actions))
        ks = np.array(knowledge[state])
        # probs = [1 / len(indices)] * len(indices)
        #
        # knowledge_bias = 1 / ks
        # knowledge_bias = knowledge_bias / sum(knowledge_bias)
        #
        # probs = knowledge_bias + probs
        # probs = probs / sum(probs)

        # sampled_action = random.choices(population=indices, weights=probs)[0]

        # to avoid biases if two or more actions have the same value we choose randomly between them
        sampled_action = np.random.choice(np.flatnonzero(ks == ks.min()))

        return sampled_action

    return policy
