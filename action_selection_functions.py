import random

import numpy as np


def ucb1(total: int, C: float, visit_child: list, n_visits: int):
    avg_reward = np.array(total) / len(visit_child)
    ucb_score = avg_reward + C * np.sqrt(np.log(n_visits) / np.array(visit_child))
    # avoid bias

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
        # random action
        indices = list(range(n_actions))
        probs = [1 / len(indices)] * len(indices)
        sampled_action = random.choices(population=indices, weights=probs)[0]
        return sampled_action

    return policy


def grid_policy(prior_knowledge, n_actions):
    knowledge = prior_knowledge
    n_actions = n_actions

    def policy(state, **kwargs):
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

        sampled_action = np.random.choice(np.flatnonzero(ks == ks.min()))

        return sampled_action

    return policy
