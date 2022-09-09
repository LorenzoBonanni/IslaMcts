import logging
import random
import timeit

import gym
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Polygon
from islaMcts.agents.abstract_mcts import AbstractStateNode
from islaMcts.agents.mcts_action_progressive_widening_hash import StateNodeProgressiveWideningHash, \
    ActionNodeProgressiveWideningHash
from islaMcts.agents.parameters.pw_parameters import PwParameters
from islaMcts.environments.curve_env import CurveEnv


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

    def genetic(node: AbstractStateNode):
        action_space = node.param.env.action_space
        actions = [node for node in node.actions.values()]
        actions.sort(key=lambda n: n.q_value)
        best_action = actions[0].data
        second_best = actions[1].data
        # compute euclidian distance with respect to the low point of the action space
        mht_best_action = distance.euclidean(action_space.low, best_action)
        mht_second_best_action = distance.euclidean(action_space.low, second_best)
        # the point nearer to the low is the low point of the new action space
        if mht_best_action >= mht_second_best_action:
            high = best_action
            low = second_best
        else:
            high = second_best
            low = best_action
        # hybrid action between two best actions
        # sample random between the two best actions and then choose the one nearer to the best
        samples = np.random.uniform(low=low, high=high, size=[n_samples, *high.shape])
        distances = distance.cdist([best_action], samples, 'euclidean')[0]
        idx_min_dist = np.argmin(distances)
        best_found = samples[idx_min_dist]
        return best_found

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
                return genetic(node)
            else:
                return default_policy(node)

    return policy


def voo(epsilon, default_policy, n_samples, max_try=5000):
    epsilon = epsilon
    n_samples = n_samples
    default_policy = default_policy
    max_try = max_try

    def sample_best_v_cell(parent_node: AbstractStateNode):
        actions = []
        q_values = []
        for action_node in parent_node.actions.values():
            actions.append(action_node.data)
            q_values.append(action_node.q_value)
        actions = np.array(actions)
        q_values = np.array(q_values)
        # get the index of best q value, to avoid biases if there are multiple index with the same q value
        # we choose random between them
        idx_best_q = np.random.choice(np.where(q_values == q_values.max())[0])

        sampled_actions = []
        sampled_distances = []
        for _ in range(n_samples):
            n_try, a_tried, d_tried = 0, np.zeros((max_try+1, *parent_node.param.env.action_space.shape)), np.zeros((max_try, 1))
            while True:
                sample = np.random.uniform(
                    low=parent_node.param.env.action_space.low,
                    high=parent_node.param.env.action_space.high,
                    size=parent_node.param.env.action_space.shape
                )

                # compute euclidian distance between sample and points
                distances = distance.cdist([sample], actions, 'euclidean')[0]
                distance_from_best = distances[idx_best_q]

                a_tried[n_try, :] = sample
                d_tried[n_try, :] = distance_from_best
                # remove distance from best action from the distance vector
                distances = np.delete(distances, idx_best_q)
                # check if sampled point is closer to the best action than any other point
                if distance_from_best == 0:
                    break
                elif (distance_from_best <= distances).all():
                    sampled_actions.append(sample)
                    sampled_distances.append(np.array([distance_from_best]))
                    break

                n_try += 1
                if n_try >= max_try:
                    idx_min_tried = np.random.choice(np.where(d_tried == d_tried.min())[0])
                    sampled_actions.append(a_tried[idx_min_tried])
                    sampled_distances.append(d_tried[idx_min_tried])
                    break

        sampled_distances = np.array(sampled_distances)
        # get the index of action closer to the best, to avoid biases if there are multiple
        # actions with the same distance we choose random between them

        idx_action = np.random.choice(np.where(sampled_distances == sampled_distances.min())[0])
        return sampled_actions[idx_action]

    def policy(node: AbstractStateNode):
        p = np.random.random()
        if p <= epsilon or len(node.actions) == 0:
            return default_policy(node)
        else:
            return sample_best_v_cell(node)

    return policy


# if __name__ == '__main__':
#     env = CurveEnv()
#     env.reset()
#     param = PwParameters(
#         root_data=None,
#         env=env,
#         n_sim=None,
#         C=None,
#         action_selection_fn=None,
#         gamma=None,
#         rollout_selection_fn=None,
#         action_expansion_function=None,
#         max_depth=None,
#         n_actions=None,
#         alpha=None,
#         k=None
#     )
#     np.random.seed(1)
#     env.seed(1)
#     node = StateNodeProgressiveWideningHash(
#         data=np.array([-4.3416102, -19.63017764]),
#         param=param
#     )
#     node.actions = {
#         np.array([-0.91029103, - 27.39435862]).tobytes(): ActionNodeProgressiveWideningHash(
#             data=np.array([-0.91029103, - 27.39435862]), param=param
#         ),
#         np.array([4.23188117, -30.92624101]).tobytes(): ActionNodeProgressiveWideningHash(
#             data=np.array([4.23188117, -30.92624101]), param=param
#         ),
#         np.array([2.40952619, 24.08882647]).tobytes(): ActionNodeProgressiveWideningHash(
#             data=np.array([2.40952619, 24.08882647]), param=param
#         ),
#         # np.array([-1.90805691, -24.78609096]).tobytes(): ActionNodeProgressiveWideningHash(
#         #     data=np.array([-1.90805691, -24.78609096]), param=param
#         # )
#     }
#     actions = list(node.actions.values())
#     actions[0].total = 10
#     actions[0].na = 2
#     actions[1].total = 9
#     actions[1].na = 2
#     actions[2].total = 8
#     actions[2].na = 2
#     # actions[3].total = 10
#     # actions[3].na = 2
#     voronoi = voo(0.5, continuous_default_policy, 4, max_try=5000)
#     print(timeit.timeit(lambda: voronoi(node), number=1000))


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
