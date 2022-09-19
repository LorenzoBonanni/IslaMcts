import argparse
import os
import random

import gym
import numpy as np
from gym.spaces import Discrete

import wandb
from islaMcts.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from islaMcts.agents.mcts_double_progressive_widening_hash import MctsDoubleProgressiveWideningHash
from islaMcts.agents.mcts_hash import MctsHash
from islaMcts.agents.mcts_state_progressive_widening_hash import MctsStateProgressiveWideningHash
from islaMcts.agents.parameters.dpw_parameters import DpwParameters
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
from islaMcts.agents.parameters.pw_parameters import PwParameters
from islaMcts.environments.curve_env import DiscreteCurveEnv, CurveEnv
from islaMcts.environments.utils.curve_utils import plot_final_trajectory, plot_simulation_trajectory
from islaMcts.utils.action_selection_functions import ucb1, continuous_default_policy, genetic_policy, voo, \
    genetic_policy2


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--environment', default=None, type=str, help='The algorithm to run')
    parser.add_argument('--algorithm', default="vanilla", type=str, help='The algorithm to run')
    parser.add_argument('--nsim', default=1000, type=int, help='The number of simulation the algorithm will run')
    parser.add_argument('--c', default=1, type=float, help='exploration-exploitation factor')
    parser.add_argument('--as', default="ucb", type=str, help='the function to select actions during simulations')
    parser.add_argument('--ae', default="random", type=str, help='the function to select actions to add to the tree')
    parser.add_argument('--gamma', default=0.5, type=float, help='discount factor')
    parser.add_argument('--rollout', default='random', type=str, help='the function to select actions during rollout')
    parser.add_argument('--max_depth', default=500, type=int, help='max number of steps during rollout')
    parser.add_argument('--n_actions', default=10, type=int, help='number of actions for vanilla mcts')
    parser.add_argument('--alpha1', default=0, type=float, help='alpha')
    parser.add_argument('--alpha2', default=0.1, type=float, help='alpha2')
    parser.add_argument('--k1', default=1, type=int, help='k1')
    parser.add_argument('--k2', default=1, type=int, help='k2')
    parser.add_argument('--epsilon', default=0.7, type=float, help='epsilon value for epsilon greedy strategies')
    parser.add_argument('--genetic_default', default="random", type=str,
                        help='the default policy for the genetic algorithm')
    parser.add_argument('--n_sample', default=5, type=int,
                        help='the number of samples taken by the genetic algorithm or by voo')
    parser.add_argument('--n_episodes', default=100, type=int,
                        help='the number of episodes for each experiment')
    return parser


def get_function(function_name, genetic_default=None):
    dict_args = args.__dict__
    functions = {
        "ucb": ucb1,
        "random": continuous_default_policy,
        "genetic": genetic_policy(
            epsilon=dict_args["epsilon"],
            default_policy=genetic_default,
            n_samples=dict_args["n_sample"]
        ),
        "genetic2": genetic_policy2(
            epsilon=dict_args["epsilon"],
            default_policy=genetic_default
        ),
        "voo": voo(
            epsilon=dict_args["epsilon"],
            default_policy=genetic_default,
            n_samples=dict_args["n_sample"]
        )
    }
    return functions[function_name]


def create_agent():
    dict_args = args.__dict__
    agent_name = dict_args["algorithm"]
    match agent_name:
        case "vanilla":
            param = MctsParameters(
                root_data=None,
                env=None,
                n_sim=dict_args["nsim"],
                C=dict_args["c"],
                action_selection_fn=get_function(dict_args["as"]),
                gamma=dict_args["gamma"],
                rollout_selection_fn=get_function(dict_args["rollout"]),
                max_depth=dict_args["max_depth"],
                n_actions=dict_args["n_actions"],
                x_values=None,
                y_values=None
            )
            return MctsHash(param)
        case "apw":
            default_genetic = get_function(dict_args["genetic_default"])
            param = PwParameters(
                root_data=None,
                env=None,
                n_sim=dict_args["nsim"],
                C=dict_args["c"],
                action_selection_fn=get_function(dict_args["as"]),
                gamma=dict_args["gamma"],
                rollout_selection_fn=get_function(dict_args["rollout"]),
                max_depth=dict_args["max_depth"],
                n_actions=dict_args["n_actions"],
                alpha=dict_args["alpha1"],
                k=dict_args["k1"],
                action_expansion_function=get_function(dict_args["ae"], genetic_default=default_genetic),
                x_values=None,
                y_values=None
            )
            return MctsActionProgressiveWideningHash(param)
        case "spw":
            param = PwParameters(
                root_data=None,
                env=None,
                n_sim=dict_args["nsim"],
                C=dict_args["c"],
                action_selection_fn=get_function(dict_args["as"]),
                gamma=dict_args["gamma"],
                rollout_selection_fn=get_function(dict_args["rollout"]),
                max_depth=dict_args["max_depth"],
                n_actions=dict_args["n_actions"],
                alpha=dict_args["alpha1"],
                k=dict_args["k1"],
                action_expansion_function=get_function(dict_args["ae"]),
                x_values=None,
                y_values=None
            )
            return MctsStateProgressiveWideningHash(param)
        case "dpw":
            param = DpwParameters(
                root_data=None,
                env=None,
                n_sim=dict_args["nsim"],
                C=dict_args["c"],
                action_selection_fn=get_function(dict_args["as"]),
                gamma=dict_args["gamma"],
                rollout_selection_fn=get_function(dict_args["rollout"]),
                max_depth=dict_args["max_depth"],
                n_actions=dict_args["n_actions"],
                alphaApw=dict_args["alpha1"],
                kApw=dict_args["k1"],
                alphaSpw=dict_args["alpha2"],
                kSpw=dict_args["k2"],
                x_values=None,
                y_values=None
            )
            return MctsDoubleProgressiveWideningHash(param)


def get_group_name():
    dict_args = args.__dict__
    agent_name = dict_args["algorithm"]
    group_name = {
        "vanilla": f"{agent_name}_{dict_args['n_actions']}",
        "apw": f"{agent_name}_{dict_args['ae']}_{dict_args['k1']}",
        "spw": f"{agent_name}_{dict_args['rollout']}",
        "dpw": f"{agent_name}_{dict_args['rollout']}_{dict_args['ae']}",
    }
    return group_name[agent_name]


def main():
    dict_args = args.__dict__
    rewards = []
    os.environ["WANDB_RUN_GROUP"] = get_group_name()
    for _ in range(dict_args["n_episodes"]):
        wandb.init(config=dict_args, entity="lorenzobonanni", project="car-game", reinit=True)
        # env = gym.make(dict_args["environment"]).unwrapped
        # env = DiscreteCurveEnv([5, 19])
        env = CurveEnv()
        observation = env.reset()
        # np.random.seed(1)
        # random.seed(1)
        # env.action_space.seed(1)
        done = False
        total_reward = 0
        x_states, y_states = [], []
        n_steps = 0
        points_x, points_y = [], []
        while not done:
            agent = create_agent()
            agent.param.env = env.unwrapped
            agent.param.root_data = observation
            agent.root.data = observation
            action = agent.fit()
            observation, reward, done, extra = env.step(action)
            if isinstance(env.action_space, Discrete):
                action_node = agent.root.actions[action]
            else:
                action_node = agent.root.actions[action.tobytes()]
            max_depth = action_node.get_depth_max(0)
            mean_depth = action_node.get_depth_mean(0, True)
            total_reward += reward
            x_states.append(observation[0])
            y_states.append(observation[1])
            points_x.extend(agent.trajectories_x)
            points_y.extend(agent.trajectories_y)
            n_steps += 1

            if n_steps >= 1000:
                rewards[-1] = -1000
                break

            if isinstance(env.action_space, Discrete):
                action = env.actions[action]
            wandb.log(
                {
                    "reward": reward,
                    "velocity": observation[2],
                    "angle": observation[3],
                    "acceleration": action[0],
                    "angle_change": action[1],
                    "max_depth": max_depth,
                    "mean_depth": mean_depth
                }
            )
            if n_steps == 1:
                wandb.log(
                    {
                        "simTrajectory0": wandb.Image(plot_simulation_trajectory(points_x, points_y))
                    }
                )
        wandb.log(
            {
                "total_reward": total_reward,
                "final_trajectory": wandb.Image(plot_final_trajectory(x_states, y_states)),
                "n_steps": n_steps,
                "simulation_trajectory": wandb.Image(plot_simulation_trajectory(points_x, points_y))
            }
        )
        rewards.append(total_reward)
    wandb.log(
        {"mean_reward": np.mean(rewards)}
    )


if __name__ == '__main__':
    global args
    args = argument_parser().parse_args()
    main()
