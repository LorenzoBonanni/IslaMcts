import argparse

import numpy as np
import wandb
from islaMcts.utils.action_selection_functions import ucb1, continuous_default_policy, genetic_policy
from islaMcts.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from islaMcts.agents.mcts_double_progressive_widening_hash import MctsDoubleProgressiveWideningHash
from islaMcts.agents.mcts_hash import MctsHash
from islaMcts.agents.mcts_state_progressive_widening_hash import MctsStateProgressiveWideningHash
from islaMcts.agents.parameters.dpw_parameters import DpwParameters
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
from islaMcts.agents.parameters.pw_parameters import PwParameters
from islaMcts.enviroments.curve_env import CurveEnv
from islaMcts.utils.mcts_utils import LazyDict


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
    parser.add_argument('--genetic_n_sample', default=5, type=int,
                        help='the number of samples taken by the genetic algorithm')
    return parser


def get_function(function_name, genetic_default=None):
    dict_args = args.__dict__
    functions = {
            "ucb": ucb1,
            "random": continuous_default_policy,
            "genetic": genetic_policy(
                epsilon=dict_args["epsilon"],
                default_policy=genetic_default,
                n_samples=dict_args["genetic_n_sample"]
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
                n_actions=dict_args["n_actions"]
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
                action_expansion_function=get_function(dict_args["ae"], genetic_default=default_genetic)
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
                action_expansion_function=get_function(dict_args["ae"])
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
            )
            return MctsDoubleProgressiveWideningHash(param)


def main():
    # env = gym.make(dict_args["environment"])
    rewards = []
    for _ in range(5):
        env = CurveEnv()
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            # action = env.action_space.sample()
            agent = create_agent()
            agent.param.env = env.unwrapped
            agent.param.root_data = observation
            action = agent.fit()
            observation, reward, done, extra = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    wandb.log("mean_reward", np.mean(rewards))


if __name__ == '__main__':
    global args
    args = argument_parser().parse_args()
    wandb.init(config=args.__dict__, entity="lorenzobonanni")
    main()