import gym
import numpy as np
from tqdm import tqdm

from action_selection_functions import ucb1, continuous_default_policy, genetic_policy
from gym_goddard.envs.goddard_env import GoddardEnv
from islaMcts.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from islaMcts.agents.parameters.pw_parameters import PwParameters
from islaMcts.utils import my_deepcopy

time = np.arange(0, 0.4, 0.001)


def main():
    # DISCRETE
    # real_env = gym.make('gym_goddard:GoddardDiscreteNoise-v0')
    # sim_env = gym.make('gym_goddard:GoddardDiscreteNoise-v0')

    # CONTINUOUS
    # real_env = gym.make('gym_goddard:Goddard-v0')
    real_env = GoddardEnv()
    observation = real_env.reset()

    state_log = [observation]
    extra_log = []
    total_reward = 0
    params = PwParameters(
        root_data=None,
        env=None,
        n_sim=100,
        C=0.0009,
        action_selection_fn=ucb1,
        gamma=1,
        rollout_selection_fn=continuous_default_policy,
        action_expansion_function=genetic_policy(0.7, continuous_default_policy),
        state_variable="_state",
        max_depth=500,
        n_actions=11,
        alpha=0,
        k=8
    )
    act_dist = []

    for _ in tqdm(time):
        params.env = my_deepcopy(real_env)

        agent = MctsActionProgressiveWideningHash(
            param=params
        )
        action = agent.fit()

        # SAVE ACTION DISTRIBUTION
        actions = np.array([a.data[0] for a in agent.root.actions.values()])
        act_dist.append(actions)

        observation, reward, done, extra = real_env.step(action)

        total_reward += reward
        state_log.append(observation)
        extra_log.append(list(extra.values()))
        params.root_data = observation

    print(total_reward)
    state_log = np.array(state_log)
    np.savetxt(fname=f"../output/state_log_spw.csv", X=state_log, delimiter=",")
    extra_log = np.array(extra_log)
    np.savetxt(fname=f"../output/extra_log_spw.csv", X=extra_log, delimiter=",")
    np.savetxt(fname=f"act_dist_.csv", X=act_dist, delimiter=",")

if __name__ == '__main__':
    main()
