import gym
import numpy as np
from tqdm import tqdm

from action_selection_functions import ucb1, continuous_default_policy
from islaMcts.agents.mcts_state_progressive_widening_hash import MctsStateProgressiveWideningHash
from islaMcts.agents.parameters.pw_parameters import PwParameters

time = np.arange(0, 0.4, 0.001)

def main():
    # DISCRETE
    real_env = gym.make('gym_goddard:GoddardDiscreteNoise-v0')
    sim_env = gym.make('gym_goddard:GoddardDiscreteNoise-v0')

    # CONTINUOUS
    # real_env = gym.make('gym_goddard:Goddard-v0')
    # sim_env = gym.make('gym_goddard:Goddard-v0')

    observation = real_env.reset()

    state_log = [observation]
    extra_log = []
    params = PwParameters(
        root_data=observation,
        env=sim_env.unwrapped,
        n_sim=1000,
        C=0.0005,
        action_selection_fn=ucb1,
        gamma=1,
        rollout_selection_fn=continuous_default_policy,
        state_variable="_state",
        max_depth=500,
        # alpha=0.3,
        # k=5,
        alpha=0.5,
        k=3,
        n_actions=11
    )

    for _ in tqdm(time):
        sim_env.reset()
        # APW 22min
        agent = MctsStateProgressiveWideningHash(
            param=params
        )
        action = agent.fit()
        observation, reward, done, extra = real_env.step(action)
        # print(len(agent.root.actions))
        # agent.visualize()
        # break
        # state_log.append(observation)
        # extra_log.append(list(extra.values()))
        params.root_data = observation

    extra_log[-1] = extra_log[-1][:-1]
    state_log = np.array(state_log)
    np.savetxt(fname=f"../output/state_log_spw.csv", X=state_log, delimiter=",")
    extra_log = np.array(extra_log)
    np.savetxt(fname=f"../output/extra_log_spw.csv", X=extra_log, delimiter=",")

if __name__ == '__main__':
    main()
