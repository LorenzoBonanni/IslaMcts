import gym
import numpy as np
from tqdm import tqdm

from action_selection_functions import ucb1, discrete_default_policy, continuous_default_policy
from src.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from src.agents.pw_parameters import PwParameters

time = np.arange(0, 0.4, 0.001)

def main():
    # DISCRETE
    # real_env = gym.make('gym_goddard:GoddardDiscrete-v0')
    # sim_env = gym.make('gym_goddard:GoddardDiscrete-v0')

    # CONTINUOUS
    real_env = gym.make('gym_goddard:Goddard-v0')
    sim_env = gym.make('gym_goddard:Goddard-v0')

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
        alpha=0.3,
        k=5,
        # alpha=0.5,
        # k=3,
        n_actions=None
    )

    for _ in tqdm(time):
        sim_env.reset()
        # APW 22min
        agent = MctsActionProgressiveWideningHash(
            param=params
        )
        action = agent.fit()
        observation, reward, done, extra = real_env.step(action)
        state_log.append(observation)
        extra_log.append(list(extra.values()))
        params.root_data = observation

    extra_log[-1] = extra_log[-1][:-1]
    state_log = np.array(state_log)
    np.savetxt(fname=f"output/state_log.csv", X=state_log, delimiter=",")
    extra_log = np.array(extra_log)
    np.savetxt(fname=f"output/extra_log.csv", X=extra_log, delimiter=",")

if __name__ == '__main__':
    main()
