import gym
import numpy as np
from tqdm import tqdm

from action_selection_functions import ucb1, discrete_default_policy
from src.agent_factory import AgentFactory
from src.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from src.agents.mcts_continuous_hash import MctsContinuousHash
from src.agents.mcts_double_progressive_widening_hash import MctsDoubleProgressiveWideningHash
from src.agents.mcts_hash import MctsHash
from src.agents.mcts_state_progressive_widening_hash import MctsStateProgressiveWideningHash

time = np.arange(0, 0.4, 0.001)


def generate_plots(godd, state_log, extra_log):
    print("Maximum altitude reached: {}".format(godd.maximum_altitude()))
    # print("End reward: {}".format(reward_end))
    # print("Total reward: {}".format(reward))

    import matplotlib.pyplot as plt

    state_log = np.array(state_log)
    extra_log = np.array(extra_log)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    ax1.plot(time, state_log[:-1, 1])
    ax1.grid(True)
    ax1.set(xlabel='time [s]', ylabel='Altitude [m]')

    ax2.plot(time, state_log[:-1, 0])
    ax2.grid(True)
    ax2.set(xlabel='time [s]', ylabel='Velocity [m/s]')

    ax3.plot(time, state_log[:-1, 2])
    ax3.grid(True)
    ax3.set(xlabel='time [s]', ylabel='Rocket Mass [kg]')

    ax4.plot(time, extra_log)
    ax4.grid(True)
    ax4.set(xlabel='time [s]', ylabel='Forces [N]')
    ax4.legend(godd.extras_labels(), loc='upper right')
    print(godd.extras_labels())
    plt.savefig('mcts_continuous.svg')


def main():
    # DISCRETE
    # real_env = gym.make('gym_goddard:GoddardDiscrete-v0')
    # sim_env = gym.make('gym_goddard:GoddardDiscrete-v0')

    # CONTINUOUS
    real_env = gym.make('gym_goddard:Goddard-v0')
    sim_env = gym.make('gym_goddard:Goddard-v0')

    observation = real_env.reset()
    sim_env.reset()
    # real_env.render()
    rollout_fn = discrete_default_policy(11)

    state_log = [observation]
    extra_log = []
    agent = None

    for _ in tqdm(time):
        last_state = observation
        agent = MctsActionProgressiveWideningHash(
            root_data=observation,
            env=sim_env.unwrapped,
            n_sim=1000,
            C=0,
            action_selection_fn=ucb1,
            gamma=1,
            rollout_selection_fn=rollout_fn,
            state_variable="_state",
            max_depth=500,
            alpha=0.3,
            k=5,
            # alpha=0.5,
            # k=3
        )
        action = agent.fit()
        observation, reward, done, extra = real_env.step(action)
        state_log.append(observation)
        extra_log.append(list(extra.values()))
        # print()
        # print(f"S: {last_state} A: {action}, S': {observation}, R: {reward}")
        agent.visualize()
        break
        # real_env.render()
    extra_log[-1] = extra_log[-1][:-1]
    generate_plots(real_env, state_log, extra_log)

    # TODO make comparison graphs


if __name__ == '__main__':
    main()
