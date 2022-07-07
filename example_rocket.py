from collections.abc import Hashable

import gym
import numpy as np

from action_selection_functions import ucb1, discrete_default_policy
from agents.mcts_continuous import MctsHashStateContinuous

time = np.arange(0, 0.4, 0.001)
def generate_plots(godd, state_log, extra_log):
    print("Maximum altitude reached: {}".format(godd.maximum_altitude()))
    # print("End reward: {}".format(reward_end))
    # print("Total reward: {}".format(reward))

    import matplotlib.pyplot as plt

    state_log = np.array(state_log)

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
    plt.savefig('mcts_continuous.svg')


def main():
    real_env = gym.make('gym_goddard:Goddard-v0')
    sim_env = gym.make('gym_goddard:Goddard-v0')
    observation = real_env.reset()
    sim_env.reset()
    done = False
    last_state = None
    real_env.render()
    rollout_fn = discrete_default_policy(11)
    hashable = True

    # check if obs is hashable
    if not isinstance(observation, Hashable):
        hashable = False

    state_log = [observation]
    extra_log = []

    for _ in time:
        last_state = observation
        # if hashable:
        #     agent = Mcts(
        #         C=2,
        #         n_sim=50,
        #         root_data=observation,
        #         env=copy(real_env.unwrapped),
        #         action_selection_fn=ucb1,
        #         max_depth=100,
        #         gamma=0.9,
        #         rollout_selection_fn=rollout_fn,
        #         state_variable="_state",
        #     )
        # else:
        #     agent = MctsHash(
        #         C=2,
        #         n_sim=100,
        #         root_data=observation,
        #         env=copy(real_env.unwrapped),
        #         action_selection_fn=ucb1,
        #         max_depth=100,
        #         gamma=0.9,
        #         rollout_selection_fn=rollout_fn,
        #         state_variable="_state",
        #     )
        agent = MctsHashStateContinuous(
                C=2,
                n_sim=50,
                root_data=observation,
                env=sim_env,
                action_selection_fn=ucb1,
                max_depth=100,
                gamma=0.9,
                rollout_selection_fn=rollout_fn,
                state_variable="_state",
            )
        action = agent.fit()
        agent.visualize()
        observation, reward, done, extra = real_env.step(action)
        state_log.append(observation)
        extra_log.append(list(extra.values()))
        print(f"S: {last_state} A: {action}, S': {observation}, R: {reward}")
        print()
        real_env.render()
        break

    extra_log[-1] = extra_log[-1][:-1]
    # generate_plots(real_env, state_log, extra_log)

    # TODO make comparison graphs


if __name__ == '__main__':
    main()
