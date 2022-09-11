import numpy as np

from islaMcts.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from islaMcts.agents.parameters.pw_parameters import PwParameters
from islaMcts.environments.curve_env import CurveEnv
from islaMcts.environments.utils.curve_utils import plot_final_trajectory
from islaMcts.utils.action_selection_functions import ucb1, continuous_default_policy, voo

rw = []
steps = []
for experiment in range(1):
    x_states = []
    y_states = []
    # env = DiscreteCurveEnv([5, 19])
    env = CurveEnv()
    observation = env.reset()
    pos_x, pos_y, vel, angle = observation
    # x_states.append(pos_x)
    # y_states.append(pos_y)

    done = False
    params = PwParameters(
        root_data=None,
        env=None,
        n_sim=100,
        C=11.837,
        action_selection_fn=ucb1,
        gamma=0.4,
        rollout_selection_fn=continuous_default_policy,
        action_expansion_function=voo(0.5, continuous_default_policy, 4, max_try=5000),
        max_depth=500,
        n_actions=11,
        alpha=0,
        k=95,
        x_values = [],
        y_values=[]
    )
    # params = MctsParameters(
    #     root_data=None,
    #     env=None,
    #     n_sim=100,
    #     C=11.837,
    #     action_selection_fn=ucb1,
    #     gamma=0.4,
    #     rollout_selection_fn=continuous_default_policy,
    #     max_depth=500,
    #     n_actions=95
    # )
    total_reward = 0
    n_steps = 0

    # while not done:
    #     print(f"step: {n_steps}")
    #     # action = env.action_space.sample()
    #     params.env = env.unwrapped
    #     params.root_data = observation
    #     # agent = MctsHash(
    #     #     param=params
    #     # )
    #     agent = MctsActionProgressiveWideningHash(
    #         param=params
    #     )
    #     action = agent.fit()
    #     observation, reward, done, extra = env.step(action)
    #     total_reward += reward
    #     pos_x, pos_y, vel, angle = observation
    #     print(f"POSITION: ({pos_x}, {pos_y})")
    #     print(f"REWARD: {reward}")
    #     x_states.append(pos_x)
    #     y_states.append(pos_y)
    #     n_steps += 1
    #     # if n_steps == 1:
    #     #     create_rollout_trajectory(0)

    trajectory = plot_final_trajectory(x_states, y_states)
    trajectory.savefig(f'../../output/trajectory_{experiment}.png')
    # experiment
    rw.append(total_reward)
    steps.append(n_steps)

print(f"REWARD:{np.mean(rw)} ±{np.std(rw)}")
print(f"STEP:{np.mean(steps)} ±{np.std(steps)}")


#
# create_rollout_trajectory(1)
