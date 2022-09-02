import time

import numpy as np
from matplotlib import pyplot as plt, animation

from islaMcts.enviroments.curve_env import CurveEnv
# from islaMcts.utils.action_selection_functions import ucb1, continuous_default_policy, genetic_policy
# from islaMcts.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
# from islaMcts.agents.parameters.pw_parameters import PwParameters

x = np.linspace(-1, 32, 100)

# the function
higher_y = [CurveEnv.higher_bound(n) for n in x]
lower_y = [CurveEnv.lower_bound(n) for n in x]

x_states = []
y_states = []



env = CurveEnv()
observation = env.reset()
pos_x, pos_y, vel, angle = observation
x_states.append(pos_x)
y_states.append(pos_y)

done = False
# params = PwParameters(
#     root_data=None,
#     env=None,
#     n_sim=3000,
#     C=7.37,
#     action_selection_fn=ucb1,
#     gamma=0.18,
#     rollout_selection_fn=continuous_default_policy,
#     action_expansion_function=genetic_policy(0.5, continuous_default_policy, 72),
#     max_depth=255,
#     n_actions=11,
#     alpha=0,
#     k=116
# )
total_reward = 0
n_steps = 0

# while not done:
#     print(f"step: {n_steps}")
#     # action = env.action_space.sample()
#     params.env = env.unwrapped
#     params.root_data = observation
#     agent = MctsActionProgressiveWideningHash(
#         param=params
#     )
#     action = agent.fit()
#     observation, reward, done, extra = env.step(action)
#     total_reward += reward
#     pos_x, pos_y, vel_x, vel_y, angular_vel = observation
#     x_states.append(pos_x)
#     y_states.append(pos_y)
#     n_steps += 1
print(f"X: {pos_x}, Y:{pos_y}, vel:{vel}, angle:{angle}")
fig = None
while not done:
    acceleration = float(input("Acceleration? [-5, +5] "))
    input_angle = float(input("Angle? [-45, +45] "))

    observation, reward, done, extra = env.step(np.array([acceleration, input_angle]))
    pos_x, pos_y, vel, angle = observation
    print(f"X: {pos_x}, Y:{pos_y}, vel:{vel}, angle:{angle}")
    total_reward += reward
    x_states.append(pos_x)
    y_states.append(pos_y)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.axis([-1, 33, 0, 33])
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(x, higher_y, 'r')
    plt.plot(x, lower_y, 'b')
    ax.plot(x_states, y_states, 'g--')
    plt.draw()
    plt.show(block=False)
    plt.pause(0.3)

fig.savefig('trajectory.png')
print(f"REWARD: {total_reward}")