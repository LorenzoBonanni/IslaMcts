import numpy as np
from matplotlib import pyplot as plt

from islaMcts.enviroments.curve_env import CurveEnv
from islaMcts.utils.action_selection_functions import ucb1, continuous_default_policy
from islaMcts.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from islaMcts.agents.parameters.pw_parameters import PwParameters

x = np.linspace(-20, 32, 100)

# the function
higher_y = [CurveEnv.higher_bound(n) for n in x]
lower_y = [CurveEnv.lower_bound(n) for n in x]

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.axis([0, 33, 0, 15])
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


x_states = []
y_states = []
env = CurveEnv()
observation = env.reset()
pos_x, pos_y, vel_x, vel_y, angular_vel = observation
x_states.append(pos_x)
y_states.append(pos_y)

done = False
params = PwParameters(
        root_data=None,
        env=None,
        n_sim=10000,
        C=0.5,
        action_selection_fn=ucb1,
        gamma=0.9,
        rollout_selection_fn=continuous_default_policy,
        action_expansion_function=continuous_default_policy,
        state_variable="_state",
        max_depth=500,
        n_actions=11,
        alpha=0,
        k=50
    )
total_reward = 0
while not done:
    # action = env.action_space.sample()
    params.env = env.unwrapped
    params.root_data = observation
    agent = MctsActionProgressiveWideningHash(
        param=params
    )
    action = agent.fit()
    observation, reward, done, extra = env.step(action)
    total_reward+=reward
    pos_x, pos_y, vel_x, vel_y, angular_vel = observation
    x_states.append(pos_x)
    y_states.append(pos_y)


# plot the road
plt.plot(x, higher_y, 'r')
plt.plot(x, lower_y, 'b')
# plot the car trajectory
plt.plot(x_states, y_states, 'g--')
plt.show()
