import numpy as np
from matplotlib import pyplot as plt
from curve_env import CurveEnv

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

# plot the road
plt.plot(x, higher_y, 'r')
plt.plot(x, lower_y, 'b')

x_states = []
y_states = []
env = CurveEnv()
pos_x, pos_y, vel_x, vel_y, angular_vel = env.reset()
x_states.append(pos_x)
y_states.append(pos_y)
for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, done, extra = env.step(action)
    pos_x, pos_y, vel_x, vel_y, angular_vel = observation
    x_states.append(pos_x)
    y_states.append(pos_y)

# plot the car trajectory
plt.plot(x_states, y_states, 'g--')
plt.show()
