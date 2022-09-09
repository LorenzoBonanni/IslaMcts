import numpy as np
from matplotlib import pyplot as plt

from islaMcts.enviroments.curve_env import CurveEnv

plt.ion()
plt.show()
x = np.linspace(-1, 32, 100)


def plot_trajectory():
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
    plt.pause(0.5)
    return fig


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

total_reward = 0
n_steps = 0
print(f"X: {pos_x}, Y:{pos_y}, vel:{vel}, angle:{angle}")
fig = None
plot_trajectory()
while not done:
    acceleration = float(input("Acceleration? [-5, +5] "))
    input_angle = float(input("Angle? [-45, +45] "))

    observation, reward, done, extra = env.step(np.array([acceleration, input_angle]))
    pos_x, pos_y, vel, angle = observation
    print(f"X: {pos_x}, Y:{pos_y}, vel:{vel}, angle:{angle}")
    total_reward += reward
    x_states.append(pos_x)
    y_states.append(pos_y)
    fig = plot_trajectory()
fig.savefig('trajectory.png')
print(f"REWARD: {total_reward}")
