import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from islaMcts.environments.curve_env import CurveEnv


def get_figure():
    x = np.linspace(-12, 12, 500)
    # the function
    higher_y = [CurveEnv.higher_bound(n) for n in x]
    lower_y = [CurveEnv.lower_bound(n) for n in x]
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.axis([-12, 12, 20, 33])
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(x, higher_y, 'r')
    plt.plot(x, lower_y, 'b')
    return fig, ax


def plot_simulation_trajectory(points_x: list, points_y: list):
    fig, ax = get_figure()
    for i in range(len(points_x)):
        x_states = points_x[i]
        y_states = points_y[i]
        ax.plot(x_states, y_states, 'go', linestyle="-")
    return fig


def plot_final_trajectory(x_states, y_states):
    fig, ax = get_figure()
    ax.plot(x_states, y_states, 'go', linestyle="--")
    return fig


def plot_distribution_of_rewards(points_y: list):
    fig = plt.figure()
    plt.hist(points_y, bins='auto')
    plt.grid()
    return fig