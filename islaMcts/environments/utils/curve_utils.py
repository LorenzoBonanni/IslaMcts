import numpy as np
from matplotlib import pyplot as plt

from islaMcts.agents.abstract_mcts import AbstractStateNode
from islaMcts.enviroments.curve_env import CurveEnv


def get_figure():
    x = np.linspace(-1, 32, 100)
    # the function
    higher_y = [CurveEnv.higher_bound(n) for n in x]
    lower_y = [CurveEnv.lower_bound(n) for n in x]
    plt.clf()
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
    return fig, ax


def plot_simulation_trajectory(root: AbstractStateNode):
    # TODO rewrite method
    pass
    fig, ax = get_figure()
    # for i in range(len(params.x_values)):
    #     x_val = params.x_values[i]
    #     y_val = params.y_values[i]
    #     ax.plot(x_val, y_val, 'g--')
    # return fig


def plot_final_trajectory(x_states, y_states):
    fig, ax = get_figure()
    ax.plot(x_states, y_states, 'go',  linestyle="--")
    return fig