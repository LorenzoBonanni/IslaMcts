import numpy as np


def ucb1(total: int, C: float, visit_child: list, n_visits: int):
    avg_reward = np.array(total)/len(visit_child)
    result = avg_reward + C*np.sqrt(np.log(n_visits)/np.array(visit_child))
    a = np.argmax(result)
    return a
