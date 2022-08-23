import numpy as np
from islaMcts.agents.abstract_mcts import AbstractMcts
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
import gym_goddard.envs.goddard_env as env


class DefaultControlled(env.Default):

    def dv(self, v, h):
        return 2.0 * self.DC * abs(v) * np.exp(-self.HC * (h - self.H0) / self.H0)

    def dh(self, v, h):
        return -self.HC / self.H0 * self.drag(v, h)

    def dvdv(self, v, h):
        return 2.0 * np.sign(v) * self.DC * np.exp(-self.HC * (h - self.H0) / self.H0)

    def dvdh(self, v, h):
        return -self.HC / self.H0 * self.dv(v, h)

    def dgdh(self, h):
        return -2.0 * self.G0 * self.H0 ** 2 / h ** 3


class OptimalController(object):
    '''
        An optimal controller that solves the continuous time Goddard rocket problem according to
        Pontryagin's maximum principle.
    '''

    def __init__(self, rocket):
        self._r = rocket
        self._trig = False
        self._prev_sing_traj = None
        self.EPS = np.finfo(float).eps

    def Dtilde(self, v, h):
        return self._r.dv(v, h) + self._r.GAMMA * self._r.drag(v, h)

    def Dtilde_dv(self, v, h):
        return self._r.dvdv(v, h) + self._r.GAMMA * self._r.dv(v, h)

    def Dtilde_dh(self, v, h):
        return self._r.dvdh(v, h) + self._r.GAMMA * self._r.dh(v, h)

    def control(self, v, h, m):
        D = self._r.drag(v, h)
        Dtilde = self.Dtilde(v, h)

        if self._trig:
            # singular trajectory
            gdt = self._r.GAMMA * Dtilde
            numerator = self._r.dh(v, h) - gdt * self._r.g(h) - v * self.Dtilde_dh(v, h) + m * self._r.dgdh(h)
            u = m * self._r.g(h) + D + m * (numerator / (gdt + self.Dtilde_dv(v, h) + self.EPS))
        else:
            # detect singular trajectory condition, i.e. == 0 or crosses 0 between samples
            sing_traj = v * Dtilde - (D + m * self._r.g(h))
            self._trig = self._prev_sing_traj is not None and (sing_traj * self._prev_sing_traj <= 0.0)
            self._prev_sing_traj = sing_traj
            u = self._r.THRUST_MAX

        return max(0.0, min(1.0, u / self._r.THRUST_MAX))


class OptimalAgent(AbstractMcts):
    def __init__(self, param: MctsParameters):
        super().__init__(param)
        rocket = DefaultControlled()
        self.oc = OptimalController(rocket)
        self.param = param

    def fit(self) -> int | np.ndarray:
        v, h, m = self.param.root_data
        return [self.oc.control(v, h, m)]
