# 10 se arriva al goal
# -0.3 per ogni passo
# -10 se va fuori
import copy
import itertools
# x curve va da 0 a 20
# y curve va da 0 a 20

# f(x)=7+log(2,x-0.1808569375707)+0.9825372164779
# g(x)=7+log(2,((1)/(4)) (x-4.5852507647728))+0.1361611060361
import math
from math import sin, cos, log
from typing import Optional, Union, List, Iterable

import gym
import numpy as np
from gym.core import RenderFrame
from numpy import ndarray
from scipy.spatial import distance


class Car:
    def __init__(self, x, y, angle=90):
        self.max_angle = 180
        self.min_angle = 0
        self.position = np.array([x, y]).astype(float)
        self.velocity = 5
        self.angle = angle
        self.max_velocity = 10.0
        self.min_velocity = 0.0

        # Position x, Position y, velocity, angle
        self.state = np.array([*self.position, self.velocity, self.angle])

    def update(self, acceleration, angle, dt):
        # update velocity
        self.velocity += acceleration
        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity
        if self.velocity < self.min_velocity:
            self.velocity = 0

        # update angle
        self.angle += angle
        if self.angle > self.max_angle:
            self.angle = self.max_angle
        if self.angle < self.min_angle:
            self.angle = 0

        vel_x = cos(math.radians(self.angle)) * self.velocity
        vel_y = sin(math.radians(self.angle)) * self.velocity

        self.position[0] += vel_x
        self.position[1] += vel_y

        self.state = np.array([*self.position, self.velocity, self.angle])


class CurveEnv(gym.Env):
    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def __init__(self):
        self.n_step = 0
        self.car = None
        self.dt = 1
        self.target_x = 30
        self.reset()
        # Position x, Position y, Velocity , angle
        self.observation_space = gym.spaces.Box(
            low=np.array((0., 0., 0., 0.)),
            high=np.array((32., 32., 10., 180.)),
            shape=(4,),
            dtype=float
        )
        # acceleration, steering
        # acceleration -5 <-> 5
        # steering -45 <-> 45
        self.action_space = gym.spaces.Box(
            low=np.array([-5., -45.]),
            high=np.array([5., 45.]),
            shape=(2,),
            dtype=float
        )

    @staticmethod
    def higher_bound(x):
        # OTHER: f(x)=7+log(5,x-0.1808569375707)+15.9825372164779
        try:
            # f(x)=7+log(2,x-0.1808569375707)+15.4825372164779
            return 7 + log(x + 0.01, 2) + 15.4825372164779
        except ValueError:
            return 0

    @staticmethod
    def lower_bound(x):
        # OTHER: g(x)=7+log(5,((1)/(4)) (x-2.7069655165562))+14.7939060040875
        try:
            # g(x)=7+log(2,((1)/(4)) (x-0.7961772381708))+17.1094871667719
            return 7 + log((1 / 4) * (x - 1), 2) + 16.7
        except ValueError:
            return 0

    def step(self, action: np.ndarray) -> tuple[ndarray, int, bool, None]:
        self.n_step += 1
        acceleration = action[0]
        angle = action[1]
        last_pos = copy.deepcopy(self.car.position)
        self.car.update(acceleration, angle, self.dt)
        x_car, y_car = self.car.position
        WINNING_REWARD = 1000
        LOSING_REWARD = -1000
        MAX_DISTANCE = 38.22316051819891

        # Verify if the car is in the curve
        # inside = lower <= y_car <= higher
        last_x, last_y = last_pos

        N_POINTS = 100
        x_values_car = np.linspace(last_x, x_car, N_POINTS)
        y_values_car = np.linspace(last_y, y_car, N_POINTS)
        lower_y_values = np.array([self.lower_bound(val) for val in x_values_car])
        higher_y_values = np.array([self.higher_bound(val) for val in x_values_car])

        inside = (y_values_car >= lower_y_values).all() and (y_values_car <= higher_y_values).all()

        if not inside:
            terminal = True
            reward = LOSING_REWARD
        else:
            if x_car >= self.target_x:
                terminal = True
                reward = WINNING_REWARD / self.n_step
            else:
                terminal = False
                reward = (1 - (distance.euclidean([x_car, y_car], [30, 25]) / MAX_DISTANCE)) / self.n_step
                # reward = 1 - (distance.euclidean([x_car, y_car], [30, 25]) / MAX_DISTANCE)

        return self.car.state, reward, terminal, None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> ndarray:
        self.car = Car(0.5, 0.1)
        return self.car.state


class DiscreteCurveEnv(CurveEnv):
    def __init__(self, n_actions: list):
        super().__init__()
        low_as = self.action_space.low
        high_as = self.action_space.high
        min_acceleration, min_angle = low_as
        max_acceleration, max_angle = high_as
        acceleration = np.linspace(min_acceleration, max_acceleration, n_actions[0])
        angles = np.linspace(min_angle, max_angle, n_actions[1])
        self.actions = list(itertools.product(acceleration, angles))
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def step(self, discrete_action: int) -> tuple[ndarray, int, bool, None]:
        action = np.array(self.actions[discrete_action])
        return super().step(action)


# if __name__ == '__main__':
#     state = np.array((11.714410786279766, 24.898901576502166, 4, 54))
#     action = (5, -44)
#     env = CurveEnv()
#     env.reset()
#     env.car.state = state
#     env.car.position = state[:2]
#     observation, reward, done, extra = env.step(np.array(action))
#     print(
#         f"TERMINAL: {done}",
#         f"REWARD: {reward}"
#     )