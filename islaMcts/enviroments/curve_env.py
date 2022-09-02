# 10 se arriva al goal
# -0.3 per ogni passo
# -10 se va fuori

# x curve va da 0 a 20
# y curve va da 0 a 20

# f(x)=7+log(2,x-0.1808569375707)+0.9825372164779
# g(x)=7+log(2,((1)/(4)) (x-4.5852507647728))+0.1361611060361
import math
from math import sin, cos, log
from typing import Optional

import gym
import numpy as np
from numpy import ndarray
from scipy.spatial import distance


class Car:
    def __init__(self, x, y, angle=0):
        self.max_angle = 180
        self.min_angle = 0
        self.position = np.array([x, y]).astype(float)
        # velocity +5 a -5
        # angle -45 a 45
        self.velocity = 5
        self.angle = angle
        self.max_velocity = 10.0
        self.min_velocity = 0.0

        # Position x, Position y, velocity, angle
        self.state = np.array([*self.position, self.velocity, self.angle])

    def update(self, velocity, angle, dt):
        # update velocity
        self.velocity += velocity
        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity
        if self.velocity < 0:
            self.velocity = 0

        # update angle
        self.angle += angle
        if self.angle > self.max_angle:
            self.angle = self.max_angle
        if self.angle < 0:
            self.angle = 0

        vel_x = cos(math.radians(self.angle)) * self.velocity
        vel_y = sin(math.radians(self.angle)) * self.velocity

        self.position[0] += vel_x
        self.position[1] += vel_y

        self.state = np.array([*self.position, self.velocity, self.angle])


class CurveEnv(gym.Env):
    def __init__(self):
        self.n_step = 0
        self.car = None
        self.dt = 1
        self.target_x = 30
        self.reset()
        # Position x, Position y, Velocity x, Velocity y, angular velocity
        self.observation_space = gym.spaces.Box(
            low=np.array((0, 0, 0, 0)),
            high=np.array((32, 32, 150, 360)),
            shape=(4,),
            dtype=float
        )
        # acceleration, steering
        # acceleration 0 <-> 5
        # steering -30 <-> 30
        self.action_space = gym.spaces.Box(
            low=np.array([-25, -30]),
            high=np.array([25, 30]),
            shape=(2,),
        )

    @staticmethod
    def higher_bound(x):
        try:
            # f(x) = 7 + log(2, x - 0.1808569375707) + 0.9825372164779
            return 7 + log(x - 0.1808569375707, 2) + 1.4825372164779
        except ValueError:
            return 0

    @staticmethod
    def lower_bound(x):
        try:
            # g(x) = 7 + log(2, ((1) / (4))(x - 4.5852507647728)) + 0.1361611060361
            return 7 + log((1 / 4) * (x - 4.5852507647728), 2) + 0.6361611060361
        except ValueError:
            return 0

    def step(self, action: np.ndarray) -> tuple[ndarray, int, bool, None]:
        self.n_step += 1
        velocity = action[0]
        angle = action[1]
        self.car.update(velocity, angle, self.dt)
        x_car, y_car = self.car.position
        WINNING_REWARD = 100
        LIVING_PENALTY = -0.3
        LOSING_REWARD = -1000
        MAX_DISTANCE = 30.384864653310537

        if x_car >= self.target_x:
            terminal = True
            reward = WINNING_REWARD / self.n_step
        else:
            # Verify if the car is in the curve
            # lower = g(x_car)
            # higher = f(x_car)
            # inside = lower <= y_car <= higher
            higher_y = self.higher_bound(x_car)
            lower_y = self.lower_bound(x_car)
            inside = lower_y <= y_car <= higher_y
            if not inside:
                terminal = True
                reward = LOSING_REWARD
            else:
                terminal = False
                reward = 1 - (distance.euclidean([x_car, y_car], [30, 12]) / MAX_DISTANCE)
                # reward = LIVING_PENALTY

        return self.car.state, reward, terminal, None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> ndarray:
        self.car = Car(2, 0.1)
        return self.car.state
