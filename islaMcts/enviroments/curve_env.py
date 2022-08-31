# 10 se arriva al goal
# -0.3 per ogni passo
# -10 se va fuori

# x curve va da 0 a 20
# y curve va da 0 a 20

# f(x)=7+log(2,x-0.1808569375707)+0.9825372164779
# g(x)=7+log(2,((1)/(4)) (x-4.5852507647728))+0.1361611060361


import os
from typing import Optional, Union, List, Tuple
from math import sin, radians, degrees, copysign, cos, log
import gym
import numpy as np
import pygame
from gym.core import RenderFrame, ActType, ObsType
from numpy import ndarray


class Car:
    def __init__(self, x, y, angle=0.0, length=2.0, max_steering=30.0, max_acceleration=5.0):
        self.angular_velocity = 0.0
        self.position = np.array([x, y]).astype(float)
        self.velocity = np.array([0.0, 0.0])
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 10.0

        # Position x, Position y, Velocity x, Velocity y, angular_velocity
        self.state = np.array([*self.position, *self.velocity, self.angular_velocity])

        self.acceleration = 0.0
        self.steering = 0.0

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
        self.velocity[0] = max(-self.max_velocity, min(self.velocity[0], self.max_velocity))

        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            self.angular_velocity = self.velocity[0] / turning_radius
        else:
            self.angular_velocity = 0

        theta = np.deg2rad(-self.angle)
        # rotation matrix
        # https://www.atqed.com/numpy-rotate-vector
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        self.position += np.dot(rot, self.velocity) * dt
        self.angle += degrees(self.angular_velocity) * dt

        self.state = np.array([*self.position, *self.velocity, self.angular_velocity])


class CurveEnv(gym.Env):
    def __init__(self):
        self.car = None
        self.dt = 0.05
        self.max_y = 30
        self.reset()
        # Position x, Position y, Velocity x, Velocity y, angular velocity
        self.observation_space = gym.spaces.Box(
            low=np.array((0, 0, -10, -10, -2.5)),
            high=np.array((32, 32, 10, 10, 2.5)),
            shape=(5,),
            dtype=float
        )
        # acceleration, steering
        # acceleration 0 <-> 5
        # steering -30 <-> 30
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1 * self.car.max_steering]),
            high=np.array([self.car.max_acceleration, self.car.max_steering]),
            shape=(2,),
        )
        # rendering stuff
        # self.car_image = None
        # self.screen = None
        # self.setup_render()

    def setup_render(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1280
        height = 720
        self.screen = pygame.display.set_mode((width, height))

        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        self.car_image = pygame.image.load(image_path)

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
        acceleration = action[0]
        steering = action[1]
        self.car.acceleration = acceleration
        self.car.steering = steering
        self.car.update(self.dt)
        x_car, y_car = self.car.position

        if y_car >= self.max_y:
            terminal = True
            reward = 10
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
                reward = -10
            else:
                terminal = False
                reward = -0.3

        return self.car.state, reward, terminal, None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ndarray:
        self.car = Car(2, 0.2)
        return self.car.state

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:

        # ppu = 32
        # self.screen.fill((0, 0, 0))
        # rotated = pygame.transform.rotate(self.car_image, self.car.angle)
        # rect = rotated.get_rect()
        # self.screen.blit(rotated, self.car.position * ppu - (rect.width / 2, rect.height / 2))
        # pygame.display.flip()
        pass