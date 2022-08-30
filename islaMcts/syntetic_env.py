import random

import numpy as np
from typing import Optional, Union, List, Tuple, Any

import gym
from gym.core import RenderFrame, ActType, ObsType
from gym.spaces import Box, Discrete
from numpy import ndarray


class SyntheticEnv(gym.Env):
    def __init__(self):
        self.last_state = 100
        self.observation_space = Discrete(self.last_state)
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=float)
        self.state = None

    def step(self, action: ActType) -> tuple[ndarray, float, bool | Any, None]:
        act = action[0]
        self.state += 1
        # noise = random.choices([-0.0025, -0.002, -0.001, 0, 0.001, 0.002, 0.0025])[0]
        # noise = random.choices([-0.001, 0, 0.001])[0]
        optimal_action = 0.625 + 0
        reward_range = 0.005

        reward = 1 if optimal_action - reward_range <= act <= optimal_action + reward_range else 0
        reward = float(reward)
        terminal = self.state == self.last_state
        return np.array(self.state), reward, terminal, None

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        ObsType, Tuple[ObsType, dict]]:
        self.state = 0


class SyntheticEnvDiscrete(gym.Env):
    def __init__(self):
        self.last_state = 100
        self.observation_space = Discrete(self.last_state)
        self.action_space = Discrete(101)
        self.state = None

    def step(self, action: ActType) -> tuple[ndarray, float, bool | Any, None]:
        act = action * 0.01
        self.state += 1
        # noise = random.choices([-0.0025, -0.002, -0.001, 0, 0.001, 0.002, 0.0025])[0]
        optimal_action = 0.625
        reward_range = 0.005

        reward = 1 if optimal_action - reward_range <= act <= optimal_action + reward_range else 0
        reward = float(reward)
        terminal = self.state == self.last_state
        return np.array(self.state), reward, terminal, None

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        ObsType, Tuple[ObsType, dict]]:
        self.state = 0
