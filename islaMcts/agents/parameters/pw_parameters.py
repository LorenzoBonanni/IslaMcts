from dataclasses import dataclass
from typing import Callable

from islaMcts.agents.parameters.mcts_parameters import MctsParameters


@dataclass(slots=True)
class PwParameters(MctsParameters):
    alpha: float
    k: int
    action_expansion_function: Callable
