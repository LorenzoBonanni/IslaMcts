from dataclasses import dataclass

from src.agents.mcts_parameters import MctsParameters


@dataclass
class PwParameters(MctsParameters):
    alpha: float
    k: int
