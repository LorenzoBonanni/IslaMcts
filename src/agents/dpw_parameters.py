from dataclasses import dataclass

from src.agents.mcts_parameters import MctsParameters


@dataclass
class DpwParameters(MctsParameters):
    # Action Progressive Widening Parameters
    alphaApw: float
    kApw: int
    # State Progressive Widening Parameters
    alphaSpw: float
    kSpw: int
