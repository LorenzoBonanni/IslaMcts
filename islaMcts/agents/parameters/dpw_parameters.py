from dataclasses import dataclass

from islaMcts.agents.parameters.mcts_parameters import MctsParameters


@dataclass(slots=True)
class DpwParameters(MctsParameters):
    # Action Progressive Widening Parameters
    alphaApw: float
    kApw: int
    # State Progressive Widening Parameters
    alphaSpw: float
    kSpw: int
