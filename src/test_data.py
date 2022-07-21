from dataclasses import dataclass

from src.agents.dpw_parameters import DpwParameters
from src.agents.mcts_parameters import MctsParameters
from src.agents.pw_parameters import PwParameters


@dataclass
class TestData:
    agent_type: str
    param: MctsParameters | PwParameters | DpwParameters
    continuous: bool
    noise: bool
