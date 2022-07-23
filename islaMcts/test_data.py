from dataclasses import dataclass

from islaMcts.agents.parameters.dpw_parameters import DpwParameters
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
from islaMcts.agents.parameters.pw_parameters import PwParameters


@dataclass
class TestData:
    agent_type: str
    param: MctsParameters | PwParameters | DpwParameters
    continuous: bool
    noise: bool
