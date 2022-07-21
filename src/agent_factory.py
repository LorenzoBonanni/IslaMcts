from collections.abc import Hashable
from ctypes import Union

from src.agents.abstract_mcts import AbstractMcts
from src.agents.dpw_parameters import DpwParameters
from src.agents.mcts import Mcts
from src.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from src.agents.mcts_double_progressive_widening_hash import MctsDoubleProgressiveWideningHash
from src.agents.mcts_hash import MctsHash
from src.agents.mcts_parameters import MctsParameters
from src.agents.mcts_state_progressive_widening_hash import MctsStateProgressiveWideningHash
from src.agents.pw_parameters import PwParameters
from src.agents.random_agent import RandomAgent


def get_agent(agent_type: str, params: MctsParameters | PwParameters | DpwParameters) -> AbstractMcts:
    hashable = True
    if not isinstance(params.root_data, Hashable):
        hashable = False

    if agent_type == "vanilla":
        if hashable:
            return Mcts(params)
        else:
            return MctsHash(params)
    elif agent_type == "spw":
        if not hashable:
            return MctsStateProgressiveWideningHash(params)
    elif agent_type == "apw":
        if not hashable:
            return MctsActionProgressiveWideningHash(params)
    elif agent_type == "dpw":
        if not hashable:
            return MctsDoubleProgressiveWideningHash(params)
    elif agent_type == "random":
        return RandomAgent(params)
