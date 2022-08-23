from collections.abc import Hashable

from islaMcts.agents.abstract_mcts import AbstractMcts
from islaMcts.agents.optimal_agent import OptimalAgent
from islaMcts.agents.parameters.dpw_parameters import DpwParameters
from islaMcts.agents.mcts import Mcts
from islaMcts.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash
from islaMcts.agents.mcts_double_progressive_widening_hash import MctsDoubleProgressiveWideningHash
from islaMcts.agents.mcts_hash import MctsHash
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
from islaMcts.agents.mcts_state_progressive_widening_hash import MctsStateProgressiveWideningHash
from islaMcts.agents.parameters.pw_parameters import PwParameters
from islaMcts.agents.random_agent import RandomAgent


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
        else:
            raise NotImplementedError
    elif agent_type == "apw":
        if not hashable:
            return MctsActionProgressiveWideningHash(params)
        else:
            raise NotImplementedError
    elif agent_type == "dpw":
        if not hashable:
            return MctsDoubleProgressiveWideningHash(params)
        else:
            raise NotImplementedError
    elif agent_type == "random":
        return RandomAgent(params)
    elif agent_type == "optimal_goddard":
        return OptimalAgent(params)
    else:
        raise NotImplementedError
    