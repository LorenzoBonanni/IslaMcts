from collections.abc import Hashable

from agents.mcts import Mcts
from agents.mcts_continuous import MctsContinuous
from agents.mcts_continuous_hash import MctsContinuousHash
from agents.mcts_hash import MctsHash
from agents.mcts_state_progressive_widening import MctsStateProgressiveWidening
from agents.mcts_state_progressive_widening_hash import MctsStateProgressiveWideningHash


class AgentFactory:
    @staticmethod
    def get_agent(agent_type, **kwargs):
        hashable = True
        if not isinstance(kwargs["root_data"], Hashable):
            hashable = False

        if agent_type == "vanilla":
            if hashable:
                return Mcts(*kwargs)
            else:
                return MctsHash(*kwargs)
        elif agent_type == "continuous":
            if hashable:
                return MctsContinuous(*kwargs)
            else:
                return MctsContinuousHash(*kwargs)
        elif agent_type == "state_pw":
            if hashable:
                return MctsStateProgressiveWidening(*kwargs)
            else:
                return MctsStateProgressiveWideningHash(*kwargs)
