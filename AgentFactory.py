from collections.abc import Hashable

from agents.mcts import Mcts
from agents.mcts_continuous import MctsHashStateContinuous
from agents.mcts_hash import MctsHash


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
                # return MctsHashStateContinuous(*kwargs)
                # TODO: return MctsContinuous Normal
                pass
            else:
                return MctsHashStateContinuous(*kwargs)
