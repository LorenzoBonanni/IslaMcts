from src.agents.abstract_mcts import AbstractMcts


class RandomAgent(AbstractMcts):
    def fit(self) -> int:
        return self.param.env.action_space.sample()