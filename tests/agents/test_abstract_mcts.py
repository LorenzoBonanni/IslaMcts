from unittest import TestCase
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest as pytest

from islaMcts.agents.abstract_mcts import AbstractMcts, AbstractActionNode, AbstractStateNode
from islaMcts.agents.parameters.mcts_parameters import MctsParameters


class TestMcts(AbstractMcts):
    def fit(self) -> int | np.ndarray:
        pass


class TestActionNode(AbstractActionNode):

    def build_tree(self, max_depth: int) -> float:
        pass


class TestStateNode(AbstractStateNode):
    def build_tree(self, max_depth: int):
        pass


class TestEnv:
    pass


class TestAbstractActionNode(TestCase):
    def test_q_value_division_zero(self):
        node = TestActionNode(None, None)
        node.total = 10
        with self.assertRaises(ZeroDivisionError):
            result = node.q_value

    def test_q_value(self):
        node = TestActionNode(None, None)
        node.total = 10
        node.na = 2
        result = node.q_value

        assert result == 5


class TestAbstractStateNode:
    def test_rollout_max_depth_minus_1(self):
        # GIVEN
        test_param = MctsParameters(
            root_data=1,
            env=None,
            n_sim=None,
            C=None,
            action_selection_fn=None,
            gamma=None,
            rollout_selection_fn=None,
            state_variable=None,
            max_depth=None,
            n_actions=None
        )
        node = TestStateNode(
            data=1,
            param=test_param
        )
        # WHEN
        result = node.rollout(-1)

        # THEN
        assert result == 0

    @pytest.mark.parametrize("test_input", [(2, 5, False, None), (2, 5, True, None)])
    def test_rollout_max_depth_1(self, test_input):
        # GIVEN
        mock_rollout_fn = Mock()
        mock_rollout_fn.return_value = 1
        test_env = TestEnv()
        test_env.step = MagicMock(return_value=test_input)
        test_param = MctsParameters(
            root_data=1,
            env=test_env,
            n_sim=None,
            C=None,
            action_selection_fn=None,
            gamma=None,
            rollout_selection_fn=mock_rollout_fn,
            state_variable=None,
            max_depth=None,
            n_actions=None
        )
        node = TestStateNode(
            data=1,
            param=test_param
        )

        # WHEN
        result = node.rollout(1)

        # THEN
        assert result == 5

    def test_rollout_max_depth_3_terminal_true(self):
        # GIVEN
        mock_rollout_fn = Mock()
        mock_rollout_fn.return_value = 1
        test_env = TestEnv()
        test_env.step = MagicMock()
        test_env.step.side_effect = [(2, 5, False, None), (2, 10, True, None), (2, 15, False, None)]
        test_param = MctsParameters(
            root_data=1,
            env=test_env,
            n_sim=None,
            C=None,
            action_selection_fn=None,
            gamma=None,
            rollout_selection_fn=mock_rollout_fn,
            state_variable=None,
            max_depth=None,
            n_actions=None
        )
        node = TestStateNode(
            data=1,
            param=test_param
        )

        # WHEN
        result = node.rollout(3)

        # THEN
        assert result == 10

    def test_rollout_max_depth_3_terminal_false(self):
        # GIVEN
        mock_rollout_fn = Mock()
        mock_rollout_fn.return_value = 1
        test_env = TestEnv()
        test_env.step = MagicMock()
        test_env.step.side_effect = [(2, 5, False, None), (2, 10, False, None), (2, 15, False, None)]
        test_param = MctsParameters(
            root_data=1,
            env=test_env,
            n_sim=None,
            C=None,
            action_selection_fn=None,
            gamma=None,
            rollout_selection_fn=mock_rollout_fn,
            state_variable=None,
            max_depth=None,
            n_actions=None
        )
        node = TestStateNode(
            data=1,
            param=test_param
        )

        # WHEN
        result = node.rollout(3)

        # THEN
        assert result == 15
