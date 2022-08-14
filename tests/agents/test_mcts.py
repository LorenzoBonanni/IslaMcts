from unittest import TestCase, mock
from unittest.mock import Mock, patch, MagicMock

import numpy as np
from parametrize import parametrize

import islaMcts.utils as utils
from islaMcts.agents.mcts import Mcts, StateNode, ActionNode
from islaMcts.agents.parameters.mcts_parameters import MctsParameters


class TestMcts(TestCase):

    def setUp(self):
        self.test_param = MctsParameters(
            root_data=None,
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
        self.test_class = Mcts(self.test_param)
        acs = {
            1: Mock(q_value=1, data=1),
            2: Mock(q_value=3, data=3),
            4: Mock(q_value=4, data=4),
            3: Mock(q_value=3, data=2)
        }
        self.test_class.root = Mock(actions=acs)

    def test_fit_nsim_minus_one(self):
        # GIVEN
        self.test_param.n_sim = -1
        my_deepcopy_mock = utils.my_deepcopy = Mock(return_value="mocked stuff")

        # WHEN
        result = self.test_class.fit()

        # THEN
        my_deepcopy_mock.assert_called_once()
        self.assertEqual(my_deepcopy_mock.call_count, 1)
        self.assertEqual(result, 4)

    @parametrize("n_sim", [1, 2, 5, 10, 100])
    def test_fit_variable_nsim(self, n_sim):
        # GIVEN
        self.test_param.n_sim = n_sim
        my_deepcopy_mock = utils.my_deepcopy = Mock(return_value="mocked stuff")
        self.test_class.root.build_tree = Mock(return_value="mocked stuff")

        # WHEN
        result = self.test_class.fit()

        # THEN
        self.assertEqual(my_deepcopy_mock.call_count, n_sim + 1)
        self.assertEqual(result, 4)

    @mock.patch("builtins.max")
    def test_fit(self, mock_max):
        # GIVEN
        self.test_param.n_sim = -1
        mock_max.return_value = 3

        # WHEN
        result = self.test_class.fit()

        self.assertIn(result, [2, 3])


class TestStateNode(TestCase):
    def setUp(self) -> None:
        self.test_param = MctsParameters(
            root_data=None,
            env=None,
            n_sim=None,
            C=None,
            action_selection_fn=None,
            gamma=0.9,
            rollout_selection_fn=None,
            state_variable=None,
            max_depth=None,
            n_actions=1
        )
        self.test_class = StateNode(None, self.test_param)

    @patch.object(ActionNode, 'build_tree')
    def test_build_tree_zero_in_visit_action(self, mocked_build_tree):
        # GIVEN
        mocked_build_tree.return_value = 5

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(list(self.test_class.actions.keys())[0], 0)
        self.assertEqual(self.test_class.ns, 1)
        self.assertEqual(self.test_class.visit_actions[0], 1)
        self.assertEqual(self.test_class.total, 4.5)
        self.assertEqual(self.test_class.actions[0].data, 0)
        self.assertEqual(result, 5)

    def test_build_tree_non_zero_in_visit_action(self):
        # GIVEN
        self.test_class.visit_actions = np.array([1, 1])
        child = Mock(q_value=1, data=1)
        acs = {
            1: child
        }
        self.test_class.actions = acs
        self.test_param.action_selection_fn = Mock(return_value=1)
        child.build_tree = MagicMock(return_value=5)

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(self.test_class.ns, 1)
        self.assertEqual(self.test_class.visit_actions[1], 2)
        self.assertEqual(self.test_class.total, 4.5)
        self.assertEqual(result, 5)
