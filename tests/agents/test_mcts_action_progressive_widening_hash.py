from unittest import TestCase, mock
from unittest.mock import Mock, patch, MagicMock

import numpy as np
from parametrize import parametrize

from islaMcts import utils
from islaMcts.agents.mcts_action_progressive_widening_hash import MctsActionProgressiveWideningHash, \
    ActionNodeProgressiveWideningHash, StateNodeProgressiveWideningHash
from islaMcts.agents.parameters.pw_parameters import PwParameters


class TestMctsActionProgressiveWideningHash(TestCase):
    def setUp(self):
        self.test_param = PwParameters(
            root_data=None,
            env=None,
            n_sim=None,
            C=None,
            action_selection_fn=None,
            gamma=None,
            rollout_selection_fn=None,
            state_variable=None,
            max_depth=None,
            n_actions=None,
            alpha=None,
            k=None
        )
        self.test_class = MctsActionProgressiveWideningHash(self.test_param)
        acs = {
            b'\x01\x00\x00\x00\x00\x00\x00\x00': Mock(q_value=1, data=np.array(1)),
            b'\x03\x00\x00\x00\x00\x00\x00\x00': Mock(q_value=3, data=np.array(3)),
            b'\x04\x00\x00\x00\x00\x00\x00\x00': Mock(q_value=4, data=np.array(4)),
            b'\x02\x00\x00\x00\x00\x00\x00\x00': Mock(q_value=3, data=np.array(2))
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
        self.assertEqual(np.array(4), result)

    @parametrize("n_sim", [0, 1, 2, 5, 10, 100])
    def test_fit_variable_nsim(self, n_sim):
        # GIVEN
        self.test_param.n_sim = n_sim
        my_deepcopy_mock = utils.my_deepcopy = Mock(return_value="mocked stuff")
        self.test_class.root.build_tree = Mock(return_value="mocked stuff")

        # WHEN
        result = self.test_class.fit()

        # THEN
        self.assertEqual(my_deepcopy_mock.call_count, n_sim + 1)
        self.assertEqual(np.array(4), result)

    @mock.patch("builtins.max")
    def test_fit(self, mock_max):
        # GIVEN
        self.test_param.n_sim = -1
        mock_max.return_value = 3

        # WHEN
        result = self.test_class.fit()

        self.assertIn(result, [np.array(2), np.array(3)])


class TestStateNodeProgressiveWideningHash(TestCase):
    def setUp(self) -> None:
        self.test_param = PwParameters(
            root_data=None,
            env=None,
            n_sim=None,
            C=None,
            action_selection_fn=None,
            gamma=0.9,
            rollout_selection_fn=None,
            state_variable=None,
            max_depth=None,
            n_actions=None,
            alpha=None,
            k=None
        )
        self.test_class = StateNodeProgressiveWideningHash(None, self.test_param)

    # TODO 4 tests
    # 1. len actions 0 child None
    # 2. len actions <= k*ns^alpha child None
    # 3. len actions <= k*ns^alpha child not None
    # 4. len actions >= k*ns^alpha
    @patch.object(ActionNodeProgressiveWideningHash, 'build_tree')
    def test_build_tree_zero_len_actions_child_none(self, mocked_build_tree):
        # GIVEN
        action = np.array(2)
        action_bytes = b'\x02\x00\x00\x00\x00\x00\x00\x00'
        self.test_param.env = Mock()
        self.test_param.env.action_space.sample = MagicMock(return_value=action)
        mocked_build_tree.return_value = 5

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(action_bytes, list(self.test_class.actions.keys())[0])
        self.assertEqual(1, self.test_class.ns)
        self.assertEqual(1, self.test_class.visit_actions[action_bytes])
        self.assertEqual(5, self.test_class.total)
        self.assertEqual(action, self.test_class.actions[action_bytes].data)
        self.assertEqual(5, result)

    @patch.object(ActionNodeProgressiveWideningHash, 'build_tree')
    def test_build_tree_widening_child_none(self, mocked_build_tree):
        # GIVEN
        action = np.array(2)
        action_bytes = b'\x02\x00\x00\x00\x00\x00\x00\x00'
        self.test_param.env = Mock()
        self.test_param.env.action_space.sample = MagicMock(return_value=action)
        mocked_build_tree.return_value = 5
        self.test_param.alpha = 0
        self.test_param.k = 3
        self.test_class.ns = 1

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(action_bytes, list(self.test_class.actions.keys())[0])
        self.assertEqual(2, self.test_class.ns)
        self.assertEqual(1, self.test_class.visit_actions[action_bytes])
        self.assertEqual(5, self.test_class.total)
        self.assertEqual(action, self.test_class.actions[action_bytes].data)
        self.assertEqual(5, result)
