from unittest import TestCase, mock
from unittest.mock import Mock, patch, MagicMock

import numpy as np
from parametrize import parametrize

import islaMcts.utils as utils
from islaMcts.agents.mcts_hash import MctsHash, StateNodeHash, ActionNodeHash
from islaMcts.agents.parameters.mcts_parameters import MctsParameters


class TestMctsHash(TestCase):

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
        self.test_class = MctsHash(self.test_param)
        acs = {
            1: Mock(q_value=1, data=1),
            3: Mock(q_value=3, data=3),
            4: Mock(q_value=4, data=4),
            2: Mock(q_value=3, data=2)
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
        self.assertEqual(4, result)

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

        self.assertIn(result, [2, 3])


class TestStateNodeHash(TestCase):
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
        self.test_class = StateNodeHash(None, self.test_param)

    @patch.object(ActionNodeHash, 'build_tree')
    def test_build_tree_zero_in_visit_action(self, mocked_build_tree):
        # GIVEN
        mocked_build_tree.return_value = 5

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(0, list(self.test_class.actions.keys())[0])
        self.assertEqual(1, self.test_class.ns)
        self.assertEqual(1, self.test_class.visit_actions[0])
        self.assertEqual(4.5, self.test_class.total)
        self.assertEqual(0, self.test_class.actions[0].data)
        self.assertEqual(5, result)

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
        self.assertEqual(1, self.test_class.ns)
        self.assertEqual(2, self.test_class.visit_actions[1])
        self.assertEqual(4.5, self.test_class.total)
        self.assertEqual(5, result)


class TestActionNodeHash(TestCase):
    def setUp(self) -> None:
        self.test_param = MctsParameters(
            root_data=None,
            env=Mock(),
            n_sim=None,
            C=None,
            action_selection_fn=None,
            gamma=0.9,
            rollout_selection_fn=None,
            state_variable=None,
            max_depth=None,
            n_actions=1
        )
        self.test_class = ActionNodeHash(None, self.test_param)

    # 4 cases
    # Case 1 - Terminal State, State None
    # Case 2 - Terminal State, State Not None
    # Case 3 - Non Terminal State, State None
    # Case 4 - Non Terminal State, State Not None
    def test_build_tree_terminal_state_none(self):
        # GIVEN
        # return values: (observation, instant_reward, terminal, _)
        observation = np.array(1)
        observation_hash = b'\x01\x00\x00\x00\x00\x00\x00\x00'
        self.test_param.env.step = MagicMock(return_value=(observation, 5, True, None))

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(observation_hash, list(self.test_class.children.keys())[0])
        self.assertEqual(observation, self.test_class.children[observation_hash].data)
        self.assertTrue(self.test_class.children[observation_hash].terminal)
        self.assertEqual(1, self.test_class.children[observation_hash].ns)
        self.assertEqual(5, self.test_class.total)
        self.assertEqual(1, self.test_class.na)
        self.assertEqual(5, result)

    def test_build_tree_terminal_state_not_none(self):
        # GIVEN
        observation = np.array(1)
        observation_hash = b'\x01\x00\x00\x00\x00\x00\x00\x00'
        # return values: (observation, instant_reward, terminal, _)
        self.test_param.env.step = MagicMock(return_value=(observation, 5, True, None))
        mock_child = Mock(data=observation, param=self.test_param, ns=0)
        self.test_class.children[observation_hash] = mock_child

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(1, mock_child.ns)
        self.assertEqual(5, self.test_class.total)
        self.assertEqual(1, self.test_class.na)
        self.assertEqual(5, result)

    @patch.object(StateNodeHash, 'rollout')
    def test_build_tree_not_terminal_state_none(self, mocked_rollout):
        # GIVEN
        observation = np.array(1)
        observation_hash = b'\x01\x00\x00\x00\x00\x00\x00\x00'
        # return values: (observation, instant_reward, terminal, _)
        self.test_param.env.step = MagicMock(return_value=(observation, 5, False, None))
        mocked_rollout.return_value = 5

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(observation_hash, list(self.test_class.children.keys())[0])
        self.assertEqual(observation, self.test_class.children[observation_hash].data)
        self.assertFalse(self.test_class.children[observation_hash].terminal)
        self.assertEqual(1, self.test_class.children[observation_hash].ns)
        self.assertEqual(9.5, self.test_class.total)
        self.assertEqual(1, self.test_class.na)
        self.assertEqual(9.5, result)

    def test_build_tree_not_terminal_state_not_none(self):
        # GIVEN
        observation = np.array(1)
        observation_hash = b'\x01\x00\x00\x00\x00\x00\x00\x00'
        # return values: (observation, instant_reward, terminal, _)
        self.test_param.env.step = MagicMock(return_value=(observation, 5, False, None))
        mock_child = Mock(data=observation, param=self.test_param, ns=0)
        mock_child.build_tree = MagicMock(return_value=5)
        self.test_class.children[observation_hash] = mock_child

        # WHEN
        result = self.test_class.build_tree(10)

        # THEN
        self.assertEqual(9.5, self.test_class.total)
        self.assertEqual(1, self.test_class.na)
        self.assertEqual(9.5, result)
