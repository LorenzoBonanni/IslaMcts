from unittest import TestCase, mock
from unittest.mock import Mock

from parametrize import parametrize

import islaMcts.utils as utils
from islaMcts.agents.mcts import Mcts
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
        self.assertEqual(my_deepcopy_mock.call_count, n_sim+1)
        self.assertEqual(result, 4)

    @mock.patch("builtins.max")
    def test_fit(self, mock_max):
        # GIVEN
        self.test_param.n_sim = -1
        mock_max.return_value = 3

        # WHEN
        result = self.test_class.fit()

        self.assertIn(result, [2, 3])
