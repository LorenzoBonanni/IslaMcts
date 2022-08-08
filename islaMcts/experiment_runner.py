import dataclasses
import logging
import math

import gym
import numpy as np
import pandas as pd
from joblib import Parallel, parallel_backend, delayed

from islaMcts.action_selection_functions import continuous_default_policy, ucb1
from islaMcts.agent_factory import get_agent
from islaMcts.agents.parameters.dpw_parameters import DpwParameters
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
from islaMcts.agents.parameters.pw_parameters import PwParameters
from islaMcts.experiment_data import ExperimentData
from islaMcts.utils import my_deepcopy

time = np.arange(0, 0.4, 0.001)
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run(agent_type: str, params: MctsParameters | PwParameters | DpwParameters, noise: bool, number: int,
        real_env: gym.Env):
    observation = real_env.reset()

    state_log = [observation]
    extra_log = []
    total_reward = 0
    it = 0
    act_dist = []
    text = []

    for _ in time:
        params.env = my_deepcopy(real_env)
        params.root_data = observation
        agent = get_agent(
            agent_type=agent_type,
            params=params
        )

        action = agent.fit()

        # SAVE Q_VALUE
        text.append(f"Step {it}")
        for node in agent.root.actions.values():
            text.append(f"{np.array2string(node.data, precision=10)}: {node.q_value}\t{node.data.tobytes()}")
        text.append(f"Chosen: {np.array2string(action, precision=10)}")
        text.append("\n")

        # SAVE ACTION DISTRIBUTION
        actions = np.array([a.data[0] for a in agent.root.actions.values()])
        act_dist.append(actions)

        # # SAVE TREE
        agent.visualize(str(it))

        observation, reward, done, extra = real_env.step(action)
        total_reward += reward
        state_log.append(observation)
        extra_log.append(list(extra.values()))
        it += 1

    extra_log[-1] = extra_log[-1][:-1]
    state_log = np.array(state_log)
    np.savetxt(fname=f"../output/log/state_log_{agent_type}_{number}{'_Noise' if noise else ''}.csv", X=state_log,
               delimiter=",")
    extra_log = np.array(extra_log)
    np.savetxt(fname=f"../output/log/extra_log_{agent_type}_{number}{'_Noise' if noise else ''}.csv", X=extra_log,
               delimiter=",")
    np.savetxt(fname=f"../output/act_dist_{number}.csv", X=act_dist, delimiter=",")

    # SAVE Q_VALUE TO FILE
    with open(f'q_val_{number}.txt', 'w') as f:
        f.write('\n'.join(text))

    return params, total_reward


def run_experiment_instance(experiment_data: ExperimentData):
    # First Key -> Continuous
    # Second Key -> Noise
    env_selector = {
        # Continuous
        True: {
            # Continuous + Noise
            True: 'gym_goddard:GoddardNoise-v0',
            # Continuous Simple
            False: 'gym_goddard:Goddard-v0'
        },
        # Discrete
        False: {
            # Discrete + Noise
            True: 'gym_goddard:GoddardDiscreteNoise-v0',
            # Discrete Simple
            False: 'gym_goddard:GoddardDiscrete-v0'
        }
    }
    env_name = env_selector[experiment_data.continuous][experiment_data.noise]
    real_env = gym.make(env_name)

    # TODO seed things

    return run(experiment_data.agent_type, experiment_data.param, experiment_data.noise, experiment_data.number,
               real_env)


if __name__ == '__main__':
    n_jobs = 1

    tests = [
        ExperimentData(
            agent_type="apw",
            param=PwParameters(
                root_data=None,
                env=None,
                n_sim=1000,
                C=0.0009,
                action_selection_fn=ucb1,
                gamma=1,
                rollout_selection_fn=continuous_default_policy,
                state_variable="_state",
                max_depth=500,
                n_actions=11,
                alpha=0,
                k=8
            ),
            continuous=True,
            noise=False,
            number=1
        ),
    ]

    with parallel_backend('loky', n_jobs=n_jobs):
        results = Parallel(verbose=100)(delayed(run_experiment_instance)(t) for t in tests)

    df = pd.DataFrame(
        [{
            **dataclasses.asdict(r[0]),
            "reward": r[1],
            "# actions": math.ceil(r[0].k * (r[0].n_sim ** r[0].alpha)) if r[0] is PwParameters else None
        } for r in results]
    )
    df.to_csv('../output/results.csv', index=False)
