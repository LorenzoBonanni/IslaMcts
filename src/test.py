import gym
import numpy as np
from joblib import Parallel, parallel_backend, delayed

from src.action_selection_functions import continuous_default_policy, ucb1, discrete_default_policy
from src.agent_factory import get_agent
from src.agents.dpw_parameters import DpwParameters
from src.agents.mcts_parameters import MctsParameters
from src.agents.pw_parameters import PwParameters
from src.test_data import TestData

time = np.arange(0, 0.4, 0.001)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def run(agent_type: str, params: MctsParameters | PwParameters | DpwParameters, number: int, real_env: gym.Env,
        sim_env: gym.Env):
    observation = real_env.reset()
    params.env = sim_env.unwrapped

    state_log = [observation]
    extra_log = []

    for _ in time:
        params.root_data = observation
        sim_env.reset()
        agent = get_agent(
            agent_type=agent_type,
            params=params
        )
        action = agent.fit()
        observation, reward, done, extra = real_env.step(action)
        state_log.append(observation)
        extra_log.append(list(extra.values()))

    extra_log[-1] = extra_log[-1][:-1]
    state_log = np.array(state_log)
    np.savetxt(fname=f"output/state_log_{agent_type}_{number}.csv", X=state_log, delimiter=",")
    extra_log = np.array(extra_log)
    np.savetxt(fname=f"output/extra_log_{agent_type}_{number}.csv", X=extra_log, delimiter=",")


@static_vars(counter=0)
def run_test_instance(testData: TestData):
    if testData.continuous:
        if testData.noise:
            real_env = gym.make('gym_goddard:GoddardNoise-v0')
            sim_env = gym.make('gym_goddard:GoddardNoise-v0')
        else:
            real_env = gym.make('gym_goddard:Goddard-v0')
            sim_env = gym.make('gym_goddard:Goddard-v0')
    else:
        if testData.noise:
            real_env = gym.make('gym_goddard:GoddardDiscreteNoise-v0')
            sim_env = gym.make('gym_goddard:GoddardDiscreteNoise-v0')
        else:
            real_env = gym.make('gym_goddard:GoddardDiscrete-v0')
            sim_env = gym.make('gym_goddard:GoddardDiscrete-v0')

    run_test_instance.counter += 1
    run(testData.agent_type, testData.param, run_test_instance.counter, real_env, sim_env)


if __name__ == '__main__':
    n_jobs = 1
    tests = [
        TestData(
            agent_type="apw",
            param=PwParameters(
                root_data=None,
                env=None,
                n_sim=1000,
                C=0.0005,
                action_selection_fn=ucb1,
                gamma=1,
                rollout_selection_fn=continuous_default_policy,
                state_variable="_state",
                max_depth=500,
                alpha=0.3,
                k=5,
                n_actions=None
            ),
            continuous=True,
            noise=False
        ),
        TestData(
            agent_type="vanilla",
            param=MctsParameters(
                root_data=None,
                env=None,
                n_sim=1000,
                C=0.0005,
                action_selection_fn=ucb1,
                gamma=1,
                rollout_selection_fn=discrete_default_policy,
                state_variable="_state",
                max_depth=500,
                n_actions=11
            ),
            continuous=False,
            noise=False
        ),
        TestData(
            agent_type="spw",
            param=PwParameters(
                root_data=None,
                env=None,
                n_sim=1000,
                C=0.0005,
                action_selection_fn=ucb1,
                gamma=1,
                rollout_selection_fn=discrete_default_policy,
                state_variable="_state",
                max_depth=500,
                alpha=0.5,
                k=3,
                n_actions=11
            ),
            continuous=False,
            noise=True
        ),
        TestData(
            agent_type="dpw",
            param=DpwParameters(
                root_data=None,
                env=None,
                n_sim=1000,
                C=0.0005,
                action_selection_fn=ucb1,
                gamma=1,
                rollout_selection_fn=continuous_default_policy,
                state_variable="_state",
                max_depth=500,
                alphaSpw=0.5,
                kSpw=3,
                alphaApw=0.3,
                kApw=5,
                n_actions=None
            ),
            continuous=True,
            noise=True
        )
    ]
    with parallel_backend('loky', n_jobs=n_jobs):
        result = Parallel(verbose=100)(delayed(run_test_instance)(t) for t in tests)
