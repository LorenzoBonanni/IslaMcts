# import wandb
from tqdm import tqdm

from islaMcts.utils.action_selection_functions import ucb1, continuous_default_policy
from islaMcts.agents.mcts_hash import MctsHash
from islaMcts.agents.parameters.mcts_parameters import MctsParameters
from islaMcts.enviroments.syntetic_env import SyntheticEnvDiscrete


def main():
    rewards = []
    for i in range(10):
        real_env = SyntheticEnvDiscrete()
        observation = real_env.reset()

        total_reward = 0
        # params = PwParameters(
        #     root_data=None,
        #     env=None,
        #     n_sim=100,
        #     C=1,
        #     action_selection_fn=ucb1,
        #     gamma=0.2,
        #     rollout_selection_fn=continuous_default_policy,
        #     action_expansion_function=genetic_policy(0.5, continuous_default_policy, 10),
        #     state_variable="_state",
        #     max_depth=500,
        #     n_actions=11,
        #     alpha=0,
        #     k=15
        # )
        params = MctsParameters(
            root_data=None,
            env=None,
            n_sim=100,
            C=1,
            action_selection_fn=ucb1,
            gamma=0.2,
            rollout_selection_fn=continuous_default_policy,
            state_variable="_state",
            max_depth=500,
            n_actions=101,
        )
        for _ in tqdm(range(10)):
            params.env = real_env
            params.root_data = observation

            agent = MctsHash(
                param=params
            )
            action = agent.fit()

            observation, reward, done, extra = real_env.step(action)

            total_reward += reward
        print(total_reward)
        rewards.append(total_reward)
    print(f"MEAN: {sum(rewards) / 10}")


if __name__ == '__main__':
    main()
