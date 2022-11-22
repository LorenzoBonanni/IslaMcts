
# IslaMcts

A small python library that implements Montecarlo Tree Search togheter with some modifications to enable it to work with continuous states and actions.


## Installation

The project requires python 3.10 or higher.

One you have done that you can install the required libraries using pip
```bash
  cd IslaMcts
  pip install -r requirements.txt
```
## Getting Started

To get started there are two ways:
1. From Command Line
2. From Python 

### Command Line
To use the software as a sort of "black box" you can use the Command Line interface to interact with the software by using Command line parameters.

To do so you have to execute the `sweep.py` file and pass it the parameters of the experiment:
- **_environment_**: The name of the gym environment to run the experiments on
- **_algorithm_**: The name of the Montecarlo Tree Search version you want to run, default _vanilla_, the available values are:
  - _vanilla_: Montecarlo Tree Search with discrete actions [1]
  - _apw_: Montecarlo Tree Search with Action Progressive Widening, a tecnique to deal with continuous actions [2]
  - _spw_: Montecarlo Tree Search with State Progressive Widening, a tecnique to deal with continuous and stochastic states [2]
  - _dpw_: Montecarlo Tree Search with Double Progressive Widening, a tecnique that joins Actions Progressive Widening and State Progressive Widening [2]
- **_nsim_**: Number of simulations from the root of the tree
- **_c_**: Value of the exploring parameter of the UCT formula
- **_as_**: Function to select actions during simulations, the available values are:
    - _ucb_: Upper Confidence Bound(UCB1) Equation
- **_ae_**: parameter for apw, and dpw that specifies the function to use to select which action node to add to the tree. The available values are:
    - _random_: it samples random from all the action space
    - _voo_: it uses Voronoi Optimistic Opmization[3] to select from the action space.
    - _genetic2_: the first three actions are the highest, the lowest and the center points of the action space. Then it uses an  epsilon greedy strategy that with probability ε chooses accorging to the `genetic_default` function, with probability 1-ε it chooses the action in between the two best.
- **_gamma_**: discount factor of the MDP
- **_rollout_**: function to select actions during rollout, available values:
    - _random_: it samples a random action from the action space
- **_max_depth_**: max number of steps during rollout
- **_n_actions_**: number of actions available at each node
- **_alpha1_**: alpha parameter for the Action Expansion Phase available in Action Progressive Widening and Double Progressive Widening, it ranges from 0 to 1
- **_alpha2_**: alpha parameter for the State Expansion Phase available in State Progressive Widening and Double Progressive Widening, it ranges from 0 to 1
- **_k1_**: k parameter for the Action Expansion Phase available in Action Progressive Widening and Double Progressive Widening
- **_k2_**: k parameter for the State Expansion Phase available in State Progressive Widening and Double Progressive Widening
- **_epsilon_**: epsilon value for the epsilon greedy strategies
- **_genetic_default_**: the default function for the _genetic2_ function.
- **_n_sample_**: the number of samples collected by Voo
- **_n_episodes_**: the number of episodes for each experiment
- **_group_**: parameter for tracking experiments with [wandb.ai](wandb.ai/), it specifies the group name so that then you can group experiments by group name

**Example**
```sh
python3 sweep.py --ae=random --algorithm=apw --alpha1=0.8 --c=11 --gamma=0.99 --k1=60 --max_depth=500 --n_episodes=1 --nsim=100 --group=test
```
### Python
To use the software with python first you need to create an environment with Gym and then reset it and obtain the starting observation like this:
```py
real_env = gym.make("FrozenLake-v1", is_slippery=False, new_step_api=True).unwrapped
observation = real_env.reset()
```
Once you have done that you have to create the class that will hold the model parameters, this will be `MctsParameters` for vanilla Mcts, `PwParameters` for Action and State Progressive Widening and `DpwParameters` for Double Progressive Widening.
```py
parameters = MctsParameters(
            C=0.5,
            n_sim=100,
            root_data=observation,
            env=real_env,
            action_selection_fn=ucb1,
            max_depth=1000,
            gamma=1,
            rollout_selection_fn=continuous_default_policy,
            n_actions=real_env.action_space.n
        )
```
Finally you have to write the experiment loop:
```py
done = False
# Loop until you reach a terminal state
while not done:
    # At each iteration update the enviroment inside Mcts
    parameters.env = real_env
    # Create the Agent and give it the parametes
    agent = Mcts(
        param=parameters,
    )
    # Let it compute the policy
    action = agent.fit()
    # Apply to the policy to the enviroment
    observation, reward, done, _, _ = real_env.step(action)
    # Update the Algorithm root
    parameters.root_data = observation
```
**Complete Example:**
```py
real_env = gym.make("FrozenLake-v1", is_slippery=False, new_step_api=True).unwrapped
observation = real_env.reset()
parameters = MctsParameters(
            C=0.5,
            n_sim=100,
            root_data=observation,
            env=real_env,
            action_selection_fn=ucb1,
            max_depth=1000,
            gamma=1,
            rollout_selection_fn=continuous_default_policy,
            n_actions=real_env.action_space.n
        )
done = False
while not done:
    parameters.env = real_env
    agent = Mcts(
        param=parameters,
    )
    action = agent.fit()
    observation, reward, done, _, _ = real_env.step(action)
    parameters.root_data = observation
```

## References

[1] [UCT]()

[2]  [Adrien Coutoux, Jean-Baptiste Hoock, Nataliya Sokolovska, Olivier Teytaud, Nicolas Bonnard - Continuous Upper Confidence Trees.](https://hal.archives-ouvertes.fr/hal-00542673v2)

[3]  [Lim, Michael H. et al. Voronoi Progressive Widening: Efficient Online Solvers for Continuous Space MDPs and POMDPs with Provably Optimal Components.](https://www.semanticscholar.org/paper/Voronoi-Progressive-Widening%3A-Efficient-Online-for-Lim-Tomlin/d5b63d3c4dceb1dc79179dcaf470980561fdafcb)