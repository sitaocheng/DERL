# DERL: Differentiable Evolutionary Reinforcement Learning

Official implementation of **Differentiable Evolutionary Reinforcement Learning** (DERL), a novel approach that combines evolutionary strategies with gradient-based reinforcement learning for robust policy optimization.

## Overview

DERL integrates two powerful optimization paradigms:
- **Evolution Strategies (ES)**: Population-based search that explores the parameter space
- **Reinforcement Learning (RL)**: Gradient-based optimization using policy gradients and value functions

By combining these approaches, DERL achieves:
- More robust policy learning
- Better exploration of the solution space
- Improved sample efficiency
- Resilience to local optima

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.8+
- OpenAI Gym
- NumPy
- Matplotlib
- TensorBoard

### Setup

```bash
# Clone the repository
git clone https://github.com/sitaocheng/DERL.git
cd DERL

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
import gym
from derl import DERL

# Create environment
env_name = 'Pendulum-v1'
env = gym.make(env_name)

# Initialize DERL agent
agent = DERL(
    env_name=env_name,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    population_size=50,
    sigma=0.1,
    lr_es=0.01,
    lr_policy=3e-4,
    lr_value=3e-4,
    gamma=0.99,
    rl_weight=0.5,
    es_weight=0.5
)

# Train the agent
agent.train(
    num_iterations=1000,
    num_episodes_per_iter=10,
    batch_size=256,
    log_interval=10
)

# Save the model
agent.save('derl_model.pth')
```

### Running Examples

Train on Pendulum environment:
```bash
python examples/train_pendulum.py
```

Train on CartPole environment:
```bash
python examples/train_cartpole.py
```

## Architecture

### Core Components

1. **PolicyNetwork**: Neural network that maps states to actions
2. **ValueNetwork**: Neural network that estimates state values
3. **EvolutionStrategy**: Implements ES with parameter perturbation
4. **RLAgent**: Gradient-based RL agent with experience replay
5. **DERL**: Main algorithm combining ES and RL

### Algorithm Flow

1. **ES Phase**: Generate population by perturbing policy parameters, evaluate fitness, and update
2. **RL Phase**: Collect experiences using current policy, update via gradient descent
3. **Combination**: Merge ES and RL policies using weighted average
4. **Repeat**: Iterate until convergence

## Configuration

Configuration files are located in `configs/`. You can customize:
- Environment parameters
- Network architecture
- ES hyperparameters (population size, noise sigma)
- RL hyperparameters (learning rates, discount factor)
- Training settings

## Project Structure

```
DERL/
├── derl/
│   ├── algorithms/
│   │   ├── derl.py          # Main DERL algorithm
│   │   ├── es.py            # Evolution Strategy
│   │   └── rl.py            # RL Agent
│   ├── models/
│   │   ├── policy.py        # Policy network
│   │   └── value.py         # Value network
│   ├── utils/
│   │   ├── logger.py        # Training logger
│   │   └── replay_buffer.py # Experience replay
│   └── envs/                # Environment wrappers
├── examples/
│   ├── train_pendulum.py    # Pendulum training example
│   └── train_cartpole.py    # CartPole training example
├── configs/
│   └── default.yaml         # Default configuration
├── tests/                   # Unit tests
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Key Features

- **Hybrid Optimization**: Combines evolutionary and gradient-based methods
- **Flexible Architecture**: Easily adaptable to different environments
- **Efficient Implementation**: Optimized for both CPU and GPU
- **Comprehensive Logging**: TensorBoard integration for monitoring
- **Modular Design**: Easy to extend and customize

## Hyperparameters

### Evolution Strategy
- `population_size`: Number of policy variants (default: 50)
- `sigma`: Noise standard deviation (default: 0.1)
- `lr_es`: ES learning rate (default: 0.01)

### Reinforcement Learning
- `lr_policy`: Policy network learning rate (default: 3e-4)
- `lr_value`: Value network learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `batch_size`: Training batch size (default: 256)

### DERL
- `rl_weight`: Weight for RL updates (default: 0.5)
- `es_weight`: Weight for ES updates (default: 0.5)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{derl2024,
  title={Differentiable Evolutionary Reinforcement Learning},
  author={Author Names},
  journal={Journal/Conference Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors
