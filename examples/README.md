# DERL Examples

This directory contains example scripts for training DERL agents on various environments.

## Available Examples

### 1. CartPole Environment
Train DERL on the CartPole-v1 environment:
```bash
python train_cartpole.py
```

### 2. Pendulum Environment
Train DERL on the Pendulum-v1 environment:
```bash
python train_pendulum.py
```

## Customizing Training

You can modify the hyperparameters in each script:
- `population_size`: Number of policy variants in ES
- `sigma`: Noise standard deviation for ES
- `lr_es`: Learning rate for ES updates
- `lr_policy`: Learning rate for policy network
- `lr_value`: Learning rate for value network
- `gamma`: Discount factor
- `rl_weight`: Weight for RL updates (0.0 to 1.0)
- `es_weight`: Weight for ES updates (0.0 to 1.0)

## Monitoring Training

Training metrics are logged to TensorBoard. To view them:
```bash
tensorboard --logdir=logs
```

Then open your browser to http://localhost:6006

## Output

Trained models are saved as `.pth` files in the current directory:
- `cartpole_derl.pth` - CartPole trained model
- `pendulum_derl.pth` - Pendulum trained model
