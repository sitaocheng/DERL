"""Example training script for Pendulum environment"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
from derl import DERL


def main():
    # Environment configuration
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize DERL agent
    agent = DERL(
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
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
    print(f"Training DERL on {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    agent.train(
        num_iterations=200,
        num_episodes_per_iter=10,
        batch_size=256,
        log_interval=10
    )
    
    # Save the trained model
    agent.save('pendulum_derl.pth')
    print("Training completed. Model saved to pendulum_derl.pth")


if __name__ == '__main__':
    main()
