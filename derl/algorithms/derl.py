import numpy as np
import torch
import gym
from .es import EvolutionStrategy
from .rl import RLAgent
from ..models.policy import PolicyNetwork
from ..utils.logger import Logger


class DERL:
    """Differentiable Evolutionary Reinforcement Learning
    
    Combines evolution strategies with gradient-based RL for robust policy learning.
    """
    
    def __init__(self, env_name, state_dim, action_dim, 
                 population_size=50, sigma=0.1, lr_es=0.01,
                 lr_policy=3e-4, lr_value=3e-4, gamma=0.99,
                 rl_weight=0.5, es_weight=0.5):
        """
        Args:
            env_name: Name of the gym environment
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            population_size: Size of ES population
            sigma: Noise standard deviation for ES
            lr_es: Learning rate for ES
            lr_policy: Learning rate for policy network
            lr_value: Learning rate for value network
            gamma: Discount factor
            rl_weight: Weight for RL gradient updates
            es_weight: Weight for ES updates
        """
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize policy
        self.policy = PolicyNetwork(state_dim, action_dim)
        
        # Initialize ES and RL components
        self.es = EvolutionStrategy(self.policy, population_size, sigma, lr_es)
        self.rl_agent = RLAgent(state_dim, action_dim, lr_policy=lr_policy, 
                               lr_value=lr_value, gamma=gamma)
        
        self.rl_weight = rl_weight
        self.es_weight = es_weight
        
        self.logger = Logger()
    
    def evaluate_policy(self, policy, env, num_episodes=1):
        """Evaluate a policy on the environment"""
        total_reward = 0
        
        for _ in range(num_episodes):
            result = env.reset()
            state = result[0] if isinstance(result, tuple) else result
            done = False
            episode_reward = 0
            
            while not done:
                action = policy.get_action(state)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result
                episode_reward += reward
                state = next_state
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def train(self, num_iterations=1000, num_episodes_per_iter=10, 
              batch_size=256, log_interval=10):
        """Train DERL agent
        
        Args:
            num_iterations: Number of training iterations
            num_episodes_per_iter: Number of episodes per iteration
            batch_size: Batch size for RL updates
            log_interval: Interval for logging
        """
        env = gym.make(self.env_name)
        
        for iteration in range(num_iterations):
            # ES Update
            population, noises = self.es.generate_population()
            fitness_scores = []
            
            for perturbed_policy in population:
                fitness = self.evaluate_policy(perturbed_policy, env)
                fitness_scores.append(fitness)
            
            self.es.update(noises, fitness_scores)
            
            # RL Update
            for episode in range(num_episodes_per_iter):
                result = env.reset()
                state = result[0] if isinstance(result, tuple) else result
                done = False
                episode_reward = 0
                
                while not done:
                    action = self.rl_agent.select_action(state)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, _ = step_result
                    
                    self.rl_agent.store_transition(state, action, reward, 
                                                   next_state, done)
                    episode_reward += reward
                    state = next_state
                    
                    # Update RL agent
                    policy_loss, value_loss = self.rl_agent.update(batch_size)
            
            # Combine ES and RL policies
            self._combine_policies()
            
            # Log progress
            if iteration % log_interval == 0:
                avg_fitness = np.mean(fitness_scores)
                print(f"Iteration {iteration}: Avg Fitness = {avg_fitness:.2f}, "
                      f"Episode Reward = {episode_reward:.2f}")
                self.logger.log_scalar('fitness', avg_fitness, iteration)
                self.logger.log_scalar('episode_reward', episode_reward, iteration)
        
        env.close()
        self.logger.close()
    
    def _combine_policies(self):
        """Combine ES and RL policies using weighted average"""
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                self.policy.named_parameters(),
                self.rl_agent.policy.named_parameters()
            ):
                combined = self.es_weight * param1 + self.rl_weight * param2
                param1.copy_(combined)
                param2.copy_(combined)
    
    def save(self, path):
        """Save trained model"""
        torch.save({
            'policy': self.policy.state_dict(),
            'rl_policy': self.rl_agent.policy.state_dict(),
            'value': self.rl_agent.value.state_dict(),
        }, path)
    
    def load(self, path):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.rl_agent.policy.load_state_dict(checkpoint['rl_policy'])
        self.rl_agent.value.load_state_dict(checkpoint['value'])
