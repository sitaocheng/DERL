import numpy as np
import torch


class EvolutionStrategy:
    """Evolution Strategy for policy optimization"""
    
    def __init__(self, policy, population_size=50, sigma=0.1, learning_rate=0.01):
        self.policy = policy
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
    
    def generate_population(self):
        """Generate population by adding noise to policy parameters"""
        population = []
        noises = []
        
        for _ in range(self.population_size):
            noise = {}
            for name, param in self.policy.named_parameters():
                noise[name] = torch.randn_like(param) * self.sigma
            noises.append(noise)
            
            perturbed_policy = self._create_perturbed_policy(noise)
            population.append(perturbed_policy)
        
        return population, noises
    
    def _create_perturbed_policy(self, noise):
        """Create a perturbed version of the policy"""
        import copy
        perturbed = copy.deepcopy(self.policy)
        
        with torch.no_grad():
            for name, param in perturbed.named_parameters():
                param.add_(noise[name])
        
        return perturbed
    
    def update(self, noises, fitness_scores):
        """Update policy parameters based on fitness scores"""
        fitness_scores = np.array(fitness_scores)
        fitness_scores = (fitness_scores - fitness_scores.mean()) / (fitness_scores.std() + 1e-8)
        
        with torch.no_grad():
            for name, param in self.policy.named_parameters():
                gradient = torch.zeros_like(param)
                for noise, fitness in zip(noises, fitness_scores):
                    gradient += fitness * noise[name]
                gradient /= (self.population_size * self.sigma)
                param.add_(gradient, alpha=self.learning_rate)
