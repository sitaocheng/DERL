import torch
import torch.nn.functional as F
from ..models.policy import PolicyNetwork
from ..models.value import ValueNetwork
from ..utils.replay_buffer import ReplayBuffer


class RLAgent:
    """Reinforcement Learning agent using gradient-based optimization"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 lr_policy=3e-4, lr_value=3e-4, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state):
        """Select action using current policy"""
        return self.policy.get_action(state)
    
    def update(self, batch_size=256):
        """Update policy and value networks"""
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Update value network
        values = self.value(states)
        with torch.no_grad():
            next_values = self.value(next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)
        
        value_loss = F.mse_loss(values, target_values)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        pred_actions = self.policy(states)
        policy_loss = F.mse_loss(pred_actions, actions)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
