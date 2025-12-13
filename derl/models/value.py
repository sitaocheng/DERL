import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """Neural network for value function in DERL"""
    
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
