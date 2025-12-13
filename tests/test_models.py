"""Tests for neural network models"""

import unittest
import torch
import numpy as np
from derl.models.policy import PolicyNetwork
from derl.models.value import ValueNetwork


class TestPolicyNetwork(unittest.TestCase):
    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
    
    def test_forward(self):
        """Test forward pass"""
        state = torch.randn(1, self.state_dim)
        action = self.policy(state)
        self.assertEqual(action.shape, (1, self.action_dim))
    
    def test_get_action(self):
        """Test action selection"""
        state = np.random.randn(self.state_dim)
        action = self.policy.get_action(state)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))


class TestValueNetwork(unittest.TestCase):
    def setUp(self):
        self.state_dim = 4
        self.value = ValueNetwork(self.state_dim)
    
    def test_forward(self):
        """Test forward pass"""
        state = torch.randn(1, self.state_dim)
        value = self.value(state)
        self.assertEqual(value.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
