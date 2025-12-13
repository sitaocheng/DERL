"""Tests for utility functions"""

import unittest
import numpy as np
from derl.utils.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer(capacity=100)
    
    def test_push(self):
        """Test adding experiences to buffer"""
        state = np.random.randn(4)
        action = np.random.randn(2)
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        
        self.buffer.push(state, action, reward, next_state, done)
        self.assertEqual(len(self.buffer), 1)
    
    def test_sample(self):
        """Test sampling from buffer"""
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.randn(2)
            reward = float(i)
            next_state = np.random.randn(4)
            done = False
            self.buffer.push(state, action, reward, next_state, done)
        
        batch = self.buffer.sample(5)
        states, actions, rewards, next_states, dones = batch
        
        self.assertEqual(states.shape, (5, 4))
        self.assertEqual(actions.shape, (5, 2))
        self.assertEqual(rewards.shape, (5,))
        self.assertEqual(next_states.shape, (5, 4))
        self.assertEqual(dones.shape, (5,))
    
    def test_capacity(self):
        """Test buffer capacity limit"""
        capacity = 10
        buffer = ReplayBuffer(capacity=capacity)
        
        for i in range(20):
            buffer.push(
                np.random.randn(4),
                np.random.randn(2),
                1.0,
                np.random.randn(4),
                False
            )
        
        self.assertEqual(len(buffer), capacity)


if __name__ == '__main__':
    unittest.main()
