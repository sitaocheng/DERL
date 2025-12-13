"""Differentiable Evolutionary Reinforcement Learning (DERL)

This package implements DERL, which combines evolutionary algorithms
with differentiable reinforcement learning for robust policy optimization.
"""

__version__ = '0.1.0'

from .algorithms.derl import DERL
from .algorithms.es import EvolutionStrategy
from .algorithms.rl import RLAgent

__all__ = ['DERL', 'EvolutionStrategy', 'RLAgent']
