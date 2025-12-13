"""Neural network models for DERL"""

from .policy import PolicyNetwork
from .value import ValueNetwork

__all__ = ['PolicyNetwork', 'ValueNetwork']
