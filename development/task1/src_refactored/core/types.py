"""
Type definitions and common types used throughout the framework.
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Protocol, TypeVar
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np

# Type aliases for clarity
TensorType = Tensor
StateType = TensorType
ActionType = TensorType
RewardType = TensorType
DoneType = TensorType
QValueType = TensorType

# Generic type variables
T = TypeVar('T')
AgentType = TypeVar('AgentType', bound='BaseAgent')
NetworkType = TypeVar('NetworkType', bound=torch.nn.Module)

# Common data structures
@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: StateType
    action: ActionType
    reward: RewardType
    next_state: StateType
    done: DoneType
    
@dataclass
class BatchExperience:
    """Batch of experiences for training."""
    states: StateType
    actions: ActionType
    rewards: RewardType
    next_states: StateType
    dones: DoneType
    indices: Optional[TensorType] = None
    weights: Optional[TensorType] = None

@dataclass
class TrainingStats:
    """Training statistics and metrics."""
    step: int
    episode_reward: float
    critic_loss: float
    actor_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    exploration_rate: Optional[float] = None
    
@dataclass
class NetworkConfig:
    """Configuration for neural networks."""
    net_dims: List[int]
    state_dim: int
    action_dim: int
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    
@dataclass
class AgentConfig:
    """Base configuration for agents."""
    name: str
    network_config: NetworkConfig
    learning_rate: float = 2e-4
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 100000
    target_update_freq: int = 1000
    exploration_config: Dict[str, Any] = None
    
@dataclass
class TrainingConfig:
    """Configuration for training process."""
    max_steps: int
    eval_frequency: int
    save_frequency: int
    log_frequency: int
    num_parallel_envs: int = 1
    device: str = "cuda"
    
# Enums for different strategies
class ExplorationStrategy:
    EPSILON_GREEDY = "epsilon_greedy"
    NOISY_NETWORKS = "noisy_networks"
    UCB = "ucb"
    
class OptimizationStrategy:
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    
class EnsembleStrategy:
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    STACKING = "stacking"
    UNCERTAINTY_BASED = "uncertainty_based"
    
class ReplayBufferType:
    STANDARD = "standard"
    PRIORITIZED = "prioritized"
    
# Action space definitions
@dataclass
class ActionSpace:
    """Defines the action space for the environment."""
    n_actions: int
    action_names: List[str]
    
# Standard action space for trading
TRADING_ACTION_SPACE = ActionSpace(
    n_actions=3,
    action_names=["sell", "hold", "buy"]
)