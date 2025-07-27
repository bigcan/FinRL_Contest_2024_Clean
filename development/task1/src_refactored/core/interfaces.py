"""
Protocol definitions and interfaces for the framework components.
These define the contracts that different components must implement.
"""

from typing import Protocol, runtime_checkable, Dict, Any, Tuple, Optional, List
import torch
from torch import Tensor

from .types import (
    StateType, ActionType, RewardType, DoneType, QValueType,
    BatchExperience, TrainingStats, AgentConfig, NetworkConfig
)


@runtime_checkable
class ReplayBufferProtocol(Protocol):
    """Protocol for replay buffer implementations."""
    
    def add(self, state: StateType, action: ActionType, reward: RewardType, 
            next_state: StateType, done: DoneType) -> None:
        """Add a single experience to the buffer."""
        ...
    
    def sample(self, batch_size: int) -> BatchExperience:
        """Sample a batch of experiences from the buffer."""
        ...
    
    def update_priorities(self, indices: Tensor, priorities: Tensor) -> None:
        """Update priorities for prioritized replay (optional)."""
        ...
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        ...
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        ...


@runtime_checkable
class NetworkProtocol(Protocol):
    """Protocol for neural network implementations."""
    
    def forward(self, state: StateType) -> Tensor:
        """Forward pass through the network."""
        ...
    
    def get_action(self, state: StateType) -> ActionType:
        """Get action from network output."""
        ...
    
    def parameters(self):
        """Return network parameters for optimization."""
        ...
    
    def to(self, device: torch.device):
        """Move network to device."""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """Get network state dictionary."""
        ...
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load network state dictionary."""
        ...


@runtime_checkable
class QNetworkProtocol(NetworkProtocol, Protocol):
    """Protocol specifically for Q-networks."""
    
    def get_q_values(self, state: StateType) -> QValueType:
        """Get Q-values for all actions."""
        ...
    
    def get_q1_q2(self, state: StateType) -> Tuple[QValueType, QValueType]:
        """Get Q-values from twin networks (for Double DQN)."""
        ...


@runtime_checkable
class ExplorationProtocol(Protocol):
    """Protocol for exploration strategies."""
    
    def select_action(self, q_values: QValueType, step: int) -> ActionType:
        """Select action based on Q-values and exploration strategy."""
        ...
    
    def update(self, step: int) -> None:
        """Update exploration parameters based on training step."""
        ...
    
    def reset(self) -> None:
        """Reset exploration state (e.g., noise in noisy networks)."""
        ...


@runtime_checkable
class TrainingProtocol(Protocol):
    """Protocol for training orchestration."""
    
    def train(self, config: Any) -> Any:
        """Train the agent/ensemble with given configuration."""
        ...
    
    def evaluate(self, episodes: int) -> Dict[str, Any]:
        """Evaluate performance over given episodes."""
        ...
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        ...
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        ...


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol for optimizer wrappers."""
    
    def step(self, loss: Tensor, performance: Optional[float] = None) -> None:
        """Perform optimization step."""
        ...
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        ...
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        ...
    
    def get_grad_norm(self) -> float:
        """Get current gradient norm."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for RL agents."""
    
    def select_action(self, state: StateType, training: bool = True) -> ActionType:
        """Select action given current state."""
        ...
    
    def update(self, batch: BatchExperience) -> TrainingStats:
        """Update agent with a batch of experiences."""
        ...
    
    def save(self, path: str) -> None:
        """Save agent state to disk."""
        ...
    
    def load(self, path: str) -> None:
        """Load agent state from disk."""
        ...
    
    def set_training_mode(self, training: bool) -> None:
        """Set agent to training or evaluation mode."""
        ...
    
    @property
    def config(self) -> AgentConfig:
        """Get agent configuration."""
        ...


@runtime_checkable
class EnsembleProtocol(Protocol):
    """Protocol for ensemble strategies."""
    
    def predict(self, state: StateType) -> ActionType:
        """Make ensemble prediction."""
        ...
    
    def add_agent(self, agent: AgentProtocol) -> None:
        """Add agent to ensemble."""
        ...
    
    def update_weights(self, performances: List[float]) -> None:
        """Update agent weights based on performance."""
        ...
    
    def get_agent_weights(self) -> List[float]:
        """Get current agent weights."""
        ...


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Protocol for trading environments."""
    
    def reset(self) -> StateType:
        """Reset environment and return initial state."""
        ...
    
    def step(self, action: ActionType) -> Tuple[StateType, RewardType, DoneType, Dict[str, Any]]:
        """Take environment step and return results."""
        ...
    
    @property
    def state_dim(self) -> int:
        """Get state dimensionality."""
        ...
    
    @property
    def action_dim(self) -> int:
        """Get action dimensionality."""
        ...


@runtime_checkable
class ConfigManagerProtocol(Protocol):
    """Protocol for configuration management."""
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        ...
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to file."""
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        ...
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for specific agent."""
        ...
    
    def get_network_config(self, network_name: str) -> NetworkConfig:
        """Get configuration for specific network."""
        ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for training orchestration."""
    
    def train(self, agent: AgentProtocol, env: EnvironmentProtocol, 
              num_steps: int) -> List[TrainingStats]:
        """Train agent in environment for specified steps."""
        ...
    
    def evaluate(self, agent: AgentProtocol, env: EnvironmentProtocol, 
                 num_episodes: int) -> Dict[str, float]:
        """Evaluate agent performance."""
        ...
    
    def save_checkpoint(self, step: int, agent: AgentProtocol) -> None:
        """Save training checkpoint."""
        ...
    
    def load_checkpoint(self, path: str) -> Tuple[int, AgentProtocol]:
        """Load training checkpoint."""
        ...


# Factory protocols for creating components
@runtime_checkable
class NetworkFactoryProtocol(Protocol):
    """Protocol for network factory."""
    
    def create_network(self, network_type: str, config: NetworkConfig) -> NetworkProtocol:
        """Create network of specified type with configuration."""
        ...
    
    def list_available_networks(self) -> List[str]:
        """List available network types."""
        ...


@runtime_checkable
class AgentFactoryProtocol(Protocol):
    """Protocol for agent factory."""
    
    def create_agent(self, agent_type: str, config: AgentConfig) -> AgentProtocol:
        """Create agent of specified type with configuration."""
        ...
    
    def list_available_agents(self) -> List[str]:
        """List available agent types."""
        ...


@runtime_checkable
class ReplayBufferFactoryProtocol(Protocol):
    """Protocol for replay buffer factory."""
    
    def create_buffer(self, buffer_type: str, **kwargs) -> ReplayBufferProtocol:
        """Create replay buffer of specified type."""
        ...
    
    def list_available_buffers(self) -> List[str]:
        """List available buffer types."""
        ...