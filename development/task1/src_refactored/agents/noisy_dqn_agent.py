"""
Noisy DQN agent implementations for the FinRL Contest 2024 framework.

This module provides DQN agents that use NoisyNet for parameter space exploration
instead of epsilon-greedy action selection.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor

from .base_dqn_agent import BaseDQNAgent
from ..core.types import TrainingStats, StateType, ActionType
from ..config import NoisyDQNConfig, NoisyDuelDQNConfig


class NoisyDQNAgent(BaseDQNAgent):
    """
    Double DQN with Noisy Networks for exploration.
    
    Replaces epsilon-greedy exploration with learned parameter noise:
    - Uses NoisyLinear layers in the network
    - No epsilon decay needed
    - Automatic exploration through parameter perturbation
    - Better exploration in complex environments
    """
    
    def __init__(self, 
                 config: Optional[NoisyDQNConfig] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize Noisy DQN agent.
        
        Args:
            config: Agent configuration (will create default if None)
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Computing device
            **kwargs: Additional configuration parameters
        """
        # Create default config if not provided
        if config is None:
            config = NoisyDQNConfig(**kwargs)
        
        # Ensure agent type is set correctly
        config.agent_type = "AgentNoisyDQN"
        
        # Store noise parameters
        self.noise_std_init = getattr(config, 'noise_std_init', 0.5)
        
        # Override exploration rate (not used with noisy networks)
        config.explore_rate = 0.0
        
        super().__init__(config, state_dim, action_dim, device)
    
    def _build_networks(self):
        """Build noisy twin Q-networks."""
        from ..networks import QNetTwinNoisy
        
        # Create online network
        self.online_network = QNetTwinNoisy(
            dims=self.config.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            noise_std_init=self.noise_std_init
        ).to(self.device)
        
        # Create target network (copy of online network)
        from copy import deepcopy
        self.target_network = deepcopy(self.online_network)
        
        # Set exploration rate (not used but kept for compatibility)
        self.online_network.explore_rate = 0.0
        
        print(f"Built Noisy DQN networks: {self.config.net_dims} dims, "
              f"{self.state_dim} state_dim, {self.action_dim} action_dim, "
              f"noise_std={self.noise_std_init}")
    
    def select_action(self, state: StateType, deterministic: bool = False) -> ActionType:
        """
        Select action using noisy networks (no epsilon-greedy needed).
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy (disables noise)
            
        Returns:
            Selected action
        """
        if isinstance(state, (list, tuple)):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure correct shape [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        # Reset noise before action selection (for exploration)
        if not deterministic and self.training:
            self.online_network.reset_noise()
        
        with torch.no_grad():
            # Noisy networks provide exploration automatically
            q1_values, q2_values = self.online_network.get_q1_q2(state)
            q_values = torch.min(q1_values, q2_values)
            action = q_values.argmax(dim=1, keepdim=True)
        
        return action.squeeze().cpu().numpy() if action.shape[0] == 1 else action.cpu().numpy()
    
    def _update_networks(self) -> Tuple[float, float]:
        """
        Update networks with noisy network considerations.
        
        Returns:
            Tuple of (critic_loss, average_q_value)
        """
        # Reset noise before each update step
        self.online_network.reset_noise()
        self.target_network.reset_noise()
        
        # Use parent method for the actual update
        return super()._update_networks()
    
    def update(self, batch_data: Tuple[Tensor, ...]) -> TrainingStats:
        """
        Update agent with noise reset before training.
        
        Args:
            batch_data: Tuple of (states, actions, rewards, dones)
            
        Returns:
            Training statistics
        """
        # Reset noise before each batch update
        if hasattr(self.online_network, 'reset_noise'):
            self.online_network.reset_noise()
        
        # Call parent update method
        return super().update(batch_data)
    
    def get_algorithm_info(self) -> dict:
        """Get algorithm-specific information."""
        return {
            'algorithm': 'Noisy Double DQN',
            'description': 'Double DQN with NoisyNet for parameter space exploration',
            'features': [
                'NoisyNet parameter space exploration',
                'No epsilon-greedy needed',
                'Factorized Gaussian noise',
                'Automatic exploration scaling',
                'Twin Q-networks for reduced overestimation',
                'Target network soft updates',
                'Uniform experience replay'
            ],
            'network_type': 'QNetTwinNoisy',
            'replay_type': 'Uniform',
            'exploration_type': 'NoisyNet',
            'noise_config': {
                'std_init': self.noise_std_init,
            }
        }
    
    def set_noise_parameters(self, std_init: Optional[float] = None):
        """
        Update noise parameters (for experimentation).
        
        Args:
            std_init: New initial standard deviation for noise
        """
        if std_init is not None:
            self.noise_std_init = std_init
            # Note: This doesn't update existing networks, only affects new ones
    
    def __repr__(self) -> str:
        return (f"NoisyDQNAgent("
               f"state_dim={self.state_dim}, "
               f"action_dim={self.action_dim}, "
               f"noise_std={self.noise_std_init}, "
               f"device={self.device}, "
               f"training_step={self.training_step})")


class NoisyDuelDQNAgent(BaseDQNAgent):
    """
    Dueling Double DQN with Noisy Networks.
    
    Combines the benefits of:
    - Dueling architecture for better value estimation
    - NoisyNet for parameter space exploration
    - Double DQN for reduced overestimation
    """
    
    def __init__(self, 
                 config: Optional[NoisyDuelDQNConfig] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize Noisy Dueling DQN agent.
        
        Args:
            config: Agent configuration (will create default if None)
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Computing device
            **kwargs: Additional configuration parameters
        """
        # Create default config if not provided
        if config is None:
            config = NoisyDuelDQNConfig(**kwargs)
        
        # Ensure agent type is set correctly
        config.agent_type = "AgentNoisyDuelDQN"
        
        # Store noise parameters
        self.noise_std_init = getattr(config, 'noise_std_init', 0.5)
        
        # Override exploration rate (not used with noisy networks)
        config.explore_rate = 0.0
        
        super().__init__(config, state_dim, action_dim, device)
    
    def _build_networks(self):
        """Build noisy dueling twin Q-networks."""
        from ..networks import QNetTwinDuelNoisy
        
        # Create online network
        self.online_network = QNetTwinDuelNoisy(
            dims=self.config.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            noise_std_init=self.noise_std_init
        ).to(self.device)
        
        # Create target network (copy of online network)
        from copy import deepcopy
        self.target_network = deepcopy(self.online_network)
        
        # Set exploration rate (not used but kept for compatibility)
        self.online_network.explore_rate = 0.0
        
        print(f"Built Noisy Dueling DQN networks: {self.config.net_dims} dims, "
              f"{self.state_dim} state_dim, {self.action_dim} action_dim, "
              f"noise_std={self.noise_std_init}")
    
    def select_action(self, state: StateType, deterministic: bool = False) -> ActionType:
        """
        Select action using noisy dueling networks.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy (disables noise)
            
        Returns:
            Selected action
        """
        if isinstance(state, (list, tuple)):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure correct shape [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        # Reset noise before action selection (for exploration)
        if not deterministic and self.training:
            self.online_network.reset_noise()
        
        with torch.no_grad():
            # Noisy dueling networks provide exploration automatically
            q1_values, q2_values = self.online_network.get_q1_q2(state)
            q_values = torch.min(q1_values, q2_values)
            action = q_values.argmax(dim=1, keepdim=True)
        
        return action.squeeze().cpu().numpy() if action.shape[0] == 1 else action.cpu().numpy()
    
    def _update_networks(self) -> Tuple[float, float]:
        """
        Update networks with noisy network considerations.
        
        Returns:
            Tuple of (critic_loss, average_q_value)
        """
        # Reset noise before each update step
        self.online_network.reset_noise()
        self.target_network.reset_noise()
        
        # Use parent method for the actual update
        return super()._update_networks()
    
    def update(self, batch_data: Tuple[Tensor, ...]) -> TrainingStats:
        """
        Update agent with noise reset before training.
        
        Args:
            batch_data: Tuple of (states, actions, rewards, dones)
            
        Returns:
            Training statistics
        """
        # Reset noise before each batch update
        if hasattr(self.online_network, 'reset_noise'):
            self.online_network.reset_noise()
        
        # Call parent update method
        return super().update(batch_data)
    
    def get_value_advantage_estimates(self, state):
        """
        Get separate value and advantage estimates from noisy dueling networks.
        
        Args:
            state: Input state
            
        Returns:
            Tuple of (value1, advantage1, value2, advantage2)
        """
        if isinstance(state, (list, tuple)):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        # Reset noise for consistent evaluation
        if hasattr(self.online_network, 'reset_noise'):
            self.online_network.reset_noise()
        
        with torch.no_grad():
            return self.online_network.get_value_advantage_estimates(state)
    
    def get_algorithm_info(self) -> dict:
        """Get algorithm-specific information."""
        return {
            'algorithm': 'Noisy Dueling Double DQN',
            'description': 'Dueling Double DQN with NoisyNet for advanced exploration',
            'features': [
                'Noisy dueling network architecture',
                'Value and advantage stream separation with noise',
                'NoisyNet parameter space exploration',
                'No epsilon-greedy needed',
                'Factorized Gaussian noise',
                'Twin Q-networks for reduced overestimation',
                'Target network soft updates',
                'Uniform experience replay'
            ],
            'network_type': 'QNetTwinDuelNoisy',
            'replay_type': 'Uniform',
            'exploration_type': 'NoisyNet',
            'noise_config': {
                'std_init': self.noise_std_init,
            }
        }
    
    def set_noise_parameters(self, std_init: Optional[float] = None):
        """
        Update noise parameters (for experimentation).
        
        Args:
            std_init: New initial standard deviation for noise
        """
        if std_init is not None:
            self.noise_std_init = std_init
            # Note: This doesn't update existing networks, only affects new ones
    
    def __repr__(self) -> str:
        return (f"NoisyDuelDQNAgent("
               f"state_dim={self.state_dim}, "
               f"action_dim={self.action_dim}, "
               f"noise_std={self.noise_std_init}, "
               f"device={self.device}, "
               f"training_step={self.training_step})")


# Factory function for noisy agents
def create_noisy_dqn_agent(agent_type: str, 
                          state_dim: int, 
                          action_dim: int,
                          device: Optional[torch.device] = None,
                          **kwargs):
    """
    Factory function to create noisy DQN agents.
    
    Args:
        agent_type: Type of noisy agent to create
        state_dim: State space dimensionality
        action_dim: Action space dimensionality
        device: Computing device
        **kwargs: Additional configuration parameters
        
    Returns:
        Noisy DQN agent instance
        
    Raises:
        ValueError: If agent_type is not supported
    """
    if agent_type == "AgentNoisyDQN":
        return NoisyDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs
        )
    elif agent_type == "AgentNoisyDuelDQN":
        return NoisyDuelDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported noisy DQN agent type: {agent_type}")


# Agent registry for noisy DQN family
NOISY_DQN_AGENT_REGISTRY = {
    "AgentNoisyDQN": NoisyDQNAgent,
    "AgentNoisyDuelDQN": NoisyDuelDQNAgent,
}