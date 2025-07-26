"""
Double DQN agent implementation for the FinRL Contest 2024 framework.

This module provides the standard Double DQN agent that extends the base DQN
with proper action selection and evaluation separation.
"""

from typing import Optional
import torch

from .base_dqn_agent import BaseDQNAgent
from ..core.types import AgentType
from ..config import DoubleDQNConfig


class DoubleDQNAgent(BaseDQNAgent):
    """
    Double DQN Agent implementation.
    
    Implements Double DQN algorithm to reduce overestimation bias:
    - Uses online network for action selection
    - Uses target network for Q-value evaluation
    - Maintains standard DQN training loop
    
    This is the standard baseline DQN implementation.
    """
    
    def __init__(self, 
                 config: Optional[DoubleDQNConfig] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize Double DQN agent.
        
        Args:
            config: Agent configuration (will create default if None)
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Computing device
            **kwargs: Additional configuration parameters
        """
        # Create default config if not provided
        if config is None:
            config = DoubleDQNConfig(**kwargs)
        
        # Ensure agent type is set correctly
        config.agent_type = "AgentDoubleDQN"
        
        super().__init__(config, state_dim, action_dim, device)
    
    def _build_networks(self):
        """Build standard twin Q-networks for Double DQN."""
        from ..networks import QNetTwin
        
        # Create online network
        self.online_network = QNetTwin(
            dims=self.config.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        # Create target network (copy of online network)
        from copy import deepcopy
        self.target_network = deepcopy(self.online_network)
        
        # Set exploration rate for online network
        self.online_network.explore_rate = self.explore_rate
        
        print(f"Built Double DQN networks: {self.config.net_dims} dims, "
              f"{self.state_dim} state_dim, {self.action_dim} action_dim")
    
    def get_algorithm_info(self) -> dict:
        """Get algorithm-specific information."""
        return {
            'algorithm': 'Double DQN',
            'description': 'Double Deep Q-Network with action selection/evaluation separation',
            'features': [
                'Twin Q-networks for reduced overestimation',
                'Target network soft updates',
                'Uniform experience replay',
                'Epsilon-greedy exploration'
            ],
            'network_type': 'QNetTwin',
            'replay_type': 'Uniform',
            'exploration_type': 'Epsilon-greedy',
        }


class D3QNAgent(BaseDQNAgent):
    """
    Dueling Double DQN (D3QN) Agent implementation.
    
    Combines Double DQN with Dueling architecture:
    - Separates state value and action advantage estimation
    - Uses dueling aggregation formula
    - Reduces variance in Q-value estimation
    """
    
    def __init__(self, 
                 config: Optional[DoubleDQNConfig] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize D3QN agent.
        
        Args:
            config: Agent configuration (will create default if None)
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Computing device
            **kwargs: Additional configuration parameters
        """
        # Create default config if not provided
        if config is None:
            config = DoubleDQNConfig(**kwargs)
        
        # Ensure agent type is set correctly
        config.agent_type = "AgentD3QN"
        
        super().__init__(config, state_dim, action_dim, device)
    
    def _build_networks(self):
        """Build dueling twin Q-networks for D3QN."""
        from ..networks import QNetTwinDuel
        
        # Create online network
        self.online_network = QNetTwinDuel(
            dims=self.config.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        # Create target network (copy of online network)
        from copy import deepcopy
        self.target_network = deepcopy(self.online_network)
        
        # Set exploration rate for online network
        self.online_network.explore_rate = self.explore_rate
        
        print(f"Built D3QN dueling networks: {self.config.net_dims} dims, "
              f"{self.state_dim} state_dim, {self.action_dim} action_dim")
    
    def get_value_advantage_estimates(self, state):
        """
        Get separate value and advantage estimates for analysis.
        
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
        
        with torch.no_grad():
            return self.online_network.get_value_advantage_estimates(state)
    
    def get_algorithm_info(self) -> dict:
        """Get algorithm-specific information."""
        return {
            'algorithm': 'Dueling Double DQN (D3QN)',
            'description': 'Double DQN with dueling architecture for value/advantage separation',
            'features': [
                'Dueling network architecture',
                'Value and advantage stream separation',
                'Twin Q-networks for reduced overestimation',
                'Target network soft updates',
                'Uniform experience replay',
                'Epsilon-greedy exploration'
            ],
            'network_type': 'QNetTwinDuel',
            'replay_type': 'Uniform',
            'exploration_type': 'Epsilon-greedy',
        }


# Factory function for easy agent creation
def create_dqn_agent(agent_type: AgentType, 
                     state_dim: int, 
                     action_dim: int,
                     device: Optional[torch.device] = None,
                     **kwargs) -> BaseDQNAgent:
    """
    Factory function to create DQN agents.
    
    Args:
        agent_type: Type of DQN agent to create
        state_dim: State space dimensionality
        action_dim: Action space dimensionality
        device: Computing device
        **kwargs: Additional configuration parameters
        
    Returns:
        DQN agent instance
        
    Raises:
        ValueError: If agent_type is not supported
    """
    if agent_type == "AgentDoubleDQN":
        return DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs
        )
    elif agent_type == "AgentD3QN":
        return D3QNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported DQN agent type: {agent_type}")


# Agent registry for the DQN family
DQN_AGENT_REGISTRY = {
    "AgentDoubleDQN": DoubleDQNAgent,
    "AgentD3QN": D3QNAgent,
}