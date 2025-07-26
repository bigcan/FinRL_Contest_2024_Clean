"""
Neural network architectures and components for the FinRL Contest 2024 framework.

This module provides comprehensive network implementations including:
- Base network classes and utilities
- DQN variants (Double DQN, Dueling DQN)
- Noisy networks for parameter space exploration
- PPO networks for policy optimization
- Utility functions for network construction and initialization

Example usage:
    from src_refactored.networks import (
        QNetTwin, QNetTwinDuel, 
        QNetTwinNoisy, QNetTwinDuelNoisy,
        ActorDiscretePPO, CriticAdv,
        build_mlp, layer_init_with_orthogonal
    )
    
    # Create a standard Double DQN network
    dqn_net = QNetTwin(dims=[128, 128], state_dim=42, action_dim=3)
    
    # Create a noisy dueling network
    noisy_net = QNetTwinDuelNoisy(dims=[128, 128], state_dim=42, action_dim=3)
    
    # Create PPO networks
    actor = ActorDiscretePPO(dims=[128, 64], state_dim=42, action_dim=3)
    critic = CriticAdv(dims=[128, 64], state_dim=42)
"""

# Base network components and utilities
from .base_networks import (
    QNetBase,
    build_mlp,
    layer_init_with_orthogonal,
    layer_init_with_xavier,
    layer_init_with_kaiming,
    NetworkUtils,
)

# DQN network architectures
from .dqn_networks import (
    QNetTwin,
    QNetTwinDuel,
)

# Noisy network components
from .noisy_networks import (
    NoisyLinear,
    QNetTwinNoisy,
    QNetTwinDuelNoisy,
)

# PPO network architectures
from .ppo_networks import (
    ActorDiscretePPO,
    CriticAdv,
    ActorCriticPPO,
)

__all__ = [
    # Base components
    'QNetBase',
    'build_mlp',
    'layer_init_with_orthogonal',
    'layer_init_with_xavier',
    'layer_init_with_kaiming',
    'NetworkUtils',
    
    # DQN networks
    'QNetTwin',
    'QNetTwinDuel',
    
    # Noisy networks
    'NoisyLinear',
    'QNetTwinNoisy',
    'QNetTwinDuelNoisy',
    
    # PPO networks
    'ActorDiscretePPO',
    'CriticAdv',
    'ActorCriticPPO',
]

# Network type mappings for factory creation
NETWORK_REGISTRY = {
    # DQN variants
    'QNetTwin': QNetTwin,
    'QNetTwinDuel': QNetTwinDuel,
    'QNetTwinNoisy': QNetTwinNoisy,
    'QNetTwinDuelNoisy': QNetTwinDuelNoisy,
    
    # PPO variants
    'ActorDiscretePPO': ActorDiscretePPO,
    'CriticAdv': CriticAdv,
    'ActorCriticPPO': ActorCriticPPO,
}


def create_network(network_type: str, **kwargs):
    """
    Factory function to create network instances.
    
    Args:
        network_type: Type of network to create
        **kwargs: Network-specific arguments
        
    Returns:
        Network instance
        
    Raises:
        ValueError: If network_type is not supported
    """
    if network_type not in NETWORK_REGISTRY:
        raise ValueError(f"Unsupported network type: {network_type}. "
                        f"Available types: {list(NETWORK_REGISTRY.keys())}")
    
    network_class = NETWORK_REGISTRY[network_type]
    return network_class(**kwargs)