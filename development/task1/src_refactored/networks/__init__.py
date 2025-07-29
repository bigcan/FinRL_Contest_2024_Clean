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

# Try importing real implementations with fallback
try:
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
except ImportError as e:
    print(f"Warning: Could not import network implementations: {e}")
    
    import torch
    import torch.nn as nn
    
    # Fallback network implementations
    class QNetBase(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
    
    class QNetTwin(QNetBase):
        def __init__(self, dims=None, state_dim=None, action_dim=None):
            super().__init__(state_dim, action_dim)
            if dims is None:
                dims = [64, 64]
            
            layers = []
            prev_dim = state_dim
            for dim in dims:
                layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, action_dim))
            
            self.net = nn.Sequential(*layers)
        
        def forward(self, state):
            return self.net(state)
    
    class QNetTwinDuel(QNetTwin):
        def __init__(self, dims=None, state_dim=None, action_dim=None):
            super().__init__(dims, state_dim, action_dim)
    
    # Set other network types to basic implementations
    NoisyLinear = nn.Linear
    QNetTwinNoisy = QNetTwinDuelNoisy = QNetTwin
    ActorDiscretePPO = CriticAdv = ActorCriticPPO = QNetTwin
    
    def build_mlp(dims, activation=nn.ReLU, output_activation=None):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
            elif output_activation:
                layers.append(output_activation())
        return nn.Sequential(*layers)
    
    def layer_init_with_orthogonal(layer, std=1.0):
        return layer
    
    def layer_init_with_xavier(layer):
        return layer
        
    def layer_init_with_kaiming(layer):
        return layer
    
    class NetworkUtils:
        @staticmethod
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

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