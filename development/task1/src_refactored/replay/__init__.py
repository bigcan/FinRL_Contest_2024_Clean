"""
Replay buffer implementations and strategies for the FinRL Contest 2024 framework.

This module provides comprehensive replay buffer functionality including:
- Base buffer classes and interfaces
- Uniform random sampling buffers
- Prioritized Experience Replay (PER) buffers
- Sum tree data structures for efficient priority sampling
- Support for vectorized environments

Example usage:
    from src_refactored.replay import (
        UniformReplayBuffer, 
        PrioritizedReplayBuffer,
        create_buffer
    )
    
    # Create a standard uniform buffer
    uniform_buffer = UniformReplayBuffer(
        max_size=100000,
        state_dim=42,
        action_dim=3,
        device="cuda"
    )
    
    # Create a prioritized buffer
    per_buffer = PrioritizedReplayBuffer(
        max_size=100000,
        state_dim=42,
        action_dim=3,
        alpha=0.6,
        beta=0.4,
        device="cuda"
    )
    
    # Or use the factory function
    buffer = create_buffer(
        buffer_type="prioritized",
        max_size=100000,
        state_dim=42,
        action_dim=3
    )
"""

# Base buffer components
from .base_buffer import BaseReplayBuffer

# Buffer implementations
from .uniform_buffer import UniformReplayBuffer
from .prioritized_buffer import PrioritizedReplayBuffer, SumTree

# Type aliases for convenience
from typing import Union

BufferType = Union[UniformReplayBuffer, PrioritizedReplayBuffer]

__all__ = [
    # Base classes
    'BaseReplayBuffer',
    
    # Buffer implementations
    'UniformReplayBuffer',
    'PrioritizedReplayBuffer',
    'SumTree',
    
    # Type aliases
    'BufferType',
    
    # Factory functions
    'create_buffer',
]

# Buffer registry for factory creation
BUFFER_REGISTRY = {
    'uniform': UniformReplayBuffer,
    'standard': UniformReplayBuffer,  # Alias
    'prioritized': PrioritizedReplayBuffer,
    'per': PrioritizedReplayBuffer,   # Alias
}


def create_buffer(buffer_type: str, **kwargs) -> BaseReplayBuffer:
    """
    Factory function to create replay buffer instances.
    
    Args:
        buffer_type: Type of buffer to create ('uniform', 'prioritized')
        **kwargs: Buffer-specific arguments
        
    Returns:
        Replay buffer instance
        
    Raises:
        ValueError: If buffer_type is not supported
        
    Example:
        # Create uniform buffer
        buffer = create_buffer(
            buffer_type="uniform",
            max_size=100000,
            state_dim=42,
            action_dim=3
        )
        
        # Create prioritized buffer
        buffer = create_buffer(
            buffer_type="prioritized",
            max_size=100000,
            state_dim=42,
            action_dim=3,
            alpha=0.6,
            beta=0.4
        )
    """
    if buffer_type not in BUFFER_REGISTRY:
        raise ValueError(f"Unsupported buffer type: {buffer_type}. "
                        f"Available types: {list(BUFFER_REGISTRY.keys())}")
    
    buffer_class = BUFFER_REGISTRY[buffer_type]
    return buffer_class(**kwargs)


def get_recommended_buffer_type(agent_type: str) -> str:
    """
    Get recommended buffer type for given agent.
    
    Args:
        agent_type: Type of agent
        
    Returns:
        Recommended buffer type
    """
    # Agents that benefit from prioritized replay
    per_agents = {
        'AgentPrioritizedDQN',
        'AgentRainbowDQN',
        'AgentD3QN',  # Can benefit from PER
    }
    
    if agent_type in per_agents:
        return 'prioritized'
    else:
        return 'uniform'


def get_buffer_config_for_agent(agent_type: str, base_config: dict) -> dict:
    """
    Get buffer configuration optimized for specific agent type.
    
    Args:
        agent_type: Type of agent
        base_config: Base buffer configuration
        
    Returns:
        Optimized buffer configuration
    """
    config = base_config.copy()
    
    # Set buffer type based on agent
    config['buffer_type'] = get_recommended_buffer_type(agent_type)
    
    # Agent-specific optimizations
    if agent_type == 'AgentPrioritizedDQN':
        config.update({
            'alpha': 0.6,
            'beta': 0.4,
            'beta_annealing_steps': 100000,
        })
    elif agent_type == 'AgentRainbowDQN':
        config.update({
            'alpha': 0.6,
            'beta': 0.4,
            'beta_annealing_steps': 200000,  # Longer annealing for Rainbow
        })
    elif agent_type in ['AgentNoisyDQN', 'AgentNoisyDuelDQN']:
        # Noisy networks work well with uniform sampling
        config['buffer_type'] = 'uniform'
    
    return config