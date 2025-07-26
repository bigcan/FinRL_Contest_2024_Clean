"""
Agent factory and registry system for the FinRL Contest 2024 framework.

This module provides a unified interface for creating all types of DQN agents
and manages the agent registry for the refactored architecture.
"""

from typing import Optional, Dict, Type, Any
import torch

from ..core.types import AgentType
from ..core.base_agent import BaseAgent

# Import all agent implementations
from .base_dqn_agent import BaseDQNAgent
from .double_dqn_agent import DoubleDQNAgent, D3QNAgent, DQN_AGENT_REGISTRY
from .prioritized_dqn_agent import PrioritizedDQNAgent
from .noisy_dqn_agent import NoisyDQNAgent, NoisyDuelDQNAgent, NOISY_DQN_AGENT_REGISTRY
from .rainbow_dqn_agent import RainbowDQNAgent
from .adaptive_dqn_agent import AdaptiveDQNAgent

# Import factory functions
from .double_dqn_agent import create_dqn_agent
from .noisy_dqn_agent import create_noisy_dqn_agent

# Consolidated agent registry
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    # Standard DQN variants
    "AgentDoubleDQN": DoubleDQNAgent,
    "AgentD3QN": D3QNAgent,
    
    # Prioritized DQN
    "AgentPrioritizedDQN": PrioritizedDQNAgent,
    
    # Noisy DQN variants
    "AgentNoisyDQN": NoisyDQNAgent,
    "AgentNoisyDuelDQN": NoisyDuelDQNAgent,
    
    # Advanced variants
    "AgentRainbowDQN": RainbowDQNAgent,
    "AgentAdaptiveDQN": AdaptiveDQNAgent,
}

# Merge individual registries
AGENT_REGISTRY.update(DQN_AGENT_REGISTRY)
AGENT_REGISTRY.update(NOISY_DQN_AGENT_REGISTRY)


def create_agent(agent_type: AgentType,
                state_dim: int,
                action_dim: int,
                device: Optional[torch.device] = None,
                **kwargs) -> BaseAgent:
    """
    Universal factory function to create any DQN agent.
    
    Args:
        agent_type: Type of agent to create
        state_dim: State space dimensionality
        action_dim: Action space dimensionality  
        device: Computing device
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured agent instance
        
    Raises:
        ValueError: If agent_type is not supported
        
    Example:
        >>> agent = create_agent("AgentRainbowDQN", state_dim=84, action_dim=4)
        >>> agent = create_agent("AgentAdaptiveDQN", state_dim=100, action_dim=3, 
        ...                      lr_strategy="cosine_annealing")
    """
    if agent_type not in AGENT_REGISTRY:
        available_types = list(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unsupported agent type: {agent_type}. "
            f"Available types: {available_types}"
        )
    
    agent_class = AGENT_REGISTRY[agent_type]
    
    return agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **kwargs
    )


def create_ensemble_agents(agent_configs: Dict[str, Dict[str, Any]],
                          state_dim: int,
                          action_dim: int,
                          device: Optional[torch.device] = None) -> Dict[str, BaseAgent]:
    """
    Create multiple agents for ensemble learning.
    
    Args:
        agent_configs: Dictionary mapping agent names to their configurations
        state_dim: State space dimensionality
        action_dim: Action space dimensionality
        device: Computing device
        
    Returns:
        Dictionary mapping agent names to agent instances
        
    Example:
        >>> configs = {
        ...     "rainbow": {"agent_type": "AgentRainbowDQN", "learning_rate": 1e-4},
        ...     "adaptive": {"agent_type": "AgentAdaptiveDQN", "lr_strategy": "plateau"},
        ...     "noisy": {"agent_type": "AgentNoisyDQN", "noise_std_init": 0.3}
        ... }
        >>> agents = create_ensemble_agents(configs, state_dim=84, action_dim=4)
    """
    agents = {}
    
    for agent_name, config in agent_configs.items():
        agent_type = config.pop("agent_type")
        
        try:
            agent = create_agent(
                agent_type=agent_type,
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                **config
            )
            agents[agent_name] = agent
            
        except Exception as e:
            print(f"Warning: Failed to create agent '{agent_name}' of type '{agent_type}': {e}")
            continue
    
    return agents


def get_agent_info(agent_type: AgentType) -> Dict[str, Any]:
    """
    Get information about a specific agent type.
    
    Args:
        agent_type: Agent type to get information for
        
    Returns:
        Dictionary containing agent information
        
    Raises:
        ValueError: If agent_type is not supported
    """
    if agent_type not in AGENT_REGISTRY:
        available_types = list(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unsupported agent type: {agent_type}. "
            f"Available types: {available_types}"
        )
    
    agent_class = AGENT_REGISTRY[agent_type]
    
    # Create temporary instance to get info (without initializing networks)
    try:
        temp_agent = agent_class.__new__(agent_class)
        if hasattr(temp_agent, 'get_algorithm_info'):
            return temp_agent.get_algorithm_info()
        else:
            return {
                'algorithm': agent_type,
                'description': f'{agent_type} implementation',
                'class': agent_class.__name__,
                'module': agent_class.__module__,
            }
    except Exception:
        return {
            'algorithm': agent_type,
            'description': f'{agent_type} implementation',
            'class': agent_class.__name__,
            'module': agent_class.__module__,
        }


def list_available_agents() -> Dict[str, Dict[str, Any]]:
    """
    List all available agent types with their information.
    
    Returns:
        Dictionary mapping agent types to their information
    """
    agent_info = {}
    
    for agent_type in AGENT_REGISTRY.keys():
        try:
            agent_info[agent_type] = get_agent_info(agent_type)
        except Exception as e:
            agent_info[agent_type] = {
                'algorithm': agent_type,
                'error': str(e),
                'class': AGENT_REGISTRY[agent_type].__name__,
            }
    
    return agent_info


def validate_agent_type(agent_type: str) -> bool:
    """
    Validate if an agent type is supported.
    
    Args:
        agent_type: Agent type to validate
        
    Returns:
        True if supported, False otherwise
    """
    return agent_type in AGENT_REGISTRY


def get_agent_categories() -> Dict[str, list]:
    """
    Get agents organized by categories.
    
    Returns:
        Dictionary mapping categories to lists of agent types
    """
    categories = {
        'standard_dqn': ['AgentDoubleDQN', 'AgentD3QN'],
        'prioritized': ['AgentPrioritizedDQN'],
        'noisy_exploration': ['AgentNoisyDQN', 'AgentNoisyDuelDQN'],
        'advanced': ['AgentRainbowDQN', 'AgentAdaptiveDQN'],
    }
    
    # Filter to only include available agents
    available_categories = {}
    for category, agent_types in categories.items():
        available_agents = [
            agent_type for agent_type in agent_types 
            if agent_type in AGENT_REGISTRY
        ]
        if available_agents:
            available_categories[category] = available_agents
    
    return available_categories


def create_default_ensemble(state_dim: int,
                           action_dim: int,
                           device: Optional[torch.device] = None,
                           num_agents: int = 4) -> Dict[str, BaseAgent]:
    """
    Create a default ensemble with diverse agent types.
    
    Args:
        state_dim: State space dimensionality
        action_dim: Action space dimensionality
        device: Computing device
        num_agents: Number of agents to include (max 6)
        
    Returns:
        Dictionary mapping agent names to agent instances
    """
    # Default configurations for diverse ensemble
    default_configs = {
        "double_dqn": {
            "agent_type": "AgentDoubleDQN",
            "learning_rate": 1e-4,
            "batch_size": 64,
        },
        "d3qn": {
            "agent_type": "AgentD3QN", 
            "learning_rate": 1e-4,
            "batch_size": 64,
        },
        "prioritized": {
            "agent_type": "AgentPrioritizedDQN",
            "learning_rate": 1e-4,
            "per_alpha": 0.6,
            "per_beta": 0.4,
        },
        "noisy": {
            "agent_type": "AgentNoisyDQN",
            "learning_rate": 1e-4,
            "noise_std_init": 0.5,
        },
        "rainbow": {
            "agent_type": "AgentRainbowDQN",
            "learning_rate": 1e-4,
            "n_step": 3,
            "per_alpha": 0.6,
        },
        "adaptive": {
            "agent_type": "AgentAdaptiveDQN",
            "learning_rate": 1e-4,
            "lr_strategy": "cosine_annealing",
        },
    }
    
    # Select subset based on num_agents
    config_keys = list(default_configs.keys())[:num_agents]
    selected_configs = {
        key: default_configs[key] for key in config_keys
    }
    
    return create_ensemble_agents(
        selected_configs,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )


# Export all public interface
__all__ = [
    # Agent classes
    'BaseAgent',
    'BaseDQNAgent',
    'DoubleDQNAgent', 
    'D3QNAgent',
    'PrioritizedDQNAgent',
    'NoisyDQNAgent',
    'NoisyDuelDQNAgent', 
    'RainbowDQNAgent',
    'AdaptiveDQNAgent',
    
    # Factory functions
    'create_agent',
    'create_ensemble_agents',
    'create_default_ensemble',
    'create_dqn_agent',
    'create_noisy_dqn_agent',
    
    # Registry and utilities
    'AGENT_REGISTRY',
    'get_agent_info',
    'list_available_agents',
    'validate_agent_type',
    'get_agent_categories',
]


# Module information
def get_module_info() -> Dict[str, Any]:
    """Get information about this agents module."""
    return {
        'module': 'src_refactored.agents',
        'description': 'Unified agent factory and registry system',
        'total_agents': len(AGENT_REGISTRY),
        'agent_types': list(AGENT_REGISTRY.keys()),
        'categories': get_agent_categories(),
        'version': '1.0.0',
        'features': [
            'Unified agent creation interface',
            'Ensemble creation utilities', 
            'Agent registry and validation',
            'Composition-based architecture',
            'Type-safe agent factory',
            'Comprehensive agent information',
        ]
    }


if __name__ == "__main__":
    # Demo usage
    print("FinRL Contest 2024 - Agent Factory Demo")
    print("=" * 50)
    
    # List available agents
    print("\nAvailable Agents:")
    for agent_type in AGENT_REGISTRY.keys():
        print(f"  - {agent_type}")
    
    # Show categories
    print("\nAgent Categories:")
    for category, agents in get_agent_categories().items():
        print(f"  {category}: {agents}")
    
    # Create example agent
    try:
        agent = create_agent("AgentRainbowDQN", state_dim=84, action_dim=4)
        info = agent.get_algorithm_info()
        print(f"\nCreated {info['algorithm']}: {info['description']}")
        print(f"Features: {info['features'][:3]}...")  # Show first 3 features
    except Exception as e:
        print(f"\nError creating example agent: {e}")