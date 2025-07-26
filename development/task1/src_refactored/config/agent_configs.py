"""
Agent-specific configuration definitions for the FinRL Contest 2024 framework.
Provides structured configuration classes for different agent types.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ..core.types import AgentType


@dataclass
class BaseAgentConfig:
    """Base configuration for all agents"""
    agent_type: AgentType
    net_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
    gamma: float = 0.995
    learning_rate: float = 2e-6
    batch_size: int = 512
    repeat_times: int = 2
    reward_scale: float = 1.0
    clip_grad_norm: float = 1.0
    soft_update_tau: float = 2e-6
    state_value_tau: float = 0.01
    explore_rate: float = 0.005
    if_off_policy: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_type': self.agent_type,
            'net_dims': self.net_dims,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'repeat_times': self.repeat_times,
            'reward_scale': self.reward_scale,
            'clip_grad_norm': self.clip_grad_norm,
            'soft_update_tau': self.soft_update_tau,
            'state_value_tau': self.state_value_tau,
            'explore_rate': self.explore_rate,
            'if_off_policy': self.if_off_policy
        }


@dataclass
class DoubleDQNConfig(BaseAgentConfig):
    """Configuration for Double DQN agents"""
    agent_type: AgentType = "AgentDoubleDQN"


@dataclass
class D3QNConfig(BaseAgentConfig):
    """Configuration for Dueling Double DQN agents"""
    agent_type: AgentType = "AgentD3QN"


@dataclass
class PrioritizedDQNConfig(BaseAgentConfig):
    """Configuration for Prioritized Experience Replay DQN"""
    agent_type: AgentType = "AgentPrioritizedDQN"
    per_alpha: float = 0.6  # Prioritization exponent
    per_beta: float = 0.4   # Importance sampling initial value
    per_beta_annealing_steps: int = 100000  # Steps to anneal beta to 1.0
    buffer_type: str = "prioritized"


@dataclass
class NoisyDQNConfig(BaseAgentConfig):
    """Configuration for Noisy Networks DQN"""
    agent_type: AgentType = "AgentNoisyDQN"
    noise_std_init: float = 0.5  # Initial standard deviation for noise
    explore_rate: float = 0.0    # No epsilon exploration needed


@dataclass
class NoisyDuelDQNConfig(BaseAgentConfig):
    """Configuration for Noisy Dueling DQN"""
    agent_type: AgentType = "AgentNoisyDuelDQN"
    noise_std_init: float = 0.5
    explore_rate: float = 0.0


@dataclass
class RainbowDQNConfig(BaseAgentConfig):
    """Configuration for Rainbow DQN with multiple enhancements"""
    agent_type: AgentType = "AgentRainbowDQN"
    n_step: int = 3              # Multi-step learning
    noise_std_init: float = 0.5  # Noisy networks
    per_alpha: float = 0.6       # Prioritized replay
    per_beta: float = 0.4
    per_beta_annealing_steps: int = 100000
    explore_rate: float = 0.0    # No epsilon exploration
    buffer_type: str = "prioritized"


@dataclass
class AdaptiveDQNConfig(BaseAgentConfig):
    """Configuration for Adaptive Learning Rate DQN"""
    agent_type: AgentType = "AgentAdaptiveDQN"
    lr_strategy: str = "cosine_annealing"  # LR scheduling strategy
    lr_T_max: int = 10000                 # Cosine annealing period
    lr_patience: int = 100                # Patience for plateau reduction
    lr_factor: float = 0.8                # LR reduction factor
    adaptive_grad_clip: bool = True       # Adaptive gradient clipping
    lr_min: float = 1e-8                  # Minimum learning rate


# Configuration factory for easy agent creation
AGENT_CONFIG_REGISTRY: Dict[AgentType, type] = {
    "AgentDoubleDQN": DoubleDQNConfig,
    "AgentD3QN": D3QNConfig,
    "AgentPrioritizedDQN": PrioritizedDQNConfig,
    "AgentNoisyDQN": NoisyDQNConfig,
    "AgentNoisyDuelDQN": NoisyDuelDQNConfig,
    "AgentRainbowDQN": RainbowDQNConfig,
    "AgentAdaptiveDQN": AdaptiveDQNConfig,
}


def create_agent_config(agent_type: AgentType, **kwargs) -> BaseAgentConfig:
    """
    Factory function to create agent configuration
    
    Args:
        agent_type: Type of agent to create config for
        **kwargs: Additional configuration parameters
        
    Returns:
        Agent configuration instance
        
    Raises:
        ValueError: If agent_type is not supported
    """
    if agent_type not in AGENT_CONFIG_REGISTRY:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    config_class = AGENT_CONFIG_REGISTRY[agent_type]
    return config_class(agent_type=agent_type, **kwargs)


def load_agent_configs_from_ensemble_config(config_path: Union[str, Path]) -> List[BaseAgentConfig]:
    """
    Load agent configurations from ensemble configuration file
    
    Args:
        config_path: Path to ensemble configuration JSON file
        
    Returns:
        List of agent configurations
    """
    import json
    
    with open(config_path, 'r') as f:
        ensemble_config = json.load(f)
    
    agent_configs = []
    base_training_config = ensemble_config.get('training', {})
    
    for agent_type in ensemble_config.get('agents', []):
        # Create base config from ensemble settings
        kwargs = {
            'net_dims': base_training_config.get('net_dims', [128, 128, 128]),
            'gamma': base_training_config.get('gamma', 0.995),
            'learning_rate': base_training_config.get('learning_rate', 2e-6),
            'batch_size': base_training_config.get('batch_size', 512),
            'repeat_times': base_training_config.get('repeat_times', 2),
            'soft_update_tau': base_training_config.get('soft_update_tau', 2e-6),
            'state_value_tau': base_training_config.get('state_value_tau', 0.01),
            'explore_rate': base_training_config.get('explore_rate', 0.005),
        }
        
        # Add agent-specific configurations
        if agent_type in ["AgentPrioritizedDQN", "AgentRainbowDQN"]:
            per_config = ensemble_config.get('per_config', {})
            kwargs.update({
                'per_alpha': per_config.get('alpha', 0.6),
                'per_beta': per_config.get('beta', 0.4),
                'per_beta_annealing_steps': per_config.get('beta_annealing_steps', 100000),
            })
        
        if agent_type == "AgentRainbowDQN":
            rainbow_config = ensemble_config.get('rainbow_config', {})
            kwargs.update({
                'n_step': rainbow_config.get('n_step', 3),
            })
        
        if agent_type == "AgentAdaptiveDQN":
            adaptive_config = ensemble_config.get('adaptive_lr_config', {})
            kwargs.update({
                'lr_strategy': adaptive_config.get('lr_strategy', 'cosine_annealing'),
                'lr_T_max': adaptive_config.get('lr_T_max', 10000),
                'lr_patience': adaptive_config.get('lr_patience', 100),
                'lr_factor': adaptive_config.get('lr_factor', 0.8),
                'adaptive_grad_clip': adaptive_config.get('adaptive_grad_clip', True),
            })
        
        agent_config = create_agent_config(agent_type, **kwargs)
        agent_configs.append(agent_config)
    
    return agent_configs


# Predefined configurations for common use cases
DEFAULT_ENSEMBLE_CONFIGS = {
    "diverse_ensemble": [
        DoubleDQNConfig(),
        D3QNConfig(),
        PrioritizedDQNConfig(),
        NoisyDuelDQNConfig(),
        RainbowDQNConfig(),
        AdaptiveDQNConfig(),
    ],
    "noisy_ensemble": [
        NoisyDQNConfig(),
        NoisyDuelDQNConfig(),
        RainbowDQNConfig(),
    ],
    "advanced_ensemble": [
        PrioritizedDQNConfig(),
        RainbowDQNConfig(),
        AdaptiveDQNConfig(),
    ],
}


def get_default_ensemble_config(ensemble_type: str = "diverse_ensemble") -> List[BaseAgentConfig]:
    """
    Get predefined ensemble configuration
    
    Args:
        ensemble_type: Type of ensemble configuration
        
    Returns:
        List of agent configurations
        
    Raises:
        ValueError: If ensemble_type is not supported
    """
    if ensemble_type not in DEFAULT_ENSEMBLE_CONFIGS:
        raise ValueError(f"Unsupported ensemble type: {ensemble_type}")
    
    return DEFAULT_ENSEMBLE_CONFIGS[ensemble_type].copy()