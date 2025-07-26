"""
Configuration management module for the FinRL Contest 2024 framework.

This module provides comprehensive configuration management including:
- Centralized configuration loading and validation
- Agent-specific configurations with type safety
- Training and experiment configurations
- Data and environment configurations
- Support for both JSON and programmatic configuration

Example usage:
    from src_refactored.config import (
        ConfigManager, 
        create_agent_config, 
        get_default_experiment_config
    )
    
    # Load configuration from existing ensemble config
    config_manager = ConfigManager()
    config_manager.load_from_ensemble_config("ensemble_config.json")
    
    # Create specific agent configuration
    agent_config = create_agent_config("AgentRainbowDQN", learning_rate=1e-4)
    
    # Get predefined experiment configuration
    experiment_config = get_default_experiment_config("full_training")
"""

from .config_manager import ConfigManager
from .agent_configs import (
    BaseAgentConfig,
    DoubleDQNConfig,
    D3QNConfig,
    PrioritizedDQNConfig,
    NoisyDQNConfig,
    NoisyDuelDQNConfig,
    RainbowDQNConfig,
    AdaptiveDQNConfig,
    AGENT_CONFIG_REGISTRY,
    create_agent_config,
    load_agent_configs_from_ensemble_config,
    get_default_ensemble_config,
    DEFAULT_ENSEMBLE_CONFIGS,
)
from .training_configs import (
    EnvironmentConfig,
    DataConfig,
    ReplayBufferConfig,
    TrainingConfig,
    EnsembleConfig,
    ExperimentConfig,
    EnsembleStrategy,
    BufferType,
    load_experiment_config_from_ensemble_json,
    get_default_experiment_config,
    DEFAULT_EXPERIMENT_CONFIGS,
)

__all__ = [
    # Core configuration manager
    'ConfigManager',
    
    # Agent configurations
    'BaseAgentConfig',
    'DoubleDQNConfig',
    'D3QNConfig', 
    'PrioritizedDQNConfig',
    'NoisyDQNConfig',
    'NoisyDuelDQNConfig',
    'RainbowDQNConfig',
    'AdaptiveDQNConfig',
    'AGENT_CONFIG_REGISTRY',
    'create_agent_config',
    'load_agent_configs_from_ensemble_config',
    'get_default_ensemble_config',
    'DEFAULT_ENSEMBLE_CONFIGS',
    
    # Training configurations
    'EnvironmentConfig',
    'DataConfig',
    'ReplayBufferConfig',
    'TrainingConfig',
    'EnsembleConfig',
    'ExperimentConfig',
    'EnsembleStrategy',
    'BufferType',
    'load_experiment_config_from_ensemble_json',
    'get_default_experiment_config', 
    'DEFAULT_EXPERIMENT_CONFIGS',
]