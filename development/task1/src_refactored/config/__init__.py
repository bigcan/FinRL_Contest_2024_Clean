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

# Try importing real implementations with fallback
try:
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
except ImportError as e:
    print(f"Warning: Could not import config implementations: {e}")
    
    # Fallback minimal config implementations
    from dataclasses import dataclass
    from typing import List, Any, Dict, Optional
    from enum import Enum
    
    class EnsembleStrategy(Enum):
        MAJORITY_VOTE = "majority_vote"
        WEIGHTED_VOTE = "weighted_vote"
        STACKING = "stacking"
    
    class BufferType(Enum):
        UNIFORM = "uniform"
        PRIORITIZED = "prioritized"
    
    @dataclass
    class BaseAgentConfig:
        learning_rate: float = 1e-4
        batch_size: int = 64
        gamma: float = 0.99
    
    @dataclass 
    class DoubleDQNConfig(BaseAgentConfig):
        target_update_freq: int = 100
        exploration_noise: float = 0.1
        
    @dataclass
    class D3QNConfig(DoubleDQNConfig):
        dueling: bool = True
    
    @dataclass
    class PrioritizedDQNConfig(DoubleDQNConfig):
        alpha: float = 0.6
        beta: float = 0.4
        
    # Set all other configs to basic implementation
    NoisyDQNConfig = NoisyDuelDQNConfig = RainbowDQNConfig = AdaptiveDQNConfig = DoubleDQNConfig
    
    @dataclass
    class TrainingConfig:
        total_episodes: int = 1000
        individual_episodes: int = 300
        ensemble_episodes: int = 500
        fine_tuning_episodes: int = 200
    
    # Minimal other configs
    EnvironmentConfig = DataConfig = ReplayBufferConfig = EnsembleConfig = ExperimentConfig = dict
    
    AGENT_CONFIG_REGISTRY = {}
    DEFAULT_ENSEMBLE_CONFIGS = {}
    DEFAULT_EXPERIMENT_CONFIGS = {}
    
    def create_agent_config(agent_type, **kwargs):
        return DoubleDQNConfig(**kwargs)
    
    def load_agent_configs_from_ensemble_config(path):
        return {}
        
    def get_default_ensemble_config():
        return {}
        
    def load_experiment_config_from_ensemble_json(path):
        return ExperimentConfig()
        
    def get_default_experiment_config(name):
        return ExperimentConfig()
        
    class ConfigManager:
        def __init__(self):
            pass
        def load_from_ensemble_config(self, path):
            pass

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