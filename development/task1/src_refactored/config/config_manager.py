"""
Centralized configuration management system.
Handles loading, validation, and providing configurations for all components.
"""

import json
import os
from typing import Dict, Any, Optional, Type, TypeVar, Union
from dataclasses import asdict, is_dataclass
from pathlib import Path

from ..core.interfaces import ConfigManagerProtocol
from ..core.types import (
    AgentConfig, NetworkConfig, TrainingConfig, 
    ExplorationStrategy, OptimizationStrategy, EnsembleStrategy, ReplayBufferType
)

T = TypeVar('T')


class ConfigManager:
    """
    Centralized configuration manager that handles all framework configurations.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.configs = {}
        self.default_configs = self._get_default_configs()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        full_path = self.config_dir / config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {full_path}")
        
        try:
            with open(full_path, 'r') as f:
                config = json.load(f)
            
            # Validate configuration
            if self.validate_config(config):
                self.configs[config_path] = config
                return config
            else:
                raise ValueError(f"Invalid configuration in {config_path}")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}")
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        full_path = self.config_dir / config_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclasses to dicts if needed
        serializable_config = self._make_serializable(config)
        
        with open(full_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        self.configs[config_path] = config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required top-level keys
            required_keys = ['agents', 'training', 'ensemble']
            if not all(key in config for key in required_keys):
                missing = [key for key in required_keys if key not in config]
                print(f"Missing required configuration keys: {missing}")
                return False
            
            # Validate agent configurations
            for agent_name, agent_config in config.get('agents', {}).items():
                if not self._validate_agent_config(agent_config):
                    print(f"Invalid agent configuration for {agent_name}")
                    return False
            
            # Validate training configuration
            if not self._validate_training_config(config.get('training', {})):
                print("Invalid training configuration")
                return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def get_agent_config(self, agent_name: str, config_file: str = "default.json") -> AgentConfig:
        """
        Get configuration for specific agent.
        
        Args:
            agent_name: Name of the agent
            config_file: Configuration file to load from
            
        Returns:
            Agent configuration
        """
        if config_file not in self.configs:
            self.load_config(config_file)
        
        config = self.configs[config_file]
        agent_config_dict = config.get('agents', {}).get(agent_name)
        
        if not agent_config_dict:
            # Return default configuration
            agent_config_dict = self.default_configs['agents'][agent_name]
        
        return self._dict_to_agent_config(agent_config_dict)
    
    def get_network_config(self, network_name: str, config_file: str = "default.json") -> NetworkConfig:
        """
        Get configuration for specific network.
        
        Args:
            network_name: Name of the network
            config_file: Configuration file to load from
            
        Returns:
            Network configuration
        """
        if config_file not in self.configs:
            self.load_config(config_file)
        
        config = self.configs[config_file]
        network_config_dict = config.get('networks', {}).get(network_name)
        
        if not network_config_dict:
            # Return default configuration
            network_config_dict = self.default_configs['networks'][network_name]
        
        return self._dict_to_network_config(network_config_dict)
    
    def get_training_config(self, config_file: str = "default.json") -> TrainingConfig:
        """
        Get training configuration.
        
        Args:
            config_file: Configuration file to load from
            
        Returns:
            Training configuration
        """
        if config_file not in self.configs:
            self.load_config(config_file)
        
        config = self.configs[config_file]
        training_config_dict = config.get('training', self.default_configs['training'])
        
        return self._dict_to_training_config(training_config_dict)
    
    def _validate_agent_config(self, agent_config: Dict[str, Any]) -> bool:
        """Validate agent configuration structure."""
        required_keys = ['network_config', 'learning_rate', 'gamma']
        return all(key in agent_config for key in required_keys)
    
    def _validate_training_config(self, training_config: Dict[str, Any]) -> bool:
        """Validate training configuration structure."""
        required_keys = ['max_steps', 'eval_frequency', 'save_frequency']
        return all(key in training_config for key in required_keys)
    
    def _dict_to_agent_config(self, config_dict: Dict[str, Any]) -> AgentConfig:
        """Convert dictionary to AgentConfig dataclass."""
        network_config = self._dict_to_network_config(config_dict['network_config'])
        
        return AgentConfig(
            name=config_dict.get('name', 'unnamed_agent'),
            network_config=network_config,
            learning_rate=config_dict.get('learning_rate', 2e-4),
            gamma=config_dict.get('gamma', 0.99),
            batch_size=config_dict.get('batch_size', 256),
            buffer_size=config_dict.get('buffer_size', 100000),
            target_update_freq=config_dict.get('target_update_freq', 1000),
            exploration_config=config_dict.get('exploration_config', {})
        )
    
    def _dict_to_network_config(self, config_dict: Dict[str, Any]) -> NetworkConfig:
        """Convert dictionary to NetworkConfig dataclass."""
        return NetworkConfig(
            net_dims=config_dict.get('net_dims', [128, 128]),
            state_dim=config_dict.get('state_dim', 10),
            action_dim=config_dict.get('action_dim', 3),
            activation=config_dict.get('activation', 'relu'),
            dropout=config_dict.get('dropout', 0.0),
            batch_norm=config_dict.get('batch_norm', False)
        )
    
    def _dict_to_training_config(self, config_dict: Dict[str, Any]) -> TrainingConfig:
        """Convert dictionary to TrainingConfig dataclass."""
        return TrainingConfig(
            max_steps=config_dict.get('max_steps', 100000),
            eval_frequency=config_dict.get('eval_frequency', 5000),
            save_frequency=config_dict.get('save_frequency', 10000),
            log_frequency=config_dict.get('log_frequency', 1000),
            num_parallel_envs=config_dict.get('num_parallel_envs', 1),
            device=config_dict.get('device', 'cuda')
        )
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert dataclasses and other objects to JSON-serializable format."""
        if is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _get_default_configs(self) -> Dict[str, Any]:
        """Get default configurations for all components."""
        return {
            'agents': {
                'DoubleDQN': {
                    'name': 'DoubleDQN',
                    'network_config': {
                        'net_dims': [128, 128],
                        'state_dim': 10,
                        'action_dim': 3,
                        'activation': 'relu'
                    },
                    'learning_rate': 2e-4,
                    'gamma': 0.99,
                    'batch_size': 256,
                    'buffer_size': 100000,
                    'target_update_freq': 1000,
                    'exploration_config': {
                        'strategy': ExplorationStrategy.EPSILON_GREEDY,
                        'epsilon_start': 1.0,
                        'epsilon_end': 0.01,
                        'epsilon_decay': 0.995
                    }
                },
                'DuelingDQN': {
                    'name': 'DuelingDQN',
                    'network_config': {
                        'net_dims': [128, 128],
                        'state_dim': 10,
                        'action_dim': 3,
                        'activation': 'relu'
                    },
                    'learning_rate': 2e-4,
                    'gamma': 0.99,
                    'batch_size': 256,
                    'buffer_size': 100000,
                    'target_update_freq': 1000,
                    'exploration_config': {
                        'strategy': ExplorationStrategy.EPSILON_GREEDY,
                        'epsilon_start': 1.0,
                        'epsilon_end': 0.01,
                        'epsilon_decay': 0.995
                    }
                },
                'PrioritizedDQN': {
                    'name': 'PrioritizedDQN',
                    'network_config': {
                        'net_dims': [128, 128],
                        'state_dim': 10,
                        'action_dim': 3,
                        'activation': 'relu'
                    },
                    'learning_rate': 2e-4,
                    'gamma': 0.99,
                    'batch_size': 256,
                    'buffer_size': 100000,
                    'target_update_freq': 1000,
                    'exploration_config': {
                        'strategy': ExplorationStrategy.EPSILON_GREEDY,
                        'epsilon_start': 1.0,
                        'epsilon_end': 0.01,
                        'epsilon_decay': 0.995
                    },
                    'per_config': {
                        'alpha': 0.6,
                        'beta': 0.4,
                        'beta_annealing_steps': 100000
                    }
                },
                'NoisyDQN': {
                    'name': 'NoisyDQN',
                    'network_config': {
                        'net_dims': [128, 128],
                        'state_dim': 10,
                        'action_dim': 3,
                        'activation': 'relu'
                    },
                    'learning_rate': 2e-4,
                    'gamma': 0.99,
                    'batch_size': 256,
                    'buffer_size': 100000,
                    'target_update_freq': 1000,
                    'exploration_config': {
                        'strategy': ExplorationStrategy.NOISY_NETWORKS,
                        'noise_std': 0.5
                    }
                },
                'RainbowDQN': {
                    'name': 'RainbowDQN',
                    'network_config': {
                        'net_dims': [128, 128],
                        'state_dim': 10,
                        'action_dim': 3,
                        'activation': 'relu'
                    },
                    'learning_rate': 2e-4,
                    'gamma': 0.99,
                    'batch_size': 256,
                    'buffer_size': 100000,
                    'target_update_freq': 1000,
                    'exploration_config': {
                        'strategy': ExplorationStrategy.NOISY_NETWORKS,
                        'noise_std': 0.5
                    },
                    'n_step': 3,
                    'per_config': {
                        'alpha': 0.6,
                        'beta': 0.4,
                        'beta_annealing_steps': 100000
                    }
                }
            },
            'networks': {
                'QNetTwin': {
                    'net_dims': [128, 128],
                    'state_dim': 10,
                    'action_dim': 3,
                    'activation': 'relu'
                },
                'QNetTwinDuel': {
                    'net_dims': [128, 128],
                    'state_dim': 10,
                    'action_dim': 3,
                    'activation': 'relu'
                },
                'QNetTwinNoisy': {
                    'net_dims': [128, 128],
                    'state_dim': 10,
                    'action_dim': 3,
                    'activation': 'relu',
                    'noise_std': 0.5
                }
            },
            'training': {
                'max_steps': 100000,
                'eval_frequency': 5000,
                'save_frequency': 10000,
                'log_frequency': 1000,
                'num_parallel_envs': 1,
                'device': 'cuda'
            },
            'ensemble': {
                'strategy': EnsembleStrategy.WEIGHTED_VOTE,
                'agents': ['DoubleDQN', 'DuelingDQN', 'PrioritizedDQN'],
                'performance_window': 1000,
                'weight_update_frequency': 100
            }
        }


# Convenience function for getting global config manager
_global_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration manager instance.
    
    Args:
        config_dir: Configuration directory (only used on first call)
        
    Returns:
        Global ConfigManager instance
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_dir)
    
    return _global_config_manager