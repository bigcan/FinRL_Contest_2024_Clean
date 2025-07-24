"""
Enhanced Hyperparameter Configuration for 16-Feature State Space
Optimized hyperparameter ranges for enhanced features and improved architectures
"""

import numpy as np
from itertools import product
from typing import Dict, List, Any

class EnhancedHyperparameterConfig:
    """Hyperparameter configuration optimized for enhanced features"""
    
    def __init__(self):
        # Base configuration optimized for 16-feature state space
        self.base_config = {
            'state_dim': 16,
            'action_dim': 3,
            'max_position': 1,
            'num_sims': 64,  # Reduced for memory efficiency with larger networks
        }
        
        # Network architecture parameters
        self.network_configs = {
            'enhanced_small': {
                'net_dims': [192, 96, 48],
                'description': 'Smaller enhanced network'
            },
            'enhanced_medium': {
                'net_dims': [256, 128, 64],
                'description': 'Medium enhanced network (recommended)'
            },
            'enhanced_large': {
                'net_dims': [384, 192, 96],
                'description': 'Large enhanced network'
            },
            'attention_based': {
                'net_dims': [256, 128],
                'n_heads': 4,
                'description': 'Attention-based architecture'
            }
        }
        
        # Hyperparameter search spaces
        self.search_spaces = {
            'learning_rate': {
                'values': [1e-6, 2e-6, 5e-6, 1e-5, 2e-5],
                'type': 'categorical',
                'description': 'Learning rate (reduced for stability with larger networks)'
            },
            'gamma': {
                'values': [0.99, 0.995, 0.999],
                'type': 'categorical',
                'description': 'Discount factor'
            },
            'explore_rate': {
                'values': [0.001, 0.005, 0.01, 0.02],
                'type': 'categorical',
                'description': 'Exploration rate (lower for enhanced features)'
            },
            'batch_size': {
                'values': [256, 512, 1024],
                'type': 'categorical',
                'description': 'Batch size (larger for stable gradients)'
            },
            'buffer_size_multiplier': {
                'values': [4, 6, 8, 10],
                'type': 'categorical',
                'description': 'Buffer size as multiple of max_step'
            },
            'soft_update_tau': {
                'values': [1e-6, 2e-6, 5e-6, 1e-5],
                'type': 'categorical',
                'description': 'Soft update rate for target networks'
            },
            'state_value_tau': {
                'values': [0.005, 0.01, 0.02, 0.05],
                'type': 'categorical',
                'description': 'State value normalization rate'
            },
            'repeat_times': {
                'values': [1, 2, 4],
                'type': 'categorical',
                'description': 'Number of gradient updates per environment step'
            }
        }
        
        # Agent-specific configurations
        self.agent_configs = {
            'AgentD3QN': {
                'compatible_networks': ['enhanced_small', 'enhanced_medium', 'enhanced_large'],
                'recommended_lr': 2e-6,
                'recommended_explore': 0.005
            },
            'AgentDoubleDQN': {
                'compatible_networks': ['enhanced_small', 'enhanced_medium', 'attention_based'],
                'recommended_lr': 1e-6,
                'recommended_explore': 0.01
            },
            'AgentTwinD3QN': {
                'compatible_networks': ['enhanced_medium', 'enhanced_large'],
                'recommended_lr': 2e-6,
                'recommended_explore': 0.005
            }
        }
        
    def get_recommended_config(self, agent_type: str = 'AgentD3QN') -> Dict[str, Any]:
        """Get recommended configuration for enhanced features"""
        config = self.base_config.copy()
        
        # Network architecture
        if agent_type in self.agent_configs:
            recommended_net = self.agent_configs[agent_type]['compatible_networks'][1]  # Use medium as default
            config['net_dims'] = self.network_configs[recommended_net]['net_dims']
            config['learning_rate'] = self.agent_configs[agent_type]['recommended_lr']
            config['explore_rate'] = self.agent_configs[agent_type]['recommended_explore']
        else:
            config['net_dims'] = self.network_configs['enhanced_medium']['net_dims']
            config['learning_rate'] = 2e-6
            config['explore_rate'] = 0.005
        
        # Other optimized parameters
        config.update({
            'gamma': 0.995,
            'batch_size': 512,
            'buffer_size_multiplier': 8,
            'soft_update_tau': 2e-6,
            'state_value_tau': 0.01,
            'repeat_times': 2,
            'break_step': 16,  # Reduced for enhanced features
            'horizon_len_multiplier': 2,  # Horizon as multiple of max_step
            'eval_per_step_multiplier': 1,  # Evaluation frequency
        })
        
        return config
    
    def generate_search_grid(self, agent_type: str, max_combinations: int = 100) -> List[Dict[str, Any]]:
        """Generate hyperparameter search grid"""
        base_config = self.get_recommended_config(agent_type)
        
        # Select key parameters for grid search
        key_params = ['learning_rate', 'explore_rate', 'batch_size', 'gamma']
        
        # Generate all combinations
        param_combinations = []
        param_values = [self.search_spaces[param]['values'] for param in key_params]
        
        for combination in product(*param_values):
            config = base_config.copy()
            for i, param in enumerate(key_params):
                config[param] = combination[i]
            param_combinations.append(config)
        
        # Limit combinations if too many
        if len(param_combinations) > max_combinations:
            # Use random sampling to reduce combinations
            np.random.seed(42)
            indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
            param_combinations = [param_combinations[i] for i in indices]
        
        return param_combinations
    
    def get_quick_test_configs(self) -> List[Dict[str, Any]]:
        """Get quick test configurations for validation"""
        base_config = self.get_recommended_config()
        
        # Reduce training time for quick tests
        quick_config = base_config.copy()
        quick_config.update({
            'break_step': 4,  # Very short training
            'num_sims': 32,   # Fewer parallel environments
            'horizon_len_multiplier': 1,
            'eval_per_step_multiplier': 1,
        })
        
        # Test different network sizes
        configs = []
        for net_name, net_config in self.network_configs.items():
            if net_name != 'attention_based':  # Skip attention for quick tests
                config = quick_config.copy()
                config['net_dims'] = net_config['net_dims']
                config['config_name'] = f'quick_test_{net_name}'
                configs.append(config)
        
        return configs
    
    def get_ablation_configs(self, feature_combinations: List[List[int]]) -> List[Dict[str, Any]]:
        """Get configurations for ablation studies with different feature sets"""
        base_config = self.get_recommended_config()
        configs = []
        
        for i, feature_indices in enumerate(feature_combinations):
            config = base_config.copy()
            
            # Adjust network size based on number of features
            n_features = len(feature_indices)
            if n_features <= 8:
                config['net_dims'] = [128, 64, 32]
            elif n_features <= 12:
                config['net_dims'] = [192, 96, 48]
            else:
                config['net_dims'] = [256, 128, 64]
            
            config['state_dim'] = n_features
            config['feature_indices'] = feature_indices
            config['config_name'] = f'ablation_features_{n_features}_{i}'
            configs.append(config)
        
        return configs
    
    def print_recommended_config(self, agent_type: str = 'AgentD3QN'):
        """Print recommended configuration"""
        config = self.get_recommended_config(agent_type)
        
        print(f"RECOMMENDED CONFIGURATION FOR {agent_type}")
        print("=" * 50)
        print(f"Network Architecture: {config['net_dims']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Exploration Rate: {config['explore_rate']}")
        print(f"Gamma: {config['gamma']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Buffer Size Multiplier: {config['buffer_size_multiplier']}")
        print(f"Soft Update Tau: {config['soft_update_tau']}")
        print(f"State Value Tau: {config['state_value_tau']}")
        print(f"Repeat Times: {config['repeat_times']}")
        print()
        
        # Print rationale
        print("RATIONALE:")
        print("- Larger network architecture to handle 16-feature state space")
        print("- Lower learning rate for stability with enhanced features")
        print("- Reduced exploration for more precise trading decisions")
        print("- Larger batch size for stable gradients")
        print("- Conservative update rates to prevent instability")
    
    def save_config_file(self, config: Dict[str, Any], filepath: str):
        """Save configuration to Python file"""
        with open(filepath, 'w') as f:
            f.write("# Enhanced Hyperparameter Configuration\n")
            f.write("# Generated automatically for enhanced features\n\n")
            
            f.write("def get_enhanced_config():\n")
            f.write("    \"\"\"Get enhanced configuration for training\"\"\"\n")
            f.write("    return {\n")
            
            for key, value in config.items():
                if isinstance(value, str):
                    f.write(f"        '{key}': '{value}',\n")
                elif isinstance(value, list):
                    f.write(f"        '{key}': {value},\n")
                else:
                    f.write(f"        '{key}': {value},\n")
            
            f.write("    }\n")
        
        print(f"Configuration saved to: {filepath}")

def main():
    """Demonstrate hyperparameter configuration"""
    print("Enhanced Hyperparameter Configuration for FinRL Contest 2024")
    print("=" * 60)
    
    config_manager = EnhancedHyperparameterConfig()
    
    # Show recommended configurations for each agent
    for agent_type in ['AgentD3QN', 'AgentDoubleDQN', 'AgentTwinD3QN']:
        config_manager.print_recommended_config(agent_type)
        print()
    
    # Generate search grid example
    print("HYPERPARAMETER SEARCH GRID (first 5 combinations):")
    print("-" * 50)
    search_configs = config_manager.generate_search_grid('AgentD3QN', max_combinations=20)
    for i, config in enumerate(search_configs[:5]):
        print(f"Config {i+1}:")
        print(f"  LR: {config['learning_rate']}, Explore: {config['explore_rate']}, "
              f"Batch: {config['batch_size']}, Gamma: {config['gamma']}")
    
    print(f"\nTotal search configurations generated: {len(search_configs)}")
    
    # Save recommended config
    recommended = config_manager.get_recommended_config('AgentD3QN')
    config_manager.save_config_file(recommended, 'recommended_enhanced_config.py')

if __name__ == "__main__":
    main()