"""
Enhanced Task 1 Ensemble Training with HPO Integration
Seamlessly integrates HPO-optimized parameters with existing training pipeline
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Import existing modules
from task1_ensemble import run
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN

# Add shared directory to path for HPO utilities
sys.path.append('../shared')
sys.path.append('.')

try:
    from hpo_config import Task1HPOSearchSpace, HPOResultsManager
    from hpo_utils import HPOAnalyzer
    HPO_AVAILABLE = True
except ImportError:
    print("⚠️ HPO modules not available. Running in standard mode.")
    HPO_AVAILABLE = False


class HPOIntegratedTrainer:
    """Trainer that can use HPO-optimized parameters or default parameters"""
    
    def __init__(self, hpo_results_dir: Optional[str] = None):
        """
        Initialize HPO integrated trainer
        
        Args:
            hpo_results_dir: Directory containing HPO results (None for default params)
        """
        self.hpo_results_dir = hpo_results_dir
        self.hpo_available = HPO_AVAILABLE and hpo_results_dir is not None
        
        # Load HPO results if available
        self.best_params = self._load_best_parameters()
        
        # Agent mapping
        self.agent_mapping = {
            'AgentD3QN': AgentD3QN,
            'AgentDoubleDQN': AgentDoubleDQN,
            'AgentTwinD3QN': AgentTwinD3QN
        }
    
    def _load_best_parameters(self) -> Dict[str, Any]:
        """Load best parameters from HPO results"""
        if not self.hpo_available:
            return {}
        
        try:
            results_manager = HPOResultsManager(self.hpo_results_dir)
            return results_manager.load_best_parameters("task1")
        except Exception as e:
            print(f"⚠️ Could not load HPO parameters: {e}")
            return {}
    
    def get_training_configuration(self, override_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get training configuration combining HPO results, defaults, and overrides
        
        Args:
            override_params: Parameters to override
            
        Returns:
            Complete training configuration
        """
        # Start with default configuration
        config = {
            'gpu_id': 0,
            'num_sims': 64,
            'num_ignore_step': 60,
            'max_position': 1,
            'step_gap': 2,
            'slippage': 7e-7,
            'starting_cash': 1e6,
            'net_dims': (128, 128, 128),
            'gamma': 0.995,
            'explore_rate': 0.005,
            'state_value_tau': 0.01,
            'soft_update_tau': 2e-6,
            'learning_rate': 2e-6,
            'batch_size': 512,
            'break_step': 16,
            'buffer_size_multiplier': 8,
            'repeat_times': 2,
            'horizon_len_multiplier': 2,
            'eval_per_step_multiplier': 1,
            'num_workers': 1,
            'save_gap': 8,
            'data_length': 4800
        }
        
        # Apply HPO parameters if available
        if self.best_params:
            print("🎯 Using HPO-optimized parameters:")
            hpo_config = Task1HPOSearchSpace.convert_to_config(self.best_params)
            config.update(hpo_config)
            
            # Display applied parameters
            for key, value in hpo_config.items():
                print(f"   {key}: {value}")
        else:
            print("📊 Using default parameters (no HPO results found)")
        
        # Apply manual overrides
        if override_params:
            print("🔧 Applying manual overrides:")
            config.update(override_params)
            for key, value in override_params.items():
                print(f"   {key}: {value}")
        
        return config
    
    def get_ensemble_agents(self) -> list:
        """Get ensemble agent classes (potentially from HPO results)"""
        if self.best_params and 'n_ensemble_agents' in self.best_params:
            # Extract agent configuration from HPO results
            n_agents = self.best_params['n_ensemble_agents']
            agent_classes = []
            
            for i in range(n_agents):
                agent_key = f'agent_{i}'
                if agent_key in self.best_params:
                    agent_name = self.best_params[agent_key]
                    if agent_name in self.agent_mapping:
                        agent_classes.append(self.agent_mapping[agent_name])
            
            if agent_classes:
                print(f"🤖 Using HPO-optimized ensemble: {[cls.__name__ for cls in agent_classes]}")
                return agent_classes
        
        # Default ensemble configuration
        default_agents = [AgentD3QN, AgentDoubleDQN, AgentDoubleDQN, AgentTwinD3QN]
        print(f"🤖 Using default ensemble: {[cls.__name__ for cls in default_agents]}")
        return default_agents
    
    def train(
        self, 
        save_path: str, 
        log_rules: bool = False,
        override_params: Optional[Dict[str, Any]] = None
    ):
        """
        Run training with HPO-integrated configuration
        
        Args:
            save_path: Path to save ensemble models
            log_rules: Whether to log trading rules
            override_params: Parameters to override
        """
        print("🚀 Starting HPO-Integrated Ensemble Training")
        print("="*60)
        
        # Get configuration and agents
        config = self.get_training_configuration(override_params)
        agent_classes = self.get_ensemble_agents()
        
        print(f"💾 Save path: {save_path}")
        print(f"🔧 Total configuration parameters: {len(config)}")
        print("="*60)
        
        # Run training
        run(
            save_path=save_path,
            agent_list=agent_classes,
            log_rules=log_rules,
            config_dict=config
        )
        
        # Save training configuration for reproducibility
        config_save_path = os.path.join(save_path, "training_config_used.json")
        config_to_save = {
            'config': config,
            'agent_classes': [cls.__name__ for cls in agent_classes],
            'hpo_used': bool(self.best_params),
            'hpo_params': self.best_params
        }
        
        with open(config_save_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"✅ Training configuration saved to {config_save_path}")
    
    def show_hpo_summary(self):
        """Display HPO results summary"""
        if not self.hpo_available or not self.best_params:
            print("❌ No HPO results available")
            return
        
        print("\n" + "="*60)
        print("🎯 HPO OPTIMIZATION SUMMARY")
        print("="*60)
        
        try:
            analyzer = HPOAnalyzer(self.hpo_results_dir)
            
            if analyzer.task1_results and 'stats' in analyzer.task1_results:
                stats = analyzer.task1_results['stats']
                print(f"Best Value: {stats.get('best_value', 'N/A')}")
                print(f"Total Trials: {stats.get('n_trials', 'N/A')}")
                print(f"Best Trial: {stats.get('best_trial_number', 'N/A')}")
            
            print("\nOptimized Parameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
                
        except Exception as e:
            print(f"Error displaying HPO summary: {e}")
        
        print("="*60)


def main():
    """Main function with HPO integration"""
    parser = argparse.ArgumentParser(description='Task 1 Ensemble Training with HPO Integration')
    
    # Basic arguments
    parser.add_argument('gpu_id', nargs='?', type=int, default=0, 
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--save-path', type=str, default='ensemble_teamname', 
                       help='Path to save ensemble models')
    parser.add_argument('--log-rules', action='store_true', 
                       help='Enable trading rules logging')
    
    # HPO integration arguments
    parser.add_argument('--hpo-results', type=str, 
                       help='Path to HPO results directory')
    parser.add_argument('--show-hpo-summary', action='store_true',
                       help='Show HPO optimization summary')
    
    # Configuration overrides
    parser.add_argument('--learning-rate', type=float,
                       help='Override learning rate')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size')
    parser.add_argument('--num-sims', type=int,
                       help='Override number of parallel simulations')
    parser.add_argument('--break-step', type=int,
                       help='Override break step')
    parser.add_argument('--net-dims', type=str,
                       help='Override network dimensions (e.g., "128,128,128")')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = HPOIntegratedTrainer(hpo_results_dir=args.hpo_results)
    
    # Show HPO summary if requested
    if args.show_hpo_summary:
        trainer.show_hpo_summary()
        return
    
    # Collect override parameters
    override_params = {'gpu_id': args.gpu_id}
    
    if args.learning_rate is not None:
        override_params['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        override_params['batch_size'] = args.batch_size
    if args.num_sims is not None:
        override_params['num_sims'] = args.num_sims
    if args.break_step is not None:
        override_params['break_step'] = args.break_step
    if args.net_dims is not None:
        # Parse network dimensions
        try:
            dims = tuple(map(int, args.net_dims.split(',')))
            override_params['net_dims'] = dims
        except ValueError:
            print(f"❌ Invalid net-dims format: {args.net_dims}. Use format: '128,128,128'")
            return
    
    # Run training
    trainer.train(
        save_path=args.save_path,
        log_rules=args.log_rules,
        override_params=override_params
    )


if __name__ == "__main__":
    main()