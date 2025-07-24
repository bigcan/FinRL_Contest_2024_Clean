import os
import torch
import numpy as np
from erl_config import Config, build_env
from reward_shaped_training_simulator import RewardShapedTrainingSimulator
from training_reward_config import TrainingRewardConfig
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from task1_eval import EnsembleEvaluator
from erl_run import train_agent
import time


class RewardShapedEnsemble:
    def __init__(self, save_path, agent_classes, reward_config=None, training_config=None):
        self.save_path = save_path
        self.agent_classes = agent_classes
        self.reward_config = reward_config or TrainingRewardConfig.balanced_training_config()
        self.training_config = training_config or self._get_default_training_config()
        
        # Shared training step tracker for curriculum learning
        self.training_step_tracker = {'step': 0}
        
        print(f"RewardShapedEnsemble initialized with {len(agent_classes)} agent types")
        print(f"Save path: {save_path}")
        self.reward_config.print_training_config()

    def _get_default_training_config(self):
        """Get default training configuration"""
        return {
            'num_sims': 64,  # Vectorized environments for faster training
            'num_ignore_step': 60,
            'max_position': 1,
            'step_gap': 2,
            'slippage': 7e-7,
            'max_step': (4800 - 60) // 2,  # Same as original
            'dataset_path': "data/BTC_1sec_predict.npy",
            'starting_cash': 1e6,
            'net_dims': (128, 128, 128),
            'gamma': 0.995,
            'explore_rate': 0.01,  # Lower than original to let reward shaping guide exploration
            'state_value_tau': 0.01,
            'target_step': 2**12,  # Steps per training iteration
            'eval_times': 2**6,    # Evaluation frequency
            'break_step': 2**14,   # Early stopping
        }

    def train_ensemble(self, gpu_id=-1):
        """Train the ensemble of agents with reward shaping"""
        
        print("\n" + "="*60)
        print("STARTING REWARD-SHAPED ENSEMBLE TRAINING")
        print("="*60)
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # Train each agent type
        trained_agents = []
        for i, agent_class in enumerate(self.agent_classes):
            agent_name = agent_class.__name__
            print(f"\n[{i+1}/{len(self.agent_classes)}] Training {agent_name}...")
            
            agent_save_path = os.path.join(self.save_path, agent_name)
            os.makedirs(agent_save_path, exist_ok=True)
            
            # Create environment args with reward shaping
            env_args = {
                "env_name": "TradeSimulator-v0",
                "num_envs": self.training_config['num_sims'],
                "max_step": self.training_config['max_step'],
                "state_dim": 8 + 2,
                "action_dim": 3,
                "if_discrete": True,
                "max_position": self.training_config['max_position'],
                "slippage": self.training_config['slippage'],
                "num_sims": self.training_config['num_sims'],
                "step_gap": self.training_config['step_gap'],
                "dataset_path": self.training_config['dataset_path'],
                "reward_config": self.reward_config,
                "training_step_tracker": self.training_step_tracker
            }
            
            # Create training configuration
            args = Config(agent_class=agent_class, 
                         env_class=RewardShapedTrainingSimulator, 
                         env_args=env_args)
            args.gpu_id = gpu_id
            args.random_seed = gpu_id + i  # Different seed for each agent
            args.net_dims = self.training_config['net_dims']
            args.starting_cash = self.training_config['starting_cash']
            
            # Agent-specific hyperparameters
            args.gamma = self.training_config['gamma']
            args.explore_rate = self.training_config['explore_rate']
            args.state_value_tau = self.training_config['state_value_tau']
            args.target_step = self.training_config['target_step']
            args.eval_times = self.training_config['eval_times']
            args.break_step = self.training_config['break_step']
            
            # Additional training parameters
            args.if_allow_break = True
            args.if_remove = True
            args.cwd = agent_save_path
            
            try:
                print(f"Starting training for {agent_name}...")
                start_time = time.time()
                
                # Train the agent
                train_agent(args)
                
                end_time = time.time()
                training_time = end_time - start_time
                print(f"✓ {agent_name} training completed in {training_time:.1f} seconds")
                
                # Test the trained agent - simplified for now
                print(f"✓ {agent_name} training completed (testing skipped for now)")
                test_results = "Training completed successfully"
                
                trained_agents.append({
                    'name': agent_name,
                    'class': agent_class,
                    'save_path': agent_save_path,
                    'training_time': training_time,
                    'test_results': test_results
                })
                
            except Exception as e:
                print(f"✗ Error training {agent_name}: {e}")
                continue
        
        # Print training summary
        self._print_training_summary(trained_agents)
        
        # Save training configuration
        self._save_training_config(trained_agents)
        
        return trained_agents

    def _print_training_summary(self, trained_agents):
        """Print comprehensive training summary"""
        print("\n" + "="*60)
        print("REWARD-SHAPED ENSEMBLE TRAINING SUMMARY")
        print("="*60)
        
        if not trained_agents:
            print("No agents were successfully trained!")
            return
        
        print(f"Successfully trained {len(trained_agents)}/{len(self.agent_classes)} agents")
        print(f"Total training steps: {self.training_step_tracker['step']:,}")
        
        total_time = sum(agent['training_time'] for agent in trained_agents)
        print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        print("\nAgent Results:")
        for agent in trained_agents:
            print(f"  {agent['name']:15s}: {agent['training_time']:6.1f}s | "
                  f"Results: {agent['test_results']}")
        
        print(f"\nReward shaping configuration:")
        print(f"  Activity bonus weight: {self.reward_config.activity_bonus_weight}")
        print(f"  Opportunity cost weight: {self.reward_config.opportunity_cost_weight}")
        print(f"  Timing bonus weight: {self.reward_config.timing_bonus_weight}")
        if self.reward_config.curriculum_learning:
            print(f"  Curriculum learning: Enabled ({self.reward_config.curriculum_steps:,} steps)")
        
        print(f"\nModels saved to: {self.save_path}")

    def _save_training_config(self, trained_agents):
        """Save training configuration and results"""
        import json
        
        config_data = {
            'reward_config': self.reward_config.get_training_summary(),
            'training_config': self.training_config,
            'trained_agents': [
                {
                    'name': agent['name'],
                    'save_path': agent['save_path'],
                    'training_time': agent['training_time'],
                    'test_results': str(agent['test_results'])  # Convert to string for JSON
                }
                for agent in trained_agents
            ],
            'total_training_steps': self.training_step_tracker['step'],
            'training_timestamp': time.time()
        }
        
        config_file = os.path.join(self.save_path, 'reward_shaped_training_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Training configuration saved to: {config_file}")

    def evaluate_ensemble(self, gpu_id=-1):
        """Evaluate the trained ensemble"""
        print("\n" + "="*60)
        print("EVALUATING REWARD-SHAPED ENSEMBLE")
        print("="*60)
        
        # Use standard evaluation environment (not reward-shaped for fair comparison)
        from trade_simulator import EvalTradeSimulator
        
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": 1,
            "max_step": self.training_config['max_step'],
            "state_dim": 8 + 2,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": self.training_config['max_position'],
            "slippage": self.training_config['slippage'],
            "num_sims": 1,
            "step_gap": self.training_config['step_gap'],
            "dataset_path": self.training_config['dataset_path']
        }
        
        args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
        args.gpu_id = gpu_id
        args.random_seed = gpu_id
        args.net_dims = self.training_config['net_dims']
        args.starting_cash = self.training_config['starting_cash']
        
        # Create evaluator
        evaluator = EnsembleEvaluator(
            save_path=self.save_path,
            agent_classes=self.agent_classes,
            args=args
        )
        
        try:
            evaluator.load_agents()
            evaluator.multi_trade()
            print("✓ Ensemble evaluation completed")
        except Exception as e:
            print(f"✗ Error during evaluation: {e}")


def run_reward_shaped_ensemble_training(config_type="balanced", gpu_id=-1):
    """Run the complete reward-shaped ensemble training pipeline"""
    
    # Select reward configuration
    if config_type == "conservative":
        reward_config = TrainingRewardConfig.conservative_training_config()
    elif config_type == "aggressive":
        reward_config = TrainingRewardConfig.aggressive_training_config()
    elif config_type == "ultra_aggressive":
        reward_config = TrainingRewardConfig.ultra_aggressive_training_config()
    elif config_type == "curriculum":
        reward_config = TrainingRewardConfig.curriculum_config()
    else:
        reward_config = TrainingRewardConfig.balanced_training_config()
    
    # Define save path with config type
    save_path = f"ensemble_reward_shaped_{config_type}"
    
    # Define agent classes to train
    agent_classes = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    
    print(f"Starting reward-shaped ensemble training with {config_type} configuration...")
    
    # Create and run ensemble training
    ensemble = RewardShapedEnsemble(
        save_path=save_path,
        agent_classes=agent_classes,
        reward_config=reward_config
    )
    
    # Train the ensemble
    trained_agents = ensemble.train_ensemble(gpu_id=gpu_id)
    
    if trained_agents:
        print(f"\n✓ Training completed successfully!")
        print(f"Trained agents: {[agent['name'] for agent in trained_agents]}")
        
        # Evaluate the ensemble
        ensemble.evaluate_ensemble(gpu_id=gpu_id)
        
        return ensemble, trained_agents
    else:
        print("\n✗ Training failed - no agents were successfully trained")
        return None, []


if __name__ == "__main__":
    import sys
    
    # Get GPU ID from command line arguments
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    
    # Get configuration type from command line arguments
    config_type = sys.argv[2] if len(sys.argv) > 2 else "balanced"
    
    print(f"Running reward-shaped ensemble training:")
    print(f"  GPU ID: {gpu_id}")
    print(f"  Configuration: {config_type}")
    print(f"  Available configs: balanced, conservative, aggressive, ultra_aggressive, curriculum")
    
    # Run the training
    ensemble, trained_agents = run_reward_shaped_ensemble_training(config_type, gpu_id)
    
    if ensemble and trained_agents:
        print(f"\n🎉 Success! Reward-shaped ensemble training completed.")
        print(f"Check results in: {ensemble.save_path}")
    else:
        print(f"\n❌ Training failed. Check logs for details.")