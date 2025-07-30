"""
Training script with profit-focused rewards and reduced features
Implements Phase 2 of profitability enhancement plan
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from erl_config import Config
from erl_agent import AgentD3QN, AgentDoubleDQN
from erl_run import train_agent
from trade_simulator import TradeSimulator
from reward_functions import create_reward_calculator

def load_profit_config(config_path: str = "../configs/profit_focused_config.json") -> dict:
    """Load profit-focused configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def update_trade_simulator_for_reduced_features():
    """
    Monkey patch TradeSimulator to prioritize reduced features
    """
    original_init = TradeSimulator.__init__
    
    def new_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Override feature loading priority
        args = self.args if hasattr(self, 'args') else None
        if args:
            # Check for reduced features first
            reduced_path = args.predict_ary_path.replace('.npy', '_reduced.npy')
            if os.path.exists(reduced_path):
                print(f"Loading reduced features from {reduced_path}")
                self.factor_ary = np.load(reduced_path)
                
                # Load metadata
                metadata_path = reduced_path.replace('.npy', '_metadata.npy')
                if os.path.exists(metadata_path):
                    metadata = np.load(metadata_path, allow_pickle=True).item()
                    self.feature_names = metadata.get('feature_names', [])
                    print(f"Reduced features loaded: {len(self.feature_names)} features")
                
                # Convert to tensor
                self.factor_ary = torch.tensor(
                    self.factor_ary, 
                    dtype=torch.float32, 
                    device=self.device
                )
    
    TradeSimulator.__init__ = new_init

def setup_profit_focused_training(config_dict: dict):
    """
    Setup training with profit-focused configuration
    """
    # Update TradeSimulator for reduced features
    update_trade_simulator_for_reduced_features()
    
    # Create Config object
    config = Config()
    
    # Update with profit-focused settings
    agent_cfg = config_dict["agent_config"]
    config.net_dims = tuple(agent_cfg["net_dims"])
    config.learning_rate = agent_cfg["learning_rate"]
    config.batch_size = agent_cfg["batch_size"]
    config.horizon_len = agent_cfg["horizon_len"]
    config.buffer_size = int(agent_cfg["buffer_size"])
    config.clip_grad_norm = agent_cfg["clip_grad_norm"]
    config.soft_update_tau = agent_cfg["soft_update_tau"]
    
    # Device settings
    device_cfg = config_dict["device_config"]
    config.gpu_id = device_cfg["gpu_id"]
    config.num_workers = device_cfg["num_workers"]
    config.num_threads = device_cfg["num_threads"]
    config.random_seed = device_cfg["random_seed"]
    
    # Training settings
    train_cfg = config_dict["training_config"]
    config.break_step = train_cfg["num_episodes"] * train_cfg["samples_per_episode"]
    config.eval_per_step = train_cfg["eval_frequency"] * train_cfg["samples_per_episode"]
    config.save_gap = train_cfg["save_frequency"]
    
    # Set reward type
    reward_cfg = config_dict["reward_config"]
    config.reward_type = reward_cfg["reward_type"]
    
    return config

def train_profit_focused_agent(
    agent_type: str = "D3QN",
    config_path: str = "../configs/profit_focused_config.json",
    output_dir: str = "./profit_focused_results"):
    """
    Train agent with profit-focused rewards and reduced features
    """
    print("="*60)
    print("Profit-Focused Training with Reduced Features")
    print("="*60)
    
    # Load configuration
    config_dict = load_profit_config(config_path)
    
    # Setup configuration
    config = setup_profit_focused_training(config_dict)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{agent_type}_profit_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Set working directory
    config.cwd = run_dir
    config.if_remove = False  # Don't remove existing results
    
    # Initialize environment
    env = TradeSimulator(
        num_sims=1,
        device=torch.device(f"cuda:{config.gpu_id}" if config.gpu_id >= 0 else "cpu"),
        gpu_id=config.gpu_id
    )
    
    # Set environment info in config
    config.env_name = "TradeSimulator-v0"
    config.state_dim = env.state_dim
    config.action_dim = env.action_dim
    config.if_discrete = True
    config.max_step = 10000  # Steps per episode
    
    # Select agent
    if agent_type == "D3QN":
        config.agent_class = AgentD3QN
    elif agent_type == "DoubleDQN":
        config.agent_class = AgentDoubleDQN
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Initialize before training
    config.init_before_training()
    
    # Create agent
    agent = config.agent_class(
        net_dims=config.net_dims,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        gpu_id=config.gpu_id,
        args=config
    )
    
    # Create reward calculator
    reward_calculator = create_reward_calculator(
        reward_type=config.reward_type,
        device=agent.device
    )
    
    # Attach reward calculator to environment
    env.reward_calculator = reward_calculator
    
    print(f"\nTraining Configuration:")
    print(f"  Agent: {agent_type}")
    print(f"  Features: Reduced ({config_dict['feature_config']['n_features']} features)")
    print(f"  Reward: {config.reward_type}")
    print(f"  Network: {config.net_dims}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Device: {'GPU' if config.gpu_id >= 0 else 'CPU'}")
    print(f"  Output: {run_dir}")
    
    # Train agent
    print("\nStarting training...")
    train_agent(config)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("Results saved to:", run_dir)
    print("="*60)
    
    return run_dir

def evaluate_profit_performance(model_dir: str):
    """
    Evaluate the profit performance of trained model
    """
    print("\nEvaluating profit performance...")
    
    # Load metrics if available
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("Metrics file not found")

if __name__ == "__main__":
    # Train multiple agents for ensemble
    agents = ["D3QN", "DoubleDQN"]
    results_dirs = []
    
    for agent_type in agents:
        print(f"\n{'='*60}")
        print(f"Training {agent_type} with profit-focused rewards")
        print(f"{'='*60}")
        
        result_dir = train_profit_focused_agent(
            agent_type=agent_type,
            config_path="../configs/profit_focused_config.json",
            output_dir="./profit_focused_results"
        )
        results_dirs.append(result_dir)
        
        # Evaluate performance
        evaluate_profit_performance(result_dir)
    
    print("\n" + "="*60)
    print("All training completed!")
    print("Results directories:")
    for dir in results_dirs:
        print(f"  - {dir}")
    print("="*60)