"""
Test script to validate aggressive hyperparameter configuration
Ensures all parameters are properly set and compatible
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
import torch as th

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_configuration():
    """Test loading and validity of aggressive configuration"""
    print("="*60)
    print("Testing Aggressive Hyperparameter Configuration")
    print("="*60)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "aggressive_profit_config.json"
    
    print(f"\n1. Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("   ✓ Configuration loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load configuration: {e}")
        return False
    
    # Test reward configuration
    print("\n2. Validating Reward Configuration:")
    reward_config = config["reward_config"]
    
    tests = [
        ("Profit amplifier", reward_config["profit_amplifier"], lambda x: x > 3.0),
        ("Speed multiplier", reward_config["max_speed_multiplier"], lambda x: x > 5.0),
        ("Blend factor", reward_config["blend_factor"], lambda x: 0.7 <= x <= 0.9),
        ("Profit speed enabled", reward_config["profit_speed_enabled"], lambda x: x is True)
    ]
    
    for name, value, test_func in tests:
        if test_func(value):
            print(f"   ✓ {name}: {value}")
        else:
            print(f"   ✗ {name}: {value} (failed validation)")
    
    # Test agent configuration
    print("\n3. Validating Agent Configuration:")
    agent_config = config["agent_config"]
    
    # Network architecture
    net_dims = agent_config["net_dims"]
    print(f"   Network dimensions: {net_dims}")
    print(f"   Total parameters: ~{sum(net_dims[i]*net_dims[i+1] for i in range(len(net_dims)-1)):,}")
    
    # Learning parameters
    print(f"   Learning rate: {agent_config['learning_rate']} (baseline: 3e-5)")
    print(f"   Batch size: {agent_config['batch_size']} (baseline: 128)")
    print(f"   Horizon length: {agent_config['horizon_len']} (baseline: 1024)")
    
    # Exploration
    print(f"   Exploration: {agent_config['explore_rate']} → {agent_config['explore_min']}")
    print(f"   Decay rate: {agent_config['explore_decay']}")
    
    # Test training configuration
    print("\n4. Validating Training Configuration:")
    train_config = config["training_config"]
    
    total_samples = train_config["num_episodes"] * train_config["samples_per_episode"]
    print(f"   Episodes: {train_config['num_episodes']}")
    print(f"   Samples per episode: {train_config['samples_per_episode']:,}")
    print(f"   Total training samples: {total_samples:,}")
    print(f"   Target Sharpe ratio: {train_config['target_sharpe_ratio']}")
    
    # Test environment configuration
    print("\n5. Validating Environment Configuration:")
    env_config = config["environment_config"]
    
    print(f"   Max position: {env_config['max_position']} (baseline: 2)")
    print(f"   Transaction cost: {env_config['transaction_cost']} (baseline: 0.001)")
    print(f"   Max holding time: {env_config['max_holding_time']}s (baseline: 3600s)")
    print(f"   Stop loss: {env_config.get('stop_loss', 'Not set')}")
    print(f"   Take profit: {env_config.get('take_profit', 'Not set')}")
    
    # Test device configuration
    print("\n6. Validating Device Configuration:")
    device_config = config["device_config"]
    
    print(f"   GPU ID: {device_config['gpu_id']}")
    print(f"   CUDA available: {th.cuda.is_available()}")
    if th.cuda.is_available():
        print(f"   GPU name: {th.cuda.get_device_name(0)}")
        print(f"   GPU memory: {th.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Calculate memory requirements
    print("\n7. Estimated Memory Requirements:")
    
    # Network memory (rough estimate)
    total_params = sum(net_dims[i]*net_dims[i+1] for i in range(len(net_dims)-1))
    param_memory = total_params * 4 / 1e6  # 4 bytes per float32, in MB
    
    # Batch memory
    state_dim = 15  # Reduced features
    batch_memory = (agent_config["batch_size"] * state_dim * 4) / 1e6
    
    # Buffer memory
    buffer_memory = (agent_config["buffer_size"] * state_dim * 4) / 1e6
    
    print(f"   Network parameters: ~{param_memory:.1f} MB")
    print(f"   Batch memory: ~{batch_memory:.1f} MB")
    print(f"   Buffer memory: ~{buffer_memory:.1f} MB")
    print(f"   Total estimate: ~{(param_memory + batch_memory + buffer_memory):.1f} MB")
    
    # Summary
    print("\n" + "="*60)
    print("Configuration Summary:")
    print("="*60)
    
    aggressive_score = 0
    
    # Calculate aggressiveness score
    if reward_config["profit_amplifier"] >= 5.0:
        aggressive_score += 2
    if agent_config["learning_rate"] >= 1e-4:
        aggressive_score += 2
    if agent_config["net_dims"][0] >= 512:
        aggressive_score += 1
    if env_config["max_position"] >= 3:
        aggressive_score += 1
    if train_config["target_sharpe_ratio"] >= 1.5:
        aggressive_score += 1
    
    print(f"Aggressiveness Score: {aggressive_score}/7")
    
    if aggressive_score >= 5:
        print("✓ Configuration is AGGRESSIVE - Ready for profit maximization!")
    elif aggressive_score >= 3:
        print("⚠ Configuration is MODERATE - Consider more aggressive parameters")
    else:
        print("✗ Configuration is CONSERVATIVE - Not aggressive enough")
    
    return True

def test_compatibility():
    """Test compatibility with existing codebase"""
    print("\n" + "="*60)
    print("Testing Codebase Compatibility")
    print("="*60)
    
    # Test imports
    print("\n1. Testing imports:")
    
    try:
        from reward_functions import create_reward_calculator
        print("   ✓ Reward functions imported")
    except ImportError as e:
        print(f"   ✗ Failed to import reward_functions: {e}")
    
    try:
        from profit_focused_rewards import ProfitFocusedRewardCalculator
        print("   ✓ Profit-focused rewards imported")
    except ImportError as e:
        print(f"   ✗ Failed to import profit_focused_rewards: {e}")
    
    try:
        from task1_optuna_hpo import LOBEnvironment
        print("   ✓ LOB Environment imported")
    except ImportError as e:
        print(f"   ✗ Failed to import LOBEnvironment: {e}")
    
    # Test data availability
    print("\n2. Testing data availability:")
    
    data_paths = [
        ("Original features", "../task1_data/BTC_1sec_predict.npy"),
        ("Reduced features", "../task1_data/BTC_1sec_predict_reduced.npy"),
        ("Feature selection", "../feature_selection/selected_features.json")
    ]
    
    for name, path in data_paths:
        full_path = Path(__file__).parent / path
        if full_path.exists():
            print(f"   ✓ {name}: Found")
        else:
            print(f"   ⚠ {name}: Not found at {full_path}")
    
    # Test reward calculator creation
    print("\n3. Testing reward calculator creation:")
    
    try:
        calc = create_reward_calculator("profit_focused", device="cpu")
        print("   ✓ Profit-focused calculator created")
        
        if hasattr(calc, 'profit_calculator'):
            print("   ✓ Profit calculator integrated")
        else:
            print("   ✗ Profit calculator not properly integrated")
    except Exception as e:
        print(f"   ✗ Failed to create calculator: {e}")
    
    print("\n" + "="*60)
    print("Compatibility test complete!")
    print("="*60)

if __name__ == "__main__":
    # Run tests
    config_valid = test_configuration()
    
    if config_valid:
        test_compatibility()
    
    print("\nReady to train? Run: python src/train_aggressive_profit.py")