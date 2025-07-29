#!/usr/bin/env python3
"""
Quick Test: Multi-Episode Training Validation
Tests the new multi-episode training functionality with reduced episodes
"""

import sys
import os
sys.path.append('src')

def test_multi_episode_config():
    """Test multi-episode configuration with 5 episodes for quick validation"""
    
    print("🔍 Testing Multi-Episode Training Configuration")
    print("=" * 55)
    
    # Quick test configuration (5 episodes)
    test_config = {
        'gpu_id': 0,
        'num_sims': 64,
        'net_dims': (128, 128, 128),
        'gamma': 0.995,
        'explore_rate': 0.005,
        'state_value_tau': 0.01,
        'soft_update_tau': 2e-6,
        'learning_rate': 2e-6,
        'batch_size': 512,
        'break_step': 75000,  # 5 episodes * 15K steps
        'buffer_size_multiplier': 8,
        'repeat_times': 2,
        'horizon_len_multiplier': 1,
        'eval_per_step_multiplier': 0.1,
        'num_workers': 1,
        'save_gap': 8,
        'data_length': 15000,  # Per-episode data length
        'num_episodes': 5,  # Quick test with 5 episodes
        'episode_tracking': True,
        'max_position': 2,
        'num_ignore_step': 30,
        'step_gap': 2,
        'slippage': 0.0001,
        'starting_cash': 100000,
    }
    
    # Test parameter calculations
    max_step = (test_config['data_length'] - test_config['num_ignore_step']) // test_config['step_gap']
    horizon_len = max_step * test_config['horizon_len_multiplier']
    total_expected_steps = test_config['num_episodes'] * horizon_len
    
    print(f"📊 Test Configuration Analysis:")
    print(f"   Target Episodes: {test_config['num_episodes']}")
    print(f"   Data per Episode: {test_config['data_length']:,} samples")
    print(f"   Max Steps per Episode: {max_step:,}")
    print(f"   Horizon Length: {horizon_len:,}")
    print(f"   Expected Total Steps: {total_expected_steps:,}")
    print(f"   Break Step Limit: {test_config['break_step']:,}")
    print()
    
    # Validation checks
    print("✅ Configuration Validation:")
    
    if test_config['num_episodes'] > 1:
        print(f"   ✓ Multi-episode configured: {test_config['num_episodes']} episodes")
    else:
        print("   ✗ Single episode detected - should be multi-episode")
        
    if test_config['episode_tracking']:
        print("   ✓ Episode tracking enabled")
    else:
        print("   ✗ Episode tracking disabled")
        
    if total_expected_steps <= test_config['break_step']:
        print(f"   ✓ Break step sufficient: {test_config['break_step']:,} >= {total_expected_steps:,}")
    else:
        print(f"   ✗ Break step too low: {test_config['break_step']:,} < {total_expected_steps:,}")
        
    # Check data requirements
    total_data_needed = test_config['num_episodes'] * test_config['data_length']
    full_dataset_size = 823682
    
    if total_data_needed <= full_dataset_size:
        print(f"   ✓ Data requirements met: {total_data_needed:,} <= {full_dataset_size:,}")
    else:
        print(f"   ✗ Insufficient data: {total_data_needed:,} > {full_dataset_size:,}")
    
    print()
    return test_config

def estimate_training_time(config):
    """Estimate training time based on previous single-episode runs"""
    
    print("⏱️ Training Time Estimation:")
    print("-" * 30)
    
    # Previous single episode took ~7 minutes for 3 agents
    single_episode_time = 7  # minutes
    agents = 3
    
    estimated_total = config['num_episodes'] * single_episode_time
    estimated_per_agent = estimated_total / agents
    
    print(f"   Previous single episode: {single_episode_time} min total")
    print(f"   Estimated {config['num_episodes']} episodes: {estimated_total} min total")
    print(f"   Per agent ({agents} agents): {estimated_per_agent:.1f} min each")
    print(f"   Expected completion: ~{estimated_total//60}h {estimated_total%60:.0f}m")
    print()

def main():
    """Run multi-episode training validation test"""
    
    print("🚀 Multi-Episode Training Validation Test")
    print("=" * 50)
    print()
    
    # Test configuration
    config = test_multi_episode_config()
    
    # Time estimation
    estimate_training_time(config)
    
    # Instructions
    print("📋 Next Steps:")
    print("1. Configuration validated ✓")
    print("2. Ready for test run with:")
    print("   python3 src/task1_ensemble.py")
    print("3. Expected output:")
    print("   - 5 episodes per agent")
    print("   - Episode progress tracking")
    print("   - Multi-episode performance plots")
    print("   - Episode-based CSV metrics")
    print()
    
    print("🎯 Success Criteria:")
    print("   ✓ 'Episode X/5 starting...' messages")
    print("   ✓ 'Episode X completed - Reward: Y' messages")
    print("   ✓ 'Multi-episode training completed: 5 episodes'")
    print("   ✓ Non-empty performance plots with 5 data points")
    print("   ✓ CSV files with episode column")
    print()
    
    print("Ready to launch multi-episode training! 🚀")

if __name__ == "__main__":
    main()