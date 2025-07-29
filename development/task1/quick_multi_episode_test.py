#!/usr/bin/env python3
"""
Quick Multi-Episode Test - 3 Episodes for Fast Validation
"""

import sys
import os
sys.path.append('src')
from task1_ensemble import run

def main():
    """Run quick 3-episode test"""
    
    print("🚀 Quick Multi-Episode Training Test (3 Episodes)")
    print("=" * 55)
    
    # Quick test configuration
    quick_config = {
        'gpu_id': 0,
        'num_sims': 64,
        'data_length': 8000,  # Smaller episodes for faster testing
        'num_episodes': 3,    # Quick test with 3 episodes
        'break_step': 36000,  # 3 episodes * 12K steps
        'horizon_len_multiplier': 1,
        'episode_tracking': True,
        'learning_rate': 2e-6,
        'batch_size': 512,
    }
    
    # Calculate expected parameters
    max_step = (quick_config['data_length'] - 30) // 2  # Using defaults
    horizon_len = max_step * quick_config['horizon_len_multiplier']
    
    print(f"📊 Quick Test Parameters:")
    print(f"   Episodes: {quick_config['num_episodes']}")
    print(f"   Data per episode: {quick_config['data_length']:,}")
    print(f"   Steps per episode: {max_step:,}")
    print(f"   Expected duration: ~15 minutes")
    print()
    
    # Import agent classes
    from erl_agent import AgentD3QN
    
    # Agent list for quick test (just one agent for speed)
    agent_list = [AgentD3QN]
    
    save_path = "quick_multi_episode_test"
    
    print("🎯 Starting quick multi-episode test...")
    print("Expected behavior:")
    print("   - 3 episodes of training")
    print("   - Episode progress messages")
    print("   - Performance tracking per episode")
    print()
    
    try:
        # Run the training
        run(
            save_path=save_path,
            agent_list=agent_list,
            log_rules=False,
            config_dict=quick_config
        )
        
        print("✅ Quick multi-episode test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()