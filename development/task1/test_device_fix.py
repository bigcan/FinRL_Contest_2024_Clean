#!/usr/bin/env python3
"""
Test Device Consistency Fix
Quick validation that the CUDA device mismatch is resolved
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the training module
from src.task1_ensemble import run
from src.erl_agent import AgentD3QN, AgentDoubleDQN

def test_device_fix():
    """Test the device consistency fix with 3 episodes"""
    
    print("🧪 Testing Device Consistency Fix")
    print("=" * 50)
    
    # Configuration for quick test
    test_config = {
        'gpu_id': 0,
        'num_sims': 8,  # Smaller for quick test
        'num_episodes': 3,  # Just test episode transitions
        'data_length': 1000,  # Small data length for speed
        'break_step': 3000,  # Small break step
        'episode_tracking': True,
        'batch_size': 128,  # Smaller batch
        'horizon_len_multiplier': 0.5,  # Reduce horizon length
    }
    
    print(f"🔧 Test Configuration:")
    print(f"   Episodes: {test_config['num_episodes']}")
    print(f"   Data per episode: {test_config['data_length']}")
    print(f"   GPU ID: {test_config['gpu_id']}")
    print(f"   Parallel envs: {test_config['num_sims']}")
    print()
    
    try:
        # Test with single agent first
        print("🎯 Testing with D3QN agent...")
        run(
            save_path='device_fix_test',
            agent_list=[AgentD3QN],  # Single agent for speed
            log_rules=False,
            config_dict=test_config
        )
        
        print("✅ Device fix test PASSED!")
        print("✅ Multi-episode transitions working correctly")
        return True
        
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print("❌ Device fix test FAILED!")
            print(f"❌ Still getting CUDA device mismatch: {e}")
            return False
        else:
            print(f"❌ Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_device_fix()
    sys.exit(0 if success else 1)