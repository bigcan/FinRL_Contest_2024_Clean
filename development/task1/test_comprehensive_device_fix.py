#!/usr/bin/env python3
"""
Test Comprehensive Device Consistency Fix
Validates that the CUDA device mismatch issue is resolved
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from task1_ensemble import run, AgentD3QN

def test_comprehensive_device_fix():
    """Test that Episodes 1 and 2 both complete successfully"""
    
    print("🧪 Testing Comprehensive Device Consistency Fix")
    print("=" * 60)
    print("This test validates that Episode 2+ can complete without CUDA device mismatch")
    print()
    
    # Quick 3-episode test configuration
    test_config = {
        'gpu_id': 0,
        'num_sims': 4,  # Small for speed
        'num_episodes': 3,  # Test episode transitions
        'data_length': 500,  # Small data for speed
        'break_step': 1500,  # Total steps for 3 episodes
        'batch_size': 64,  # Small batch
        'horizon_len_multiplier': 0.5,  # Reduce horizon
    }
    
    print(f"🔧 Test Configuration:")
    print(f"   Episodes: {test_config['num_episodes']}")
    print(f"   Data per episode: {test_config['data_length']}")
    print(f"   Total steps: {test_config['break_step']}")
    print()
    
    try:
        print("🎯 Starting multi-episode device consistency test...")
        run(
            save_path='comprehensive_device_test',
            agent_list=[AgentD3QN],  # Single agent for speed
            log_rules=False,
            config_dict=test_config
        )
        
        print()
        print("✅ COMPREHENSIVE DEVICE FIX SUCCESS!")
        print("✅ All 3 episodes completed without CUDA device mismatch")
        print("✅ Multi-episode training is now fully functional")
        return True
        
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print()
            print("❌ DEVICE FIX FAILED!")
            print(f"❌ CUDA device mismatch still occurs: {e}")
            return False
        else:
            print(f"❌ Unexpected RuntimeError: {e}")
            return False
    except Exception as e:
        print(f"❌ Test failed with error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    success = test_comprehensive_device_fix()
    if success:
        print("\n🚀 Ready for full 65-episode training!")
    sys.exit(0 if success else 1)