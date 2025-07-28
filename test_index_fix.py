#!/usr/bin/env python3
"""
Test script to validate the index out of bounds fix in replay buffers
"""

import torch
import sys
import numpy as np

# Add the development path for imports
sys.path.append('development/task1/src')

from erl_replay_buffer import ReplayBuffer
from erl_per_buffer import PrioritizedReplayBuffer

def test_replay_buffer_fix():
    """Test the fixed ReplayBuffer"""
    print("🧪 Testing ReplayBuffer Fix")
    print("=" * 40)
    
    # Test configuration similar to the failing case
    max_size = 200000  # Large buffer
    state_dim = 16
    action_dim = 1
    num_seqs = 16
    batch_size = 256
    
    buffer = ReplayBuffer(
        max_size=max_size,
        state_dim=state_dim, 
        action_dim=action_dim,
        num_seqs=num_seqs,
        gpu_id=-1  # Use CPU for testing
    )
    
    print(f"Buffer initialized: max_size={max_size}, state_dim={state_dim}")
    print(f"Action_dim={action_dim}, num_seqs={num_seqs}")
    
    # Simulate the problematic scenario - fill buffer to a specific size
    target_size = 164737  # From the original error
    steps_to_add = target_size // 100  # Add in chunks
    
    for step in range(steps_to_add):
        # Generate random data
        states = torch.randn(100, num_seqs, state_dim)
        actions = torch.randn(100, num_seqs, action_dim)  
        rewards = torch.randn(100, num_seqs)
        undones = torch.ones(100, num_seqs)
        
        buffer.update((states, actions, rewards, undones))
        
        if step % 100 == 0:
            print(f"  Step {step}: cur_size={buffer.cur_size}")
    
    print(f"Final buffer size: {buffer.cur_size}")
    
    # Now test sampling - this should not cause index errors
    try:
        for test_batch in [64, 128, 256, 512]:
            print(f"  Testing batch_size={test_batch}...")
            states, actions, rewards, undones, next_states = buffer.sample(test_batch)
            
            # Validate shapes
            expected_state_shape = (test_batch, state_dim)
            expected_action_shape = (test_batch, action_dim)
            expected_reward_shape = (test_batch,)
            
            assert states.shape == expected_state_shape, f"States shape mismatch: {states.shape} != {expected_state_shape}"
            assert actions.shape == expected_action_shape, f"Actions shape mismatch: {actions.shape} != {expected_action_shape}"
            assert rewards.shape == expected_reward_shape, f"Rewards shape mismatch: {rewards.shape} != {expected_reward_shape}"
            assert undones.shape == expected_reward_shape, f"Undones shape mismatch: {undones.shape} != {expected_reward_shape}"
            assert next_states.shape == expected_state_shape, f"Next states shape mismatch: {next_states.shape} != {expected_state_shape}"
            
            print(f"    ✅ Batch size {test_batch} successful")
        
        print("✅ ReplayBuffer fix working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ ReplayBuffer test failed: {str(e)}")
        return False

def test_per_buffer_fix():
    """Test the fixed PrioritizedReplayBuffer"""
    print("\n🧪 Testing PrioritizedReplayBuffer Fix")
    print("=" * 40)
    
    # Test configuration
    max_size = 100000
    state_dim = 16
    action_dim = 1
    num_seqs = 16
    batch_size = 256
    
    buffer = PrioritizedReplayBuffer(
        max_size=max_size,
        state_dim=state_dim,
        action_dim=action_dim,
        num_seqs=num_seqs,
        gpu_id=-1
    )
    
    print(f"PER Buffer initialized: max_size={max_size}, state_dim={state_dim}")
    
    # Add some data
    target_size = 50000
    steps_to_add = target_size // 100
    
    for step in range(steps_to_add):
        states = torch.randn(100, num_seqs, state_dim)
        actions = torch.randn(100, num_seqs, action_dim)
        rewards = torch.randn(100, num_seqs)
        undones = torch.ones(100, num_seqs)
        
        buffer.update((states, actions, rewards, undones))
        
        if step % 100 == 0:
            print(f"  Step {step}: cur_size={buffer.cur_size}")
    
    print(f"Final PER buffer size: {buffer.cur_size}")
    
    # Test sampling
    try:
        for test_batch in [64, 128, 256]:
            print(f"  Testing PER batch_size={test_batch}...")
            states, actions, rewards, undones, next_states, indices, weights = buffer.sample(test_batch)
            
            # Validate shapes
            expected_state_shape = (test_batch, state_dim)  
            expected_action_shape = (test_batch, action_dim)
            expected_reward_shape = (test_batch,)
            
            assert states.shape == expected_state_shape
            assert actions.shape == expected_action_shape
            assert rewards.shape == expected_reward_shape
            assert undones.shape == expected_reward_shape
            assert next_states.shape == expected_state_shape
            assert indices.shape == expected_reward_shape
            assert weights.shape == expected_reward_shape
            
            print(f"    ✅ PER Batch size {test_batch} successful")
        
        print("✅ PrioritizedReplayBuffer fix working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ PrioritizedReplayBuffer test failed: {str(e)}")
        return False

def simulate_original_error():
    """Simulate the exact conditions that caused the original error"""
    print("\n🎯 Simulating Original Error Conditions")
    print("=" * 45)
    
    # Exact numbers from the error
    trying_to_access = 275140
    available_size = 164737
    
    print(f"Original error: trying to access index {trying_to_access}")
    print(f"Available size: {available_size}")
    
    # Create a buffer with this exact size
    buffer = ReplayBuffer(
        max_size=available_size + 1000,  # Slightly larger
        state_dim=16,
        action_dim=1,
        num_seqs=16,
        gpu_id=-1
    )
    
    # Fill to exact size  
    states = torch.randn(available_size, 16, 16)
    actions = torch.randn(available_size, 16, 1)
    rewards = torch.randn(available_size, 16)
    undones = torch.ones(available_size, 16)
    
    buffer.update((states, actions, rewards, undones))
    
    print(f"Buffer filled to size: {buffer.cur_size}")
    
    # Now try the sampling that would have failed before
    try:
        # This was the problematic call
        batch_size = 256
        states, actions, rewards, undones, next_states = buffer.sample(batch_size)
        
        print(f"✅ Sampling succeeded with batch_size={batch_size}")
        print(f"   States shape: {states.shape}")
        print(f"   Next states shape: {next_states.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sampling still fails: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🔧 Replay Buffer Index Fix Validation")
    print("=" * 50)
    
    torch.manual_seed(42)  # Reproducible results
    
    tests = [
        ("ReplayBuffer Fix", test_replay_buffer_fix),
        ("PrioritizedReplayBuffer Fix", test_per_buffer_fix), 
        ("Original Error Simulation", simulate_original_error),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
    
    print(f"\n📊 Final Results:")
    print(f"   Passed: {passed}/{total} tests")
    
    if passed == total:
        print("   🎉 ALL TESTS PASSED! Index fix working correctly.")
        print("   The training pipeline should now work without index errors.")
        return 0
    else:
        print("   ⚠️  Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())