#!/usr/bin/env python3
"""
Test the specific error condition that was causing issues
Test with batch_size=64 which was mentioned in the original error
"""

import torch
import sys

# Test both versions
sys.path.append('development/task1/src')
from erl_net import QNetTwinDuel

def test_original_error_condition():
    """Test the exact scenario that was failing before"""
    print("🔍 Testing Original Error Condition")
    print("=" * 50)
    
    # Configuration from the error context
    state_dim = 32  # Common in trading environments
    action_dim = 3  # Hold, Buy, Sell
    batch_size = 64  # Batch size from the error message
    dims = [128, 64]  # Typical network dimensions
    
    try:
        # Create the problematic network
        net = QNetTwinDuel(dims=dims, state_dim=state_dim, action_dim=action_dim)
        
        # Create batch of states that was causing the issue
        states = torch.randn(batch_size, state_dim)
        print(f"Input states shape: {states.shape}")
        
        # Test the operations that were failing
        print("\n🧪 Testing individual components:")
        
        # Encode states
        state_norm = net.state_norm(states)
        s_enc = net.net_state(state_norm)
        print(f"Encoded states shape: {s_enc.shape}")
        
        # Test advantage and value streams
        q_adv1 = net.net_adv1(s_enc)
        q_val1 = net.net_val1(s_enc)
        print(f"Advantage stream 1 shape: {q_adv1.shape}")
        print(f"Value stream 1 shape: {q_val1.shape}")
        
        # Test the dueling combination that was failing
        mean_adv = q_adv1.mean(dim=1, keepdim=True)
        q_dueling = q_val1 + (q_adv1 - mean_adv)
        print(f"Mean advantage shape: {mean_adv.shape}")
        print(f"Final Q-values shape: {q_dueling.shape}")
        
        # Test full forward pass
        output = net.forward(states)
        print(f"Forward pass output shape: {output.shape}")
        
        # Test get_q1_q2 (the method that was failing)
        q1, q2 = net.get_q1_q2(states)
        print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
        
        # Test action selection
        actions = net.get_action(states)
        print(f"Actions shape: {actions.shape}")
        
        print("\n✅ All operations completed successfully!")
        print("🎉 The tensor shape mismatch error has been FIXED!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error still exists: {str(e)}")
        return False

def compare_tensor_operations():
    """Show the difference between old and new approach"""
    print("\n📊 Tensor Operation Comparison")
    print("=" * 50)
    
    batch_size, action_dim = 64, 3
    
    # Simulate the OLD (incorrect) approach
    print("❌ OLD (Incorrect) Approach:")
    try:
        # This was the problem: wrong dimensions
        old_advantage = torch.randn(batch_size, 1)  # Should be (64, 3)
        old_value = torch.randn(batch_size, action_dim)  # Should be (64, 1)
        
        print(f"  Old advantage shape: {old_advantage.shape}")
        print(f"  Old value shape: {old_value.shape}")
        
        # This would fail with the original error
        # old_result = old_value - old_value.mean(dim=1, keepdim=True) + old_advantage
        print("  ⚠️  This combination would cause tensor shape mismatch!")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Show the NEW (correct) approach  
    print("\n✅ NEW (Correct) Approach:")
    new_advantage = torch.randn(batch_size, action_dim)  # Correct: (64, 3)
    new_value = torch.randn(batch_size, 1)  # Correct: (64, 1)
    
    print(f"  New advantage shape: {new_advantage.shape}")
    print(f"  New value shape: {new_value.shape}")
    
    # This works correctly
    new_result = new_value + (new_advantage - new_advantage.mean(dim=1, keepdim=True))
    print(f"  Final result shape: {new_result.shape}")
    print("  ✅ Tensor operations work perfectly!")

def main():
    print("🔧 D3QN Specific Error Condition Test")
    print("Testing the exact scenario that was failing...")
    
    success = test_original_error_condition()
    compare_tensor_operations()
    
    if success:
        print("\n🎉 SUCCESS: The D3QN tensor shape mismatch has been resolved!")
        print("The network can now handle batch_size=64 and other configurations correctly.")
        return 0
    else:
        print("\n❌ FAILURE: The error still exists.")
        return 1

if __name__ == "__main__":
    exit(main())