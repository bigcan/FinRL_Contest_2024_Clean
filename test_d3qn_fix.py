#!/usr/bin/env python3
"""
Test script to validate D3QN tensor shape fixes
Tests the corrected QNetTwinDuel implementation with various batch sizes
"""

import torch
import sys
import os

# Add the development path for imports
sys.path.append('development/task1/src')

try:
    from erl_net import QNetTwinDuel, QNetTwin
    print("✅ Successfully imported fixed networks from development/task1/src")
except ImportError:
    # Fallback to original
    sys.path.append('original/Task_1_starter_kit')
    from erl_net import QNetTwinDuel, QNetTwin
    print("⚠️  Using original networks from original/Task_1_starter_kit")

def test_network_shapes():
    """Test network with various configurations"""
    
    # Test configurations
    configs = [
        {"state_dim": 32, "action_dim": 3, "batch_size": 1, "dims": [64, 32]},
        {"state_dim": 32, "action_dim": 3, "batch_size": 64, "dims": [64, 32]},
        {"state_dim": 16, "action_dim": 5, "batch_size": 32, "dims": [128, 64]},
        {"state_dim": 8, "action_dim": 2, "batch_size": 128, "dims": [32, 16]},
    ]
    
    print("\n🧪 Testing QNetTwinDuel (D3QN) Architecture:")
    print("=" * 60)
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: {config}")
        try:
            # Create network
            net = QNetTwinDuel(
                dims=config["dims"],
                state_dim=config["state_dim"], 
                action_dim=config["action_dim"]
            )
            
            # Create test input
            test_state = torch.randn(config["batch_size"], config["state_dim"])
            
            # Test forward pass
            print(f"  Input shape: {test_state.shape}")
            output = net.forward(test_state)
            print(f"  Forward output shape: {output.shape}")
            expected_shape = (config["batch_size"], config["action_dim"])
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            # Test get_q1_q2
            q1, q2 = net.get_q1_q2(test_state)
            print(f"  Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
            assert q1.shape == expected_shape, f"Q1 expected {expected_shape}, got {q1.shape}"
            assert q2.shape == expected_shape, f"Q2 expected {expected_shape}, got {q2.shape}"
            
            # Test get_action
            action = net.get_action(test_state)
            print(f"  Action shape: {action.shape}")
            expected_action_shape = (config["batch_size"], 1)
            assert action.shape == expected_action_shape, f"Action expected {expected_action_shape}, got {action.shape}"
            
            # Verify action values are within valid range
            assert torch.all(action >= 0) and torch.all(action < config["action_dim"]), "Actions out of range"
            
            print(f"  ✅ Test {i+1} PASSED")
            
        except Exception as e:
            print(f"  ❌ Test {i+1} FAILED: {str(e)}")
            return False
    
    return True

def test_gradient_flow():
    """Test that gradients flow properly through the network"""
    print("\n🔄 Testing Gradient Flow:")
    print("=" * 40)
    
    try:
        # Create network
        net = QNetTwinDuel(dims=[64, 32], state_dim=16, action_dim=3)
        
        # Create test data
        state = torch.randn(32, 16, requires_grad=True)
        target = torch.randn(32, 3)
        
        # Forward pass
        output = net.forward(state)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        param_count = 0
        grad_count = 0
        for param in net.parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
        
        print(f"  Parameters with gradients: {grad_count}/{param_count}")
        
        if grad_count > 0:
            print("  ✅ Gradient flow test PASSED")
            return True
        else:
            print("  ❌ Gradient flow test FAILED: No gradients found")
            return False
            
    except Exception as e:
        print(f"  ❌ Gradient flow test FAILED: {str(e)}")
        return False

def test_dueling_formula():
    """Test that the dueling formula produces expected results"""
    print("\n🎯 Testing Dueling Formula Logic:")
    print("=" * 45)
    
    try:
        net = QNetTwinDuel(dims=[32, 16], state_dim=8, action_dim=3)
        
        # Create test state
        state = torch.randn(4, 8)
        
        # Get network components
        s_enc = net.net_state(net.state_norm(state))
        q_adv = net.net_adv1(s_enc)  # [4, 3]
        q_val = net.net_val1(s_enc)  # [4, 1]
        
        print(f"  Encoded state shape: {s_enc.shape}")
        print(f"  Advantage shape: {q_adv.shape}")
        print(f"  Value shape: {q_val.shape}")
        
        # Test dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        mean_adv = q_adv.mean(dim=1, keepdim=True)  # [4, 1]
        q_dueling = q_val + (q_adv - mean_adv)  # [4, 3]
        
        print(f"  Mean advantage shape: {mean_adv.shape}")
        print(f"  Final Q-values shape: {q_dueling.shape}")
        
        # Verify the advantage property: mean advantage should be 0
        final_mean_adv = (q_dueling - q_val).mean(dim=1)
        print(f"  Final mean advantage (should be ~0): {final_mean_adv.abs().max().item():.6f}")
        
        if final_mean_adv.abs().max().item() < 1e-6:
            print("  ✅ Dueling formula test PASSED")
            return True
        else:
            print("  ❌ Dueling formula test FAILED: Mean advantage not zero")
            return False
            
    except Exception as e:
        print(f"  ❌ Dueling formula test FAILED: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🔧 D3QN Tensor Shape Fix Validation")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    tests = [
        ("Network Shape Tests", test_network_shapes),
        ("Gradient Flow Test", test_gradient_flow),
        ("Dueling Formula Test", test_dueling_formula),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🚀 Running {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\n📊 Final Results:")
    print(f"   Passed: {passed}/{total} tests")
    
    if passed == total:
        print("   🎉 ALL TESTS PASSED! D3QN fix is working correctly.")
        return 0
    else:
        print("   ⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())