#!/usr/bin/env python3

# Quick test of the optimized ensemble training
import os
import sys
import torch

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_basic_training():
    """Test basic training components"""
    
    print("🧪 Quick Training Test")
    
    try:
        # Test data loading
        from trade_simulator import TradeSimulator
        
        print("📊 Testing TradeSimulator...")
        temp_sim = TradeSimulator(num_sims=1)
        state_dim = temp_sim.state_dim
        print(f"   State dimension: {state_dim}")
        
        # Test if optimized features are available
        if hasattr(temp_sim, 'feature_names'):
            print(f"   Features: {temp_sim.feature_names}")
        
        # Test agent initialization
        from erl_agent import AgentD3QN
        
        print("🤖 Testing Agent initialization...")
        agent = AgentD3QN(
            net_dims=(64, 32, 16),  # Smaller for testing
            state_dim=state_dim,
            action_dim=3,
            gpu_id=0
        )
        print("   Agent created successfully")
        
        # Test environment reset
        print("🔄 Testing environment...")
        state = temp_sim.reset()
        print(f"   Initial state shape: {state.shape if hasattr(state, 'shape') else type(state)}")
        
        print("✅ Basic components test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_training()