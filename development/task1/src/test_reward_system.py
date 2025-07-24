#!/usr/bin/env python3
"""
Test the Enhanced Reward System
Quick validation of the new risk-adjusted reward functions
"""

import torch as th
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_simulator import TradeSimulator
from reward_functions import create_reward_calculator

def test_reward_system():
    """Test the enhanced reward system with different variants"""
    
    print("🧪 Testing Enhanced Reward System")
    print("=" * 60)
    
    # Test all reward types
    reward_types = ["simple", "transaction_cost_adjusted", "sharpe_adjusted", "multi_objective"]
    
    # Create simulator with 1 environment for testing
    try:
        simulator = TradeSimulator(num_sims=1, device=th.device("cpu"))
        print(f"✅ TradeSimulator created successfully")
        print(f"   State dimension: {simulator.state_dim}")
        print(f"   Default reward type: {simulator.reward_type}")
        
        # Test each reward type
        for reward_type in reward_types:
            print(f"\n🎯 Testing '{reward_type}' reward:")
            
            # Switch reward type
            simulator.set_reward_type(reward_type)
            
            # Reset and take some test actions
            state = simulator.reset()
            print(f"   Initial state shape: {state.shape}")
            
            # Simulate some trading actions
            total_reward = 0
            test_actions = [
                th.tensor([[1]], dtype=th.long),  # Buy
                th.tensor([[0]], dtype=th.long),  # Hold  
                th.tensor([[2]], dtype=th.long),  # Sell
                th.tensor([[0]], dtype=th.long),  # Hold
                th.tensor([[1]], dtype=th.long),  # Buy
            ]
            
            for i, action in enumerate(test_actions):
                state, reward, done, info = simulator.step(action)
                total_reward += reward.item()
                print(f"   Step {i+1}: Action={action.item()}, Reward={reward.item():.4f}")
                
                if done.any():
                    break
            
            print(f"   📊 Total Reward: {total_reward:.4f}")
            
            # Get performance metrics
            simulator.print_reward_performance()
        
        print(f"\n✅ All reward types tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing reward system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_calculator_standalone():
    """Test the reward calculator in isolation"""
    
    print(f"\n🔬 Testing RewardCalculator Standalone")
    print("-" * 40)
    
    device = "cpu"
    
    # Test scenario: profitable trade
    old_asset = th.tensor([1000000.0], device=device)
    new_asset = th.tensor([1005000.0], device=device)  # $5000 profit
    action_int = th.tensor([1], device=device)  # Buy action
    mid_price = th.tensor([50000.0], device=device)  # BTC at $50k
    slippage = 7e-7
    
    for reward_type in ["simple", "transaction_cost_adjusted", "sharpe_adjusted", "multi_objective"]:
        calc = create_reward_calculator(reward_type, device=device)
        
        reward = calc.calculate_reward(old_asset, new_asset, action_int, mid_price, slippage)
        print(f"   {reward_type:25}: {reward.item():8.2f}")
        
        # Get metrics
        metrics = calc.get_performance_metrics()
        if 'total_transaction_costs' in metrics:
            print(f"   {'':25}  Transaction costs: ${metrics['total_transaction_costs']:.2f}")

def compare_with_baseline():
    """Compare new rewards with the original simple reward"""
    
    print(f"\n📊 Baseline Comparison Analysis")
    print("-" * 40)
    
    # Test scenarios
    scenarios = [
        {"name": "Small Profit", "old": 1000000, "new": 1001000, "action": 1},
        {"name": "Large Profit", "old": 1000000, "new": 1010000, "action": 1},
        {"name": "Small Loss", "old": 1000000, "new": 999000, "action": 2},
        {"name": "Large Loss", "old": 1000000, "new": 990000, "action": 2},
        {"name": "No Change", "old": 1000000, "new": 1000000, "action": 0},
    ]
    
    device = "cpu"
    mid_price = th.tensor([50000.0], device=device)
    slippage = 7e-7
    
    for scenario in scenarios:
        print(f"\n📈 {scenario['name']}:")
        old_asset = th.tensor([float(scenario['old'])], device=device)
        new_asset = th.tensor([float(scenario['new'])], device=device)
        action_int = th.tensor([scenario['action']], device=device)
        
        # Test each reward type
        for reward_type in ["simple", "multi_objective"]:
            calc = create_reward_calculator(reward_type, device=device)
            reward = calc.calculate_reward(old_asset, new_asset, action_int, mid_price, slippage)
            print(f"   {reward_type:20}: {reward.item():8.2f}")

if __name__ == "__main__":
    print("🚀 Enhanced Reward System Testing")
    print("=" * 60)
    
    # Test 1: Standalone reward calculator
    test_reward_calculator_standalone()
    
    # Test 2: Baseline comparison
    compare_with_baseline()
    
    # Test 3: Full integration test
    success = test_reward_system()
    
    if success:
        print(f"\n🎉 All tests passed! Enhanced reward system is ready.")
        print(f"📋 Next steps:")
        print(f"   1. Update training configuration to use new rewards")
        print(f"   2. Extend training duration")
        print(f"   3. Run comparative analysis vs baseline")
    else:
        print(f"\n⚠️  Some tests failed. Please check the implementation.")