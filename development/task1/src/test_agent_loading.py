#!/usr/bin/env python3
"""
Quick test to verify agent loading and prediction functionality
"""

import os
import torch
import numpy as np
from comprehensive_backtester import ComprehensiveBacktester, BacktestConfig

def test_agent_loading():
    """Test that agents load correctly and can make predictions"""
    
    print("🧪 Testing Agent Loading and Predictions")
    print("=" * 60)
    
    # Create minimal config
    config = BacktestConfig(
        ensemble_path="ensemble_optimized_phase2/ensemble_models",
        walk_forward_window=500,  # Small window for testing
        monte_carlo_runs=5
    )
    
    try:
        # Initialize backtester
        print("📊 Initializing backtester...")
        backtester = ComprehensiveBacktester(config)
        
        # Check if agents loaded
        print(f"✅ Loaded {len(backtester.agents)} agents")
        
        if not backtester.agents:
            print("❌ No agents loaded - cannot test predictions")
            return False
        
        # Test prediction with dummy state
        print("🎯 Testing agent predictions...")
        dummy_state = torch.zeros(1, 8, dtype=torch.float32)  # 8-feature state
        
        for i, agent in enumerate(backtester.agents):
            try:
                # Test agent prediction
                with torch.no_grad():
                    q_values = agent.act(dummy_state)
                    action = q_values.argmax(dim=1)
                    
                print(f"✅ Agent {i+1} ({agent.__class__.__name__}): Action {action.item()}")
                
            except Exception as e:
                print(f"❌ Agent {i+1} prediction failed: {e}")
                return False
        
        # Test ensemble action
        print("🤝 Testing ensemble action...")
        actions = []
        for agent in backtester.agents:
            with torch.no_grad():
                q_values = agent.act(dummy_state)
                action = q_values.argmax(dim=1).unsqueeze(1)
                actions.append(action)
        
        ensemble_action = backtester._ensemble_action(actions)
        print(f"✅ Ensemble action: {ensemble_action.item()}")
        
        # Test a very small backtest
        print("🚀 Testing minimal backtest...")
        result = backtester.run_standard_backtest(start_idx=0, end_idx=200)
        
        print(f"✅ Backtest completed!")
        print(f"   Period: {result.period_name}")
        print(f"   Total Return: {result.total_return:.4f}")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"   Trades: {result.num_trades}")
        print(f"   Trade Log Length: {len(result.trade_log)}")
        
        # Verify trade log has actual trades with real data
        if result.trade_log:
            sample_trade = result.trade_log[0]
            print(f"   Sample Trade: {sample_trade}")
            
            # Check if trade has realistic data
            if 'price' in sample_trade and sample_trade['price'] > 0:
                print("✅ Trade log contains realistic price data")
            else:
                print("❌ Trade log has invalid price data")
                return False
        else:
            print("⚠️  No trades in trade log")
        
        print("\n🎉 All tests passed! Agents are loading and making predictions correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_loading()
    if success:
        print("\n✅ Agent loading test successful - backtesting framework is ready!")
    else:
        print("\n❌ Agent loading test failed - check configuration and models")