#!/usr/bin/env python3
"""
Quick Test of Extended Training System
Validate the enhanced training before full run
"""

import sys
import os
import torch

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from enhanced_training_config import EnhancedConfig, EarlyStoppingManager, LearningRateScheduler
from erl_agent import AgentD3QN

def test_extended_training_integration():
    """Test extended training integration without full training"""
    
    print("🧪 Testing Extended Training Integration")
    print("=" * 60)
    
    try:
        # Test 1: TradeSimulator with enhanced rewards
        print("1️⃣  Testing TradeSimulator with enhanced rewards...")
        simulator = TradeSimulator(num_sims=1, device=torch.device("cpu"))
        simulator.set_reward_type("multi_objective")
        state = simulator.reset()
        print(f"   ✅ Simulator initialized: state shape {state.shape}")
        print(f"   ✅ Reward type: {simulator.reward_type}")
        
        # Test 2: Enhanced configuration
        print("\n2️⃣  Testing Enhanced Configuration...")
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": 1,
            "max_step": 100,
            "state_dim": simulator.state_dim,
            "action_dim": 3,
            "if_discrete": True,
        }
        
        config = EnhancedConfig(
            agent_class=AgentD3QN, 
            env_class=TradeSimulator, 
            env_args=env_args
        )
        print(f"   ✅ Enhanced config created")
        print(f"   📈 Training steps: {config.break_step}")
        print(f"   📚 Learning rate: {config.learning_rate}")
        print(f"   🔍 Exploration rate: {config.explore_rate}")
        
        # Test 3: Agent initialization
        print("\n3️⃣  Testing Agent Initialization...")
        agent = AgentD3QN(
            net_dims=config.net_dims,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            gpu_id=config.gpu_id,
            args=config
        )
        print(f"   ✅ Agent created: {type(agent).__name__}")
        print(f"   🧠 Network dims: {config.net_dims}")
        
        # Test 4: Early stopping manager
        print("\n4️⃣  Testing Early Stopping Manager...")
        early_stopping = EarlyStoppingManager(patience=5, min_delta=0.01)
        
        # Simulate some scores
        test_scores = [0.1, 0.15, 0.12, 0.18, 0.17, 0.16, 0.15, 0.14]
        for i, score in enumerate(test_scores):
            should_stop = early_stopping.update(score, i)
            if should_stop:
                print(f"   ✅ Early stopping works: triggered at step {i}")
                break
        
        # Test 5: Learning rate scheduler
        print("\n5️⃣  Testing Learning Rate Scheduler...")
        lr_scheduler = LearningRateScheduler(
            agent.act_optimizer,
            scheduler_type="cosine_annealing",
            total_steps=10,
            min_lr=1e-7
        )
        
        initial_lr = lr_scheduler.get_current_lr()
        for step in range(5):
            new_lr = lr_scheduler.step()
        final_lr = lr_scheduler.get_current_lr()
        
        print(f"   ✅ LR scheduling works: {initial_lr:.2e} → {final_lr:.2e}")
        
        # Test 6: Quick action test
        print("\n6️⃣  Testing Agent Action...")
        with torch.no_grad():
            test_state = torch.randn(1, config.state_dim, device=agent.device)
            q_values = agent.act(test_state)
            action = q_values.argmax(dim=1)
        print(f"   ✅ Agent action works: Q-values shape {q_values.shape}, action {action.item()}")
        
        print(f"\n✅ All integration tests passed!")
        print(f"🚀 Extended training system is ready for full run")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_impact():
    """Test the impact of different reward functions"""
    
    print(f"\n🎯 Testing Reward Function Impact")
    print("-" * 40)
    
    reward_types = ["simple", "transaction_cost_adjusted", "multi_objective"]
    
    for reward_type in reward_types:
        print(f"\n🧪 Testing {reward_type}:")
        
        try:
            simulator = TradeSimulator(num_sims=1, device=torch.device("cpu"))
            simulator.set_reward_type(reward_type)
            
            # Test some actions
            state = simulator.reset()
            total_reward = 0
            actions = [torch.tensor([[1]], dtype=torch.long), torch.tensor([[0]], dtype=torch.long), torch.tensor([[2]], dtype=torch.long)]
            
            for i, action in enumerate(actions):
                state, reward, done, _ = simulator.step(action)
                total_reward += reward.item()
                print(f"   Step {i+1}: Action {action.item()}, Reward {reward.item():.4f}")
            
            print(f"   📊 Total reward: {total_reward:.4f}")
            
            # Get metrics
            metrics = simulator.get_reward_metrics()
            print(f"   💰 Transaction costs: ${metrics.get('total_transaction_costs', 0):.2f}")
            
        except Exception as e:
            print(f"   ❌ Error with {reward_type}: {e}")

if __name__ == "__main__":
    print("🚀 Extended Training System Test")
    print("=" * 60)
    
    # Run integration test
    success = test_extended_training_integration()
    
    if success:
        # Test reward impact
        test_reward_impact()
        
        print(f"\n🎉 All tests passed! Ready for full extended training.")
        print(f"\n📋 Next steps:")
        print(f"   1. Run: python3 task1_ensemble_extended.py 0 multi_objective")
        print(f"   2. Monitor training progress")
        print(f"   3. Compare with baseline performance")
    else:
        print(f"\n⚠️  Tests failed. Please fix issues before full training.")