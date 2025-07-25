#!/usr/bin/env python3
"""
Diagnostic script to understand why the model isn't trading
"""

import os
import sys
import torch
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from erl_agent import AgentD3QN
from reward_functions import create_reward_calculator


def diagnose_model_behavior(model_path="ensemble_extended_phase1_20250724_233428/AgentD3QN"):
    """Diagnose why the model isn't trading"""
    
    print("🔍 TRADING BEHAVIOR DIAGNOSIS")
    print("=" * 60)
    
    # Load model
    temp_sim = TradeSimulator(num_sims=1)
    state_dim = temp_sim.state_dim
    
    agent = AgentD3QN(
        net_dims=(128, 64, 32),
        state_dim=state_dim,
        action_dim=3,
        gpu_id=-1,  # CPU for diagnosis
    )
    
    if os.path.exists(model_path):
        agent.save_or_load_agent(model_path, if_save=False)
        print(f"✅ Model loaded from: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        return
    
    # Test environment
    env = TradeSimulator(num_sims=1)
    env.set_reward_type("multi_objective")
    
    print(f"\n📊 Environment Setup:")
    print(f"   State dimension: {state_dim}")
    print(f"   Action space: {env.action_dim} (0=Hold, 1=Buy, 2=Sell)")
    print(f"   Starting cash: $1,000,000")
    
    # Collect actions over 100 steps
    state = env.reset()
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    
    actions = []
    q_values_history = []
    rewards = []
    
    print(f"\n🎮 Testing 100 trading steps...")
    
    for step in range(100):
        with torch.no_grad():
            q_values = agent.act(state)
            action = q_values.argmax(dim=1, keepdim=True)
            
            actions.append(action[0].item())
            q_values_history.append(q_values[0].cpu().numpy())
            
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward.item())
            
            if step < 10:  # Show first 10 steps
                print(f"   Step {step+1}: Q=[{q_values[0][0]:.3f}, {q_values[0][1]:.3f}, {q_values[0][2]:.3f}] → Action={action[0].item()} → Reward={reward.item():.3f}")
            
            state = next_state
            
            if done.any():
                break
    
    # Analyze results
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in actions:
        action_counts[action] += 1
    
    print(f"\n📈 DIAGNOSIS RESULTS:")
    print(f"   Total steps: {len(actions)}")
    print(f"   Action distribution:")
    print(f"     Hold (0): {action_counts[0]} ({action_counts[0]/len(actions)*100:.1f}%)")
    print(f"     Buy (1):  {action_counts[1]} ({action_counts[1]/len(actions)*100:.1f}%)")
    print(f"     Sell (2): {action_counts[2]} ({action_counts[2]/len(actions)*100:.1f}%)")
    
    print(f"\n💡 Q-Value Analysis:")
    q_array = np.array(q_values_history)
    avg_q = np.mean(q_array, axis=0)
    std_q = np.std(q_array, axis=0)
    
    print(f"   Average Q-values: Hold={avg_q[0]:.3f}, Buy={avg_q[1]:.3f}, Sell={avg_q[2]:.3f}")
    print(f"   Q-value std dev:  Hold={std_q[0]:.3f}, Buy={std_q[1]:.3f}, Sell={std_q[2]:.3f}")
    
    # Reward analysis
    avg_reward = np.mean(rewards)
    print(f"   Average reward: {avg_reward:.3f}")
    print(f"   Reward range: {min(rewards):.3f} to {max(rewards):.3f}")
    
    # Determine issue
    print(f"\n🧠 BEHAVIORAL ANALYSIS:")
    
    if action_counts[0] > 90:  # >90% hold
        print("   ❌ ISSUE: Model learned to be overly conservative (>90% HOLD)")
        print("   🔧 SOLUTION: Reduce transaction cost penalty or increase exploration")
        
        # Test different reward functions
        print(f"\n🧪 Testing different reward functions...")
        test_reward_functions(env, agent, state)
        
    elif max(q_array.flatten()) - min(q_array.flatten()) < 0.1:
        print("   ❌ ISSUE: Q-values too similar - poor action differentiation")
        print("   🔧 SOLUTION: Increase learning rate or training duration")
        
    else:
        print("   ✅ Model shows reasonable action diversity")
    
    return action_counts, q_array, rewards


def test_reward_functions(env, agent, initial_state):
    """Test different reward functions to see impact on behavior"""
    
    reward_types = ["simple", "transaction_cost_adjusted", "multi_objective"]
    
    for reward_type in reward_types:
        env.set_reward_type(reward_type)
        state = initial_state.clone()
        
        actions = []
        for _ in range(20):
            with torch.no_grad():
                q_values = agent.act(state)
                action = q_values.argmax(dim=1, keepdim=True)
                actions.append(action[0].item())
                
                next_state, reward, done, _ = env.step(action)
                state = next_state
                
                if done.any():
                    break
        
        action_counts = {0: 0, 1: 0, 2: 0}
        for action in actions:
            action_counts[action] += 1
        
        print(f"   {reward_type:20}: Hold={action_counts[0]:2d}, Buy={action_counts[1]:2d}, Sell={action_counts[2]:2d}")


def compare_models():
    """Compare extended vs baseline model behavior"""
    
    print("\n🔍 COMPARING MODEL BEHAVIORS")
    print("=" * 60)
    
    models = [
        ("Extended Model", "ensemble_extended_phase1_20250724_233428/AgentD3QN"),
        ("Baseline Model", "ensemble_optimized_phase2/ensemble_models/AgentD3QN"),
    ]
    
    for name, path in models:
        if os.path.exists(path):
            print(f"\n📊 Testing {name}:")
            action_counts, q_array, rewards = diagnose_model_behavior(path)
            
            # Brief summary
            total = sum(action_counts.values())
            hold_pct = action_counts[0] / total * 100 if total > 0 else 0
            print(f"   Summary: {hold_pct:.1f}% HOLD, Avg Q-range: {np.ptp(q_array):.3f}")
        else:
            print(f"\n❌ {name} not found at {path}")


if __name__ == "__main__":
    # Diagnose extended model
    diagnose_model_behavior()
    
    # Compare models
    compare_models()
    
    print(f"\n📋 RECOMMENDATIONS:")
    print(f"   1. Test with 'simple' reward function (no transaction costs)")
    print(f"   2. Increase exploration rate during training")
    print(f"   3. Reduce transaction cost penalty in multi_objective reward")
    print(f"   4. Add diversity bonus to reward function")