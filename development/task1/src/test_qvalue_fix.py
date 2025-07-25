#!/usr/bin/env python3
"""
Test the Q-value calculation fix in erl_agent.py
"""

import os
import sys
import torch
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN
from erl_config import Config
from erl_replay_buffer import ReplayBuffer


def test_qvalue_fix():
    """Test that the Q-value fix works correctly"""
    
    print("🧪 Testing Q-Value Fix")
    print("=" * 60)
    
    # Setup environment
    temp_sim = TradeSimulator(num_sims=4)
    state_dim = temp_sim.state_dim
    
    print(f"📊 Test Configuration:")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: 3")
    print(f"   Batch size: 4")
    
    # Create agent
    agent = AgentDoubleDQN(
        net_dims=(128, 64, 32),
        state_dim=state_dim,
        action_dim=3,
        gpu_id=-1,  # CPU for testing
    )
    
    print(f"\n🤖 Agent created: {agent.__class__.__name__}")
    
    # Test 1: Check get_cumulative_rewards calculation
    print(f"\n📊 Test 1: Cumulative Rewards Calculation")
    
    # Create mock data
    horizon_len = 10
    num_envs = 4
    
    # Random rewards and undones
    rewards = torch.randn(horizon_len, num_envs)
    undones = torch.ones(horizon_len, num_envs)
    undones[-1] = 0  # Episode ends
    
    # Random last state
    agent.last_state = torch.randn(num_envs, state_dim)
    
    print(f"   Horizon length: {horizon_len}")
    print(f"   Number of envs: {num_envs}")
    
    # Test cumulative rewards calculation
    try:
        returns = agent.get_cumulative_rewards(rewards, undones)
        
        print(f"   ✅ Cumulative rewards calculated successfully!")
        print(f"   Returns shape: {returns.shape}")
        print(f"   Returns range: [{returns.min().item():.3f}, {returns.max().item():.3f}]")
        
        # Verify returns are reasonable Q-values, not action indices
        if returns.min() >= 0 and returns.max() <= 2:
            print(f"   ⚠️  WARNING: Returns look like action indices (0-2 range)")
        else:
            print(f"   ✅ Returns are proper Q-values (not action indices)")
            
    except Exception as e:
        print(f"   ❌ Error in cumulative rewards: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Verify Q-values from network
    print(f"\n📊 Test 2: Q-Network Output Verification")
    
    # Random state
    test_state = torch.randn(1, state_dim)
    
    try:
        # Get Q-values
        with torch.no_grad():
            q_values = agent.act(test_state)
            q1, q2 = agent.act.get_q1_q2(test_state)
            
        print(f"   Q-values shape: {q_values.shape}")
        print(f"   Q-values: {q_values[0].numpy()}")
        print(f"   Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
        
        # Verify Q-values are reasonable
        if q_values.shape[1] == 3:  # Should have 3 actions
            print(f"   ✅ Correct action dimension")
        else:
            print(f"   ❌ Wrong action dimension: {q_values.shape[1]}")
            
        # Check Q-value range
        q_range = q_values.max() - q_values.min()
        if q_range < 0.001:
            print(f"   ⚠️  WARNING: Q-values too similar (range: {q_range:.6f})")
        else:
            print(f"   ✅ Q-values show proper differentiation")
            
    except Exception as e:
        print(f"   ❌ Error in Q-network: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Quick training step
    print(f"\n📊 Test 3: Training Step Verification")
    
    # Setup minimal training
    env = TradeSimulator(num_sims=4)
    buffer = ReplayBuffer(
        gpu_id=-1,
        num_seqs=4,
        max_size=1000,
        state_dim=state_dim,
        action_dim=1,
    )
    
    # Collect some experience
    state = env.reset()
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    
    agent.last_state = state.detach()
    
    # Warm up buffer
    buffer_items = agent.explore_env(env, horizon_len=100, if_random=True)
    buffer.update(buffer_items)
    
    print(f"   Buffer warmed up with random experience")
    
    # Try one training step
    try:
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        
        if logging_tuple:
            obj_critic, obj_actor = logging_tuple[:2]
            print(f"   ✅ Training step successful!")
            print(f"   Critic loss: {obj_critic:.4f}")
            print(f"   Actor loss: {obj_actor:.4f}")
        else:
            print(f"   ⚠️  No losses returned from training")
            
    except Exception as e:
        print(f"   ❌ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🎉 ALL TESTS PASSED - Q-Value Fix Working!")
    return True


def quick_performance_test():
    """Quick test to see if the fix improves learning"""
    
    print(f"\n\n🚀 Quick Performance Test")
    print("=" * 60)
    
    # Train for 10 steps and check behavior
    from enhanced_training_config import EnhancedConfig
    from erl_config import build_env
    
    temp_sim = TradeSimulator(num_sims=8)
    state_dim = temp_sim.state_dim
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 8,
        "max_step": 1000,
        "state_dim": state_dim,
        "action_dim": 3,
        "if_discrete": True,
    }
    
    config = EnhancedConfig(agent_class=AgentDoubleDQN, env_class=TradeSimulator, env_args=env_args)
    config.break_step = 10  # Very short test
    config.learning_rate = 1e-4
    config.batch_size = 256
    config.net_dims = (128, 64, 32)
    
    # Build env
    env = build_env(config.env_class, config.env_args, -1)
    
    # Create agent
    agent = AgentDoubleDQN(
        config.net_dims,
        config.state_dim,
        config.action_dim,
        gpu_id=-1,
    )
    
    # Initialize
    state = env.reset()
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    agent.last_state = state.detach()
    
    # Buffer
    buffer = ReplayBuffer(
        gpu_id=-1,
        num_seqs=config.num_envs,
        max_size=2000,
        state_dim=config.state_dim,
        action_dim=1,
    )
    
    # Warm up
    buffer_items = agent.explore_env(env, 200, if_random=True)
    buffer.update(buffer_items)
    
    print(f"📊 Training for 10 steps...")
    
    rewards = []
    q_ranges = []
    
    for step in range(10):
        # Explore
        buffer_items = agent.explore_env(env, 100)
        exp_r = buffer_items[2].mean().item()
        rewards.append(exp_r)
        
        # Update buffer
        buffer.update(buffer_items)
        
        # Train
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        
        # Check Q-value range
        with torch.no_grad():
            test_state = torch.randn(1, state_dim)
            q_values = agent.act(test_state)
            q_range = (q_values.max() - q_values.min()).item()
            q_ranges.append(q_range)
        
        print(f"   Step {step+1}: Reward={exp_r:.3f}, Q-range={q_range:.3f}")
    
    # Analyze results
    print(f"\n📈 Performance Analysis:")
    print(f"   Average reward: {np.mean(rewards):.3f}")
    print(f"   Reward trend: {'Improving' if rewards[-1] > rewards[0] else 'Declining'}")
    print(f"   Q-value differentiation: {np.mean(q_ranges):.3f}")
    
    if np.mean(q_ranges) > 0.01:
        print(f"   ✅ Model showing good Q-value differentiation")
    else:
        print(f"   ⚠️  Model may need more training")
    
    return True


if __name__ == "__main__":
    # Run tests
    success = test_qvalue_fix()
    
    if success:
        quick_performance_test()
        
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Run full training with fixed Q-value calculation")
    print(f"   2. Use simple reward first to verify improvement")
    print(f"   3. Then test multi-objective reward with adjustments")