#!/usr/bin/env python3
"""
Test training with the fixed Q-value calculation
Quick 5-minute test to verify improvement
"""

import os
import sys
import torch
import numpy as np
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from erl_agent import AgentD3QN
from enhanced_training_config import EnhancedConfig
from optimized_hyperparameters import get_optimized_hyperparameters, apply_optimized_hyperparameters
from erl_config import build_env
from erl_replay_buffer import ReplayBuffer


def test_fixed_training():
    """Quick training test with fixed Q-value calculation"""
    
    print("🧪 Testing Fixed Model Training")
    print("=" * 60)
    
    # Setup
    temp_sim = TradeSimulator(num_sims=1)
    state_dim = temp_sim.state_dim
    temp_sim.set_reward_type("simple")  # Start with simple reward
    
    print(f"📊 Configuration:")
    print(f"   State dimension: {state_dim}")
    print(f"   Reward type: simple (no penalties)")
    print(f"   Training steps: 50 (quick test)")
    print(f"   Agents: D3QN with fixed Q-values")
    
    # Environment args
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 16,  # Smaller for quick test
        "max_step": 1000,
        "state_dim": state_dim,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 16,
        "step_gap": 2,
    }
    
    # Enhanced config
    config = EnhancedConfig(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    config.gpu_id = 0 if torch.cuda.is_available() else -1
    config.random_seed = 42
    config.state_dim = state_dim
    
    # Apply optimized hyperparameters
    optimized_params = get_optimized_hyperparameters("simple")
    config = apply_optimized_hyperparameters(config, optimized_params, env_args)
    
    # Override for quick test
    config.break_step = 50  # Just 50 steps
    config.eval_per_step = 10
    config.horizon_len = 200
    config.buffer_size = 2000
    
    print(f"\n⚙️  Test Parameters:")
    print(f"   Learning rate: {config.learning_rate:.2e}")
    print(f"   Exploration: {config.initial_exploration:.3f}")
    print(f"   Network: {config.net_dims}")
    print(f"   Batch size: {config.batch_size}")
    
    # Build environment
    env = build_env(config.env_class, config.env_args, config.gpu_id)
    env.set_reward_type("simple")
    
    # Create agent
    agent = AgentD3QN(
        config.net_dims,
        config.state_dim,
        config.action_dim,
        gpu_id=config.gpu_id,
        args=config,
    )
    
    # Initialize
    state = env.reset()
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    state = state.to(agent.device)
    agent.last_state = state.detach()
    
    # Buffer
    buffer = ReplayBuffer(
        gpu_id=config.gpu_id,
        num_seqs=config.num_envs,
        max_size=config.buffer_size,
        state_dim=config.state_dim,
        action_dim=1,
    )
    
    # Warm up buffer
    print(f"\n🔄 Warming up buffer...")
    buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
    buffer.update(buffer_items)
    
    # Training metrics
    rewards = []
    q_values_history = []
    action_counts = {0: 0, 1: 0, 2: 0}
    
    print(f"\n🏋️  Starting training...")
    training_start = time.time()
    
    for step in range(config.break_step):
        # Explore
        buffer_items = agent.explore_env(env, config.horizon_len)
        exp_r = buffer_items[2].mean().item()
        rewards.append(exp_r)
        
        # Update buffer
        buffer.update(buffer_items)
        
        # Update network
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        
        # Track Q-values and actions
        if step % 5 == 0:
            with torch.no_grad():
                test_state = state[:1]  # First env
                q_values = agent.act(test_state)
                q_values_history.append(q_values[0].cpu().numpy())
                
                # Sample some actions to check diversity
                for _ in range(10):
                    q_values = agent.act(state)
                    action = q_values.argmax(dim=1, keepdim=True)
                    for a in action.cpu().numpy():
                        action_counts[int(a)] += 1
        
        # Progress
        if step % 10 == 0:
            obj_critic, obj_actor = logging_tuple[:2] if logging_tuple else (0, 0)
            avg_q = np.mean([q.mean() for q in q_values_history[-5:]]) if q_values_history else 0
            print(f"   Step {step}: Reward={exp_r:.3f}, Critic={obj_critic:.3f}, Avg Q={avg_q:.3f}")
    
    training_time = time.time() - training_start
    
    # Analyze results
    print(f"\n📊 Training Results:")
    print(f"   Training time: {training_time:.1f}s")
    print(f"   Average reward: {np.mean(rewards):.3f}")
    print(f"   Final reward: {rewards[-1]:.3f}")
    print(f"   Reward improvement: {rewards[-1] - rewards[0]:.3f}")
    
    # Action diversity
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        print(f"\n🎯 Action Distribution:")
        print(f"   Hold: {action_counts[0]/total_actions*100:.1f}%")
        print(f"   Buy: {action_counts[1]/total_actions*100:.1f}%")
        print(f"   Sell: {action_counts[2]/total_actions*100:.1f}%")
        
        # Check for diversity
        action_variety = len([v for v in action_counts.values() if v > 0])
        if action_variety >= 2:
            print(f"   ✅ Good action diversity ({action_variety}/3 actions used)")
        else:
            print(f"   ⚠️  Limited action diversity ({action_variety}/3 actions used)")
    
    # Q-value analysis
    if q_values_history:
        q_array = np.array(q_values_history)
        print(f"\n💡 Q-Value Evolution:")
        print(f"   Initial Q-values: {q_array[0]}")
        print(f"   Final Q-values: {q_array[-1]}")
        print(f"   Q-value range: {q_array[-1].max() - q_array[-1].min():.3f}")
        
        if q_array[-1].max() - q_array[-1].min() > 0.01:
            print(f"   ✅ Good Q-value differentiation")
        else:
            print(f"   ⚠️  Q-values too similar")
    
    # Quick evaluation
    print(f"\n🎮 Quick Evaluation (20 steps)...")
    eval_rewards = []
    eval_actions = []
    
    state = env.reset()
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    state = state.to(agent.device)
    
    for _ in range(20):
        with torch.no_grad():
            q_values = agent.act(state)
            action = q_values.argmax(dim=1, keepdim=True)
            eval_actions.extend(action.cpu().numpy())
        
        next_state, reward, done, _ = env.step(action)
        eval_rewards.append(reward.mean().item())
        
        if done.any():
            break
        state = next_state
    
    print(f"   Average eval reward: {np.mean(eval_rewards):.3f}")
    
    # Check if model is trading
    eval_action_counts = {0: 0, 1: 0, 2: 0}
    for a in eval_actions:
        eval_action_counts[int(a)] += 1
    
    total_eval = sum(eval_action_counts.values())
    if total_eval > 0:
        non_hold_pct = (eval_action_counts[1] + eval_action_counts[2]) / total_eval * 100
        print(f"   Trading activity: {non_hold_pct:.1f}% (Buy + Sell)")
        
        if non_hold_pct > 10:
            print(f"   ✅ Model is actively trading!")
        else:
            print(f"   ⚠️  Model may be too conservative")
    
    # Summary
    print(f"\n🎉 SUMMARY:")
    if rewards[-1] > rewards[0] and action_variety >= 2:
        print(f"   ✅ Model is learning and showing improvement!")
        print(f"   ✅ Q-value fix appears to be working!")
        return True
    else:
        print(f"   ⚠️  Model needs more training or hyperparameter tuning")
        return False


if __name__ == "__main__":
    success = test_fixed_training()
    
    print(f"\n📋 RECOMMENDATIONS:")
    if success:
        print(f"   1. ✅ Q-value fix is working - proceed with full training")
        print(f"   2. Run full 200-step training with simple reward")
        print(f"   3. Then test multi-objective reward with adjusted penalties")
        print(f"   4. Compare results with baseline")
    else:
        print(f"   1. Consider adjusting hyperparameters")
        print(f"   2. Try different reward functions")
        print(f"   3. Increase training duration")