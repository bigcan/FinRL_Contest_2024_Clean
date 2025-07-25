#!/usr/bin/env python3
"""
Quick reward function validation test
Tests all 3 reward functions for basic functionality
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


def quick_test_reward(reward_type: str, steps: int = 5) -> dict:
    """Quick test of a single reward function"""
    
    print(f"\n🧪 Testing {reward_type} reward...")
    test_start = time.time()
    
    try:
        # Setup
        temp_sim = TradeSimulator(num_sims=1)
        state_dim = temp_sim.state_dim
        temp_sim.set_reward_type(reward_type)
        
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": 2,  # Minimal
            "max_step": 100,  # Very short
            "state_dim": state_dim,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "num_sims": 2,
            "step_gap": 2,
        }
        
        # Create config
        config = EnhancedConfig(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
        config.gpu_id = 0 if torch.cuda.is_available() else -1
        config.random_seed = 42
        config.state_dim = state_dim
        
        # Apply optimized params
        optimized_params = get_optimized_hyperparameters(reward_type)
        config = apply_optimized_hyperparameters(config, optimized_params, env_args)
        
        # Override for quick test
        config.break_step = steps
        config.eval_per_step = max(1, steps // 2)
        config.horizon_len = 50
        config.buffer_size = 200
        config.early_stopping_enabled = False
        
        # Build environment
        env = build_env(config.env_class, config.env_args, config.gpu_id)
        env.set_reward_type(reward_type)
        
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
        
        # Warm up
        buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
        buffer.update(buffer_items)
        
        # Quick training
        rewards = []
        for step in range(steps):
            buffer_items = agent.explore_env(env, config.horizon_len)
            exp_r = buffer_items[2].mean().item()
            rewards.append(exp_r)
            
            buffer.update(buffer_items)
            
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
        
        # Quick eval
        eval_rewards = []
        state = env.reset()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(agent.device)
        
        for _ in range(5):
            with torch.no_grad():
                q_values = agent.act(state)
                action = q_values.argmax(dim=1, keepdim=True)
            
            next_state, reward, done, _ = env.step(action)
            eval_rewards.append(reward.mean().item())
            
            if done.any():
                break
            state = next_state
        
        # Results
        avg_training_reward = np.mean(rewards)
        avg_eval_reward = np.mean(eval_rewards)
        improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0
        
        result = {
            'reward_type': reward_type,
            'avg_training_reward': avg_training_reward,
            'avg_eval_reward': avg_eval_reward,
            'improvement': improvement,
            'training_time': time.time() - test_start,
            'success': True
        }
        
        print(f"   ✅ {reward_type}: Avg reward={avg_training_reward:.4f}, Eval={avg_eval_reward:.4f}, Time={result['training_time']:.1f}s")
        
        env.close() if hasattr(env, "close") else None
        return result
        
    except Exception as e:
        print(f"   ❌ {reward_type} failed: {e}")
        return {
            'reward_type': reward_type,
            'avg_training_reward': -999,
            'avg_eval_reward': -999,
            'improvement': -999,
            'training_time': time.time() - test_start,
            'success': False,
            'error': str(e)
        }


def main():
    """Quick test all reward functions"""
    
    print("🚀 QUICK REWARD FUNCTION VALIDATION")
    print("=" * 50)
    
    reward_functions = ["simple", "transaction_cost_adjusted", "multi_objective"]
    results = {}
    
    for reward_type in reward_functions:
        result = quick_test_reward(reward_type, steps=3)
        results[reward_type] = result
    
    # Analysis
    print(f"\n📊 QUICK TEST RESULTS:")
    print("-" * 50)
    
    successful_tests = [r for r in results.values() if r['success']]
    
    if successful_tests:
        # Rank by average eval reward
        best_reward = max(successful_tests, key=lambda x: x['avg_eval_reward'])
        
        print(f"\n🥇 BEST PERFORMER: {best_reward['reward_type']}")
        print(f"   Eval reward: {best_reward['avg_eval_reward']:.4f}")
        print(f"   Training reward: {best_reward['avg_training_reward']:.4f}")
        print(f"   Improvement: {best_reward['improvement']:.4f}")
        
        print(f"\n📋 ALL RESULTS:")
        for reward_type, result in results.items():
            if result['success']:
                status = "✅"
                eval_reward = result['avg_eval_reward']
            else:
                status = "❌"  
                eval_reward = "FAILED"
            
            print(f"   {status} {reward_type:25}: {eval_reward}")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Use '{best_reward['reward_type']}' for full training")
        print(f"   Expected to work with fixed Q-value calculation")
        
        return best_reward['reward_type']
    else:
        print(f"\n❌ All reward function tests failed")
        print(f"   Check system configuration and dependencies")
        return None


if __name__ == "__main__":
    best_reward = main()
    
    if best_reward:
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Run full training with {best_reward} reward")
        print(f"   2. Use: python3 task1_ensemble_extended.py 0 {best_reward}")
        print(f"   3. Compare with baseline performance")
    else:
        print(f"\n📋 DEBUG STEPS:")
        print(f"   1. Check imports and dependencies")
        print(f"   2. Verify CUDA setup if using GPU")
        print(f"   3. Run test_fixed_training.py first")