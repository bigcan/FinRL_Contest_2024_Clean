#!/usr/bin/env python3
"""
Quick Extended Training Test (5 steps)
Validate the extended training works before full 200-step run
"""

import os
import sys
import torch
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from enhanced_training_config import EnhancedConfig
from erl_agent import AgentD3QN
from erl_replay_buffer import ReplayBuffer
from erl_config import build_env


def quick_extended_training_test():
    """Quick test with just 5 training steps"""
    
    print("🧪 Quick Extended Training Test (5 steps)")
    print("=" * 60)
    
    try:
        # Setup configuration
        temp_sim = TradeSimulator(num_sims=1)
        state_dim = temp_sim.state_dim
        temp_sim.set_reward_type("multi_objective")
        
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": 2,  # Small for testing
            "max_step": 100,  # Small for testing
            "state_dim": state_dim,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "num_sims": 2,
            "step_gap": 2,
        }
        
        config = EnhancedConfig(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
        config.gpu_id = -1  # Use CPU for testing
        config.state_dim = state_dim
        config.break_step = 5  # Only 5 steps for quick test
        config.net_dims = (64, 32)  # Smaller for speed
        config.learning_rate = 1e-4
        config.batch_size = 32
        config.horizon_len = 50
        config.buffer_size = 200
        
        print(f"✅ Configuration setup complete")
        print(f"   State dim: {state_dim}")
        print(f"   Training steps: {config.break_step}")
        print(f"   Network: {config.net_dims}")
        
        # Initialize agent
        agent = AgentD3QN(
            config.net_dims,
            config.state_dim,
            config.action_dim,
            gpu_id=config.gpu_id,
            args=config,
        )
        print(f"✅ Agent created: {type(agent).__name__}")
        
        # Build environment
        env = build_env(config.env_class, config.env_args, config.gpu_id)
        print(f"✅ Environment built")
        
        # Initialize state
        state = env.reset()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(agent.device)
        agent.last_state = state.detach()
        
        # Initialize buffer
        buffer = ReplayBuffer(
            gpu_id=config.gpu_id,
            num_seqs=config.num_envs,
            max_size=config.buffer_size,
            state_dim=config.state_dim,
            action_dim=1,
        )
        print(f"✅ Buffer initialized")
        
        # Warm up buffer
        print(f"🔄 Warming up buffer...")
        buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
        buffer.update(buffer_items)
        print(f"   Buffer warmed up")
        
        # Quick training loop
        print(f"🏋️  Starting quick training ({config.break_step} steps)...")
        
        for step in range(config.break_step):
            step_start = time.time()
            
            # Collect experience
            buffer_items = agent.explore_env(env, config.horizon_len)
            exp_r = buffer_items[2].mean().item()
            
            # Update buffer
            buffer.update(buffer_items)
            
            # Update network
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
            
            step_time = time.time() - step_start
            
            obj_critic = obj_actor = 0.0
            if logging_tuple:
                obj_critic, obj_actor = logging_tuple[:2]
            
            print(f"   Step {step+1}/{config.break_step}: Reward={exp_r:.4f}, Critic={obj_critic:.4f}, Actor={obj_actor:.4f}, Time={step_time:.1f}s")
        
        print(f"✅ Quick training completed successfully!")
        
        # Quick evaluation
        print(f"🧪 Quick evaluation...")
        state = env.reset()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(agent.device)
        
        action_counts = {0: 0, 1: 0, 2: 0}
        total_reward = 0
        
        for i in range(10):
            with torch.no_grad():
                q_values = agent.act(state)
                action = q_values.argmax(dim=1, keepdim=True)
                action_counts[action[0].item()] += 1
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward.mean().item()
            
            if done.any():
                break
                
            state = next_state
        
        action_variety = len([v for v in action_counts.values() if v > 0])
        
        print(f"   📊 Total reward: {total_reward:.4f}")
        print(f"   🎯 Action distribution: {action_counts}")
        print(f"   🎪 Action variety: {action_variety}/3")
        
        env.close() if hasattr(env, "close") else None
        
        print(f"\n🎉 Quick extended training test PASSED!")
        print(f"📋 Ready for full extended training with 200 steps")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_extended_training_test()
    
    if success:
        print(f"\n🚀 Full extended training command:")
        print(f"   python3 task1_ensemble_extended.py 0 multi_objective")
    else:
        print(f"\n⚠️  Fix issues before running full training")