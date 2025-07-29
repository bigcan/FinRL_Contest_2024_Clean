#!/usr/bin/env python3
"""
Quick test to validate the comprehensive device fix for multi-episode training
"""

import os
import sys
import torch
import numpy as np
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN

def test_device_consistency():
    """Test device consistency during episode transitions"""
    
    print("🧪 Testing Device Consistency Fix")
    
    # Minimal config for testing
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    num_sims = 8  # Small for quick testing
    data_length = 1000  # Small dataset for speed
    
    # Environment setup
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": 400,
        "state_dim": 41,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": num_sims,
        "step_gap": 2,
        "data_length": data_length,
    }
    
    # Agent setup
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.net_dims = (64, 64)  # Smaller networks for speed
    args.gamma = 0.995
    args.learning_rate = 2e-6
    args.batch_size = 128
    args.horizon_len = 200  # Short episodes
    
    args.init_before_training()
    
    # Create environment and agent
    env = build_env(args.env_class, args.env_args, args.gpu_id)
    agent = args.agent_class(
        args.net_dims,
        args.state_dim,
        args.action_dim,
        gpu_id=args.gpu_id,
        args=args,
    )
    
    # Setup buffer
    buffer = ReplayBuffer(
        gpu_id=args.gpu_id,
        num_seqs=args.num_envs,
        max_size=args.horizon_len * 4,
        state_dim=args.state_dim,
        action_dim=1,
    )
    
    # Initialize agent state
    state = env.reset()
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0) if args.num_envs == 1 else torch.tensor(state, dtype=torch.float32, device=agent.device)
    else:
        state = state.to(agent.device)
    agent.last_state = state.detach()
    
    # Warm up buffer
    buffer_items = agent.explore_env(env, args.horizon_len, if_random=True)
    buffer.update(buffer_items)
    
    print(f"✅ Initial setup complete - Agent device: {agent.device}")
    print(f"   Actor network device: {next(agent.act.parameters()).device}")
    print(f"   Critic network device: {next(agent.cri.parameters()).device}")
    
    # Test 3 episodes to validate device consistency
    for episode in range(3):
        print(f"\n📈 Episode {episode + 1}/3 starting...")
        
        # CRITICAL FIX: Ensure networks remain on GPU before episode reset
        target_device = agent.device
        
        # Force all networks to correct device before reset
        agent.act = agent.act.to(target_device)
        agent.act_target = agent.act_target.to(target_device)
        if hasattr(agent, 'cri') and agent.cri is not agent.act:
            agent.cri = agent.cri.to(target_device)
            agent.cri_target = agent.cri_target.to(target_device)
        
        # Verify networks are on correct device
        act_device = next(agent.act.parameters()).device
        if act_device != target_device:
            print(f"❌ CRITICAL: Networks still on wrong device {act_device} after forced move!")
            # Emergency recovery
            agent.act = agent.act.to(target_device)
            agent.act_target = agent.act_target.to(target_device)
            if hasattr(agent, 'cri') and agent.cri is not agent.act:
                agent.cri = agent.cri.to(target_device)
                agent.cri_target = agent.cri_target.to(target_device)
            print(f"✅ Emergency network recovery completed")
        
        # Reset environment
        state = env.reset()
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0) if args.num_envs == 1 else torch.tensor(state, dtype=torch.float32, device=agent.device)
        else:
            state = state.to(agent.device)
        agent.last_state = state.detach()
        
        # CRITICAL: Verify networks didn't move during reset
        post_reset_device = next(agent.act.parameters()).device
        if post_reset_device != target_device:
            print(f"❌ CRITICAL: Environment reset moved networks from {target_device} to {post_reset_device}!")
            # Force networks back to correct device
            agent.act = agent.act.to(target_device)
            agent.act_target = agent.act_target.to(target_device)
            if hasattr(agent, 'cri') and agent.cri is not agent.act:
                agent.cri = agent.cri.to(target_device)
                agent.cri_target = agent.cri_target.to(target_device)
            print(f"✅ Networks restored to {target_device} after reset")
        
        print(f"   ✅ Pre-exploration - Networks on: {next(agent.act.parameters()).device}")
        
        # Explore environment
        buffer_items = agent.explore_env(env, args.horizon_len)
        
        # CRITICAL: Verify networks didn't move during exploration
        post_explore_device = next(agent.act.parameters()).device
        if post_explore_device != target_device:
            print(f"❌ CRITICAL: Exploration moved networks from {target_device} to {post_explore_device}!")
            # Force networks back to correct device
            agent.act = agent.act.to(target_device)
            agent.act_target = agent.act_target.to(target_device)
            if hasattr(agent, 'cri') and agent.cri is not agent.act:
                agent.cri = agent.cri.to(target_device)
                agent.cri_target = agent.cri_target.to(target_device)
            print(f"✅ Networks restored to {target_device} after exploration")
        
        print(f"   ✅ Post-exploration - Networks on: {next(agent.act.parameters()).device}")
        
        # Update buffer and train
        buffer.update(buffer_items)
        
        torch.set_grad_enabled(True)
        try:
            logging_tuple = agent.update_net(buffer)
            print(f"   ✅ Episode {episode + 1} training successful!")
            
            # Verify networks after training
            post_training_device = next(agent.act.parameters()).device
            print(f"   ✅ Post-training - Networks on: {post_training_device}")
            
            if post_training_device != target_device:
                print(f"❌ WARNING: Networks moved during training!")
        except Exception as e:
            print(f"   ❌ Episode {episode + 1} training failed: {e}")
            break
        finally:
            torch.set_grad_enabled(False)
    
    env.close() if hasattr(env, "close") else None
    print(f"\n🏁 Device consistency test completed!")

if __name__ == "__main__":
    test_device_consistency()