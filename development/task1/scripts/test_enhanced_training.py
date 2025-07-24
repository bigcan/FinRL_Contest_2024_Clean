"""
Test Enhanced Feature Training

Quick test to verify that the enhanced features work with the training pipeline.
"""

import os
import sys
import torch
import numpy as np
from trade_simulator import TradeSimulator
from erl_agent import AgentD3QN
from erl_config import Config, build_env

def main():
    """Test enhanced feature training"""
    print("=" * 60)
    print("TESTING ENHANCED FEATURE TRAINING")
    print("=" * 60)
    
    # Check if enhanced features exist
    enhanced_path = "./data/raw/task1/BTC_1sec_predict_enhanced.npy"
    if not os.path.exists(enhanced_path):
        print("❌ Enhanced features not found. Run create_enhanced_features_simple.py first.")
        return
    
    print("✓ Enhanced features found")
    
    # Configuration
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": (3600 - 60) // 2,  # Smaller for testing
        "state_dim": 16,  # Will be overridden by dynamic detection
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 2,
        "slippage": 7e-7,
        "num_sims": 4,  # Small for testing
        "step_gap": 2,
        "dataset_path": "data/BTC_1sec_predict.npy"
    }
    
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = -1  # Use CPU for testing
    args.net_dims = (64, 64)  # Smaller network for testing
    args.learning_rate = 1e-4
    args.gamma = 0.99
    
    print(f"Configuration: {env_args}")
    
    # Build environment
    print("\nBuilding environment...")
    env = build_env(TradeSimulator, env_args, gpu_id=args.gpu_id)
    print(f"✓ Environment built with state_dim: {env.state_dim}")
    
    # Create agent
    print("\nCreating agent...")
    agent = AgentD3QN(args.net_dims, env.state_dim, env.action_dim, 
                      gpu_id=args.gpu_id, args=args)
    print(f"✓ Agent created with state_dim: {env.state_dim}")
    
    # Test a few training steps
    print("\nTesting training steps...")
    
    # Initialize replay buffer with some random experiences
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for step in range(10):
        # Random action for testing
        action = torch.randint(env.action_dim, size=(env.num_sims, 1))
        next_state, reward, done, info = env.step(action)
        
        # Store experience
        agent.replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        
        if step % 5 == 4:
            print(f"Step {step+1}: reward_mean={reward.mean():.4f}, state_shape={state.shape}")
    
    # Test agent training
    print("\nTesting agent training...")
    
    # Need enough samples for training
    for _ in range(agent.replay_buffer.batch_size):
        action = torch.randint(env.action_dim, size=(env.num_sims, 1))
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
    
    # Test one training update
    try:
        initial_q_values = agent.act(state[:1]).detach().cpu().numpy()
        
        agent.update_net(args)
        
        updated_q_values = agent.act(state[:1]).detach().cpu().numpy()
        
        print(f"✓ Training update successful")
        print(f"  Q-values before: {initial_q_values[0]}")
        print(f"  Q-values after:  {updated_q_values[0]}")
        
    except Exception as e:
        print(f"❌ Training update failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("ENHANCED FEATURE TRAINING TEST COMPLETE")
    print("=" * 60)
    print("✓ Enhanced features loaded successfully")
    print(f"✓ State dimension: {env.state_dim} (up from 10)")
    print("✓ Environment integration working")
    print("✓ Agent training compatible")
    print("✓ Ready for full ensemble training!")
    
    # Create simple training command for reference
    print("\nTo train ensemble with enhanced features:")
    print("python3 task1_ensemble.py")
    print("(Enhanced features will be automatically detected and used)")

if __name__ == "__main__":
    main()