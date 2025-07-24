"""
Test script for reward-shaped training

This script runs a quick test of the reward-shaped training environment
to verify everything works correctly before running the full training.
"""

import torch
import numpy as np
from erl_config import Config, build_env
from reward_shaped_training_simulator import RewardShapedTrainingSimulator
from training_reward_config import TrainingRewardConfig
from erl_agent import AgentD3QN
import time


def test_reward_shaped_environment():
    """Test the reward-shaped training environment"""
    print("="*60)
    print("TESTING REWARD-SHAPED TRAINING ENVIRONMENT")
    print("="*60)
    
    # Create reward configuration
    reward_config = TrainingRewardConfig.balanced_training_config()
    training_step_tracker = {'step': 0}
    
    # Create environment args
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 4,  # Small number for testing
        "max_step": 100,  # Short episode for testing
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 4,
        "step_gap": 2,
        "dataset_path": "data/raw/task1/BTC_1sec_predict.npy",
        "reward_config": reward_config,
        "training_step_tracker": training_step_tracker
    }
    
    # Build environment
    print("Building reward-shaped training environment...")
    try:
        env = build_env(RewardShapedTrainingSimulator, env_args, gpu_id=-1)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        return False
    
    # Test environment reset
    print("\nTesting environment reset...")
    try:
        state = env.reset()
        print(f"✓ Reset successful, state shape: {state.shape}")
    except Exception as e:
        print(f"✗ Error during reset: {e}")
        return False
    
    # Test environment steps with different actions
    print("\nTesting environment steps...")
    try:
        rewards = []
        # Get actual number of environments from the created environment
        actual_num_envs = env.num_sims if hasattr(env, 'num_sims') else state.shape[0]
        print(f"  Detected {actual_num_envs} environments")
        
        for step in range(10):
            # Test different actions: hold, buy, sell (vectorized for all environments)
            if step < 3:
                action = torch.ones((actual_num_envs, 1), dtype=torch.int32)  # Hold for all envs
                action_name = "HOLD"
            elif step < 6:
                action = torch.full((actual_num_envs, 1), 2, dtype=torch.int32)  # Buy for all envs
                action_name = "BUY"
            else:
                action = torch.zeros((actual_num_envs, 1), dtype=torch.int32)  # Sell for all envs
                action_name = "SELL"
            
            state, reward, done, info = env.step(action)
            rewards.append(reward.cpu().mean().item())
            
            print(f"  Step {step+1}: {action_name:4s} | Reward: {reward.cpu().mean().item():8.4f} | Done: {done.any().item()}")
            
            if done.any():
                print("  Episode ended early")
                break
        
        print(f"✓ Environment steps completed")
        print(f"  Average reward: {np.mean(rewards):.4f}")
        print(f"  Reward std: {np.std(rewards):.4f}")
        print(f"  Total training steps tracked: {training_step_tracker['step']}")
        
    except Exception as e:
        print(f"✗ Error during environment steps: {e}")
        return False
    
    # Test reward analysis
    print("\nTesting reward analysis...")
    try:
        if hasattr(env, 'print_training_reward_summary'):
            env.print_training_reward_summary()
        else:
            print("  Reward analysis not available")
    except Exception as e:
        print(f"  Warning: Could not print reward analysis: {e}")
    
    return True


def test_agent_creation():
    """Test creating an agent with reward-shaped environment"""
    print("\n" + "="*60)
    print("TESTING AGENT CREATION WITH REWARD-SHAPED ENVIRONMENT")
    print("="*60)
    
    # Create configuration
    reward_config = TrainingRewardConfig.conservative_training_config()
    training_step_tracker = {'step': 0}
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 2,
        "max_step": 50,
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 2,
        "step_gap": 2,
        "dataset_path": "data/raw/task1/BTC_1sec_predict.npy",
        "reward_config": reward_config,
        "training_step_tracker": training_step_tracker
    }
    
    args = Config(agent_class=AgentD3QN, 
                  env_class=RewardShapedTrainingSimulator, 
                  env_args=env_args)
    args.gpu_id = -1
    args.random_seed = 42
    args.net_dims = (64, 64)  # Smaller networks for testing
    args.starting_cash = 1e6
    args.gamma = 0.99
    args.explore_rate = 0.1
    
    print("Creating agent...")
    try:
        agent = AgentD3QN(args.net_dims, args.env_args['state_dim'], 
                         args.env_args['action_dim'], gpu_id=args.gpu_id, args=args)
        print("✓ Agent created successfully")
    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        return False
    
    print("Building environment...")
    try:
        env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)
        print("✓ Environment built successfully")
    except Exception as e:
        print(f"✗ Error building environment: {e}")
        return False
    
    print("Testing agent-environment interaction...")
    try:
        state = env.reset()
        
        for step in range(5):
            # Get action from agent
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
            tensor_q_values = agent.act(tensor_state)
            tensor_action = tensor_q_values.argmax(dim=1, keepdim=True)
            
            # Step environment
            state, reward, done, info = env.step(tensor_action)
            
            print(f"  Step {step+1}: Q-values: {tensor_q_values[0].detach().cpu().numpy()}")
            print(f"            Action: {tensor_action[0].item()}, Reward: {reward[0].item():.4f}")
            
            if done.any():
                break
        
        print("✓ Agent-environment interaction successful")
        return True
        
    except Exception as e:
        print(f"✗ Error in agent-environment interaction: {e}")
        return False


def test_curriculum_learning():
    """Test curriculum learning functionality"""
    print("\n" + "="*60)
    print("TESTING CURRICULUM LEARNING")
    print("="*60)
    
    # Create curriculum config
    reward_config = TrainingRewardConfig.curriculum_config()
    
    print("Testing curriculum multipliers over training steps...")
    test_steps = [0, 10000, 25000, 50000, 75000, 100000, 150000]
    
    for step in test_steps:
        activity_mult, opportunity_mult, timing_mult = reward_config.get_curriculum_multipliers(step)
        print(f"  Step {step:6d}: Activity={activity_mult:.3f}, "
              f"Opportunity={opportunity_mult:.3f}, Timing={timing_mult:.3f}")
    
    print("✓ Curriculum learning test completed")
    return True


def run_all_tests():
    """Run all tests"""
    print("STARTING COMPREHENSIVE REWARD-SHAPED TRAINING TESTS")
    print("="*80)
    
    tests = [
        ("Environment Creation and Steps", test_reward_shaped_environment),
        ("Agent Creation and Interaction", test_agent_creation),
        ("Curriculum Learning", test_curriculum_learning)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Print final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for full training.")
        return True
    else:
        print("⚠️  Some tests failed. Check issues before running full training.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)