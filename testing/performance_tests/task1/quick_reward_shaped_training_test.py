"""
Quick Reward-Shaped Training Test

Run a very short training session to verify the complete pipeline works
before running the full training.
"""

import os
import torch
import numpy as np
from erl_config import Config, build_env
from reward_shaped_training_simulator import RewardShapedTrainingSimulator
from training_reward_config import TrainingRewardConfig
from erl_agent import AgentD3QN
import time


def quick_training_test():
    """Run a quick training test with minimal parameters"""
    print("="*60)
    print("QUICK REWARD-SHAPED TRAINING TEST")
    print("="*60)
    
    # Create simple reward configuration for testing
    reward_config = TrainingRewardConfig.balanced_training_config()
    training_step_tracker = {'step': 0}
    
    # Minimal training parameters for quick test
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 8,  # Small number for quick test
        "max_step": 50,  # Very short episodes
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 8,
        "step_gap": 2,
        "dataset_path": "data/BTC_1sec_predict.npy",  # Fixed path
        "reward_config": reward_config,
        "training_step_tracker": training_step_tracker
    }
    
    # Create agent configuration
    args = Config(agent_class=AgentD3QN, 
                  env_class=RewardShapedTrainingSimulator, 
                  env_args=env_args)
    args.gpu_id = -1  # Use CPU for quick test
    args.random_seed = 42
    args.net_dims = (32, 32)  # Very small networks for speed
    args.starting_cash = 1e6
    
    # Training hyperparameters - very minimal for quick test
    args.gamma = 0.99
    args.explore_rate = 0.2  # Higher exploration for quick test
    args.state_value_tau = 0.01
    args.target_step = 200   # Very small target steps
    args.eval_times = 50     # Quick evaluation
    args.break_step = 1000   # Early stopping
    args.if_allow_break = True
    
    print("Creating agent and environment...")
    try:
        # Create agent
        agent = AgentD3QN(args.net_dims, args.env_args['state_dim'], 
                         args.env_args['action_dim'], gpu_id=args.gpu_id, args=args)
        print("✓ Agent created")
        
        # Create environment
        env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)
        print("✓ Environment created")
        
    except Exception as e:
        print(f"✗ Error in setup: {e}")
        return False
    
    print("\nRunning quick training loop...")
    try:
        start_time = time.time()
        
        # Simple training loop
        state = env.reset()
        total_reward = 0
        total_steps = 0
        
        for episode in range(3):  # Just 3 episodes
            print(f"  Episode {episode + 1}/3...")
            episode_reward = 0
            episode_steps = 0
            
            for step in range(args.target_step // 8):  # Short episodes
                # Get action from agent
                tensor_state = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                
                # Use exploration for training
                if np.random.random() < args.explore_rate:
                    # Random action - use actual number of environments
                    actual_num_envs = env.num_sims if hasattr(env, 'num_sims') else state.shape[0]
                    action = torch.randint(0, args.env_args['action_dim'], 
                                         (actual_num_envs, 1), 
                                         dtype=torch.int32)
                else:
                    # Agent action
                    tensor_q_values = agent.act(tensor_state)
                    action = tensor_q_values.argmax(dim=1, keepdim=True)
                
                # Step environment
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward.mean().item()
                episode_steps += 1
                total_steps += 1
                
                # Simple experience collection (no replay buffer for this test)
                state = next_state
                
                if done.any() or episode_steps >= 20:  # Short episodes
                    break
            
            total_reward += episode_reward
            print(f"    Steps: {episode_steps}, Avg Reward: {episode_reward/episode_steps:.4f}")
            
            # Reset for next episode
            state = env.reset()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n✓ Quick training completed in {training_time:.2f} seconds")
        print(f"  Total steps: {total_steps}")
        print(f"  Average reward per step: {total_reward/total_steps:.4f}")
        print(f"  Training step tracker: {training_step_tracker['step']}")
        
        # Test reward analysis
        if hasattr(env, 'print_training_reward_summary'):
            env.print_training_reward_summary()
        
        return True
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_training_script():
    """Test the ensemble training script with minimal parameters"""
    print("\n" + "="*60)
    print("TESTING ENSEMBLE TRAINING SCRIPT")
    print("="*60)
    
    try:
        import sys
        sys.path.append('.')
        from task1_ensemble_reward_shaped import RewardShapedEnsemble
        
        # Create minimal configuration
        reward_config = TrainingRewardConfig.conservative_training_config()
        
        # Override with very small parameters for testing
        training_config = {
            'num_sims': 4,
            'num_ignore_step': 60,
            'max_position': 1,
            'step_gap': 2,
            'slippage': 7e-7,
            'max_step': 30,  # Very short for testing
            'dataset_path': "data/BTC_1sec_predict.npy",  # Fixed path
            'starting_cash': 1e6,
            'net_dims': (16, 16),  # Tiny networks
            'gamma': 0.99,
            'explore_rate': 0.2,
            'state_value_tau': 0.01,
            'target_step': 100,  # Very small
            'eval_times': 20,
            'break_step': 500,   # Early stopping
        }
        
        # Test only one agent type for speed
        agent_classes = [AgentD3QN]
        save_path = "test_reward_shaped_ensemble"
        
        print("Creating ensemble...")
        ensemble = RewardShapedEnsemble(
            save_path=save_path,
            agent_classes=agent_classes,
            reward_config=reward_config,
            training_config=training_config
        )
        
        print("✓ Ensemble created successfully")
        
        # Don't actually run training - just verify setup
        print("✓ Ensemble training script setup test passed")
        
        # Cleanup
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing ensemble script: {e}")
        return False


def run_all_quick_tests():
    """Run all quick tests"""
    print("RUNNING QUICK REWARD-SHAPED TRAINING TESTS")
    print("="*80)
    
    tests = [
        ("Quick Training Loop", quick_training_test),
        ("Ensemble Training Script", test_ensemble_training_script)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
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
    
    # Summary
    print("\n" + "="*80)
    print("QUICK TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL QUICK TESTS PASSED! Ready for full reward-shaped training.")
        return True
    else:
        print("⚠️  Some tests failed. Check issues before running full training.")
        return False


if __name__ == "__main__":
    success = run_all_quick_tests()
    
    if success:
        print("\n" + "="*80)
        print("🚀 READY TO RUN FULL REWARD-SHAPED TRAINING!")
        print("="*80)
        print("To run full training, use:")
        print("  python3 task1_ensemble_reward_shaped.py [GPU_ID] [CONFIG_TYPE]")
        print("  Available configs: balanced, conservative, aggressive, curriculum")
        print("  Example: python3 task1_ensemble_reward_shaped.py 0 balanced")
    
    exit(0 if success else 1)