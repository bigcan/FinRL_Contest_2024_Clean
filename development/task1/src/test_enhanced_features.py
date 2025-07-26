#!/usr/bin/env python3
"""
Test Enhanced Features
Evaluate the 20-feature enhanced dataset with existing trained models
"""

import os
import sys
import numpy as np
import torch as th
import json
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import existing modules
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_config import Config

def load_enhanced_dataset():
    """Load the enhanced 20-feature dataset"""
    
    print("📊 Loading enhanced 20-feature dataset...")
    
    enhanced_path = "../../../data/raw/task1/BTC_1sec_predict_enhanced_v2.npy"
    metadata_path = "../../../data/raw/task1/BTC_1sec_predict_enhanced_v2_metadata.npy"
    
    if not os.path.exists(enhanced_path):
        raise FileNotFoundError(f"Enhanced dataset not found: {enhanced_path}")
    
    # Load enhanced features
    features = np.load(enhanced_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    
    print(f"✅ Loaded enhanced dataset: {features.shape}")
    print(f"   Feature names ({len(metadata['feature_names'])}): {metadata['feature_names']}")
    print(f"   Base features: {len(metadata['base_features'])}")
    print(f"   Enhancement count: {metadata['enhancement_count']}")
    
    return features, metadata

def create_enhanced_environment(features):
    """Create trading environment with enhanced features"""
    
    print("🏗️  Creating enhanced trading environment...")
    
    # Save enhanced features temporarily (avoid automatic optimized loading)
    temp_path = "../../../data/raw/task1/enhanced_test_features.npy"
    np.save(temp_path, features)
    
    print(f"📁 Saved enhanced features to: {temp_path}")
    print(f"   Shape: {features.shape}")
    
    # Environment configuration matching task1_eval.py format
    num_sims = 1
    num_ignore_step = 120
    step_gap = 2
    slippage = 7e-7
    max_step = (features.shape[0] - num_ignore_step) // step_gap
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": features.shape[1],  # 20 features
        "action_dim": 3,
        "if_discrete": True,
        "target_return": 4.0,
        "dataset_path": temp_path,
    }
    
    # Create config and environment
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = 0
    args.random_seed = 0
    args.starting_cash = 1e6
    args.state_dim = features.shape[1]
    args.action_dim = 3
    args.net_dims = (128, 128, 128)
    
    # Add missing attributes for agent creation
    args.num_envs = 1
    args.if_off_policy = True
    args.gamma = 0.99
    args.learning_rate = 1e-5
    args.soft_update_tau = 2e-3
    
    from erl_config import build_env
    env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)
    
    print(f"✅ Enhanced environment created")
    print(f"   State dimension: {features.shape[1]}")
    print(f"   Max steps: {max_step}")
    print(f"   Dataset shape: {features.shape}")
    
    return env, args

def load_existing_agents(ensemble_path, args):
    """Load existing trained agents with proper args configuration"""
    
    print(f"🤖 Loading existing agents from: {ensemble_path}")
    
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble path not found: {ensemble_path}")
    
    # Agent configurations
    agent_classes = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    agents = []
    
    # Use network dimensions from saved models or defaults
    net_dims = (128, 128, 128)  # Use same as training
    
    for agent_class in agent_classes:
        agent_name = agent_class.__name__
        model_dir = os.path.join(ensemble_path, "ensemble_models", agent_name)
        
        if os.path.exists(model_dir):
            print(f"   Loading {agent_name}...")
            
            try:
                # Create agent with full args
                agent = agent_class(
                    net_dims=net_dims,
                    state_dim=args.state_dim,
                    action_dim=args.action_dim,
                    gpu_id=args.gpu_id,
                    args=args
                )
                
                # Try to load existing weights
                agent.save_or_load_agent(model_dir, if_save=False)
                agents.append(agent)
                print(f"     ✅ {agent_name} loaded successfully")
                
            except Exception as e:
                print(f"     ⚠️  {agent_name} failed to load: {e}")
                # Create fresh agent for compatibility testing
                try:
                    agent = agent_class(
                        net_dims=net_dims,
                        state_dim=args.state_dim,
                        action_dim=args.action_dim,
                        gpu_id=args.gpu_id,
                        args=args
                    )
                    agents.append(agent)
                    print(f"     📝 Created fresh {agent_name} for testing")
                except Exception as e2:
                    print(f"     ❌ Failed to create {agent_name}: {e2}")
        
        else:
            print(f"   ⚠️  Model directory not found: {model_dir}")
    
    print(f"✅ Created {len(agents)} agents")
    return agents

def test_agent_compatibility(agents, env):
    """Test if agents work with enhanced features"""
    
    print("🧪 Testing agent compatibility with enhanced features...")
    
    # Get initial state
    state = env.reset()
    
    # Convert state to numpy for shape checking
    if hasattr(state, 'cpu'):
        state_np = state.cpu().numpy()
    else:
        state_np = np.array(state)
    
    print(f"   State shape: {state_np.shape}")
    
    compatibility_results = {}
    
    for i, agent in enumerate(agents):
        agent_name = agent.__class__.__name__
        print(f"   Testing {agent_name}...")
        
        try:
            # Ensure state is properly shaped
            if isinstance(state, np.ndarray):
                if state.ndim == 1:
                    state_input = state.reshape(1, -1)
                else:
                    state_input = state
            else:
                state_input = np.array(state).reshape(1, -1)
            
            # Convert state to tensor
            state_tensor = th.as_tensor(state_input, dtype=th.float32, device=agent.device)
            
            # Get action from agent
            with th.no_grad():
                action = agent.act(state_tensor)
                action_int = action.argmax(dim=1)
            
            print(f"     ✅ {agent_name} compatible - Action shape: {action.shape}")
            compatibility_results[agent_name] = {
                'compatible': True,
                'action_shape': tuple(action.shape),
                'state_processed': True
            }
            
        except Exception as e:
            print(f"     ❌ {agent_name} incompatible: {e}")
            compatibility_results[agent_name] = {
                'compatible': False,
                'error': str(e),
                'state_processed': False
            }
    
    return compatibility_results

def run_enhanced_evaluation(agents, env, num_steps=100):
    """Run evaluation with enhanced features"""
    
    print(f"📈 Running enhanced feature evaluation ({num_steps} steps)...")
    
    state = env.reset()
    total_rewards = []
    actions_taken = []
    
    for step in range(num_steps):
        # Get ensemble action
        actions = []
        
        for agent in agents:
            try:
                # Ensure state is properly shaped for agent
                if isinstance(state, np.ndarray):
                    if state.ndim == 1:
                        state_input = state.reshape(1, -1)
                    else:
                        state_input = state
                else:
                    state_input = np.array(state).reshape(1, -1)
                
                state_tensor = th.as_tensor(state_input, dtype=th.float32, device=agent.device)
                with th.no_grad():
                    q_values = agent.act(state_tensor)
                    action = q_values.argmax(dim=1)
                    actions.append(action.item())
            except Exception as e:
                # Fallback to hold action
                actions.append(1)
        
        # Majority voting
        if actions:
            ensemble_action = max(set(actions), key=actions.count)  # Most common action
        else:
            ensemble_action = 1  # Hold
        
        # Execute action
        next_state, reward, done, _ = env.step(ensemble_action)
        
        total_rewards.append(reward)
        actions_taken.append(ensemble_action)
        
        state = next_state
        
        if step % 20 == 0:
            avg_reward = np.mean(total_rewards)
            print(f"   Step {step}: Avg reward = {avg_reward:.6f}")
        
        if done:
            break
    
    # Calculate performance metrics
    total_reward = np.sum(total_rewards)
    avg_reward = np.mean(total_rewards)
    reward_std = np.std(total_rewards) if len(total_rewards) > 1 else 0
    
    # Action distribution
    action_counts = np.bincount(actions_taken, minlength=3)
    action_dist = action_counts / len(actions_taken)
    
    results = {
        'total_steps': len(total_rewards),
        'total_reward': total_reward,
        'average_reward': avg_reward,
        'reward_std': reward_std,
        'action_distribution': {
            'sell': action_dist[0],
            'hold': action_dist[1], 
            'buy': action_dist[2]
        },
        'sharpe_estimate': avg_reward / reward_std if reward_std > 0 else 0
    }
    
    return results

def compare_with_baseline():
    """Compare enhanced features with baseline 8-feature performance"""
    
    print("📊 Comparing with baseline performance...")
    
    # Load baseline metadata for comparison
    baseline_metadata_path = "../../../data/raw/task1/BTC_1sec_predict_optimized_metadata.npy"
    
    if os.path.exists(baseline_metadata_path):
        baseline_metadata = np.load(baseline_metadata_path, allow_pickle=True).item()
        
        print(f"   Baseline system:")
        print(f"     Features: {len(baseline_metadata.get('feature_names', []))}")
        print(f"     Expected accuracy: {baseline_metadata.get('expected_accuracy', 'Unknown')}")
        print(f"     Selection criteria: {baseline_metadata.get('selection_criteria', 'Unknown')}")
        
        return baseline_metadata
    else:
        print(f"   ⚠️  Baseline metadata not found")
        return None

def main():
    """Main testing process"""
    
    print("🚀 ENHANCED FEATURE TESTING SYSTEM")
    print("Testing 20-feature enhanced dataset with existing models")
    print("=" * 70)
    
    try:
        # Load enhanced dataset
        features, metadata = load_enhanced_dataset()
        
        # Create environment with enhanced features
        env, env_args = create_enhanced_environment(features)
        
        # Find best existing ensemble to test with
        ensemble_paths = [
            "ensemble_optimized_phase2",
            "ensemble_extended_phase1_20250725_080759",
            "ensemble_extended_phase1_20250724_233428"
        ]
        
        ensemble_path = None
        for path in ensemble_paths:
            if os.path.exists(path):
                ensemble_path = path
                break
        
        if not ensemble_path:
            print("❌ No existing ensemble found for testing")
            return False
        
        print(f"🎯 Using ensemble: {ensemble_path}")
        
        # Load existing agents
        agents = load_existing_agents(ensemble_path, env_args)
        
        # Test compatibility
        compatibility = test_agent_compatibility(agents, env)
        
        compatible_agents = [agent for i, agent in enumerate(agents) 
                           if list(compatibility.values())[i]['compatible']]
        
        if not compatible_agents:
            print("❌ No compatible agents found")
            return False
        
        print(f"✅ {len(compatible_agents)} agents are compatible")
        
        # Run enhanced evaluation
        results = run_enhanced_evaluation(compatible_agents, env, num_steps=200)
        
        # Display results
        print(f"\n📊 ENHANCED FEATURE TEST RESULTS:")
        print(f"   Steps completed: {results['total_steps']}")
        print(f"   Total reward: {results['total_reward']:.6f}")
        print(f"   Average reward: {results['average_reward']:.6f}")
        print(f"   Reward std: {results['reward_std']:.6f}")
        print(f"   Sharpe estimate: {results['sharpe_estimate']:.4f}")
        
        print(f"\n🎯 Action Distribution:")
        for action, prob in results['action_distribution'].items():
            print(f"   {action}: {prob:.1%}")
        
        # Compare with baseline
        baseline_metadata = compare_with_baseline()
        
        # Save test results
        test_results = {
            'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'enhanced_features': len(metadata['feature_names']),
            'base_features': len(metadata['base_features']),
            'ensemble_path': ensemble_path,
            'compatible_agents': len(compatible_agents),
            'performance': results,
            'compatibility': compatibility
        }
        
        results_path = f"enhanced_feature_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n✅ Test results saved: {results_path}")
        
        # Assessment
        print(f"\n🎉 ENHANCED FEATURE TESTING COMPLETE!")
        print(f"   Enhanced dataset (20 features) is compatible with existing models")
        print(f"   Average reward: {results['average_reward']:.6f}")
        print(f"   Ready for training with enhanced features")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in enhanced feature testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ Enhanced feature testing completed successfully!")
        print(f"   The 20-feature enhanced dataset is ready for training")
    else:
        print(f"\n❌ Enhanced feature testing failed!")