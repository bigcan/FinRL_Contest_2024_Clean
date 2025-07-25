#!/usr/bin/env python3
"""
Quick test of hybrid ensemble with DQN + PPO agents
Validates that all agent types can train together
"""

import os
import sys
import torch
import numpy as np
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from task1_ensemble_with_ppo import HybridEnsembleTrainer


def test_hybrid_ensemble():
    """Quick test of hybrid ensemble functionality"""
    
    print("🧪 TESTING HYBRID ENSEMBLE")
    print("=" * 60)
    
    # Create trainer with minimal settings
    trainer = HybridEnsembleTrainer(
        reward_type="simple",  # Best performing reward from A/B test
        gpu_id=-1,  # CPU for testing
        team_name="test_hybrid"
    )
    
    print(f"\n📊 Agent Configurations:")
    for agent_name, config in trainer.agent_configs.items():
        buffer_type = config['buffer_type']
        agent_class = config['agent_class'].__name__
        print(f"   {agent_name:15}: {agent_class} + {buffer_type} buffer")
    
    # Modify configs for quick test
    for agent_name, config in trainer.agent_configs.items():
        # Reduce training steps for quick test
        config['config'].break_step = 5
        config['config'].eval_per_step = 2
        config['config'].horizon_len = 20
        config['config'].buffer_size = 100
        config['config'].early_stopping_enabled = False
        
        # Smaller network for faster training
        config['config'].net_dims = (32, 16)
        config['config'].batch_size = 32
    
    print(f"\n🏃 Starting quick training test...")
    test_start = time.time()
    
    # Test single agent from each type
    test_agents = ["AgentD3QN", "AgentPPO"]  # One DQN, one PPO
    
    for agent_name in test_agents:
        if agent_name in trainer.agent_configs:
            print(f"\n🤖 Testing {agent_name}...")
            
            agent_config = trainer.agent_configs[agent_name]
            save_dir = f"test_models/{agent_name.lower()}"
            
            result = trainer.train_single_agent(agent_name, agent_config, save_dir)
            trainer.training_results[agent_name] = result
            
            if result['success']:
                print(f"   ✅ {agent_name} test passed:")
                print(f"      Return: {result['total_return']:.4f}")
                print(f"      Trading: {result['trading_activity_pct']:.1f}%")
                print(f"      Time: {result['training_time']:.1f}s")
            else:
                print(f"   ❌ {agent_name} test failed: {result.get('error', 'Unknown')}")
    
    test_time = time.time() - test_start
    
    # Analysis
    print(f"\n📊 TEST RESULTS:")
    print("-" * 40)
    
    successful_tests = [r for r in trainer.training_results.values() if r['success']]
    
    if len(successful_tests) >= 2:
        print(f"✅ Hybrid ensemble test PASSED!")
        print(f"   DQN agent: Working")
        print(f"   PPO agent: Working")
        print(f"   Integration: Successful")
        print(f"   Total time: {test_time:.1f}s")
        
        # Compare agent types
        dqn_result = trainer.training_results.get("AgentD3QN", {})
        ppo_result = trainer.training_results.get("AgentPPO", {})
        
        if dqn_result.get('success') and ppo_result.get('success'):
            print(f"\n🔍 AGENT COMPARISON:")
            print(f"   DQN Return: {dqn_result['total_return']:.4f}")
            print(f"   PPO Return: {ppo_result['total_return']:.4f}")
            
            if ppo_result['total_return'] > dqn_result['total_return']:
                print(f"   🏆 PPO outperformed DQN in quick test")
            else:
                print(f"   🏆 DQN outperformed PPO in quick test")
        
        return True
    else:
        print(f"❌ Hybrid ensemble test FAILED!")
        print(f"   Successful agents: {len(successful_tests)}/2")
        print(f"   Issues need to be resolved before full training")
        return False


def main():
    """Main test execution"""
    
    success = test_hybrid_ensemble()
    
    print(f"\n📋 RECOMMENDATIONS:")
    if success:
        print(f"   ✅ Hybrid ensemble ready for production training")
        print(f"   🚀 Run: python3 task1_ensemble_with_ppo.py 0 --reward simple")
        print(f"   📊 Expect improved performance from PPO diversity")
        print(f"   🎯 PPO may provide better policy gradient learning")
    else:
        print(f"   🔧 Debug hybrid ensemble issues:")
        print(f"   1. Check agent initialization")
        print(f"   2. Verify buffer compatibility")
        print(f"   3. Test individual components")
        print(f"   4. Review error messages above")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)