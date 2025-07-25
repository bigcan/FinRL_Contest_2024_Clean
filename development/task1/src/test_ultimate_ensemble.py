#!/usr/bin/env python3
"""
Quick test of the ultimate ensemble with all agent types
Validates DQN + PPO + Rainbow integration
"""

import os
import sys
import torch
import numpy as np
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from task1_ensemble_ultimate import UltimateEnsembleTrainer


def test_ultimate_ensemble():
    """Quick test of ultimate ensemble functionality"""
    
    print("🧪 TESTING ULTIMATE ENSEMBLE")
    print("=" * 60)
    
    # Create trainer with minimal settings
    trainer = UltimateEnsembleTrainer(
        reward_type="simple",  # Best performing reward from A/B test
        gpu_id=-1,  # CPU for testing
        team_name="test_ultimate"
    )
    
    print(f"\n📊 Agent Configurations:")
    for agent_name, config in trainer.agent_configs.items():
        category = config['category']
        buffer_type = config['buffer_type']
        agent_class = config['agent_class'].__name__
        print(f"   {agent_name:15}: {category:15} + {buffer_type:12} buffer ({agent_class})")
    
    # Modify configs for quick test
    for agent_name, config in trainer.agent_configs.items():
        # Reduce training steps for quick test
        config['config'].break_step = 3
        config['config'].eval_per_step = 1
        config['config'].horizon_len = 10
        config['config'].buffer_size = 50
        config['config'].early_stopping_enabled = False
        
        # Smaller network for faster training
        config['config'].net_dims = (16, 8)
        config['config'].batch_size = 16
    
    print(f"\n🏃 Starting quick training test...")
    test_start = time.time()
    
    # Test one agent from each category
    test_agents = ["AgentD3QN", "AgentPPO", "AgentRainbow"]
    
    for agent_name in test_agents:
        if agent_name in trainer.agent_configs:
            print(f"\n🤖 Testing {agent_name} ({trainer.agent_configs[agent_name]['category']})...")
            
            agent_config = trainer.agent_configs[agent_name]
            save_dir = f"test_ultimate_models/{agent_name.lower()}"
            
            result = trainer.train_single_agent(agent_name, agent_config, save_dir)
            trainer.training_results[agent_name] = result
            
            if result['success']:
                print(f"   ✅ {agent_name} test passed:")
                print(f"      📈 Return: {result['total_return']:.4f}")
                print(f"      🎯 Trading: {result['trading_activity_pct']:.1f}%")
                print(f"      ⏱️  Time: {result['training_time']:.1f}s")
                print(f"      🧠 Category: {result['category']}")
            else:
                print(f"   ❌ {agent_name} test failed: {result.get('error', 'Unknown')}")
    
    test_time = time.time() - test_start
    
    # Analysis
    print(f"\n📊 ULTIMATE ENSEMBLE TEST RESULTS:")
    print("-" * 60)
    
    successful_tests = [r for r in trainer.training_results.values() if r['success']]
    
    if len(successful_tests) >= 3:
        print(f"✅ Ultimate ensemble test PASSED!")
        print(f"   🔵 DQN agent: Working")
        print(f"   🟢 PPO agent: Working") 
        print(f"   🌈 Rainbow agent: Working")
        print(f"   🎯 Integration: Successful")
        print(f"   ⏱️  Total time: {test_time:.1f}s")
        
        # Performance comparison
        print(f"\n🏆 ALGORITHM PERFORMANCE COMPARISON:")
        for result in successful_tests:
            category_emoji = {
                'dqn': '🔵',
                'policy_gradient': '🟢',
                'advanced_dqn': '🌈'
            }.get(result['category'], '⚪')
            
            print(f"   {category_emoji} {result['agent_name']:12}: Return={result['total_return']:6.3f}, "
                  f"Trading={result['trading_activity_pct']:4.1f}%, {result['category']}")
        
        # Find best performer
        best_agent = max(successful_tests, key=lambda x: x['total_return'])
        print(f"\n🥇 BEST PERFORMER: {best_agent['agent_name']} ({best_agent['category']})")
        print(f"   📈 Return: {best_agent['total_return']:.4f}")
        print(f"   🎯 Trading: {best_agent['trading_activity_pct']:.1f}%")
        
        # Diversity analysis
        categories = set([r['category'] for r in successful_tests])
        buffer_types = set([r['buffer_type'] for r in successful_tests])
        
        print(f"\n🎭 DIVERSITY METRICS:")
        print(f"   Algorithm categories: {len(categories)}/3")
        print(f"   Buffer types: {len(buffer_types)}")
        print(f"   Categories: {', '.join(categories)}")
        
        return True
    else:
        print(f"❌ Ultimate ensemble test FAILED!")
        print(f"   Successful agents: {len(successful_tests)}/3")
        print(f"   Issues need to be resolved before full training")
        
        # Show which failed
        for agent_name in test_agents:
            result = trainer.training_results.get(agent_name, {'success': False})
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {agent_name}")
        
        return False


def main():
    """Main test execution"""
    
    success = test_ultimate_ensemble()
    
    print(f"\n📋 RECOMMENDATIONS:")
    if success:
        print(f"   ✅ Ultimate ensemble ready for production training")
        print(f"   🚀 Run: python3 task1_ensemble_ultimate.py 0 --reward simple")
        print(f"   🌟 Expect state-of-the-art performance from algorithm diversity")
        print(f"   🎯 Rainbow DQN may provide the best single-agent performance")
        print(f"   🔄 PPO provides valuable policy gradient perspective")
        print(f"   🎪 Ensemble voting should be highly robust")
    else:
        print(f"   🔧 Debug ultimate ensemble issues:")
        print(f"   1. Check individual agent implementations")
        print(f"   2. Verify buffer compatibility across agent types")
        print(f"   3. Test Rainbow DQN components separately")
        print(f"   4. Review error messages above")
        print(f"   5. Consider running hybrid ensemble first")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)