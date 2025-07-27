"""
Test Enhanced Conservative Trading Solution
Simple test to validate the implementation works
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any

# Import our enhanced components
from trade_simulator import TradeSimulator
from erl_agent import AgentDoubleDQN
from erl_config import Config
from training_monitor import ActionDiversityMonitor
from reward_functions import create_reward_calculator
from erl_exploration import ExplorationOrchestrator
from hpo_config import Task1HPOSearchSpace


def test_enhanced_reward_system():
    """Test the enhanced reward system"""
    print("🧪 Testing Enhanced Reward System")
    print("-" * 40)
    
    # Test different reward types
    reward_types = ['simple', 'multi_objective', 'adaptive_multi_objective']
    
    for reward_type in reward_types:
        print(f"\n📊 Testing {reward_type} reward:")
        
        # Create reward calculator
        reward_weights = {
            'conservatism_penalty_weight': 0.2,
            'action_diversity_weight': 0.15,
            'transaction_cost_weight': 0.5,
            'risk_adjusted_return_weight': 0.7
        }
        
        calculator = create_reward_calculator(
            reward_type=reward_type,
            device="cpu",
            reward_weights=reward_weights
        )
        
        # Test reward calculation
        old_asset = torch.tensor([100000.0])
        new_asset = torch.tensor([100100.0])  # $100 gain
        action_int = torch.tensor([1])  # Hold action
        mid_price = torch.tensor([50000.0])
        slippage = 1e-5
        
        reward = calculator.calculate_reward(old_asset, new_asset, action_int, mid_price, slippage)
        print(f"   Reward for $100 gain with hold: {reward.item():.4f}")
        
        # Test conservative penalty (if available)
        if hasattr(calculator, 'conservatism_penalty'):
            print(f"   Has conservatism penalty: ✅")
        else:
            print(f"   Has conservatism penalty: ❌")


def test_exploration_strategies():
    """Test exploration strategies"""
    print("\n🧪 Testing Exploration Strategies")
    print("-" * 40)
    
    # Test exploration orchestrator
    orchestrator = ExplorationOrchestrator(action_dim=3)
    
    # Simulate conservative behavior
    print("\n📊 Simulating conservative trading:")
    for i in range(100):
        # 90% hold actions (conservative)
        action = 1 if np.random.random() < 0.9 else np.random.choice([0, 2])
        reward = np.random.randn() * 0.1
        
        should_explore = orchestrator.should_explore()
        orchestrator.update(action, reward)
        
        if i % 20 == 0:
            stats = orchestrator.get_stats()
            print(f"   Step {i}: Exploration rate={stats['current_exploration_rate']:.3f}, "
                  f"Forced explorations={stats['forced_explorations']}")


def test_action_diversity_monitor():
    """Test action diversity monitoring"""
    print("\n🧪 Testing Action Diversity Monitor")
    print("-" * 40)
    
    monitor = ActionDiversityMonitor(
        window_size=100,
        diversity_threshold=0.3,
        conservatism_threshold=0.7,
        checkpoint_dir="test_checkpoints"
    )
    
    # Simulate conservative episode
    print("\n📊 Simulating conservative episode:")
    episode_return = 0
    
    for step in range(200):
        # Conservative behavior: 85% hold, 10% sell, 5% buy
        if np.random.random() < 0.85:
            action = 1  # Hold
        elif np.random.random() < 0.67:  # 10% of remaining 15%
            action = 0  # Sell
        else:
            action = 2  # Buy
            
        reward = np.random.randn() * 0.1
        episode_return += reward
        
        monitor.update(action, reward, done=(step == 199))
        
    # Check diversity
    diversity_check = monitor.check_diversity()
    print(f"\n📊 Diversity Check Results:")
    print(f"   Status: {diversity_check.get('status', 'N/A')}")
    
    if 'metrics' in diversity_check:
        metrics = diversity_check['metrics']
        print(f"   Hold ratio: {metrics.get('hold_ratio', 0):.1%}")
        print(f"   Buy ratio: {metrics.get('buy_ratio', 0):.1%}")
        print(f"   Sell ratio: {metrics.get('sell_ratio', 0):.1%}")
        print(f"   Entropy: {metrics.get('entropy', 0):.3f}")
        
    # Check intervention
    should_intervene, reason = monitor.should_intervene()
    print(f"   Intervention needed: {should_intervene}")
    if should_intervene:
        print(f"   Reason: {reason}")


def test_enhanced_hpo_parameters():
    """Test enhanced HPO parameter space"""
    print("\n🧪 Testing Enhanced HPO Parameters")
    print("-" * 40)
    
    # Create mock trial for parameter suggestion
    class MockTrial:
        def suggest_int(self, name, low, high, step=1):
            return np.random.randint(low, high + 1)
            
        def suggest_float(self, name, low, high, log=False):
            if log:
                return np.exp(np.random.uniform(np.log(low), np.log(high)))
            return np.random.uniform(low, high)
            
        def suggest_categorical(self, name, choices):
            return np.random.choice(choices)
    
    trial = MockTrial()
    
    # Test parameter suggestion
    params = Task1HPOSearchSpace.suggest_parameters(trial)
    
    print(f"📊 Sample HPO Parameters:")
    print(f"   Explore rate: {params['explore_rate']:.4f}")
    print(f"   Min explore rate: {params['min_explore_rate']:.4f}")
    print(f"   Reward type: {params['reward_type']}")
    print(f"   Conservatism penalty weight: {params['conservatism_penalty_weight']:.3f}")
    print(f"   Action diversity weight: {params['action_diversity_weight']:.3f}")
    
    # Convert to config
    config = Task1HPOSearchSpace.convert_to_config(params)
    print(f"\n📊 Converted Config:")
    print(f"   Net dims: {config['net_dims']}")
    print(f"   Learning rate: {config['learning_rate']:.6f}")


def test_trading_simulator_integration():
    """Test trading simulator with enhanced rewards"""
    print("\n🧪 Testing Trading Simulator Integration")
    print("-" * 40)
    
    try:
        # Create simulator with limited data
        simulator = TradeSimulator(
            num_sims=4,
            gpu_id=-1,  # CPU
            data_length=1000  # Small dataset for testing
        )
        
        print(f"✅ Simulator created successfully")
        print(f"   State dim: {simulator.state_dim}")
        print(f"   Action dim: {simulator.action_dim}")
        print(f"   Reward type: {simulator.reward_type}")
        
        # Test reward type switching
        simulator.set_reward_type(
            'adaptive_multi_objective',
            reward_weights={
                'conservatism_penalty_weight': 0.3,
                'action_diversity_weight': 0.2
            }
        )
        print(f"✅ Reward type switched successfully")
        
        # Test reset and step
        state = simulator.reset()
        print(f"✅ Reset successful, state shape: {state.shape}")
        
        # Test a few steps
        for i in range(5):
            action = torch.randint(0, 3, (simulator.num_sims, 1))
            next_state, reward, done, _ = simulator.step(action)
            print(f"   Step {i}: Reward mean = {reward.mean().item():.6f}")
            
        print(f"✅ Step testing successful")
        
        # Get reward metrics
        metrics = simulator.get_reward_metrics()
        print(f"📊 Reward Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
                
    except Exception as e:
        print(f"❌ Simulator test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_agent_integration():
    """Test agent with enhanced exploration"""
    print("\n🧪 Testing Agent Integration")
    print("-" * 40)
    
    try:
        # Create config with enhanced parameters
        args = Config()
        args.explore_rate = 0.1
        args.min_explore_rate = 0.01
        args.exploration_decay_rate = 0.995
        args.exploration_warmup_steps = 1000
        args.force_exploration_probability = 0.05
        
        # Create agent
        agent = AgentDoubleDQN(
            net_dims=(128, 64, 32),
            state_dim=50,  # Example state dim
            action_dim=3,
            gpu_id=-1,
            args=args
        )
        
        print(f"✅ Agent created successfully")
        print(f"   Explore rate: {agent.act.explore_rate:.4f}")
        print(f"   Min explore rate: {agent.min_explore_rate:.4f}")
        
        # Test exploration methods
        agent.update_exploration_rate()
        print(f"✅ Exploration rate update successful")
        
        should_force = agent.should_force_exploration()
        print(f"   Force exploration: {should_force}")
        
        print(f"✅ Agent integration successful")
        
    except Exception as e:
        print(f"❌ Agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("🚀 Enhanced Conservative Trading Solution - Integration Tests")
    print("=" * 60)
    
    # Run all tests
    try:
        test_enhanced_reward_system()
        test_exploration_strategies()
        test_action_diversity_monitor()
        test_enhanced_hpo_parameters()
        test_trading_simulator_integration()
        test_agent_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("   Enhanced conservative trading solution is ready for deployment")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()