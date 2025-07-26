"""
Framework validation script for the FinRL Contest 2024 refactored architecture.

This script performs basic validation of the refactored components to ensure
they are working correctly before running the full test suite.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src_refactored to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from . import TEST_CONFIG


def validate_imports():
    """Validate that all core modules can be imported."""
    print("🔍 Validating imports...")
    
    try:
        # Core components
        from ..core.types import StateType, ActionType, TrainingStats
        from ..core.base_agent import BaseAgent
        from ..core.interfaces import NetworkProtocol, ReplayBufferProtocol
        print("  ✅ Core components imported successfully")
        
        # Agent components
        from ..agents import create_agent, create_ensemble_agents, AGENT_REGISTRY
        from ..agents.base_dqn_agent import BaseDQNAgent
        from ..agents.double_dqn_agent import DoubleDQNAgent, D3QNAgent
        print("  ✅ Agent components imported successfully")
        
        # Ensemble components
        from ..ensemble import (
            BaseEnsemble, EnsembleStrategy, VotingEnsemble, 
            StackingEnsemble, create_ensemble
        )
        print("  ✅ Ensemble components imported successfully")
        
        # Configuration components
        from ..config import DoubleDQNConfig, PrioritizedDQNConfig
        print("  ✅ Configuration components imported successfully")
        
        # Network components
        from ..networks import QNetTwin, QNetTwinDuel
        print("  ✅ Network components imported successfully")
        
        # Training components
        from ..training.ensemble_trainer import EnsembleTrainer, TrainingConfig
        print("  ✅ Training components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def validate_agent_creation():
    """Validate that agents can be created correctly."""
    print("\n🤖 Validating agent creation...")
    
    try:
        from ..agents import create_agent, AGENT_REGISTRY
        
        device = torch.device(TEST_CONFIG['device'])
        
        # Test creating different agent types
        agent_types_to_test = ['AgentDoubleDQN', 'AgentD3QN']
        
        for agent_type in agent_types_to_test:
            if agent_type in AGENT_REGISTRY:
                agent = create_agent(
                    agent_type=agent_type,
                    state_dim=TEST_CONFIG['state_dim'],
                    action_dim=TEST_CONFIG['action_dim'],
                    device=device
                )
                
                # Basic functionality test
                state = torch.randn(TEST_CONFIG['state_dim'])
                action = agent.select_action(state, deterministic=True)
                
                assert isinstance(action, (int, np.integer))
                assert 0 <= action < TEST_CONFIG['action_dim']
                
                print(f"  ✅ {agent_type} created and tested successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent creation failed: {e}")
        return False


def validate_ensemble_creation():
    """Validate that ensembles can be created correctly."""
    print("\n🤝 Validating ensemble creation...")
    
    try:
        from ..agents import create_ensemble_agents
        from ..ensemble import create_voting_ensemble, EnsembleStrategy
        
        device = torch.device(TEST_CONFIG['device'])
        
        # Create agents for ensemble
        agent_configs = {
            "agent1": {"agent_type": "AgentDoubleDQN"},
            "agent2": {"agent_type": "AgentD3QN"}
        }
        
        agents = create_ensemble_agents(
            agent_configs,
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=device
        )
        
        if len(agents) == 2:
            # Create voting ensemble
            ensemble = create_voting_ensemble(
                agents=agents,
                strategy=EnsembleStrategy.MAJORITY_VOTE,
                device=device
            )
            
            # Test ensemble functionality
            state = torch.randn(TEST_CONFIG['state_dim'])
            action = ensemble.select_action(state, deterministic=True)
            
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < TEST_CONFIG['action_dim']
            
            print("  ✅ Voting ensemble created and tested successfully")
            return True
        else:
            print(f"  ⚠️  Expected 2 agents, got {len(agents)}")
            return False
        
    except Exception as e:
        print(f"  ❌ Ensemble creation failed: {e}")
        return False


def validate_mock_environment():
    """Validate the mock environment functionality."""
    print("\n🌍 Validating mock environment...")
    
    try:
        from .utils.mock_environment import MockEnvironment
        
        env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=10,
            seed=TEST_CONFIG['seed']
        )
        
        # Test environment reset
        state = env.reset()
        assert state.shape == (TEST_CONFIG['state_dim'],)
        
        # Test environment step
        action = 0  # Valid action
        next_state, reward, done, info = env.step(action)
        
        assert next_state.shape == (TEST_CONFIG['state_dim'],)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        print("  ✅ Mock environment tested successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Mock environment failed: {e}")
        return False


def validate_training_config():
    """Validate training configuration."""
    print("\n⚙️  Validating training configuration...")
    
    try:
        from ..training.ensemble_trainer import TrainingConfig, TrainingResults
        
        # Test training config creation
        config = TrainingConfig(
            total_episodes=100,
            individual_episodes=30,
            ensemble_episodes=50,
            fine_tuning_episodes=20
        )
        
        assert config.total_episodes == 100
        assert config.individual_episodes == 30
        
        # Test training results
        results = TrainingResults()
        results.training_rewards = [0.1, 0.2, 0.3]
        assert len(results.training_rewards) == 3
        
        print("  ✅ Training configuration tested successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Training configuration failed: {e}")
        return False


def run_integration_test():
    """Run a quick integration test."""
    print("\n🔗 Running integration test...")
    
    try:
        from ..agents import create_agent
        from .utils.mock_environment import MockEnvironment
        
        device = torch.device(TEST_CONFIG['device'])
        
        # Create environment and agent
        env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=5,
            seed=TEST_CONFIG['seed']
        )
        
        agent = create_agent(
            agent_type="AgentDoubleDQN",
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=device
        )
        
        # Run short episode
        state = env.reset()
        total_reward = 0
        
        for step in range(3):
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Quick update test
            transition = env.get_last_transition()
            if transition is not None:
                batch_data = (
                    torch.tensor([transition[0]], dtype=torch.float32),
                    torch.tensor([transition[1]], dtype=torch.long),
                    torch.tensor([transition[2]], dtype=torch.float32),
                    torch.tensor([not transition[3]], dtype=torch.float32),
                    torch.tensor([transition[4]], dtype=torch.float32)
                )
                result = agent.update(batch_data)
                assert result is not None
            
            state = next_state
            if done:
                break
        
        print(f"  ✅ Integration test completed (total reward: {total_reward:.3f})")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("FinRL Contest 2024 - Framework Validation")
    print("=" * 80)
    print(f"Device: {TEST_CONFIG['device']}")
    print(f"State dim: {TEST_CONFIG['state_dim']}")
    print(f"Action dim: {TEST_CONFIG['action_dim']}")
    print()
    
    # Run validation tests
    tests = [
        ("Imports", validate_imports),
        ("Agent Creation", validate_agent_creation),
        ("Ensemble Creation", validate_ensemble_creation),
        ("Mock Environment", validate_mock_environment),
        ("Training Config", validate_training_config),
        ("Integration", run_integration_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} validation failed")
        except Exception as e:
            print(f"💥 {test_name} validation crashed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    success_rate = (passed / total) * 100
    status = "✅ PASSED" if passed == total else "⚠️  PARTIAL" if passed > 0 else "❌ FAILED"
    
    print(f"Status: {status}")
    print(f"Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 All validations passed! The framework is ready for testing.")
        print("Run 'python -m tests.run_tests' to execute the full test suite.")
    elif passed > 0:
        print(f"\n⚠️  {total - passed} validations failed. Check the errors above.")
        print("Some components may need fixes before running full tests.")
    else:
        print("\n❌ All validations failed. Check your setup and dependencies.")
    
    print("=" * 80)
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)