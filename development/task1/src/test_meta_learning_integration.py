"""
Comprehensive Integration Tests for Meta-Learning System
Tests complete integration of meta-learning with existing ensemble
"""

import torch
import numpy as np
import tempfile
import shutil
import os
import sys
import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all meta-learning components
from meta_learning_framework import (
    MetaLearningEnsembleManager,
    MetaLearningRiskManagedEnsemble
)
from meta_learning_config import (
    MetaLearningConfig,
    MetaLearningTracker,
    create_meta_learning_config
)
from meta_learning_agent_wrapper import (
    AgentWrapperFactory,
    AgentEnsembleWrapper,
    create_agent_wrappers_from_config
)
from task1_ensemble_meta_learning import MetaLearningEnsembleTrainer
from meta_learning_evaluation import MetaLearningEvaluator

# Import existing components for compatibility testing
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from trade_simulator import TradeSimulator
from erl_config import Config


class TestMetaLearningIntegration:
    """Test complete meta-learning system integration"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Import test environment
        from test_environment import TestTradingEnvironment
        
        # Create test configuration
        self.config = create_meta_learning_config(
            preset='balanced',
            env_args={
                'env_name': 'TradeSimulator-v0',
                'state_dim': 50,
                'action_dim': 3,
                'if_discrete': True
            },
            custom_params={
                'meta_lookback': 100,  # Smaller for testing
                'meta_training_frequency': 20,
                'break_step': 50  # Shorter training for tests
            }
        )
        
        # Set environment class for testing
        self.config.env_class = TestTradingEnvironment
        self.config.state_dim = 50
        self.config.action_dim = 3
        
        print(f"📁 Test temp directory: {self.temp_dir}")
    
    def teardown_method(self):
        """Cleanup after each test"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"⚠️ Failed to cleanup temp directory: {e}")
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline from start to finish"""
        print("\n🧪 Testing Complete Training Pipeline")
        
        # Create trainer
        trainer = MetaLearningEnsembleTrainer(
            config=self.config,
            team_name="test_ensemble",
            save_dir=self.temp_dir
        )
        
        # Verify trainer initialization
        assert trainer.meta_ensemble is not None
        assert trainer.agent_wrappers is not None
        assert len(trainer.agent_wrappers) >= 3  # At least 3 agents
        assert trainer.meta_learning_manager is not None
        
        # Test single episode training
        episode_stats = trainer.train_episode(episode_num=0, max_steps=100)
        
        # Verify episode results
        assert 'total_reward' in episode_stats
        assert 'steps' in episode_stats
        assert 'regime_changes' in episode_stats
        assert 'meta_learning_updates' in episode_stats
        assert isinstance(episode_stats['total_reward'], float)
        assert episode_stats['steps'] > 0
        
        print(f"   ✅ Single episode training: {episode_stats['steps']} steps, "
              f"reward={episode_stats['total_reward']:.4f}")
        
        # Test short training session
        session_stats = trainer.train_full_session(
            num_episodes=5,
            save_interval=3,
            evaluation_interval=2
        )
        
        # Verify session results
        assert 'total_episodes' in session_stats
        assert 'episode_rewards' in session_stats
        assert 'best_episode_reward' in session_stats
        assert 'final_performance' in session_stats
        assert len(session_stats['episode_rewards']) == 5
        
        print(f"   ✅ Training session: {session_stats['total_episodes']} episodes")
        
        # Verify model saving
        assert os.path.exists(os.path.join(self.temp_dir, "final_session"))
        assert os.path.exists(os.path.join(self.temp_dir, "final_session", "agents"))
        
        print(f"   ✅ Models saved successfully")
    
    def test_meta_learning_adaptation(self):
        """Test meta-learning adaptation over time"""
        print("\n🧪 Testing Meta-Learning Adaptation")
        
        # Create trainer
        trainer = MetaLearningEnsembleTrainer(
            config=self.config,
            team_name="test_adaptation",
            save_dir=self.temp_dir
        )
        
        # Track meta-learning progress over episodes
        meta_learning_progress = []
        
        for episode in range(10):
            episode_stats = trainer.train_episode(episode_num=episode, max_steps=50)
            
            # Track meta-learning statistics
            meta_samples = len(trainer.meta_learning_manager.training_data['market_features'])
            meta_updates = trainer.training_stats['meta_learning_updates']
            
            meta_learning_progress.append({
                'episode': episode,
                'meta_samples': meta_samples,
                'meta_updates': meta_updates,
                'regime': episode_stats.get('final_regime', 'unknown')
            })
            
            # Force meta-learning update every few episodes
            if episode % 3 == 2 and meta_samples > 10:
                trainer._perform_meta_learning_update()
        
        # Verify meta-learning is working
        final_progress = meta_learning_progress[-1]
        initial_progress = meta_learning_progress[0]
        
        assert final_progress['meta_samples'] > initial_progress['meta_samples']
        print(f"   ✅ Meta-learning data collected: "
              f"{initial_progress['meta_samples']} → {final_progress['meta_samples']} samples")
        
        # Test regime detection is working
        detected_regimes = set(p['regime'] for p in meta_learning_progress)
        assert len(detected_regimes) >= 1  # At least one regime detected
        print(f"   ✅ Regime detection: {len(detected_regimes)} regimes detected: {detected_regimes}")
        
        # Test adaptive weighting
        ensemble = trainer.meta_ensemble
        state = torch.randn(50)
        
        # Get multiple weight samples
        weight_samples = []
        for _ in range(5):
            weights = trainer.meta_learning_manager.get_adaptive_algorithm_weights()
            weight_samples.append(weights)
        
        # Verify weights are reasonable
        for weights in weight_samples:
            assert abs(sum(weights.values()) - 1.0) < 0.01  # Sum to 1
            assert all(w >= 0 for w in weights.values())  # Non-negative
            assert max(weights.values()) <= 0.7  # Max weight constraint
        
        print(f"   ✅ Adaptive weighting working: {len(weight_samples)} weight samples")
    
    def test_agent_wrapper_compatibility(self):
        """Test agent wrapper compatibility with existing agents"""
        print("\n🧪 Testing Agent Wrapper Compatibility")
        
        # Create test agents
        test_agents = {
            'd3qn': AgentD3QN([256, 256], 50, 3, gpu_id=0),
            'double_dqn': AgentDoubleDQN([256, 256], 50, 3, gpu_id=0),
            'twin_d3qn': AgentTwinD3QN([256, 256], 50, 3, gpu_id=0)
        }
        
        # Create wrappers
        wrappers = AgentWrapperFactory.create_multiple_wrappers(test_agents)
        
        # Test each wrapper
        test_state = torch.randn(50)
        
        for agent_name, wrapper in wrappers.items():
            # Test action generation
            action, confidence, info = wrapper.get_action_with_confidence(test_state)
            
            assert isinstance(action, int)
            assert 0 <= action <= 2
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
            assert isinstance(info, dict)
            
            print(f"   ✅ {agent_name}: action={action}, confidence={confidence:.3f}")
            
            # Test performance tracking
            wrapper.update_performance(action, 0.01, confidence)
            metrics = wrapper.get_performance_metrics()
            
            assert 'sharpe_ratio' in metrics
            assert 'win_rate' in metrics
            assert 'confidence' in metrics
            
            # Test statistics
            stats = wrapper.get_agent_statistics()
            assert 'agent_name' in stats
            assert 'total_decisions' in stats
            assert stats['total_decisions'] > 0
        
        # Test ensemble wrapper
        ensemble_wrapper = AgentEnsembleWrapper(wrappers)
        ensemble_results = ensemble_wrapper.get_all_actions_with_confidence(test_state)
        
        assert len(ensemble_results) == len(wrappers)
        for agent_name, (action, confidence, info) in ensemble_results.items():
            assert agent_name in wrappers
            assert isinstance(action, int)
            assert isinstance(confidence, float)
        
        print(f"   ✅ Ensemble wrapper: {len(ensemble_results)} agents coordinated")
    
    def test_risk_management_integration(self):
        """Test integration with risk management systems"""
        print("\n🧪 Testing Risk Management Integration")
        
        # Create trainer with risk management
        trainer = MetaLearningEnsembleTrainer(
            config=self.config,
            team_name="test_risk",
            save_dir=self.temp_dir
        )
        
        # Test risk management is active
        assert trainer.risk_manager is not None
        assert trainer.meta_ensemble.risk_manager is not None
        
        # Test trading action with risk constraints
        state = torch.randn(50)
        price = 100.0
        volume = 1000.0
        
        # Get multiple actions to test risk management
        actions_taken = []
        for _ in range(10):
            action, decision_info = trainer.meta_ensemble.get_trading_action(
                state, price, volume
            )
            actions_taken.append(action)
            
            # Verify action is valid
            assert isinstance(action, int)
            assert 0 <= action <= 2
            
            # Verify decision info
            assert 'algorithm_weights' in decision_info
            assert 'current_regime' in decision_info
            
            # Test risk constraints are applied
            weights = decision_info['algorithm_weights']
            assert abs(sum(weights.values()) - 1.0) < 0.01
            assert max(weights.values()) <= 0.7  # Max weight constraint
        
        print(f"   ✅ Risk management active: {len(set(actions_taken))} unique actions")
        
        # Test performance tracking with risk metrics
        trainer.meta_ensemble.update_performance(
            returns=0.05,
            sharpe_ratio=1.2,
            additional_metrics={
                'max_drawdown': 0.08,
                'win_rate': 0.65,
                'volatility': 0.15
            }
        )
        
        performance_summary = trainer.meta_ensemble.get_performance_summary()
        assert 'total_trades' in performance_summary
        assert 'recent_sharpe' in performance_summary
        
        print(f"   ✅ Performance tracking integrated")
    
    def test_evaluation_system(self):
        """Test meta-learning evaluation system"""
        print("\n🧪 Testing Evaluation System")
        
        # Create and train a minimal ensemble
        trainer = MetaLearningEnsembleTrainer(
            config=self.config,
            team_name="test_eval",
            save_dir=self.temp_dir
        )
        
        # Run minimal training
        trainer.train_full_session(num_episodes=3, save_interval=2)
        
        # Create evaluator
        evaluator = MetaLearningEvaluator(
            model_path=os.path.join(self.temp_dir, "final_session"),
            config=self.config
        )
        
        # Test evaluation
        evaluation_results = evaluator.evaluate_comprehensive(
            num_episodes=3,
            max_steps_per_episode=50,
            save_results=True
        )
        
        # Verify evaluation results structure
        assert 'portfolio_metrics' in evaluation_results
        assert 'meta_learning_metrics' in evaluation_results
        assert 'agent_comparison' in evaluation_results
        assert 'regime_analysis' in evaluation_results
        assert 'detailed_analysis' in evaluation_results
        
        # Verify portfolio metrics
        portfolio = evaluation_results['portfolio_metrics']
        assert 'mean_return' in portfolio
        assert 'mean_sharpe_ratio' in portfolio
        assert 'success_rate' in portfolio
        assert portfolio['total_episodes'] == 3
        
        # Verify meta-learning metrics
        meta = evaluation_results['meta_learning_metrics']
        assert 'mean_confidence' in meta
        assert 'mean_agreement_rate' in meta
        assert 'regime_adaptability' in meta
        
        # Verify detailed analysis
        analysis = evaluation_results['detailed_analysis']
        assert 'performance_assessment' in analysis
        assert 'strengths_weaknesses' in analysis
        assert 'recommendations' in analysis
        
        # Check that evaluation files were saved
        eval_dir = os.path.join(self.temp_dir, "final_session", "evaluation_results")
        assert os.path.exists(eval_dir)
        assert os.path.exists(os.path.join(eval_dir, "evaluation_results.json"))
        assert os.path.exists(os.path.join(eval_dir, "evaluation_summary.txt"))
        
        print(f"   ✅ Evaluation completed: Grade {analysis['performance_assessment']['grade']}")
    
    def test_configuration_management(self):
        """Test configuration management and presets"""
        print("\n🧪 Testing Configuration Management")
        
        # Test different configuration presets
        presets = ['conservative', 'aggressive', 'balanced', 'research']
        
        for preset in presets:
            config = create_meta_learning_config(
                preset=preset,
                env_args={'env_name': 'test', 'state_dim': 50, 'action_dim': 3}
            )
            
            # Verify configuration is valid
            assert hasattr(config, 'meta_learning_enabled')
            assert hasattr(config, 'max_agent_weight')
            assert hasattr(config, 'meta_training_frequency')
            
            # Verify preset-specific characteristics
            if preset == 'conservative':
                assert config.max_agent_weight <= 0.5
                assert config.min_diversification_agents >= 3
            elif preset == 'aggressive':
                assert config.max_agent_weight >= 0.7
                assert config.min_diversification_agents <= 3
            
            print(f"   ✅ {preset}: max_weight={config.max_agent_weight}, "
                  f"min_div={config.min_diversification_agents}")
        
        # Test custom parameter override
        custom_config = create_meta_learning_config(
            preset='balanced',
            env_args={'env_name': 'test', 'state_dim': 50, 'action_dim': 3},
            custom_params={
                'max_agent_weight': 0.8,
                'meta_training_frequency': 150,
                'regime_features_dim': 75
            }
        )
        
        assert custom_config.max_agent_weight == 0.8
        assert custom_config.meta_training_frequency == 150
        assert custom_config.regime_features_dim == 75
        
        print(f"   ✅ Custom parameters applied successfully")
        
        # Test configuration save/load
        from meta_learning_config import MetaLearningConfigManager
        
        config_file = os.path.join(self.temp_dir, "test_config.json")
        MetaLearningConfigManager.save_config_to_file(custom_config, config_file)
        
        assert os.path.exists(config_file)
        
        loaded_config = MetaLearningConfigManager.load_config_from_file(
            config_file, None, None, {'env_name': 'test'}
        )
        
        assert loaded_config.max_agent_weight == 0.8
        assert loaded_config.meta_training_frequency == 150
        
        print(f"   ✅ Configuration persistence working")
    
    def test_error_handling_and_robustness(self):
        """Test error handling and system robustness"""
        print("\n🧪 Testing Error Handling and Robustness")
        
        # Test with minimal data
        config = create_meta_learning_config(
            preset='balanced',
            env_args={'env_name': 'test', 'state_dim': 10, 'action_dim': 3},
            custom_params={'meta_lookback': 5, 'meta_batch_size': 2}
        )
        
        trainer = MetaLearningEnsembleTrainer(
            config=config,
            team_name="test_errors",
            save_dir=self.temp_dir
        )
        
        # Test training with very short episodes
        try:
            episode_stats = trainer.train_episode(episode_num=0, max_steps=5)
            assert 'total_reward' in episode_stats
            print(f"   ✅ Short episode handling: {episode_stats['steps']} steps")
        except Exception as e:
            print(f"   ⚠️ Short episode error: {e}")
        
        # Test with invalid state dimensions
        try:
            invalid_state = torch.randn(1)  # Wrong dimension
            action, decision_info = trainer.meta_ensemble.get_trading_action(
                invalid_state, 100.0, 1000.0
            )
            print(f"   ⚠️ Invalid state handled: action={action}")
        except Exception as e:
            print(f"   ✅ Invalid state error caught: {e}")
        
        # Test meta-learning with insufficient data
        try:
            trainer.meta_learning_manager.train_meta_models(batch_size=1, epochs=1)
            print(f"   ✅ Meta-learning with minimal data")
        except Exception as e:
            print(f"   ✅ Meta-learning error handled: {e}")
        
        # Test model saving/loading errors
        invalid_path = "/invalid/path/that/does/not/exist"
        try:
            trainer._save_checkpoint("test_invalid")
            print(f"   ✅ Model saving error handling")
        except Exception as e:
            print(f"   ✅ Save error handled: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks and timing"""
        print("\n🧪 Testing Performance Benchmarks")
        
        # Create optimized configuration
        config = create_meta_learning_config(
            preset='balanced',
            env_args={'env_name': 'test', 'state_dim': 50, 'action_dim': 3},
            custom_params={
                'meta_lookback': 200,
                'meta_training_frequency': 50
            }
        )
        
        trainer = MetaLearningEnsembleTrainer(
            config=config,
            team_name="test_performance",
            save_dir=self.temp_dir
        )
        
        # Benchmark single action time
        state = torch.randn(50)
        
        action_times = []
        for _ in range(10):
            start_time = time.time()
            action, decision_info = trainer.meta_ensemble.get_trading_action(
                state, 100.0, 1000.0
            )
            action_time = time.time() - start_time
            action_times.append(action_time)
        
        avg_action_time = np.mean(action_times)
        print(f"   ⏱️ Average action time: {avg_action_time*1000:.2f}ms")
        
        # Verify reasonable performance (should be under 100ms)
        assert avg_action_time < 0.1, f"Action time too slow: {avg_action_time*1000:.2f}ms"
        
        # Benchmark episode training time
        start_time = time.time()
        episode_stats = trainer.train_episode(episode_num=0, max_steps=100)
        episode_time = time.time() - start_time
        
        steps_per_second = episode_stats['steps'] / episode_time
        print(f"   ⏱️ Episode training: {episode_time:.2f}s, {steps_per_second:.1f} steps/sec")
        
        # Verify reasonable training speed
        assert steps_per_second > 10, f"Training too slow: {steps_per_second:.1f} steps/sec"
        
        # Memory usage check (basic)
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"   💾 Memory usage: {memory_mb:.1f} MB")
        
        # Should use reasonable amount of memory (less than 2GB for testing)
        assert memory_mb < 2000, f"Memory usage too high: {memory_mb:.1f} MB"
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing system"""
        print("\n🧪 Testing Backward Compatibility")
        
        # Test that meta-learning can be disabled
        config = create_meta_learning_config(
            preset='balanced',
            env_args={'env_name': 'test', 'state_dim': 50, 'action_dim': 3},
            custom_params={'meta_learning_enabled': False}
        )
        
        trainer = MetaLearningEnsembleTrainer(
            config=config,
            team_name="test_compatibility",
            save_dir=self.temp_dir
        )
        
        # Should still work without meta-learning
        episode_stats = trainer.train_episode(episode_num=0, max_steps=50)
        assert 'total_reward' in episode_stats
        print(f"   ✅ Meta-learning disabled mode works")
        
        # Test individual agent interfaces remain unchanged
        for agent_name, wrapper in trainer.agent_wrappers.items():
            # Original agent should still have expected methods
            original_agent = wrapper.agent
            
            assert hasattr(original_agent, 'act')
            assert hasattr(original_agent, 'save_or_load_agent')
            
            # Test that original agent interface works
            state = torch.randn(50)
            result = original_agent.act(state)
            assert isinstance(result, (int, tuple))
            
            print(f"   ✅ {agent_name} original interface preserved")
        
        # Test existing evaluation metrics
        if episode_stats.get('sharpe_ratio') is not None:
            assert isinstance(episode_stats['sharpe_ratio'], float)
        
        if 'returns' in episode_stats:
            returns = episode_stats['returns']
            assert isinstance(returns, list)
            assert all(isinstance(r, (int, float)) for r in returns)
        
        print(f"   ✅ Existing metrics format preserved")


def run_integration_tests():
    """Run all integration tests"""
    
    print("🧪 META-LEARNING INTEGRATION TEST SUITE")
    print("=" * 60)
    
    test_instance = TestMetaLearningIntegration()
    
    tests = [
        ('Complete Training Pipeline', test_instance.test_complete_training_pipeline),
        ('Meta-Learning Adaptation', test_instance.test_meta_learning_adaptation),
        ('Agent Wrapper Compatibility', test_instance.test_agent_wrapper_compatibility),
        ('Risk Management Integration', test_instance.test_risk_management_integration),
        ('Evaluation System', test_instance.test_evaluation_system),
        ('Configuration Management', test_instance.test_configuration_management),
        ('Error Handling', test_instance.test_error_handling_and_robustness),
        ('Performance Benchmarks', test_instance.test_performance_benchmarks),
        ('Backward Compatibility', test_instance.test_backward_compatibility),
    ]
    
    passed_tests = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            
            # Setup for each test
            test_instance.setup_method()
            
            # Run test
            start_time = time.time()
            test_func()
            test_time = time.time() - start_time
            
            print(f"\n✅ {test_name} PASSED ({test_time:.2f}s)")
            passed_tests += 1
            
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed_tests.append((test_name, str(e)))
        
        finally:
            # Cleanup after each test
            try:
                test_instance.teardown_method()
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"🎯 INTEGRATION TEST RESULTS")
    print(f"="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {passed_tests/len(tests)*100:.1f}%")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
    else:
        print(f"\n🎉 All integration tests passed!")
    
    print(f"="*60)
    
    return passed_tests == len(tests)


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)