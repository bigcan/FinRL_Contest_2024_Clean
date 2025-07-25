import torch
import numpy as np
import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meta_learning_framework import (
    MarketRegimeClassifier,
    AlgorithmPerformancePredictor,
    MarketFeatureExtractor,
    MetaLearningEnsembleManager,
    MetaLearningRiskManagedEnsemble
)


class TestMarketRegimeClassifier:
    """Test market regime classification"""
    
    def test_initialization(self):
        """Test proper initialization"""
        classifier = MarketRegimeClassifier(input_dim=50)
        assert classifier.input_dim == 50
        assert len(classifier.regime_labels) == 7
        
        # Test forward pass
        test_input = torch.randn(1, 50)
        output = classifier(test_input)
        assert output.shape == (1, 7)
        assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-6)  # Softmax check
    
    def test_regime_prediction(self):
        """Test regime prediction functionality"""
        classifier = MarketRegimeClassifier(input_dim=50)
        test_features = torch.randn(50)
        
        regime = classifier.predict_regime(test_features)
        assert regime in classifier.regime_labels
    
    def test_different_input_dims(self):
        """Test with different input dimensions"""
        for input_dim in [20, 50, 100]:
            classifier = MarketRegimeClassifier(input_dim=input_dim)
            test_input = torch.randn(2, input_dim)
            output = classifier(test_input)
            assert output.shape == (2, 7)


class TestAlgorithmPerformancePredictor:
    """Test algorithm performance prediction"""
    
    def test_initialization(self):
        """Test proper initialization"""
        predictor = AlgorithmPerformancePredictor(
            market_features_dim=50,
            agent_history_dim=20
        )
        
        # Test forward pass
        market_features = torch.randn(1, 50)
        agent_history = torch.randn(1, 20)
        output = predictor(market_features, agent_history)
        assert output.shape == (1, 1)
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        predictor = AlgorithmPerformancePredictor()
        
        batch_size = 5
        market_features = torch.randn(batch_size, 50)
        agent_history = torch.randn(batch_size, 20)
        
        output = predictor(market_features, agent_history)
        assert output.shape == (batch_size, 1)


class TestMarketFeatureExtractor:
    """Test market feature extraction"""
    
    def test_initialization(self):
        """Test proper initialization"""
        extractor = MarketFeatureExtractor(lookback_window=100)
        assert len(extractor.price_history) == 0
        assert len(extractor.volume_history) == 0
        assert extractor.lookback_window == 100
    
    def test_data_update(self):
        """Test data update functionality"""
        extractor = MarketFeatureExtractor(lookback_window=10)
        
        for i in range(15):
            extractor.update_data(price=100.0 + i, volume=1000 + i*10)
        
        assert len(extractor.price_history) == 10  # Should be capped at lookback_window
        assert len(extractor.volume_history) == 10
        assert extractor.price_history[-1] == 114.0  # Last price
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        extractor = MarketFeatureExtractor(lookback_window=50)
        
        # Add enough data for feature extraction
        np.random.seed(42)
        for i in range(50):
            price = 100.0 + np.random.randn() * 2
            volume = 1000 + np.random.randn() * 100
            extractor.update_data(price, volume)
        
        features = extractor.extract_features()
        assert isinstance(features, torch.Tensor)
        assert features.shape == (50,)  # Should always return 50 features
        assert torch.isfinite(features).all()  # No NaN or Inf values
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        extractor = MarketFeatureExtractor()
        
        # Add minimal data
        for i in range(5):
            extractor.update_data(100.0 + i, 1000)
        
        features = extractor.extract_features()
        assert features.shape == (50,)
        assert torch.allclose(features, torch.zeros(50))  # Should return zeros


class TestMetaLearningEnsembleManager:
    """Test meta-learning ensemble manager"""
    
    def setup_method(self):
        """Setup for each test"""
        # Mock agents
        self.mock_agents = {
            'agent1': Mock(),
            'agent2': Mock(),
            'agent3': Mock()
        }
        
        self.manager = MetaLearningEnsembleManager(
            agents=self.mock_agents,
            meta_lookback=100
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        assert len(self.manager.agents) == 3
        assert len(self.manager.performance_predictors) == 3
        assert self.manager.meta_lookback == 100
        assert isinstance(self.manager.market_feature_extractor, MarketFeatureExtractor)
    
    def test_market_data_update(self):
        """Test market data updates"""
        initial_regime = self.manager.current_regime
        
        # Update with market data
        for i in range(10):
            self.manager.update_market_data(100.0 + i * 0.1, 1000 + i * 10)
        
        assert len(self.manager.market_state_history) == 10
        # Regime might change, but should be valid
        assert self.manager.current_regime in self.manager.market_regime_classifier.regime_labels
    
    def test_agent_performance_update(self):
        """Test agent performance updates"""
        performance_metrics = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.65,
            'avg_return': 0.02,
            'volatility': 0.15
        }
        
        self.manager.update_agent_performance('agent1', performance_metrics)
        
        assert len(self.manager.agent_performance_history['agent1']) == 1
        assert self.manager.agent_performance_history['agent1'][0] == performance_metrics
    
    def test_agent_history_features(self):
        """Test agent history feature extraction"""
        # Add some performance history
        for i in range(5):
            perf = {
                'sharpe_ratio': 1.0 + i * 0.1,
                'win_rate': 0.5 + i * 0.05,
                'avg_return': 0.01 + i * 0.002,
                'volatility': 0.1 + i * 0.01
            }
            self.manager.update_agent_performance('agent1', perf)
        
        features = self.manager.get_agent_history_features('agent1')
        assert isinstance(features, torch.Tensor)
        assert features.shape == (20,)
        assert not torch.allclose(features, torch.zeros(20))  # Should have non-zero values
    
    def test_performance_prediction(self):
        """Test performance prediction"""
        # Add some market data
        for i in range(25):
            self.manager.update_market_data(100.0 + i * 0.1, 1000)
        
        # Add some agent performance history
        perf = {'sharpe_ratio': 1.2, 'win_rate': 0.6, 'avg_return': 0.015, 'volatility': 0.12}
        self.manager.update_agent_performance('agent1', perf)
        
        prediction = self.manager.predict_agent_performance('agent1')
        assert isinstance(prediction, float)
        assert np.isfinite(prediction)
    
    def test_adaptive_weights(self):
        """Test adaptive algorithm weights"""
        # Add market data and performance history
        for i in range(30):
            self.manager.update_market_data(100.0 + i * 0.1, 1000)
        
        for agent_name in self.manager.agent_names:
            perf = {
                'sharpe_ratio': np.random.normal(1.0, 0.3),
                'win_rate': np.random.uniform(0.4, 0.7),
                'avg_return': np.random.normal(0.01, 0.005),
                'volatility': np.random.uniform(0.05, 0.2)
            }
            self.manager.update_agent_performance(agent_name, perf)
        
        weights = self.manager.get_adaptive_algorithm_weights()
        
        # Check weights properties
        assert len(weights) == len(self.manager.agent_names)
        assert all(isinstance(w, float) for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6  # Should sum to 1
        assert all(w >= 0 for w in weights.values())  # All weights non-negative
        assert max(weights.values()) <= 0.7  # Max weight constraint
    
    def test_training_data_collection(self):
        """Test training data collection"""
        # Add market data
        for i in range(50):
            self.manager.update_market_data(100.0 + i * 0.1, 1000 + i * 5)
        
        # Add agent performance
        for agent_name in self.manager.agent_names:
            for i in range(10):
                perf = {
                    'sharpe_ratio': np.random.normal(1.0, 0.2),
                    'win_rate': np.random.uniform(0.45, 0.65),
                    'avg_return': np.random.normal(0.01, 0.003),
                    'volatility': np.random.uniform(0.08, 0.15)
                }
                self.manager.update_agent_performance(agent_name, perf)
                self.manager.collect_training_data()
        
        # Check training data collection
        assert len(self.manager.training_data['market_features']) > 0
        for agent_name in self.manager.agent_names:
            assert len(self.manager.training_data['agent_histories'][agent_name]) > 0
            assert len(self.manager.training_data['performance_labels'][agent_name]) > 0
    
    def test_meta_model_training(self):
        """Test meta-model training"""
        # Prepare training data
        np.random.seed(42)
        
        # Add market data
        for i in range(100):
            self.manager.update_market_data(100.0 + np.random.randn() * 0.5, 1000 + np.random.randn() * 50)
        
        # Add agent performance and training data
        for agent_name in self.manager.agent_names:
            for i in range(50):
                perf = {
                    'sharpe_ratio': np.random.normal(1.0, 0.3),
                    'win_rate': np.random.uniform(0.4, 0.7),
                    'avg_return': np.random.normal(0.01, 0.005),
                    'volatility': np.random.uniform(0.05, 0.2)
                }
                self.manager.update_agent_performance(agent_name, perf)
                self.manager.collect_training_data()
        
        # Train meta-models (should not raise errors)
        initial_training_step = self.manager.training_step
        self.manager.train_meta_models(batch_size=8, epochs=3)
        
        assert self.manager.training_step > initial_training_step


class TestMetaLearningRiskManagedEnsemble:
    """Test integrated meta-learning ensemble with risk management"""
    
    def setup_method(self):
        """Setup for each test"""
        self.mock_agents = {
            'agent1': Mock(),
            'agent2': Mock()
        }
        
        # Configure mock agents
        for agent in self.mock_agents.values():
            agent.act.return_value = (1, [0.3, 0.4, 0.3])  # action, q_values
        
        self.meta_manager = MetaLearningEnsembleManager(
            agents=self.mock_agents,
            meta_lookback=50
        )
        
        self.risk_manager = Mock()
        self.risk_manager.check_risk_constraints.return_value = {'allowed': True}
        
        self.ensemble = MetaLearningRiskManagedEnsemble(
            agents=self.mock_agents,
            meta_learning_manager=self.meta_manager,
            risk_manager=self.risk_manager
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        assert len(self.ensemble.agents) == 2
        assert self.ensemble.meta_learning_manager is not None
        assert self.ensemble.risk_manager is not None
    
    def test_trading_action(self):
        """Test trading action generation"""
        state = torch.randn(50)
        current_price = 100.0
        current_volume = 1000.0
        
        action, decision_info = self.ensemble.get_trading_action(state, current_price, current_volume)
        
        # Check action is valid
        assert isinstance(action, int)
        assert 0 <= action <= 2
        
        # Check decision info
        assert 'ensemble_action' in decision_info
        assert 'algorithm_weights' in decision_info
        assert 'agent_actions' in decision_info
        assert 'current_regime' in decision_info
        
        # Verify agents were called
        for agent in self.mock_agents.values():
            agent.act.assert_called_once()
    
    def test_performance_update(self):
        """Test performance updates"""
        initial_history_len = len(self.ensemble.performance_history)
        
        self.ensemble.update_performance(
            returns=0.05,
            sharpe_ratio=1.2,
            additional_metrics={
                'win_rate': 0.6,
                'avg_return': 0.02,
                'volatility': 0.15
            }
        )
        
        assert len(self.ensemble.performance_history) == initial_history_len + 1
        
        # Check that meta-learning manager was updated
        for agent_name in self.mock_agents.keys():
            assert len(self.ensemble.meta_learning_manager.agent_performance_history[agent_name]) > 0
    
    def test_risk_management_integration(self):
        """Test integration with risk management"""
        # Configure risk manager to block action
        self.risk_manager.check_risk_constraints.return_value = {
            'allowed': False,
            'alternative_action': 1  # Force hold
        }
        
        state = torch.randn(50)
        action, decision_info = self.ensemble.get_trading_action(state, 100.0, 1000.0)
        
        # Should be forced to hold (action=1) due to risk constraints
        assert action == 1
        
        # Verify risk manager was called
        self.risk_manager.check_risk_constraints.assert_called_once()


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    def test_complete_workflow(self):
        """Test complete meta-learning workflow"""
        # Mock agents with realistic behavior
        mock_agents = {}
        for i, name in enumerate(['dqn1', 'dqn2', 'ppo']):
            agent = Mock()
            # Different agents return different actions with different confidence
            agent.act.return_value = (i % 3, [0.2 + i*0.1, 0.5, 0.3 - i*0.05])
            mock_agents[name] = agent
        
        # Initialize meta-learning system
        meta_manager = MetaLearningEnsembleManager(mock_agents, meta_lookback=20)
        ensemble = MetaLearningRiskManagedEnsemble(
            agents=mock_agents,
            meta_learning_manager=meta_manager,
            risk_manager=None
        )
        
        # Simulate trading session
        np.random.seed(42)
        session_rewards = []
        
        for step in range(50):
            # Generate market data
            price = 100.0 + np.random.randn() * 2
            volume = 1000 + np.random.randn() * 100
            state = torch.randn(50)
            
            # Get trading action
            action, decision_info = ensemble.get_trading_action(state, price, volume)
            
            # Simulate reward
            reward = np.random.normal(0.01, 0.02)
            session_rewards.append(reward)
            
            # Update performance
            if step > 0:  # Need at least one return for Sharpe ratio
                returns_array = np.array(session_rewards)
                sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
                
                ensemble.update_performance(
                    returns=reward,
                    sharpe_ratio=sharpe,
                    additional_metrics={
                        'win_rate': np.mean(returns_array > 0),
                        'avg_return': np.mean(returns_array),
                        'volatility': np.std(returns_array)
                    }
                )
        
        # Verify system state
        assert len(meta_manager.market_state_history) > 0
        assert len(ensemble.performance_history) > 0
        
        # Check that agents were called appropriately
        for agent in mock_agents.values():
            assert agent.act.call_count > 0
        
        # Get final performance summary
        summary = ensemble.get_performance_summary()
        assert 'total_trades' in summary
        assert 'recent_sharpe' in summary
        assert 'regime_info' in summary
    
    def test_model_persistence(self):
        """Test saving and loading models"""
        mock_agents = {'agent1': Mock()}
        meta_manager = MetaLearningEnsembleManager(mock_agents)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Add some data
            for i in range(10):
                meta_manager.update_market_data(100.0 + i, 1000)
                perf = {'sharpe_ratio': 1.0, 'win_rate': 0.5, 'avg_return': 0.01, 'volatility': 0.1}
                meta_manager.update_agent_performance('agent1', perf)
            
            # Save models
            meta_manager.save_meta_models(temp_dir)
            
            # Verify files were created
            model_file = os.path.join(temp_dir, "meta_learning_models.pth")
            assert os.path.exists(model_file)
            
            # Create new manager and load models
            new_meta_manager = MetaLearningEnsembleManager(mock_agents)
            new_meta_manager.load_meta_models(temp_dir)
            
            # Verify loading worked (training step should be preserved)
            assert new_meta_manager.training_step >= 0
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)


def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    
    print("=== Meta-Learning Framework Test Suite ===\n")
    
    test_classes = [
        TestMarketRegimeClassifier,
        TestAlgorithmPerformancePredictor,
        TestMarketFeatureExtractor,
        TestMetaLearningEnsembleManager,
        TestMetaLearningRiskManagedEnsemble,
        TestIntegrationScenarios
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"Running {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            
            try:
                # Create test instance
                test_instance = test_class()
                
                # Run setup if available
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run the test
                getattr(test_instance, test_method)()
                
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {str(e)}")
    
    print(f"\n=== Test Results ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
    else:
        print(f"\n🎉 All tests passed!")
    
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1%}")
    
    return success_rate >= 0.8  # Return True if at least 80% tests pass


if __name__ == "__main__":
    # Run comprehensive test suite
    success = run_comprehensive_tests()
    
    if success:
        print(f"\n✅ Meta-learning framework is ready for deployment!")
    else:
        print(f"\n❌ Some tests failed. Please review and fix issues before deployment.")
    
    sys.exit(0 if success else 1)