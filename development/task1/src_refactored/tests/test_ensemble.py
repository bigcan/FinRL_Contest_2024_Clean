"""
Unit tests for ensemble implementations in the FinRL Contest 2024 refactored framework.

This module tests voting ensembles, stacking ensembles, and ensemble evaluation utilities.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from . import TEST_CONFIG
from .utils.test_helpers import (
    set_random_seeds, create_test_state, create_test_batch, create_mock_agent,
    assert_tensor_shape, validate_ensemble_interface, create_temporary_directory,
    cleanup_temporary_directory, PerformanceTimer
)
from .utils.mock_environment import MockEnvironment

# Import ensemble components to test
from ..ensemble import (
    BaseEnsemble, EnsembleStrategy, EnsembleMetrics,
    VotingEnsemble, StackingEnsemble, MetaLearnerNetwork,
    create_ensemble, create_voting_ensemble, create_stacking_ensemble,
    evaluate_ensemble_diversity, compare_ensemble_strategies
)
from ..agents import create_agent, create_ensemble_agents


class TestEnsembleMetrics(unittest.TestCase):
    """Test ensemble metrics tracking."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.metrics = EnsembleMetrics()
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        self.assertIsInstance(self.metrics.individual_rewards, dict)
        self.assertIsInstance(self.metrics.ensemble_rewards, list)
        self.assertIsInstance(self.metrics.agreement_scores, list)
        self.assertEqual(len(self.metrics.ensemble_rewards), 0)
    
    def test_add_step_metrics(self):
        """Test adding step metrics."""
        individual_rewards = {"agent1": 0.5, "agent2": 0.3}
        ensemble_reward = 0.4
        individual_losses = {"agent1": 0.1, "agent2": 0.2}
        agreement_score = 0.8
        diversity_score = 0.6
        weights = {"agent1": 0.6, "agent2": 0.4}
        confidence = 0.7
        
        self.metrics.add_step_metrics(
            individual_rewards=individual_rewards,
            ensemble_reward=ensemble_reward,
            individual_losses=individual_losses,
            agreement_score=agreement_score,
            diversity_score=diversity_score,
            weights=weights,
            confidence=confidence
        )
        
        self.assertEqual(len(self.metrics.ensemble_rewards), 1)
        self.assertEqual(self.metrics.ensemble_rewards[0], ensemble_reward)
        self.assertEqual(self.metrics.agreement_scores[0], agreement_score)
        self.assertEqual(self.metrics.diversity_scores[0], diversity_score)
        self.assertEqual(len(self.metrics.individual_rewards["agent1"]), 1)


class TestMetaLearnerNetwork(unittest.TestCase):
    """Test meta-learner network for stacking ensemble."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.input_dim = 20  # Features from multiple agents
        self.output_dim = TEST_CONFIG['action_dim']
        self.network = MetaLearnerNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[16, 8]
        )
    
    def test_network_creation(self):
        """Test meta-learner network creation."""
        self.assertIsInstance(self.network, MetaLearnerNetwork)
        self.assertEqual(self.network.input_dim, self.input_dim)
        self.assertEqual(self.network.output_dim, self.output_dim)
    
    def test_forward_pass(self):
        """Test forward pass through meta-learner."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, self.input_dim)
        
        output = self.network(input_tensor)
        
        assert_tensor_shape(output, (batch_size, self.output_dim))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_parameter_count(self):
        """Test that network has reasonable number of parameters."""
        param_count = sum(p.numel() for p in self.network.parameters())
        self.assertGreater(param_count, 0)
        # Should be reasonable size (not too large for test)
        self.assertLess(param_count, 10000)


class TestVotingEnsemble(unittest.TestCase):
    """Test voting ensemble implementations."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        
        # Create mock agents
        self.agents = {
            "agent1": create_mock_agent("Agent1"),
            "agent2": create_mock_agent("Agent2"),
            "agent3": create_mock_agent("Agent3")
        }
        
        # Configure mock agents to return different actions
        self.agents["agent1"].select_action.return_value = 0
        self.agents["agent2"].select_action.return_value = 1
        self.agents["agent3"].select_action.return_value = 0
        
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_voting_ensemble_creation(self):
        """Test voting ensemble creation."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        self.assertIsInstance(ensemble, VotingEnsemble)
        self.assertEqual(len(ensemble.agents), 3)
        self.assertEqual(ensemble.strategy, EnsembleStrategy.MAJORITY_VOTE)
    
    def test_ensemble_interface(self):
        """Test that ensemble implements required interface."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        validation_results = validate_ensemble_interface(ensemble)
        for method, is_valid in validation_results.items():
            self.assertTrue(is_valid, f"Method {method} not properly implemented")
    
    def test_majority_vote_strategy(self):
        """Test majority vote strategy."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        state = create_test_state(TEST_CONFIG['state_dim'])
        action = ensemble.select_action(state, deterministic=True)
        
        # Should return action 0 (majority: agent1=0, agent2=1, agent3=0)
        self.assertEqual(action, 0)
    
    def test_weighted_vote_strategy(self):
        """Test weighted vote strategy."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.WEIGHTED_VOTE,
            device=self.device
        )
        
        # Set specific weights
        ensemble.weights = {"agent1": 0.5, "agent2": 0.3, "agent3": 0.2}
        
        state = create_test_state(TEST_CONFIG['state_dim'])
        action = ensemble.select_action(state, deterministic=True)
        
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, TEST_CONFIG['action_dim'])
    
    def test_confidence_threshold(self):
        """Test confidence threshold functionality."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.UNCERTAINTY_WEIGHTED,
            device=self.device,
            confidence_threshold=0.8
        )
        
        self.assertEqual(ensemble.confidence_threshold, 0.8)
        
        # Test setting new threshold
        ensemble.set_confidence_threshold(0.6)
        self.assertEqual(ensemble.confidence_threshold, 0.6)
    
    def test_ensemble_update(self):
        """Test ensemble update functionality."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.WEIGHTED_VOTE,
            device=self.device
        )
        
        batch_data = create_test_batch(
            TEST_CONFIG['state_dim'],
            TEST_CONFIG['action_dim'],
            TEST_CONFIG['test_batch_size']
        )
        
        result = ensemble.update(batch_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('ensemble_reward', result)
        self.assertIn('individual_rewards', result)
        self.assertIn('agreement_score', result)
        self.assertIn('diversity_score', result)
    
    def test_voting_statistics(self):
        """Test voting statistics retrieval."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.WEIGHTED_VOTE,
            device=self.device
        )
        
        stats = ensemble.get_voting_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('voting_strategy', stats)
        self.assertIn('confidence_threshold', stats)
        self.assertIn('agent_confidences', stats)
        self.assertIn('weights', stats)
    
    def test_active_agents(self):
        """Test active agent identification."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.UNCERTAINTY_WEIGHTED,
            device=self.device,
            confidence_threshold=0.5
        )
        
        # Initially all agents should be active (default confidence = 1.0)
        active_agents = ensemble.get_active_agents()
        self.assertEqual(len(active_agents), 3)
        
        # Lower confidence for one agent
        ensemble.agent_confidences["agent1"] = 0.3
        active_agents = ensemble.get_active_agents()
        self.assertEqual(len(active_agents), 2)
        self.assertNotIn("agent1", active_agents)


class TestStackingEnsemble(unittest.TestCase):
    """Test stacking ensemble implementation."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        
        # Create mock agents with Q-value functionality
        self.agents = {}
        for i in range(3):
            agent = create_mock_agent(f"Agent{i+1}")
            # Mock get_q_values method
            agent.online_network.get_q1_q2.return_value = (
                torch.randn(1, TEST_CONFIG['action_dim']),
                torch.randn(1, TEST_CONFIG['action_dim'])
            )
            self.agents[f"agent{i+1}"] = agent
        
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_stacking_ensemble_creation(self):
        """Test stacking ensemble creation."""
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        self.assertIsInstance(ensemble, StackingEnsemble)
        self.assertEqual(len(ensemble.agents), 3)
        self.assertIsInstance(ensemble.meta_learner, MetaLearnerNetwork)
        self.assertEqual(ensemble.action_dim, TEST_CONFIG['action_dim'])
    
    def test_meta_learner_input_dimension(self):
        """Test meta-learner input dimension calculation."""
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        # Should be num_agents * (action_dim + 1) for Q-values + confidence
        expected_input_dim = len(self.agents) * (TEST_CONFIG['action_dim'] + 1)
        self.assertEqual(ensemble.meta_input_dim, expected_input_dim)
    
    def test_feature_extraction(self):
        """Test agent feature extraction."""
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        state = create_test_state(TEST_CONFIG['state_dim'])
        features = ensemble._extract_agent_features(state)
        
        assert_tensor_shape(features, (ensemble.meta_input_dim,))
        self.assertFalse(torch.isnan(features).any())
    
    def test_stacking_action_selection(self):
        """Test action selection through meta-learner."""
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        state = create_test_state(TEST_CONFIG['state_dim'])
        action = ensemble.select_action(state, deterministic=True)
        
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, TEST_CONFIG['action_dim'])
    
    def test_meta_learning_update(self):
        """Test meta-learner update functionality."""
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device,
            buffer_size=50  # Small buffer for testing
        )
        
        # Add some data to meta buffer
        for _ in range(10):
            batch_data = create_test_batch(
                TEST_CONFIG['state_dim'],
                TEST_CONFIG['action_dim'],
                4  # Small batch
            )
            ensemble.update(batch_data)
        
        # Should have some data in buffer
        self.assertGreater(len(ensemble.meta_buffer), 0)
    
    def test_meta_statistics(self):
        """Test meta-learner statistics."""
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        stats = ensemble.get_meta_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('meta_learner_params', stats)
        self.assertIn('meta_buffer_size', stats)
        self.assertIn('meta_update_count', stats)
        self.assertIn('meta_input_dim', stats)
        self.assertIn('meta_output_dim', stats)


class TestEnsembleFactory(unittest.TestCase):
    """Test ensemble factory functions."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.agents = {
            "agent1": create_mock_agent("Agent1"),
            "agent2": create_mock_agent("Agent2")
        }
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_create_ensemble_voting(self):
        """Test creating voting ensemble through factory."""
        ensemble = create_ensemble(
            ensemble_type="voting",
            agents=self.agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        self.assertIsInstance(ensemble, VotingEnsemble)
        self.assertEqual(ensemble.strategy, EnsembleStrategy.MAJORITY_VOTE)
    
    def test_create_ensemble_stacking(self):
        """Test creating stacking ensemble through factory."""
        ensemble = create_ensemble(
            ensemble_type="stacking",
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        self.assertIsInstance(ensemble, StackingEnsemble)
        self.assertEqual(ensemble.action_dim, TEST_CONFIG['action_dim'])
    
    def test_create_voting_ensemble_direct(self):
        """Test creating voting ensemble directly."""
        ensemble = create_voting_ensemble(
            agents=self.agents,
            strategy=EnsembleStrategy.WEIGHTED_VOTE,
            confidence_threshold=0.6,
            temperature=2.0,
            device=self.device
        )
        
        self.assertIsInstance(ensemble, VotingEnsemble)
        self.assertEqual(ensemble.strategy, EnsembleStrategy.WEIGHTED_VOTE)
        self.assertEqual(ensemble.confidence_threshold, 0.6)
        self.assertEqual(ensemble.temperature, 2.0)
    
    def test_create_stacking_ensemble_direct(self):
        """Test creating stacking ensemble directly."""
        ensemble = create_stacking_ensemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            meta_learning_rate=1e-3,
            meta_hidden_dims=[32, 16],
            device=self.device
        )
        
        self.assertIsInstance(ensemble, StackingEnsemble)
        self.assertEqual(ensemble.meta_learning_rate, 1e-3)
    
    def test_invalid_ensemble_type(self):
        """Test error handling for invalid ensemble type."""
        with self.assertRaises(ValueError):
            create_ensemble(
                ensemble_type="invalid_type",
                agents=self.agents,
                device=self.device
            )


class TestEnsembleEvaluation(unittest.TestCase):
    """Test ensemble evaluation utilities."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        
        # Create real agents for evaluation testing
        self.agent_configs = {
            "double_dqn": {"agent_type": "AgentDoubleDQN"},
            "d3qn": {"agent_type": "AgentD3QN"}
        }
        
        try:
            self.agents = create_ensemble_agents(
                self.agent_configs,
                state_dim=TEST_CONFIG['state_dim'],
                action_dim=TEST_CONFIG['action_dim'],
                device=torch.device(TEST_CONFIG['device'])
            )
        except Exception as e:
            # Fallback to mock agents if real agents fail
            self.agents = {
                "agent1": create_mock_agent("Agent1"),
                "agent2": create_mock_agent("Agent2")
            }
        
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_ensemble_diversity_evaluation(self):
        """Test ensemble diversity evaluation."""
        ensemble = create_voting_ensemble(
            agents=self.agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        # Create test states
        test_states = [create_test_state(TEST_CONFIG['state_dim']) for _ in range(10)]
        
        diversity_metrics = evaluate_ensemble_diversity(
            ensemble=ensemble,
            test_states=test_states,
            num_samples=5
        )
        
        self.assertIsInstance(diversity_metrics, dict)
        self.assertIn('mean_agreement', diversity_metrics)
        self.assertIn('mean_diversity', diversity_metrics)
        self.assertIn('num_samples', diversity_metrics)
        self.assertEqual(diversity_metrics['num_samples'], 5)
    
    def test_strategy_comparison(self):
        """Test comparing different ensemble strategies."""
        test_states = [create_test_state(TEST_CONFIG['state_dim']) for _ in range(5)]
        
        strategies = [
            EnsembleStrategy.MAJORITY_VOTE,
            EnsembleStrategy.WEIGHTED_VOTE
        ]
        
        results = compare_ensemble_strategies(
            agents=self.agents,
            test_states=test_states,
            strategies=strategies,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(strategies))
        
        for strategy in strategies:
            self.assertIn(strategy.value, results)
            self.assertIsInstance(results[strategy.value], dict)


class TestEnsembleIntegration(unittest.TestCase):
    """Integration tests for ensembles with mock environment."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=10,
            seed=TEST_CONFIG['seed']
        )
        
        # Create mock agents
        self.agents = {
            "agent1": create_mock_agent("Agent1"),
            "agent2": create_mock_agent("Agent2")
        }
        
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_voting_ensemble_environment_interaction(self):
        """Test voting ensemble interaction with environment."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        # Run short episode
        state = self.env.reset()
        total_reward = 0
        
        for step in range(5):
            action = ensemble.select_action(state, deterministic=False)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Verify action is valid
            self.assertIsInstance(action, (int, np.integer))
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, TEST_CONFIG['action_dim'])
            
            # Update ensemble
            transition = self.env.get_last_transition()
            if transition is not None:
                batch_data = (
                    torch.tensor([transition[0]], dtype=torch.float32),
                    torch.tensor([transition[1]], dtype=torch.long),
                    torch.tensor([transition[2]], dtype=torch.float32),
                    torch.tensor([not transition[3]], dtype=torch.float32),
                    torch.tensor([transition[4]], dtype=torch.float32)
                )
                result = ensemble.update(batch_data)
                self.assertIsInstance(result, dict)
            
            state = next_state
            if done:
                break
        
        # Verify episode completed successfully
        self.assertIsInstance(total_reward, (int, float))
    
    def test_stacking_ensemble_environment_interaction(self):
        """Test stacking ensemble interaction with environment."""
        # Add Q-value functionality to mock agents
        for agent in self.agents.values():
            agent.online_network.get_q1_q2.return_value = (
                torch.randn(1, TEST_CONFIG['action_dim']),
                torch.randn(1, TEST_CONFIG['action_dim'])
            )
        
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        # Run short episode
        state = self.env.reset()
        total_reward = 0
        
        for step in range(5):
            action = ensemble.select_action(state, deterministic=False)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Verify action is valid
            self.assertIsInstance(action, (int, np.integer))
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, TEST_CONFIG['action_dim'])
            
            state = next_state
            if done:
                break
        
        # Verify episode completed successfully
        self.assertIsInstance(total_reward, (int, float))


class TestEnsembleCheckpoints(unittest.TestCase):
    """Test ensemble save/load functionality."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.agents = {
            "agent1": create_mock_agent("Agent1"),
            "agent2": create_mock_agent("Agent2")
        }
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_voting_ensemble_save_load(self):
        """Test voting ensemble save and load."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.WEIGHTED_VOTE,
            device=self.device
        )
        
        # Modify some state
        ensemble.weights = {"agent1": 0.7, "agent2": 0.3}
        ensemble.training_step = 100
        
        temp_dir = create_temporary_directory()
        try:
            checkpoint_path = temp_dir / "ensemble_checkpoint.pth"
            
            # Save ensemble
            ensemble.save_ensemble(str(checkpoint_path))
            self.assertTrue(checkpoint_path.exists())
            
            # Create new ensemble and load
            new_ensemble = VotingEnsemble(
                agents=self.agents,
                strategy=EnsembleStrategy.MAJORITY_VOTE,  # Different strategy
                device=self.device
            )
            
            new_ensemble.load_ensemble(str(checkpoint_path))
            
            # Verify state was loaded
            self.assertEqual(new_ensemble.strategy, EnsembleStrategy.WEIGHTED_VOTE)
            self.assertEqual(new_ensemble.training_step, 100)
            self.assertEqual(new_ensemble.weights["agent1"], 0.7)
            
        finally:
            cleanup_temporary_directory(temp_dir)
    
    def test_stacking_ensemble_save_load(self):
        """Test stacking ensemble save and load."""
        # Add Q-value functionality to mock agents
        for agent in self.agents.values():
            agent.online_network.get_q1_q2.return_value = (
                torch.randn(1, TEST_CONFIG['action_dim']),
                torch.randn(1, TEST_CONFIG['action_dim'])
            )
        
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        # Add some training data
        for _ in range(5):
            batch_data = create_test_batch(
                TEST_CONFIG['state_dim'],
                TEST_CONFIG['action_dim'],
                4
            )
            ensemble.update(batch_data)
        
        temp_dir = create_temporary_directory()
        try:
            checkpoint_path = temp_dir / "stacking_checkpoint.pth"
            
            # Save ensemble
            ensemble.save_ensemble(str(checkpoint_path))
            self.assertTrue(checkpoint_path.exists())
            
            # Meta-learner checkpoint should also exist
            meta_checkpoint_path = temp_dir / "stacking_checkpoint_meta.pth"
            self.assertTrue(meta_checkpoint_path.exists())
            
            # Create new ensemble and load
            new_ensemble = StackingEnsemble(
                agents=self.agents,
                action_dim=TEST_CONFIG['action_dim'],
                device=self.device
            )
            
            new_ensemble.load_ensemble(str(checkpoint_path))
            
            # Verify some state was loaded
            self.assertIsNotNone(new_ensemble.meta_learner)
            
        finally:
            cleanup_temporary_directory(temp_dir)


class TestEnsemblePerformance(unittest.TestCase):
    """Performance tests for ensembles."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.agents = {
            "agent1": create_mock_agent("Agent1"),
            "agent2": create_mock_agent("Agent2"),
            "agent3": create_mock_agent("Agent3")
        }
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_voting_ensemble_action_selection_speed(self):
        """Test voting ensemble action selection performance."""
        ensemble = VotingEnsemble(
            agents=self.agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        state = create_test_state(TEST_CONFIG['state_dim'])
        
        # Time action selection
        with PerformanceTimer() as timer:
            for _ in range(50):
                action = ensemble.select_action(state, deterministic=True)
        
        # Should be reasonably fast
        self.assertLess(timer.elapsed, 2.0)
    
    def test_stacking_ensemble_action_selection_speed(self):
        """Test stacking ensemble action selection performance."""
        # Add Q-value functionality to mock agents
        for agent in self.agents.values():
            agent.online_network.get_q1_q2.return_value = (
                torch.randn(1, TEST_CONFIG['action_dim']),
                torch.randn(1, TEST_CONFIG['action_dim'])
            )
        
        ensemble = StackingEnsemble(
            agents=self.agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        state = create_test_state(TEST_CONFIG['state_dim'])
        
        # Time action selection
        with PerformanceTimer() as timer:
            for _ in range(50):
                action = ensemble.select_action(state, deterministic=True)
        
        # Should be reasonably fast (may be slower due to meta-learner)
        self.assertLess(timer.elapsed, 3.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)