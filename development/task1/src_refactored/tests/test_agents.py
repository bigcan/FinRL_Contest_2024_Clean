"""
Unit tests for agent implementations in the FinRL Contest 2024 refactored framework.

This module tests all agent types including Double DQN, Prioritized DQN,
Noisy DQN, Rainbow DQN, and Adaptive DQN implementations.
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
    set_random_seeds, create_test_state, create_test_batch, create_test_config,
    assert_tensor_shape, assert_agent_interface, run_agent_smoke_test,
    create_temporary_directory, cleanup_temporary_directory, PerformanceTimer
)
from .utils.mock_environment import MockEnvironment

# Import agents to test
from ..agents import (
    create_agent, create_ensemble_agents, create_default_ensemble,
    AGENT_REGISTRY, validate_agent_type, get_agent_categories
)
from ..agents.base_dqn_agent import BaseDQNAgent
from ..agents.double_dqn_agent import DoubleDQNAgent, D3QNAgent
from ..agents.prioritized_dqn_agent import PrioritizedDQNAgent
from ..agents.noisy_dqn_agent import NoisyDQNAgent, NoisyDuelDQNAgent
from ..agents.rainbow_dqn_agent import RainbowDQNAgent
from ..agents.adaptive_dqn_agent import AdaptiveDQNAgent

from ..config import DoubleDQNConfig, PrioritizedDQNConfig, NoisyDQNConfig, RainbowDQNConfig, AdaptiveDQNConfig


class TestAgentRegistry(unittest.TestCase):
    """Test the agent registry and factory functions."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
    
    def test_agent_registry_completeness(self):
        """Test that all expected agents are in the registry."""
        expected_agents = [
            'AgentDoubleDQN', 'AgentD3QN', 'AgentPrioritizedDQN',
            'AgentNoisyDQN', 'AgentNoisyDuelDQN', 'AgentRainbowDQN', 'AgentAdaptiveDQN'
        ]
        
        for agent_type in expected_agents:
            self.assertIn(agent_type, AGENT_REGISTRY)
            self.assertTrue(validate_agent_type(agent_type))
    
    def test_agent_factory_creation(self):
        """Test creating agents through the factory function."""
        for agent_type in AGENT_REGISTRY.keys():
            with self.subTest(agent_type=agent_type):
                try:
                    agent = create_agent(
                        agent_type=agent_type,
                        state_dim=TEST_CONFIG['state_dim'],
                        action_dim=TEST_CONFIG['action_dim'],
                        device=torch.device(TEST_CONFIG['device'])
                    )
                    self.assertIsNotNone(agent)
                    self.assertEqual(agent.state_dim, TEST_CONFIG['state_dim'])
                    self.assertEqual(agent.action_dim, TEST_CONFIG['action_dim'])
                except Exception as e:
                    self.fail(f"Failed to create agent {agent_type}: {e}")
    
    def test_agent_categories(self):
        """Test agent categorization."""
        categories = get_agent_categories()
        self.assertIsInstance(categories, dict)
        
        # Check that categories contain expected agent types
        all_agents = set()
        for agent_list in categories.values():
            all_agents.update(agent_list)
        
        self.assertTrue(len(all_agents) > 0)
        
        # Verify all agents in categories are valid
        for agent_type in all_agents:
            self.assertTrue(validate_agent_type(agent_type))
    
    def test_ensemble_creation(self):
        """Test creating ensemble of agents."""
        agent_configs = {
            "double_dqn": {"agent_type": "AgentDoubleDQN"},
            "d3qn": {"agent_type": "AgentD3QN"},
            "prioritized": {"agent_type": "AgentPrioritizedDQN"}
        }
        
        agents = create_ensemble_agents(
            agent_configs,
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=torch.device(TEST_CONFIG['device'])
        )
        
        self.assertEqual(len(agents), 3)
        self.assertIn("double_dqn", agents)
        self.assertIn("d3qn", agents)
        self.assertIn("prioritized", agents)
    
    def test_default_ensemble_creation(self):
        """Test creating default ensemble."""
        agents = create_default_ensemble(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=torch.device(TEST_CONFIG['device']),
            num_agents=3
        )
        
        self.assertEqual(len(agents), 3)
        # Should contain the first 3 agents from default config
        expected_names = ["double_dqn", "d3qn", "prioritized"]
        for name in expected_names:
            self.assertIn(name, agents)


class TestBaseAgentInterface(unittest.TestCase):
    """Test the base agent interface and common functionality."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.agent = DoubleDQNAgent(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=torch.device(TEST_CONFIG['device'])
        )
    
    def test_agent_interface(self):
        """Test that agent implements required interface."""
        required_methods = [
            'select_action', 'update', 'get_training_info', 
            'get_algorithm_info', 'save_checkpoint', 'load_checkpoint'
        ]
        assert_agent_interface(self.agent, required_methods)
    
    def test_action_selection(self):
        """Test action selection functionality."""
        state = create_test_state(TEST_CONFIG['state_dim'])
        
        # Test deterministic action selection
        action_det = self.agent.select_action(state, deterministic=True)
        self.assertIsInstance(action_det, (int, np.integer))
        self.assertGreaterEqual(action_det, 0)
        self.assertLess(action_det, TEST_CONFIG['action_dim'])
        
        # Test stochastic action selection
        action_stoch = self.agent.select_action(state, deterministic=False)
        self.assertIsInstance(action_stoch, (int, np.integer))
        self.assertGreaterEqual(action_stoch, 0)
        self.assertLess(action_stoch, TEST_CONFIG['action_dim'])
    
    def test_batch_action_selection(self):
        """Test action selection with batch inputs."""
        batch_size = 8
        states = create_test_state(TEST_CONFIG['state_dim'], batch_size)
        
        actions = self.agent.select_action(states, deterministic=True)
        if isinstance(actions, (int, np.integer)):
            # Single action returned
            self.assertGreaterEqual(actions, 0)
            self.assertLess(actions, TEST_CONFIG['action_dim'])
        else:
            # Array of actions returned
            self.assertEqual(len(actions), batch_size)
            for action in actions:
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, TEST_CONFIG['action_dim'])
    
    def test_update_functionality(self):
        """Test agent update functionality."""
        batch_data = create_test_batch(
            TEST_CONFIG['state_dim'], 
            TEST_CONFIG['action_dim'],
            TEST_CONFIG['test_batch_size']
        )
        
        # Test that update runs without error
        result = self.agent.update(batch_data)
        self.assertIsNotNone(result)
        
        # Check that training step increments
        initial_step = self.agent.training_step
        self.agent.update(batch_data)
        self.assertGreater(self.agent.training_step, initial_step)
    
    def test_training_info(self):
        """Test training information retrieval."""
        info = self.agent.get_training_info()
        self.assertIsInstance(info, dict)
        
        # Check for expected keys
        expected_keys = ['training_step', 'exploration_rate']
        for key in expected_keys:
            self.assertIn(key, info)
    
    def test_algorithm_info(self):
        """Test algorithm information retrieval."""
        info = self.agent.get_algorithm_info()
        self.assertIsInstance(info, dict)
        
        # Check for required keys
        required_keys = ['algorithm', 'description']
        for key in required_keys:
            self.assertIn(key, info)
            self.assertIsInstance(info[key], str)
            self.assertGreater(len(info[key]), 0)


class TestDoubleDQNAgent(unittest.TestCase):
    """Test Double DQN agent implementation."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.config = DoubleDQNConfig(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            buffer_size=TEST_CONFIG['small_buffer_size']
        )
        self.agent = DoubleDQNAgent(config=self.config, device=torch.device(TEST_CONFIG['device']))
    
    def test_double_dqn_creation(self):
        """Test Double DQN agent creation."""
        self.assertIsInstance(self.agent, DoubleDQNAgent)
        self.assertEqual(self.agent.state_dim, TEST_CONFIG['state_dim'])
        self.assertEqual(self.agent.action_dim, TEST_CONFIG['action_dim'])
        self.assertIsNotNone(self.agent.online_network)
        self.assertIsNotNone(self.agent.target_network)
    
    def test_double_dqn_smoke_test(self):
        """Run smoke test on Double DQN agent."""
        results = run_agent_smoke_test(
            self.agent,
            TEST_CONFIG['state_dim'],
            TEST_CONFIG['action_dim']
        )
        
        self.assertTrue(results['action_selection'], f"Errors: {results['errors']}")
        self.assertTrue(results['algorithm_info'], f"Errors: {results['errors']}")
    
    def test_network_architecture(self):
        """Test network architecture."""
        # Test that networks have correct input/output dimensions
        test_input = torch.randn(1, TEST_CONFIG['state_dim'])
        
        with torch.no_grad():
            q1, q2 = self.agent.online_network.get_q1_q2(test_input)
            
        assert_tensor_shape(q1, (1, TEST_CONFIG['action_dim']))
        assert_tensor_shape(q2, (1, TEST_CONFIG['action_dim']))
    
    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        temp_dir = create_temporary_directory()
        try:
            checkpoint_path = temp_dir / "test_checkpoint.pth"
            
            # Save checkpoint
            self.agent.save_checkpoint(str(checkpoint_path))
            self.assertTrue(checkpoint_path.exists())
            
            # Modify agent state
            original_step = self.agent.training_step
            self.agent.training_step = 999
            
            # Load checkpoint
            self.agent.load_checkpoint(str(checkpoint_path))
            self.assertEqual(self.agent.training_step, original_step)
            
        finally:
            cleanup_temporary_directory(temp_dir)


class TestD3QNAgent(unittest.TestCase):
    """Test Dueling Double DQN agent implementation."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.agent = D3QNAgent(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=torch.device(TEST_CONFIG['device'])
        )
    
    def test_d3qn_creation(self):
        """Test D3QN agent creation."""
        self.assertIsInstance(self.agent, D3QNAgent)
        self.assertEqual(self.agent.state_dim, TEST_CONFIG['state_dim'])
        self.assertEqual(self.agent.action_dim, TEST_CONFIG['action_dim'])
    
    def test_value_advantage_estimates(self):
        """Test value and advantage estimation."""
        if hasattr(self.agent, 'get_value_advantage_estimates'):
            state = create_test_state(TEST_CONFIG['state_dim'])
            result = self.agent.get_value_advantage_estimates(state)
            self.assertIsNotNone(result)
            # Should return tuple of (value1, advantage1, value2, advantage2)
            self.assertEqual(len(result), 4)


class TestPrioritizedDQNAgent(unittest.TestCase):
    """Test Prioritized DQN agent implementation."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.config = PrioritizedDQNConfig(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            buffer_size=TEST_CONFIG['small_buffer_size'],
            per_alpha=0.6,
            per_beta=0.4
        )
        self.agent = PrioritizedDQNAgent(config=self.config, device=torch.device(TEST_CONFIG['device']))
    
    def test_prioritized_dqn_creation(self):
        """Test Prioritized DQN agent creation."""
        self.assertIsInstance(self.agent, PrioritizedDQNAgent)
        self.assertEqual(self.agent.per_alpha, 0.6)
        self.assertEqual(self.agent.per_beta, 0.4)
    
    def test_per_parameters(self):
        """Test PER parameter management."""
        # Test setting parameters
        self.agent.set_per_parameters(alpha=0.7, beta=0.5)
        self.assertEqual(self.agent.per_alpha, 0.7)
        self.assertEqual(self.agent.per_beta, 0.5)
    
    def test_priority_reset(self):
        """Test priority reset functionality."""
        # This should not raise an error
        self.agent.reset_priorities()


class TestNoisyDQNAgent(unittest.TestCase):
    """Test Noisy DQN agent implementation."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.config = NoisyDQNConfig(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            noise_std_init=0.5
        )
        self.agent = NoisyDQNAgent(config=self.config, device=torch.device(TEST_CONFIG['device']))
    
    def test_noisy_dqn_creation(self):
        """Test Noisy DQN agent creation."""
        self.assertIsInstance(self.agent, NoisyDQNAgent)
        self.assertEqual(self.agent.noise_std_init, 0.5)
        # Exploration rate should be 0 for noisy networks
        self.assertEqual(self.agent.explore_rate, 0.0)
    
    def test_noise_parameters(self):
        """Test noise parameter management."""
        self.agent.set_noise_parameters(std_init=0.3)
        self.assertEqual(self.agent.noise_std_init, 0.3)
    
    def test_noise_diversity(self):
        """Test that noisy networks produce diverse actions."""
        state = create_test_state(TEST_CONFIG['state_dim'])
        
        # Get multiple actions from the same state
        actions = []
        for _ in range(20):
            action = self.agent.select_action(state, deterministic=False)
            actions.append(action)
        
        # Should have some diversity (not all the same)
        unique_actions = set(actions)
        self.assertGreater(len(unique_actions), 1)


class TestRainbowDQNAgent(unittest.TestCase):
    """Test Rainbow DQN agent implementation."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.config = RainbowDQNConfig(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            buffer_size=TEST_CONFIG['small_buffer_size'],
            n_step=3,
            per_alpha=0.6,
            per_beta=0.4
        )
        self.agent = RainbowDQNAgent(config=self.config, device=torch.device(TEST_CONFIG['device']))
    
    def test_rainbow_creation(self):
        """Test Rainbow DQN agent creation."""
        self.assertIsInstance(self.agent, RainbowDQNAgent)
        self.assertEqual(self.agent.n_step, 3)
        self.assertEqual(self.agent.per_alpha, 0.6)
        self.assertEqual(self.agent.per_beta, 0.4)
    
    def test_rainbow_parameters(self):
        """Test Rainbow parameter management."""
        self.agent.set_rainbow_parameters(
            n_step=5,
            noise_std=0.3,
            per_alpha=0.7,
            per_beta=0.5
        )
        self.assertEqual(self.agent.n_step, 5)
        self.assertEqual(self.agent.noise_std_init, 0.3)
        self.assertEqual(self.agent.per_alpha, 0.7)
        self.assertEqual(self.agent.per_beta, 0.5)
    
    def test_value_advantage_estimates(self):
        """Test Rainbow value and advantage estimation."""
        if hasattr(self.agent, 'get_value_advantage_estimates'):
            state = create_test_state(TEST_CONFIG['state_dim'])
            result = self.agent.get_value_advantage_estimates(state)
            self.assertIsNotNone(result)


class TestAdaptiveDQNAgent(unittest.TestCase):
    """Test Adaptive DQN agent implementation."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.config = AdaptiveDQNConfig(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            lr_strategy="cosine_annealing",
            adaptive_grad_clip=True
        )
        self.agent = AdaptiveDQNAgent(config=self.config, device=torch.device(TEST_CONFIG['device']))
    
    def test_adaptive_creation(self):
        """Test Adaptive DQN agent creation."""
        self.assertIsInstance(self.agent, AdaptiveDQNAgent)
        self.assertEqual(self.agent.lr_strategy, "cosine_annealing")
        self.assertTrue(self.agent.adaptive_grad_clip)
    
    def test_adaptive_statistics(self):
        """Test adaptive optimization statistics."""
        stats = self.agent.get_adaptive_statistics()
        self.assertIsInstance(stats, dict)
        
        expected_keys = ['lr_strategy', 'adaptive_grad_clip']
        for key in expected_keys:
            self.assertIn(key, stats)
    
    def test_learning_rate_adjustment(self):
        """Test manual learning rate adjustment."""
        self.agent.adjust_learning_rate(1e-5)
        # Should not raise an error
    
    def test_performance_summary(self):
        """Test performance summary functionality."""
        # Add some performance history first
        for _ in range(10):
            batch_data = create_test_batch(
                TEST_CONFIG['state_dim'], 
                TEST_CONFIG['action_dim'],
                TEST_CONFIG['test_batch_size']
            )
            self.agent.update(batch_data)
        
        summary = self.agent.get_performance_summary()
        self.assertIsInstance(summary, dict)


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for agents with mock environment."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=20,
            seed=TEST_CONFIG['seed']
        )
    
    def test_agent_environment_interaction(self):
        """Test agent interaction with mock environment."""
        agent = DoubleDQNAgent(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=torch.device(TEST_CONFIG['device'])
        )
        
        # Run a short episode
        state = self.env.reset()
        total_reward = 0
        
        for step in range(10):
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Test that agent can handle the environment's outputs
            self.assertIsInstance(action, (int, np.integer))
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, TEST_CONFIG['action_dim'])
            
            # Update agent if transition is available
            transition = self.env.get_last_transition()
            if transition is not None:
                # Convert to batch format
                batch_data = (
                    torch.tensor([transition[0]], dtype=torch.float32),
                    torch.tensor([transition[1]], dtype=torch.long),
                    torch.tensor([transition[2]], dtype=torch.float32),
                    torch.tensor([not transition[3]], dtype=torch.float32),
                    torch.tensor([transition[4]], dtype=torch.float32)
                )
                result = agent.update(batch_data)
                self.assertIsNotNone(result)
            
            state = next_state
            if done:
                break
        
        # Verify episode completed without errors
        self.assertIsInstance(total_reward, (int, float))
    
    def test_multiple_agent_types(self):
        """Test multiple agent types with environment."""
        agent_types = ['AgentDoubleDQN', 'AgentD3QN', 'AgentPrioritizedDQN']
        
        for agent_type in agent_types:
            with self.subTest(agent_type=agent_type):
                agent = create_agent(
                    agent_type=agent_type,
                    state_dim=TEST_CONFIG['state_dim'],
                    action_dim=TEST_CONFIG['action_dim'],
                    device=torch.device(TEST_CONFIG['device'])
                )
                
                # Quick interaction test
                state = self.env.reset()
                action = agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Verify basic functionality
                self.assertIsInstance(action, (int, np.integer))
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, TEST_CONFIG['action_dim'])


class TestAgentPerformance(unittest.TestCase):
    """Performance tests for agents."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
    
    def test_action_selection_speed(self):
        """Test action selection performance."""
        agent = DoubleDQNAgent(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=torch.device(TEST_CONFIG['device'])
        )
        
        state = create_test_state(TEST_CONFIG['state_dim'])
        
        # Time action selection
        with PerformanceTimer() as timer:
            for _ in range(100):
                action = agent.select_action(state, deterministic=True)
        
        # Should be reasonably fast (less than 1 second for 100 actions)
        self.assertLess(timer.elapsed, 1.0)
    
    def test_update_speed(self):
        """Test update performance."""
        agent = DoubleDQNAgent(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=torch.device(TEST_CONFIG['device'])
        )
        
        batch_data = create_test_batch(
            TEST_CONFIG['state_dim'],
            TEST_CONFIG['action_dim'],
            TEST_CONFIG['test_batch_size']
        )
        
        # Time updates
        with PerformanceTimer() as timer:
            for _ in range(10):
                result = agent.update(batch_data)
        
        # Should be reasonably fast (less than 5 seconds for 10 updates)
        self.assertLess(timer.elapsed, 5.0)


if __name__ == '__main__':
    # Run specific test classes or all tests
    unittest.main(verbosity=2)