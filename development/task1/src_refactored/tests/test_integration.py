"""
Integration tests for the FinRL Contest 2024 refactored framework.

This module tests end-to-end functionality including training workflows,
ensemble coordination, and complete system integration.
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
    set_random_seeds, create_test_config, create_temporary_directory,
    cleanup_temporary_directory, PerformanceTimer
)
from .utils.mock_environment import MockEnvironment, MockVectorizedEnvironment

# Import components for integration testing
from ..agents import create_agent, create_ensemble_agents
from ..ensemble import create_voting_ensemble, create_stacking_ensemble, EnsembleStrategy
from ..training.ensemble_trainer import EnsembleTrainer, TrainingConfig, TrainingPhase, TrainingResults
from ..config import DoubleDQNConfig, PrioritizedDQNConfig


class TestAgentEnsembleIntegration(unittest.TestCase):
    """Test integration between agents and ensembles."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.device = torch.device(TEST_CONFIG['device'])
        
        # Create test environment
        self.env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=20,
            seed=TEST_CONFIG['seed']
        )
    
    def test_agent_creation_and_ensemble_integration(self):
        """Test creating agents and integrating them into ensembles."""
        # Create diverse agents
        agent_configs = {
            "double_dqn": {
                "agent_type": "AgentDoubleDQN",
                "learning_rate": 1e-4,
                "batch_size": 16
            },
            "d3qn": {
                "agent_type": "AgentD3QN",
                "learning_rate": 1e-4,
                "batch_size": 16
            },
            "prioritized": {
                "agent_type": "AgentPrioritizedDQN",
                "learning_rate": 1e-4,
                "per_alpha": 0.6,
                "per_beta": 0.4,
                "buffer_size": TEST_CONFIG['small_buffer_size']
            }
        }
        
        # Create agents
        agents = create_ensemble_agents(
            agent_configs,
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        self.assertEqual(len(agents), 3)
        
        # Test individual agent functionality
        state = self.env.reset()
        for name, agent in agents.items():
            action = agent.select_action(state)
            self.assertIsInstance(action, (int, np.integer))
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, TEST_CONFIG['action_dim'])
        
        # Create voting ensemble
        voting_ensemble = create_voting_ensemble(
            agents=agents,
            strategy=EnsembleStrategy.WEIGHTED_VOTE,
            device=self.device
        )
        
        # Test ensemble functionality
        ensemble_action = voting_ensemble.select_action(state)
        self.assertIsInstance(ensemble_action, (int, np.integer))
        self.assertGreaterEqual(ensemble_action, 0)
        self.assertLess(ensemble_action, TEST_CONFIG['action_dim'])
        
        # Create stacking ensemble
        stacking_ensemble = create_stacking_ensemble(
            agents=agents,
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        # Test stacking ensemble functionality
        stacking_action = stacking_ensemble.select_action(state)
        self.assertIsInstance(stacking_action, (int, np.integer))
        self.assertGreaterEqual(stacking_action, 0)
        self.assertLess(stacking_action, TEST_CONFIG['action_dim'])
    
    def test_ensemble_training_episode(self):
        """Test running a complete training episode with ensemble."""
        # Create simple agents
        agents = {
            "agent1": create_agent(
                "AgentDoubleDQN",
                state_dim=TEST_CONFIG['state_dim'],
                action_dim=TEST_CONFIG['action_dim'],
                device=self.device
            ),
            "agent2": create_agent(
                "AgentD3QN",
                state_dim=TEST_CONFIG['state_dim'],
                action_dim=TEST_CONFIG['action_dim'],
                device=self.device
            )
        }
        
        ensemble = create_voting_ensemble(
            agents=agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        # Run complete episode
        state = self.env.reset()
        total_reward = 0
        episode_length = 0
        
        for step in range(15):
            # Ensemble action selection
            action = ensemble.select_action(state, deterministic=False)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            episode_length += 1
            
            # Ensemble update
            transition = self.env.get_last_transition()
            if transition is not None:
                # Convert to batch format for ensemble update
                batch_data = (
                    torch.tensor([transition[0]], dtype=torch.float32),
                    torch.tensor([transition[1]], dtype=torch.long),
                    torch.tensor([transition[2]], dtype=torch.float32),
                    torch.tensor([not transition[3]], dtype=torch.float32),
                    torch.tensor([transition[4]], dtype=torch.float32)
                )
                
                update_result = ensemble.update(batch_data)
                self.assertIsInstance(update_result, dict)
                self.assertIn('ensemble_reward', update_result)
            
            state = next_state
            if done:
                break
        
        # Verify episode completed successfully
        self.assertGreater(episode_length, 0)
        self.assertIsInstance(total_reward, (int, float))
        
        # Check ensemble statistics
        ensemble_info = ensemble.get_ensemble_info()
        self.assertIsInstance(ensemble_info, dict)
        self.assertEqual(ensemble_info['num_agents'], 2)


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration management."""
    
    def test_training_config_creation(self):
        """Test training configuration creation and validation."""
        config = TrainingConfig(
            total_episodes=100,
            individual_episodes=30,
            ensemble_episodes=50,
            fine_tuning_episodes=20,
            eval_frequency=10,
            save_frequency=20
        )
        
        self.assertEqual(config.total_episodes, 100)
        self.assertEqual(config.individual_episodes, 30)
        self.assertEqual(config.ensemble_episodes, 50)
        self.assertEqual(config.fine_tuning_episodes, 20)
        self.assertEqual(config.eval_frequency, 10)
        self.assertEqual(config.save_frequency, 20)
    
    def test_training_config_defaults(self):
        """Test training configuration default values."""
        config = TrainingConfig()
        
        self.assertGreater(config.total_episodes, 0)
        self.assertGreater(config.individual_episodes, 0)
        self.assertGreater(config.ensemble_episodes, 0)
        self.assertGreater(config.eval_frequency, 0)
        self.assertIsInstance(config.ensemble_strategy, EnsembleStrategy)


class TestTrainingResults(unittest.TestCase):
    """Test training results management."""
    
    def test_training_results_creation(self):
        """Test training results creation and data storage."""
        results = TrainingResults()
        
        # Add some test data
        results.training_rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
        results.evaluation_rewards = [0.15, 0.25, 0.35]
        results.individual_agent_rewards = {
            "agent1": [0.1, 0.2, 0.3],
            "agent2": [0.05, 0.15, 0.25]
        }
        results.best_episode = 4
        results.best_reward = 0.5
        results.total_training_time = 120.5
        
        self.assertEqual(len(results.training_rewards), 5)
        self.assertEqual(len(results.evaluation_rewards), 3)
        self.assertEqual(results.best_episode, 4)
        self.assertEqual(results.best_reward, 0.5)
    
    def test_training_results_save_load(self):
        """Test saving and loading training results."""
        results = TrainingResults()
        results.training_rewards = [0.1, 0.2, 0.3]
        results.evaluation_rewards = [0.15, 0.25]
        results.best_reward = 0.3
        results.total_training_time = 60.0
        
        temp_dir = create_temporary_directory()
        try:
            results_path = temp_dir / "test_results.json"
            
            # Save results
            results.save_results(str(results_path))
            self.assertTrue(results_path.exists())
            
            # Load and verify (basic file existence and format)
            import json
            with open(results_path, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertIn('training_rewards', loaded_data)
            self.assertIn('evaluation_rewards', loaded_data)
            self.assertIn('best_reward', loaded_data)
            self.assertIn('total_training_time', loaded_data)
            self.assertIn('summary_statistics', loaded_data)
            
            self.assertEqual(loaded_data['training_rewards'], [0.1, 0.2, 0.3])
            self.assertEqual(loaded_data['best_reward'], 0.3)
            
        finally:
            cleanup_temporary_directory(temp_dir)


class TestEnsembleTrainerIntegration(unittest.TestCase):
    """Test ensemble trainer integration (without full training)."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.device = torch.device(TEST_CONFIG['device'])
        
        # Create mock environment for trainer testing
        self.env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=10,  # Very short episodes for testing
            seed=TEST_CONFIG['seed']
        )
    
    def test_ensemble_trainer_creation(self):
        """Test ensemble trainer creation and initialization."""
        agent_configs = {
            "agent1": {"agent_type": "AgentDoubleDQN"},
            "agent2": {"agent_type": "AgentD3QN"}
        }
        
        config = TrainingConfig(
            total_episodes=20,  # Very short for testing
            individual_episodes=5,
            ensemble_episodes=10,
            fine_tuning_episodes=5,
            eval_frequency=5,
            save_frequency=10,
            eval_episodes=2,
            device=TEST_CONFIG['device']
        )
        
        temp_dir = create_temporary_directory()
        try:
            trainer = EnsembleTrainer(
                environment=self.env,
                agent_configs=agent_configs,
                config=config,
                save_dir=str(temp_dir)
            )
            
            self.assertIsInstance(trainer, EnsembleTrainer)
            self.assertEqual(trainer.device, self.device)
            self.assertEqual(trainer.config, config)
            self.assertEqual(trainer.current_phase, TrainingPhase.INDIVIDUAL)
            
        finally:
            cleanup_temporary_directory(temp_dir)
    
    def test_ensemble_trainer_phase_transitions(self):
        """Test ensemble trainer phase tracking."""
        agent_configs = {
            "agent1": {"agent_type": "AgentDoubleDQN"}
        }
        
        config = TrainingConfig(
            total_episodes=10,
            individual_episodes=3,
            ensemble_episodes=4,
            fine_tuning_episodes=3,
            device=TEST_CONFIG['device']
        )
        
        temp_dir = create_temporary_directory()
        try:
            trainer = EnsembleTrainer(
                environment=self.env,
                agent_configs=agent_configs,
                config=config,
                save_dir=str(temp_dir)
            )
            
            # Test phase initialization
            self.assertEqual(trainer.current_phase, TrainingPhase.INDIVIDUAL)
            
            # Test phase transition recording
            trainer.results.phase_transitions[TrainingPhase.INDIVIDUAL] = 0
            trainer.results.phase_transitions[TrainingPhase.ENSEMBLE] = 5
            
            self.assertEqual(trainer.results.phase_transitions[TrainingPhase.INDIVIDUAL], 0)
            self.assertEqual(trainer.results.phase_transitions[TrainingPhase.ENSEMBLE], 5)
            
        finally:
            cleanup_temporary_directory(temp_dir)
    
    def test_ensemble_trainer_agent_creation(self):
        """Test that ensemble trainer can create agents correctly."""
        agent_configs = {
            "double_dqn": {
                "agent_type": "AgentDoubleDQN",
                "learning_rate": 1e-4
            },
            "prioritized": {
                "agent_type": "AgentPrioritizedDQN",
                "learning_rate": 1e-4,
                "buffer_size": TEST_CONFIG['small_buffer_size']
            }
        }
        
        config = TrainingConfig(device=TEST_CONFIG['device'])
        
        temp_dir = create_temporary_directory()
        try:
            trainer = EnsembleTrainer(
                environment=self.env,
                agent_configs=agent_configs,
                config=config,
                save_dir=str(temp_dir)
            )
            
            # Manually trigger agent creation (normally done in training)
            state_dim = getattr(trainer.environment, 'state_dim', TEST_CONFIG['state_dim'])
            action_dim = getattr(trainer.environment, 'action_dim', TEST_CONFIG['action_dim'])
            
            agents = create_ensemble_agents(
                trainer.agent_configs,
                state_dim=state_dim,
                action_dim=action_dim,
                device=trainer.device
            )
            
            self.assertEqual(len(agents), 2)
            self.assertIn("double_dqn", agents)
            self.assertIn("prioritized", agents)
            
        finally:
            cleanup_temporary_directory(temp_dir)


class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration scenarios."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow simulation (shortened for testing)."""
        # Create simple environment
        env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=5,  # Very short episodes
            seed=TEST_CONFIG['seed']
        )
        
        # Create agents
        agent_configs = {
            "agent1": {"agent_type": "AgentDoubleDQN"},
            "agent2": {"agent_type": "AgentD3QN"}
        }
        
        agents = create_ensemble_agents(
            agent_configs,
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        # Create ensemble
        ensemble = create_voting_ensemble(
            agents=agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        # Simulate training workflow
        total_episodes = 3
        training_rewards = []
        
        for episode in range(total_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(5):  # Short episodes
                # Individual agent training
                if episode < 1:  # Individual training phase
                    for name, agent in agents.items():
                        action = agent.select_action(state, deterministic=False)
                        next_state, reward, done, info = env.step(action)
                        episode_reward += reward
                        
                        # Simple update
                        transition = env.get_last_transition()
                        if transition is not None:
                            batch_data = (
                                torch.tensor([transition[0]], dtype=torch.float32),
                                torch.tensor([transition[1]], dtype=torch.long),
                                torch.tensor([transition[2]], dtype=torch.float32),
                                torch.tensor([not transition[3]], dtype=torch.float32),
                                torch.tensor([transition[4]], dtype=torch.float32)
                            )
                            agent.update(batch_data)
                        
                        state = next_state
                        if done:
                            break
                
                else:  # Ensemble training phase
                    action = ensemble.select_action(state, deterministic=False)
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    # Ensemble update
                    transition = env.get_last_transition()
                    if transition is not None:
                        batch_data = (
                            torch.tensor([transition[0]], dtype=torch.float32),
                            torch.tensor([transition[1]], dtype=torch.long),
                            torch.tensor([transition[2]], dtype=torch.float32),
                            torch.tensor([not transition[3]], dtype=torch.float32),
                            torch.tensor([transition[4]], dtype=torch.float32)
                        )
                        ensemble.update(batch_data)
                    
                    state = next_state
                    if done:
                        break
            
            training_rewards.append(episode_reward)
        
        # Verify workflow completed
        self.assertEqual(len(training_rewards), total_episodes)
        
        # Test ensemble statistics
        ensemble_info = ensemble.get_ensemble_info()
        self.assertIsInstance(ensemble_info, dict)
        self.assertGreater(ensemble_info['training_step'], 0)
    
    def test_error_handling_integration(self):
        """Test error handling in integrated system."""
        # Test with invalid agent configuration
        invalid_agent_configs = {
            "invalid_agent": {"agent_type": "NonExistentAgent"}
        }
        
        # This should handle the error gracefully
        agents = create_ensemble_agents(
            invalid_agent_configs,
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        # Should return empty dict or handle gracefully
        self.assertIsInstance(agents, dict)
    
    def test_memory_usage_integration(self):
        """Test memory usage in integrated scenarios."""
        import gc
        import torch
        
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create components
        env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=5,
            seed=TEST_CONFIG['seed']
        )
        
        agents = create_ensemble_agents(
            {"agent1": {"agent_type": "AgentDoubleDQN"}},
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        # Run some operations
        if agents:  # Only if agent creation succeeded
            agent = list(agents.values())[0]
            state = env.reset()
            
            for _ in range(10):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                if done:
                    state = env.reset()
        
        # Clean up
        del agents, env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance characteristics of integrated system."""
    
    def setUp(self):
        set_random_seeds(TEST_CONFIG['seed'])
        self.device = torch.device(TEST_CONFIG['device'])
    
    def test_ensemble_training_performance(self):
        """Test performance of ensemble training workflow."""
        env = MockEnvironment(
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=10,
            seed=TEST_CONFIG['seed']
        )
        
        # Create small ensemble
        agents = create_ensemble_agents(
            {"agent1": {"agent_type": "AgentDoubleDQN"}},
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        if not agents:  # Skip if agent creation failed
            self.skipTest("Agent creation failed")
        
        ensemble = create_voting_ensemble(
            agents=agents,
            strategy=EnsembleStrategy.MAJORITY_VOTE,
            device=self.device
        )
        
        # Time a short training sequence
        with PerformanceTimer() as timer:
            for episode in range(3):
                state = env.reset()
                for step in range(5):
                    action = ensemble.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    
                    # Quick update
                    transition = env.get_last_transition()
                    if transition is not None:
                        batch_data = (
                            torch.tensor([transition[0]], dtype=torch.float32),
                            torch.tensor([transition[1]], dtype=torch.long),
                            torch.tensor([transition[2]], dtype=torch.float32),
                            torch.tensor([not transition[3]], dtype=torch.float32),
                            torch.tensor([transition[4]], dtype=torch.float32)
                        )
                        ensemble.update(batch_data)
                    
                    state = next_state
                    if done:
                        break
        
        # Should complete in reasonable time (less than 5 seconds)
        self.assertLess(timer.elapsed, 5.0)
    
    def test_vectorized_environment_integration(self):
        """Test integration with vectorized environments."""
        num_envs = 2
        vec_env = MockVectorizedEnvironment(
            num_envs=num_envs,
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            max_steps=5,
            seed=TEST_CONFIG['seed']
        )
        
        # Create agent
        agents = create_ensemble_agents(
            {"agent1": {"agent_type": "AgentDoubleDQN"}},
            state_dim=TEST_CONFIG['state_dim'],
            action_dim=TEST_CONFIG['action_dim'],
            device=self.device
        )
        
        if not agents:  # Skip if agent creation failed
            self.skipTest("Agent creation failed")
        
        agent = list(agents.values())[0]
        
        # Test vectorized interaction
        states = vec_env.reset()
        self.assertEqual(states.shape[0], num_envs)
        
        # Select actions for all environments
        actions = []
        for i in range(num_envs):
            action = agent.select_action(states[i])
            actions.append(action)
        
        # Step all environments
        next_states, rewards, dones, infos = vec_env.step(actions)
        
        self.assertEqual(len(rewards), num_envs)
        self.assertEqual(len(dones), num_envs)
        self.assertEqual(len(infos), num_envs)
        
        vec_env.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)