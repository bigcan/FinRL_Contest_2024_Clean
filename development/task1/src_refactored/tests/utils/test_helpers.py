"""
Test helper functions and utilities for the FinRL Contest 2024 testing suite.

This module provides common utilities, fixtures, and helper functions
used across all test modules.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock
import tempfile
import shutil
from pathlib import Path

from ...core.types import StateType, ActionType, TrainingStats


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducible testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_test_state(state_dim: int = 10, batch_size: Optional[int] = None) -> torch.Tensor:
    """Create a test state tensor."""
    if batch_size is None:
        return torch.randn(state_dim)
    else:
        return torch.randn(batch_size, state_dim)


def create_test_action(action_dim: int = 3, batch_size: Optional[int] = None) -> torch.Tensor:
    """Create a test action tensor."""
    if batch_size is None:
        return torch.randint(0, action_dim, (1,))
    else:
        return torch.randint(0, action_dim, (batch_size, 1))


def create_test_reward(batch_size: Optional[int] = None) -> torch.Tensor:
    """Create a test reward tensor."""
    if batch_size is None:
        return torch.randn(1)
    else:
        return torch.randn(batch_size)


def create_test_done(batch_size: Optional[int] = None) -> torch.Tensor:
    """Create a test done tensor."""
    if batch_size is None:
        return torch.tensor([False])
    else:
        return torch.randint(0, 2, (batch_size,)).bool()


def create_test_batch(state_dim: int = 10, 
                     action_dim: int = 3, 
                     batch_size: int = 16) -> Tuple[torch.Tensor, ...]:
    """Create a test batch of transitions."""
    states = create_test_state(state_dim, batch_size)
    actions = create_test_action(action_dim, batch_size)
    rewards = create_test_reward(batch_size)
    dones = create_test_done(batch_size)
    next_states = create_test_state(state_dim, batch_size)
    
    return states, actions, rewards, dones, next_states


def create_mock_agent(agent_name: str = "MockAgent") -> Mock:
    """Create a mock agent for testing."""
    mock_agent = Mock()
    mock_agent.agent_name = agent_name
    mock_agent.state_dim = 10
    mock_agent.action_dim = 3
    mock_agent.device = torch.device('cpu')
    mock_agent.training_step = 0
    
    # Mock methods
    mock_agent.select_action.return_value = 0
    mock_agent.update.return_value = TrainingStats(
        critic_loss=0.1,
        q_value=1.0,
        exploration_rate=0.1,
        learning_rate=1e-4,
        reward=0.5
    )
    mock_agent.get_algorithm_info.return_value = {
        'algorithm': agent_name,
        'description': f'Mock {agent_name} for testing'
    }
    
    return mock_agent


def create_mock_network() -> Mock:
    """Create a mock neural network for testing."""
    mock_network = Mock()
    mock_network.eval.return_value = mock_network
    mock_network.train.return_value = mock_network
    mock_network.to.return_value = mock_network
    
    # Mock forward pass
    def mock_forward(x):
        batch_size = x.shape[0] if x.dim() > 1 else 1
        return torch.randn(batch_size, 3)  # 3 actions
    
    mock_network.forward = mock_forward
    mock_network.__call__ = mock_forward
    
    # Mock Q-network methods
    def mock_get_q1_q2(x):
        batch_size = x.shape[0] if x.dim() > 1 else 1
        q1 = torch.randn(batch_size, 3)
        q2 = torch.randn(batch_size, 3)
        return q1, q2
    
    mock_network.get_q1_q2 = mock_get_q1_q2
    
    return mock_network


def create_test_config(**kwargs) -> Dict[str, Any]:
    """Create a test configuration dictionary."""
    default_config = {
        'state_dim': 10,
        'action_dim': 3,
        'net_dims': [64, 32],
        'learning_rate': 1e-4,
        'batch_size': 16,
        'buffer_size': 100,
        'gamma': 0.99,
        'tau': 0.005,
        'explore_rate': 0.1,
        'clip_grad_norm': 10.0,
        'device': 'cpu'
    }
    default_config.update(kwargs)
    return default_config


def create_temporary_directory() -> Path:
    """Create a temporary directory for testing."""
    return Path(tempfile.mkdtemp())


def cleanup_temporary_directory(temp_dir: Path):
    """Clean up a temporary directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


class TestEnvironment:
    """Simple test environment for agent testing."""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 3, max_steps: int = 100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None
        
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.current_step = 0
        self.state = np.random.randn(self.state_dim)
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment."""
        self.current_step += 1
        
        # Simple reward function based on action
        reward = np.random.randn() + (action / self.action_dim)
        
        # Update state
        self.state = np.random.randn(self.state_dim)
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        info = {'step': self.current_step}
        
        return self.state.copy(), reward, done, info
    
    def get_last_transition(self) -> Tuple[np.ndarray, int, float, bool]:
        """Get the last transition (for agent updates)."""
        # Return dummy transition
        return (
            np.random.randn(self.state_dim),  # state
            np.random.randint(0, self.action_dim),  # action
            np.random.randn(),  # reward
            False  # done
        )


def assert_tensor_equal(actual: torch.Tensor, 
                       expected: torch.Tensor, 
                       tolerance: float = 1e-6,
                       msg: str = ""):
    """Assert that two tensors are approximately equal."""
    if not torch.allclose(actual, expected, atol=tolerance):
        raise AssertionError(
            f"Tensors are not equal within tolerance {tolerance}. {msg}\n"
            f"Actual: {actual}\n"
            f"Expected: {expected}\n"
            f"Difference: {(actual - expected).abs().max().item()}"
        )


def assert_tensor_shape(tensor: torch.Tensor, 
                       expected_shape: Tuple[int, ...],
                       msg: str = ""):
    """Assert that a tensor has the expected shape."""
    if tensor.shape != expected_shape:
        raise AssertionError(
            f"Tensor shape mismatch. {msg}\n"
            f"Actual shape: {tensor.shape}\n"
            f"Expected shape: {expected_shape}"
        )


def assert_agent_interface(agent: Any, expected_methods: List[str]):
    """Assert that an agent implements the expected interface."""
    for method_name in expected_methods:
        if not hasattr(agent, method_name):
            raise AssertionError(f"Agent missing method: {method_name}")
        
        method = getattr(agent, method_name)
        if not callable(method):
            raise AssertionError(f"Agent attribute {method_name} is not callable")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def check_gradient_flow(model: torch.nn.Module) -> Dict[str, bool]:
    """Check if gradients are flowing through the model."""
    gradient_flow = {}
    for name, param in model.named_parameters():
        gradient_flow[name] = param.grad is not None and param.grad.norm() > 0
    return gradient_flow


def create_test_training_stats(**kwargs) -> TrainingStats:
    """Create test training statistics."""
    default_stats = {
        'critic_loss': 0.1,
        'q_value': 1.0,
        'exploration_rate': 0.1,
        'learning_rate': 1e-4,
        'reward': 0.5
    }
    default_stats.update(kwargs)
    return TrainingStats(**default_stats)


def run_agent_smoke_test(agent: Any, 
                        state_dim: int = 10, 
                        action_dim: int = 3,
                        num_steps: int = 10) -> Dict[str, Any]:
    """
    Run a basic smoke test on an agent to ensure it works.
    
    Returns dictionary with test results.
    """
    results = {
        'action_selection': False,
        'update': False,
        'training_info': False,
        'algorithm_info': False,
        'errors': []
    }
    
    try:
        # Test action selection
        state = create_test_state(state_dim)
        action = agent.select_action(state)
        if isinstance(action, (int, np.integer)) and 0 <= action < action_dim:
            results['action_selection'] = True
        elif hasattr(action, 'item'):
            action_val = action.item()
            if 0 <= action_val < action_dim:
                results['action_selection'] = True
    except Exception as e:
        results['errors'].append(f"Action selection failed: {e}")
    
    try:
        # Test update
        batch_data = create_test_batch(state_dim, action_dim, 16)
        update_result = agent.update(batch_data)
        if update_result is not None:
            results['update'] = True
    except Exception as e:
        results['errors'].append(f"Update failed: {e}")
    
    try:
        # Test training info
        info = agent.get_training_info()
        if isinstance(info, dict):
            results['training_info'] = True
    except Exception as e:
        results['errors'].append(f"Training info failed: {e}")
    
    try:
        # Test algorithm info
        info = agent.get_algorithm_info()
        if isinstance(info, dict) and 'algorithm' in info:
            results['algorithm_info'] = True
    except Exception as e:
        results['errors'].append(f"Algorithm info failed: {e}")
    
    return results


def validate_ensemble_interface(ensemble: Any) -> Dict[str, bool]:
    """Validate that an ensemble implements the expected interface."""
    required_methods = [
        'select_action',
        'update',
        'get_individual_actions',
        'get_ensemble_info',
        'save_ensemble',
        'load_ensemble'
    ]
    
    validation_results = {}
    for method_name in required_methods:
        validation_results[method_name] = (
            hasattr(ensemble, method_name) and 
            callable(getattr(ensemble, method_name))
        )
    
    return validation_results


class PerformanceTimer:
    """Simple performance timer for benchmarking."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time