"""
Comprehensive testing suite for the FinRL Contest 2024 refactored framework.

This module provides unit tests, integration tests, and validation utilities
for all components of the refactored architecture.
"""

import sys
import os
from pathlib import Path

# Add src_refactored to path for testing
test_dir = Path(__file__).parent
src_dir = test_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Test configuration
TEST_CONFIG = {
    'device': 'cpu',  # Use CPU for tests
    'state_dim': 10,
    'action_dim': 3,
    'test_episodes': 5,
    'small_buffer_size': 100,
    'test_batch_size': 16,
    'tolerance': 1e-6,
    'seed': 42
}

# Import test utilities
try:
    from .utils.test_helpers import *
    from .utils.mock_environment import MockEnvironment
except ImportError:
    # Handle case where utils haven't been created yet
    MockEnvironment = None

__all__ = [
    'TEST_CONFIG',
    'MockEnvironment',
]