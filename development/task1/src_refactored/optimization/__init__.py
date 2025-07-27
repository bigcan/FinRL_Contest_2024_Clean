"""
Optimization components and learning rate scheduling for the FinRL Contest 2024 framework.

This module provides comprehensive optimization functionality including:
- Base optimizer wrappers and utilities
- Adaptive learning rate scheduling strategies
- Gradient clipping and manipulation utilities
- Gradient noise injection for regularization
- Gradient accumulation for large batch training
- Gradient analysis and debugging tools

Example usage:
    from src_refactored.optimization import (
        AdaptiveLRScheduler,
        AdaptiveOptimizerWrapper,
        GradientClipper,
        OptimizerFactory,
        create_adaptive_optimizer
    )
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import numpy as np

# Base optimizer components
from .base_optimizers import (
    BaseOptimizerWrapper,
    OptimizerFactory,
    GradientStatistics,
    OptimizerUtils,
)

# Adaptive scheduling
from .adaptive_schedulers import (
    AdaptiveLRScheduler,
    AdaptiveOptimizerWrapper,
    create_adaptive_optimizer,
    SCHEDULER_CONFIGS,
)

# Gradient utilities
from .gradient_utils import (
    GradientClipper,
    GradientNoiseInjector,
    GradientAccumulator,
    GradientAnalyzer,
    apply_gradient_modification,
)

# Type aliases for convenience
from typing import Union
import torch

OptimizerType = Union[torch.optim.Optimizer, AdaptiveOptimizerWrapper]

__all__ = [
    # Base components
    'BaseOptimizerWrapper',
    'OptimizerFactory', 
    'GradientStatistics',
    'OptimizerUtils',
    
    # Adaptive scheduling
    'AdaptiveLRScheduler',
    'AdaptiveOptimizerWrapper',
    'create_adaptive_optimizer',
    'SCHEDULER_CONFIGS',
    
    # Gradient utilities
    'GradientClipper',
    'GradientNoiseInjector',
    'GradientAccumulator',
    'GradientAnalyzer',
    'apply_gradient_modification',
    
    # Type aliases
    'OptimizerType',
    
    # Factory functions
    'create_optimizer_suite',
    'get_recommended_optimizer_config',
]


def create_optimizer_suite(model: torch.nn.Module,
                          optimizer_type: str = "adamw",
                          lr: float = 1e-3,
                          scheduler_strategy: str = "cosine_annealing",
                          use_gradient_clipping: bool = True,
                          **kwargs) -> AdaptiveOptimizerWrapper:
    """
    Create a complete optimizer suite with scheduling and gradient handling.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of base optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        lr: Learning rate
        scheduler_strategy: Learning rate scheduling strategy
        use_gradient_clipping: Whether to use gradient clipping
        **kwargs: Additional configuration options
        
    Returns:
        Adaptive optimizer wrapper with full functionality
        
    Example:
        optimizer = create_optimizer_suite(
            model,
            optimizer_type="adamw",
            lr=1e-3,
            scheduler_strategy="warmup_cosine",
            use_gradient_clipping=True,
            grad_clip_norm=1.0
        )
    """
    # Create base optimizer
    if optimizer_type == "adam":
        base_optimizer = OptimizerFactory.create_adam(
            model.parameters(), 
            lr=lr,
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    elif optimizer_type == "adamw":
        base_optimizer = OptimizerFactory.create_adamw(
            model.parameters(),
            lr=lr,
            weight_decay=kwargs.get('weight_decay', 1e-2)
        )
    elif optimizer_type == "sgd":
        base_optimizer = OptimizerFactory.create_sgd(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    elif optimizer_type == "rmsprop":
        base_optimizer = OptimizerFactory.create_rmsprop(
            model.parameters(),
            lr=lr,
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Get scheduler configuration
    if scheduler_strategy in SCHEDULER_CONFIGS:
        scheduler_config = SCHEDULER_CONFIGS[scheduler_strategy].copy()
    else:
        scheduler_config = {'strategy': scheduler_strategy}
    
    # Override with custom parameters
    scheduler_config.update({k: v for k, v in kwargs.items() 
                           if k.startswith(('T_', 'eta_', 'patience', 'factor', 'warmup_'))})
    
    # Wrapper configuration
    wrapper_config = {
        'grad_clip_norm': kwargs.get('grad_clip_norm', 1.0),
        'adaptive_grad_clip': use_gradient_clipping and kwargs.get('adaptive_grad_clip', True),
        'gradient_accumulation_steps': kwargs.get('gradient_accumulation_steps', 1),
    }
    
    return create_adaptive_optimizer(base_optimizer, scheduler_config, wrapper_config)


def get_recommended_optimizer_config(agent_type: str, model_size: str = "medium") -> Dict[str, Any]:
    """
    Get recommended optimizer configuration for specific agent and model size.
    
    Args:
        agent_type: Type of RL agent
        model_size: Size category ('small', 'medium', 'large')
        
    Returns:
        Dictionary of optimizer configuration
    """
    # Base configurations by model size
    size_configs = {
        'small': {
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'grad_clip_norm': 0.5,
        },
        'medium': {
            'lr': 1e-4,
            'weight_decay': 1e-3,
            'grad_clip_norm': 1.0,
        },
        'large': {
            'lr': 3e-5,
            'weight_decay': 1e-2,
            'grad_clip_norm': 2.0,
        }
    }
    
    base_config = size_configs.get(model_size, size_configs['medium'])
    
    # Agent-specific optimizations
    if agent_type in ['AgentPrioritizedDQN', 'AgentRainbowDQN']:
        # These agents benefit from more stable learning
        base_config.update({
            'optimizer_type': 'adamw',
            'scheduler_strategy': 'reduce_on_plateau',
            'adaptive_grad_clip': True,
        })
    elif agent_type in ['AgentNoisyDQN', 'AgentNoisyDuelDQN']:
        # Noisy networks can handle higher learning rates
        base_config.update({
            'lr': base_config['lr'] * 1.5,
            'optimizer_type': 'adam',
            'scheduler_strategy': 'cosine_annealing',
        })
    elif agent_type == 'AgentAdaptiveDQN':
        # Already has adaptive components, use simpler scheduling
        base_config.update({
            'optimizer_type': 'adamw',
            'scheduler_strategy': 'cosine_annealing',
            'adaptive_grad_clip': True,
        })
    else:
        # Default configuration
        base_config.update({
            'optimizer_type': 'adamw',
            'scheduler_strategy': 'warmup_cosine',
            'adaptive_grad_clip': True,
        })
    
    return base_config


# Import shortcuts for common configurations
from typing import Dict, Any

def quick_adam(parameters, lr: float = 1e-3) -> torch.optim.Adam:
    """Quick Adam optimizer creation."""
    return OptimizerFactory.create_adam(parameters, lr=lr)

def quick_adamw(parameters, lr: float = 1e-3) -> torch.optim.AdamW:
    """Quick AdamW optimizer creation."""
    return OptimizerFactory.create_adamw(parameters, lr=lr)