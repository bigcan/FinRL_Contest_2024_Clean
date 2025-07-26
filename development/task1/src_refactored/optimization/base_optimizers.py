"""
Base optimizer components and utilities for the FinRL Contest 2024 framework.

This module provides foundational optimizer classes and utilities
that are shared across different optimization strategies.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
from torch.optim import Optimizer


class BaseOptimizerWrapper(ABC):
    """
    Abstract base class for optimizer wrappers.
    
    Provides a common interface for enhanced optimizers that add
    functionality like adaptive learning rates, gradient clipping,
    or custom update strategies on top of standard PyTorch optimizers.
    """
    
    def __init__(self, optimizer: Optimizer):
        """
        Initialize optimizer wrapper.
        
        Args:
            optimizer: Base PyTorch optimizer to wrap
        """
        self.optimizer = optimizer
        self.step_count = 0
        
    @abstractmethod
    def step(self, closure: Optional[Callable] = None, **kwargs):
        """
        Perform optimization step with additional functionality.
        
        Args:
            closure: Optional closure for optimizer
            **kwargs: Additional step arguments
        """
        pass
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dictionary."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dictionary."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step_count = state_dict.get('step_count', 0)
    
    @property
    def param_groups(self):
        """Access optimizer parameter groups."""
        return self.optimizer.param_groups
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class OptimizerFactory:
    """Factory for creating optimizers with standard configurations."""
    
    @staticmethod
    def create_adam(parameters, lr: float = 1e-3, weight_decay: float = 0.0,
                   betas: tuple = (0.9, 0.999), eps: float = 1e-8) -> torch.optim.Adam:
        """
        Create Adam optimizer with standard configuration.
        
        Args:
            parameters: Model parameters
            lr: Learning rate
            weight_decay: Weight decay coefficient
            betas: Coefficients for computing running averages
            eps: Epsilon for numerical stability
            
        Returns:
            Adam optimizer
        """
        return torch.optim.Adam(
            parameters, 
            lr=lr, 
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
    @staticmethod
    def create_adamw(parameters, lr: float = 1e-3, weight_decay: float = 1e-2,
                    betas: tuple = (0.9, 0.999), eps: float = 1e-8) -> torch.optim.AdamW:
        """
        Create AdamW optimizer with standard configuration.
        
        Args:
            parameters: Model parameters
            lr: Learning rate
            weight_decay: Weight decay coefficient
            betas: Coefficients for computing running averages
            eps: Epsilon for numerical stability
            
        Returns:
            AdamW optimizer
        """
        return torch.optim.AdamW(
            parameters, 
            lr=lr, 
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
    @staticmethod
    def create_sgd(parameters, lr: float = 1e-3, momentum: float = 0.9,
                  weight_decay: float = 0.0, nesterov: bool = False) -> torch.optim.SGD:
        """
        Create SGD optimizer with standard configuration.
        
        Args:
            parameters: Model parameters
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay coefficient
            nesterov: Whether to use Nesterov momentum
            
        Returns:
            SGD optimizer
        """
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    @staticmethod
    def create_rmsprop(parameters, lr: float = 1e-3, alpha: float = 0.99,
                      eps: float = 1e-8, weight_decay: float = 0.0,
                      momentum: float = 0.0) -> torch.optim.RMSprop:
        """
        Create RMSprop optimizer with standard configuration.
        
        Args:
            parameters: Model parameters
            lr: Learning rate
            alpha: Smoothing constant
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
            momentum: Momentum factor
            
        Returns:
            RMSprop optimizer
        """
        return torch.optim.RMSprop(
            parameters,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum
        )


class GradientStatistics:
    """Class for tracking gradient statistics across training."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize gradient statistics tracker.
        
        Args:
            window_size: Size of moving window for statistics
        """
        self.window_size = window_size
        self.grad_norms = []
        self.step_count = 0
    
    def update(self, parameters):
        """
        Update gradient statistics with current gradients.
        
        Args:
            parameters: Model parameters with gradients
        """
        # Calculate gradient norm
        total_norm = 0.0
        param_count = 0
        
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            
            # Add to history
            self.grad_norms.append(total_norm)
            if len(self.grad_norms) > self.window_size:
                self.grad_norms.pop(0)
            
            self.step_count += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get gradient statistics.
        
        Returns:
            Dictionary of gradient statistics
        """
        if not self.grad_norms:
            return {
                'mean_grad_norm': 0.0,
                'std_grad_norm': 0.0,
                'max_grad_norm': 0.0,
                'min_grad_norm': 0.0,
                'recent_grad_norm': 0.0,
            }
        
        import numpy as np
        
        return {
            'mean_grad_norm': np.mean(self.grad_norms),
            'std_grad_norm': np.std(self.grad_norms),
            'max_grad_norm': np.max(self.grad_norms),
            'min_grad_norm': np.min(self.grad_norms),
            'recent_grad_norm': self.grad_norms[-1],
        }
    
    def get_adaptive_clip_value(self, percentile: float = 95.0) -> float:
        """
        Get adaptive gradient clipping value based on statistics.
        
        Args:
            percentile: Percentile for clipping threshold
            
        Returns:
            Adaptive clipping value
        """
        if not self.grad_norms:
            return 1.0
        
        import numpy as np
        return np.percentile(self.grad_norms, percentile)
    
    def reset(self):
        """Reset statistics."""
        self.grad_norms = []
        self.step_count = 0


class OptimizerUtils:
    """Utility functions for optimizer operations."""
    
    @staticmethod
    def get_parameter_count(optimizer: Optimizer) -> int:
        """
        Get total number of parameters in optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            Total parameter count
        """
        return sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    
    @staticmethod
    def get_parameter_groups_info(optimizer: Optimizer) -> List[Dict[str, Any]]:
        """
        Get information about parameter groups.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            List of parameter group information
        """
        info = []
        for i, group in enumerate(optimizer.param_groups):
            group_info = {
                'group_id': i,
                'lr': group.get('lr', 'N/A'),
                'weight_decay': group.get('weight_decay', 'N/A'),
                'parameter_count': sum(p.numel() for p in group['params']),
                'parameter_shapes': [list(p.shape) for p in group['params']],
            }
            info.append(group_info)
        return info
    
    @staticmethod
    def scale_learning_rates(optimizer: Optimizer, scale_factor: float):
        """
        Scale learning rates of all parameter groups.
        
        Args:
            optimizer: PyTorch optimizer
            scale_factor: Factor to scale learning rates by
        """
        for group in optimizer.param_groups:
            group['lr'] *= scale_factor
    
    @staticmethod
    def set_weight_decay(optimizer: Optimizer, weight_decay: float):
        """
        Set weight decay for all parameter groups.
        
        Args:
            optimizer: PyTorch optimizer
            weight_decay: New weight decay value
        """
        for group in optimizer.param_groups:
            group['weight_decay'] = weight_decay
    
    @staticmethod
    def get_optimizer_memory_usage(optimizer: Optimizer) -> Dict[str, float]:
        """
        Estimate memory usage of optimizer state.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            Dictionary with memory usage information (in MB)
        """
        total_params = 0
        total_state_tensors = 0
        
        for group in optimizer.param_groups:
            for param in group['params']:
                total_params += param.numel()
                
                # Count state tensors (varies by optimizer type)
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            total_state_tensors += value.numel()
        
        # Estimate memory usage (assuming float32)
        param_memory = total_params * 4 / (1024 * 1024)  # MB
        state_memory = total_state_tensors * 4 / (1024 * 1024)  # MB
        
        return {
            'parameter_memory_mb': param_memory,
            'state_memory_mb': state_memory,
            'total_memory_mb': param_memory + state_memory,
            'parameter_count': total_params,
            'state_tensor_count': total_state_tensors,
        }
    
    @staticmethod
    def create_parameter_groups(model: nn.Module, 
                              layer_configs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different configurations for different layers.
        
        Args:
            model: PyTorch model
            layer_configs: Dictionary mapping layer name patterns to configurations
            
        Returns:
            List of parameter groups
            
        Example:
            layer_configs = {
                'conv': {'lr': 1e-3, 'weight_decay': 1e-4},
                'fc': {'lr': 1e-4, 'weight_decay': 1e-3},
                'bias': {'weight_decay': 0.0},
            }
        """
        groups = []
        used_params = set()
        
        # Create groups for specified patterns
        for pattern, config in layer_configs.items():
            group_params = []
            for name, param in model.named_parameters():
                if pattern in name and param not in used_params:
                    group_params.append(param)
                    used_params.add(param)
            
            if group_params:
                group = {'params': group_params, **config}
                groups.append(group)
        
        # Add remaining parameters to default group
        remaining_params = [p for p in model.parameters() if p not in used_params]
        if remaining_params:
            groups.append({'params': remaining_params})
        
        return groups