"""
Adaptive learning rate schedulers for the FinRL Contest 2024 framework.

This module provides advanced learning rate scheduling strategies including:
- Cosine annealing with restarts
- Plateau-based reduction
- Performance-based adaptation
- Warmup scheduling
- Combined strategies
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, Any, List, Union
import math

from .base_optimizers import BaseOptimizerWrapper


class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler with multiple strategies.
    
    Supports various scheduling strategies that can adapt based on:
    - Training progress (cosine annealing)
    - Performance metrics (plateau detection)
    - Gradient behavior (performance-based)
    - Training phases (warmup)
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 strategy: str = "cosine_annealing",
                 **kwargs):
        """
        Initialize adaptive learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            strategy: Scheduling strategy ('cosine_annealing', 'reduce_on_plateau', 
                     'exponential', 'performance_based', 'warmup_cosine')
            **kwargs: Strategy-specific parameters
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
        
        # Performance tracking for adaptive scheduling
        self.performance_history = []
        self.best_performance = float('-inf')
        self.patience_counter = 0
        
        # Initialize strategy-specific parameters
        self._init_strategy_params(**kwargs)
    
    def _init_strategy_params(self, **kwargs):
        """Initialize parameters based on selected strategy."""
        if self.strategy == "cosine_annealing":
            self.T_max = kwargs.get('T_max', 10000)
            self.eta_min = kwargs.get('eta_min', 1e-8)
            self.T_mult = kwargs.get('T_mult', 1)  # For restarts
            self.last_restart = 0
            
        elif self.strategy == "reduce_on_plateau":
            self.patience = kwargs.get('patience', 100)
            self.factor = kwargs.get('factor', 0.5)
            self.threshold = kwargs.get('threshold', 1e-4)
            self.min_lr = kwargs.get('min_lr', 1e-8)
            self.cooldown = kwargs.get('cooldown', 0)
            self.cooldown_counter = 0
            
        elif self.strategy == "exponential":
            self.gamma = kwargs.get('gamma', 0.9999)
            self.min_lr = kwargs.get('min_lr', 1e-8)
            
        elif self.strategy == "performance_based":
            self.target_performance = kwargs.get('target_performance', 0.1)
            self.lr_increase_factor = kwargs.get('lr_increase_factor', 1.05)
            self.lr_decrease_factor = kwargs.get('lr_decrease_factor', 0.95)
            self.adaptation_frequency = kwargs.get('adaptation_frequency', 50)
            
        elif self.strategy == "warmup_cosine":
            self.warmup_steps = kwargs.get('warmup_steps', 1000)
            self.T_max = kwargs.get('T_max', 10000)
            self.eta_min = kwargs.get('eta_min', 1e-8)
            
        elif self.strategy == "polynomial":
            self.power = kwargs.get('power', 1.0)
            self.max_steps = kwargs.get('max_steps', 100000)
            self.min_lr = kwargs.get('min_lr', 1e-8)
            
        elif self.strategy == "cyclic":
            self.base_lr = kwargs.get('base_lr', self.initial_lr * 0.1)
            self.max_lr = kwargs.get('max_lr', self.initial_lr)
            self.step_size_up = kwargs.get('step_size_up', 2000)
            self.mode = kwargs.get('mode', 'triangular')  # triangular, triangular2, exp_range
            self.gamma = kwargs.get('gamma', 1.0)  # For exp_range mode
    
    def step(self, performance: Optional[float] = None):
        """
        Update learning rate based on strategy.
        
        Args:
            performance: Current performance metric (higher is better)
        """
        self.step_count += 1
        
        if self.strategy == "cosine_annealing":
            self._cosine_annealing_step()
            
        elif self.strategy == "cosine_annealing_restarts":
            self._cosine_annealing_restarts_step()
            
        elif self.strategy == "reduce_on_plateau":
            if performance is not None:
                self._reduce_on_plateau_step(performance)
                
        elif self.strategy == "exponential":
            self._exponential_step()
            
        elif self.strategy == "performance_based":
            if performance is not None:
                self._performance_based_step(performance)
                
        elif self.strategy == "warmup_cosine":
            self._warmup_cosine_step()
            
        elif self.strategy == "polynomial":
            self._polynomial_step()
            
        elif self.strategy == "cyclic":
            self._cyclic_step()
    
    def _cosine_annealing_step(self):
        """Cosine annealing learning rate schedule."""
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.step_count / self.T_max)) / 2
        self._set_lr(lr)
    
    def _cosine_annealing_restarts_step(self):
        """Cosine annealing with warm restarts."""
        # Calculate current period
        period_step = self.step_count - self.last_restart
        current_T_max = self.T_max * (self.T_mult ** len([x for x in range(self.step_count) if x % self.T_max == 0]))
        
        if period_step >= current_T_max:
            self.last_restart = self.step_count
            period_step = 0
            
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
             (1 + math.cos(math.pi * period_step / current_T_max)) / 2
        self._set_lr(lr)
    
    def _reduce_on_plateau_step(self, performance: float):
        """Reduce learning rate on plateau."""
        self.performance_history.append(performance)
        
        # Skip if in cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        if performance > self.best_performance + self.threshold:
            self.best_performance = performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * self.factor, self.min_lr)
            self._set_lr(new_lr)
            self.patience_counter = 0
            self.cooldown_counter = self.cooldown
    
    def _exponential_step(self):
        """Exponential decay learning rate schedule."""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * self.gamma, self.min_lr)
        self._set_lr(new_lr)
    
    def _performance_based_step(self, performance: float):
        """Performance-based adaptive learning rate."""
        if self.step_count % self.adaptation_frequency == 0:
            if len(self.performance_history) > 0:
                recent_performance = sum(self.performance_history[-10:]) / min(10, len(self.performance_history))
                
                if performance > recent_performance:
                    # Performance is improving, increase learning rate slightly
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = current_lr * self.lr_increase_factor
                    self._set_lr(new_lr)
                elif performance < recent_performance - self.threshold:
                    # Performance is degrading, decrease learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = current_lr * self.lr_decrease_factor
                    self._set_lr(new_lr)
        
        self.performance_history.append(performance)
    
    def _warmup_cosine_step(self):
        """Warmup followed by cosine annealing."""
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            adjusted_step = self.step_count - self.warmup_steps
            adjusted_T_max = self.T_max - self.warmup_steps
            lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * adjusted_step / adjusted_T_max)) / 2
        self._set_lr(lr)
    
    def _polynomial_step(self):
        """Polynomial decay learning rate schedule."""
        progress = min(self.step_count / self.max_steps, 1.0)
        lr = self.initial_lr * (1 - progress) ** self.power
        lr = max(lr, self.min_lr)
        self._set_lr(lr)
    
    def _cyclic_step(self):
        """Cyclic learning rate schedule."""
        cycle = math.floor(1 + self.step_count / (2 * self.step_size_up))
        x = abs(self.step_count / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** self.step_count
        else:
            scale_factor = 1.0
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * \
             max(0, (1 - x)) * scale_factor
        self._set_lr(lr)
    
    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for saving."""
        return {
            'step_count': self.step_count,
            'performance_history': self.performance_history,
            'best_performance': self.best_performance,
            'patience_counter': self.patience_counter,
            'initial_lr': self.initial_lr,
            'strategy': self.strategy,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary."""
        self.step_count = state_dict['step_count']
        self.performance_history = state_dict['performance_history']
        self.best_performance = state_dict['best_performance']
        self.patience_counter = state_dict['patience_counter']
        self.initial_lr = state_dict['initial_lr']


class AdaptiveOptimizerWrapper(BaseOptimizerWrapper):
    """
    Wrapper that combines optimizer with adaptive learning rate scheduling.
    
    Provides integrated optimization with:
    - Learning rate scheduling
    - Gradient clipping (standard and adaptive)
    - Performance tracking
    - Gradient statistics
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: AdaptiveLRScheduler,
                 grad_clip_norm: float = 1.0,
                 adaptive_grad_clip: bool = True,
                 gradient_accumulation_steps: int = 1):
        """
        Initialize adaptive optimizer wrapper.
        
        Args:
            optimizer: Base PyTorch optimizer
            lr_scheduler: Adaptive learning rate scheduler
            grad_clip_norm: Gradient clipping norm
            adaptive_grad_clip: Whether to use adaptive gradient clipping
            gradient_accumulation_steps: Steps to accumulate gradients
        """
        super().__init__(optimizer)
        self.lr_scheduler = lr_scheduler
        self.grad_clip_norm = grad_clip_norm
        self.adaptive_grad_clip = adaptive_grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Track gradient norms for adaptive clipping
        self.grad_norm_history = []
        self.max_history_length = 100
        
        # Gradient accumulation
        self.accumulation_step = 0
    
    def step(self, performance: Optional[float] = None, 
             accumulate_gradients: bool = False):
        """
        Perform optimization step with adaptive features.
        
        Args:
            performance: Current performance metric for LR scheduling
            accumulate_gradients: Whether to accumulate gradients
        """
        self.accumulation_step += 1
        
        # Only step if we've accumulated enough gradients or not accumulating
        if not accumulate_gradients or self.accumulation_step % self.gradient_accumulation_steps == 0:
            # Apply gradient clipping
            if self.adaptive_grad_clip:
                self._adaptive_grad_clip()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group['params']], 
                    self.grad_clip_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Learning rate scheduling
            self.lr_scheduler.step(performance)
            
            self.step_count += 1
            
            # Reset accumulation counter
            if accumulate_gradients:
                self.accumulation_step = 0
    
    def _adaptive_grad_clip(self):
        """Adaptive gradient clipping based on gradient norm history."""
        # Calculate current gradient norm
        total_norm = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Track gradient norm
        self.grad_norm_history.append(total_norm)
        if len(self.grad_norm_history) > self.max_history_length:
            self.grad_norm_history.pop(0)
        
        # Adaptive clipping threshold
        if len(self.grad_norm_history) > 10:
            import numpy as np
            mean_norm = np.mean(self.grad_norm_history)
            std_norm = np.std(self.grad_norm_history)
            adaptive_threshold = mean_norm + 2 * std_norm
            clip_norm = min(self.grad_clip_norm, adaptive_threshold)
        else:
            clip_norm = self.grad_clip_norm
        
        # Apply clipping
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group['params']], 
            clip_norm
        )
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr_scheduler.get_lr()
    
    def get_grad_norm(self) -> float:
        """Get current gradient norm."""
        if self.grad_norm_history:
            return self.grad_norm_history[-1]
        return 0.0
    
    def get_grad_statistics(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.grad_norm_history:
            return {}
        
        import numpy as np
        return {
            'mean_grad_norm': np.mean(self.grad_norm_history),
            'std_grad_norm': np.std(self.grad_norm_history),
            'max_grad_norm': np.max(self.grad_norm_history),
            'min_grad_norm': np.min(self.grad_norm_history),
            'current_grad_norm': self.grad_norm_history[-1],
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary."""
        base_state = super().state_dict()
        return {
            **base_state,
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'grad_norm_history': self.grad_norm_history,
            'accumulation_step': self.accumulation_step,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary."""
        super().load_state_dict(state_dict)
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.grad_norm_history = state_dict.get('grad_norm_history', [])
        self.accumulation_step = state_dict.get('accumulation_step', 0)


def create_adaptive_optimizer(optimizer: torch.optim.Optimizer,
                            scheduler_config: Dict[str, Any],
                            wrapper_config: Optional[Dict[str, Any]] = None) -> AdaptiveOptimizerWrapper:
    """
    Factory function to create adaptive optimizer with scheduler.
    
    Args:
        optimizer: Base PyTorch optimizer
        scheduler_config: Configuration for learning rate scheduler
        wrapper_config: Optional configuration for optimizer wrapper
        
    Returns:
        Adaptive optimizer wrapper
    """
    # Create scheduler
    lr_scheduler = AdaptiveLRScheduler(optimizer, **scheduler_config)
    
    # Create wrapper
    wrapper_config = wrapper_config or {}
    return AdaptiveOptimizerWrapper(optimizer, lr_scheduler, **wrapper_config)


# Predefined configurations for common use cases
SCHEDULER_CONFIGS = {
    'cosine_annealing': {
        'strategy': 'cosine_annealing',
        'T_max': 10000,
        'eta_min': 1e-8,
    },
    'reduce_on_plateau': {
        'strategy': 'reduce_on_plateau',
        'patience': 100,
        'factor': 0.8,
        'threshold': 1e-4,
    },
    'warmup_cosine': {
        'strategy': 'warmup_cosine',
        'warmup_steps': 1000,
        'T_max': 10000,
        'eta_min': 1e-8,
    },
    'cyclic': {
        'strategy': 'cyclic',
        'step_size_up': 2000,
        'mode': 'triangular2',
    },
}