import torch
import torch.optim as optim
from typing import Optional, Dict, Any
import math


class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler with multiple strategies
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 strategy: str = "cosine_annealing",
                 **kwargs):
        """
        Initialize adaptive learning rate scheduler
        
        Args:
            optimizer: PyTorch optimizer
            strategy: Scheduling strategy ('cosine_annealing', 'reduce_on_plateau', 'exponential', 'performance_based')
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
        
        # Strategy-specific initialization
        if strategy == "cosine_annealing":
            self.T_max = kwargs.get('T_max', 10000)
            self.eta_min = kwargs.get('eta_min', 1e-8)
            
        elif strategy == "reduce_on_plateau":
            self.patience = kwargs.get('patience', 100)
            self.factor = kwargs.get('factor', 0.5)
            self.threshold = kwargs.get('threshold', 1e-4)
            self.min_lr = kwargs.get('min_lr', 1e-8)
            
        elif strategy == "exponential":
            self.gamma = kwargs.get('gamma', 0.9999)
            self.min_lr = kwargs.get('min_lr', 1e-8)
            
        elif strategy == "performance_based":
            self.target_performance = kwargs.get('target_performance', 0.1)
            self.lr_increase_factor = kwargs.get('lr_increase_factor', 1.05)
            self.lr_decrease_factor = kwargs.get('lr_decrease_factor', 0.95)
            self.adaptation_frequency = kwargs.get('adaptation_frequency', 50)
            
        elif strategy == "warmup_cosine":
            self.warmup_steps = kwargs.get('warmup_steps', 1000)
            self.T_max = kwargs.get('T_max', 10000)
            self.eta_min = kwargs.get('eta_min', 1e-8)
            
    def step(self, performance: Optional[float] = None):
        """
        Update learning rate based on strategy
        
        Args:
            performance: Current performance metric (higher is better)
        """
        self.step_count += 1
        
        if self.strategy == "cosine_annealing":
            self._cosine_annealing_step()
            
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
    
    def _cosine_annealing_step(self):
        """Cosine annealing learning rate schedule"""
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.step_count / self.T_max)) / 2
        self._set_lr(lr)
    
    def _reduce_on_plateau_step(self, performance: float):
        """Reduce learning rate on plateau"""
        self.performance_history.append(performance)
        
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
    
    def _exponential_step(self):
        """Exponential decay learning rate schedule"""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * self.gamma, self.min_lr)
        self._set_lr(new_lr)
    
    def _performance_based_step(self, performance: float):
        """Performance-based adaptive learning rate"""
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
        """Warmup followed by cosine annealing"""
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
    
    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for saving"""
        return {
            'step_count': self.step_count,
            'performance_history': self.performance_history,
            'best_performance': self.best_performance,
            'patience_counter': self.patience_counter,
            'initial_lr': self.initial_lr
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary"""
        self.step_count = state_dict['step_count']
        self.performance_history = state_dict['performance_history']
        self.best_performance = state_dict['best_performance']
        self.patience_counter = state_dict['patience_counter']
        self.initial_lr = state_dict['initial_lr']


class AdaptiveOptimizerWrapper:
    """
    Wrapper that combines optimizer with adaptive learning rate scheduling
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: AdaptiveLRScheduler,
                 grad_clip_norm: float = 1.0,
                 adaptive_grad_clip: bool = True):
        """
        Initialize adaptive optimizer wrapper
        
        Args:
            optimizer: Base PyTorch optimizer
            lr_scheduler: Adaptive learning rate scheduler
            grad_clip_norm: Gradient clipping norm
            adaptive_grad_clip: Whether to use adaptive gradient clipping
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_clip_norm = grad_clip_norm
        self.adaptive_grad_clip = adaptive_grad_clip
        
        # Track gradient norms for adaptive clipping
        self.grad_norm_history = []
        self.max_history_length = 100
        
    def step(self, performance: Optional[float] = None):
        """
        Perform optimization step with adaptive features
        
        Args:
            performance: Current performance metric for LR scheduling
        """
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
    
    def _adaptive_grad_clip(self):
        """Adaptive gradient clipping based on gradient norm history"""
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
            mean_norm = sum(self.grad_norm_history) / len(self.grad_norm_history)
            std_norm = (sum((x - mean_norm) ** 2 for x in self.grad_norm_history) / len(self.grad_norm_history)) ** 0.5
            adaptive_threshold = mean_norm + 2 * std_norm
            clip_norm = min(self.grad_clip_norm, adaptive_threshold)
        else:
            clip_norm = self.grad_clip_norm
        
        # Apply clipping
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group['params']], 
            clip_norm
        )
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.lr_scheduler.get_lr()
    
    def get_grad_norm(self) -> float:
        """Get current gradient norm"""
        if self.grad_norm_history:
            return self.grad_norm_history[-1]
        return 0.0
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'grad_norm_history': self.grad_norm_history
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.grad_norm_history = state_dict.get('grad_norm_history', [])