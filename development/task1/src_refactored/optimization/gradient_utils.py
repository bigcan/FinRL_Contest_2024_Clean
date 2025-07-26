"""
Gradient utilities and manipulation functions for the FinRL Contest 2024 framework.

This module provides utilities for gradient processing including:
- Various gradient clipping strategies
- Gradient noise injection
- Gradient accumulation
- Gradient analysis and debugging
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union, Iterable
import math
import numpy as np


class GradientClipper:
    """
    Advanced gradient clipping with multiple strategies.
    
    Supports various clipping methods to prevent gradient explosion
    and improve training stability.
    """
    
    def __init__(self, method: str = "norm", **kwargs):
        """
        Initialize gradient clipper.
        
        Args:
            method: Clipping method ('norm', 'value', 'adaptive', 'percentile')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.history = []
        self.max_history = kwargs.get('max_history', 1000)
        
        if method == "norm":
            self.max_norm = kwargs.get('max_norm', 1.0)
            self.norm_type = kwargs.get('norm_type', 2.0)
            
        elif method == "value":
            self.clip_value = kwargs.get('clip_value', 0.5)
            
        elif method == "adaptive":
            self.percentile = kwargs.get('percentile', 95.0)
            self.min_norm = kwargs.get('min_norm', 0.1)
            self.max_norm = kwargs.get('max_norm', 10.0)
            
        elif method == "percentile":
            self.percentile = kwargs.get('percentile', 95.0)
            self.window_size = kwargs.get('window_size', 100)
    
    def clip_gradients(self, parameters: Union[torch.Tensor, Iterable[torch.Tensor]]) -> float:
        """
        Clip gradients using selected method.
        
        Args:
            parameters: Model parameters or parameter tensors
            
        Returns:
            Total gradient norm before clipping
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        parameters = [p for p in parameters if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # Calculate gradient norm
        total_norm = self._calculate_grad_norm(parameters)
        self.history.append(total_norm)
        
        # Maintain history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Apply clipping based on method
        if self.method == "norm":
            self._clip_by_norm(parameters, total_norm)
        elif self.method == "value":
            self._clip_by_value(parameters)
        elif self.method == "adaptive":
            self._clip_adaptive(parameters, total_norm)
        elif self.method == "percentile":
            self._clip_by_percentile(parameters, total_norm)
        
        return total_norm
    
    def _calculate_grad_norm(self, parameters: List[torch.Tensor]) -> float:
        """Calculate total gradient norm."""
        if self.norm_type == float('inf'):
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0.0
            for p in parameters:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
            total_norm = total_norm ** (1. / self.norm_type)
        
        return total_norm
    
    def _clip_by_norm(self, parameters: List[torch.Tensor], total_norm: float):
        """Clip gradients by norm."""
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    
    def _clip_by_value(self, parameters: List[torch.Tensor]):
        """Clip gradients by value."""
        for p in parameters:
            p.grad.data.clamp_(-self.clip_value, self.clip_value)
    
    def _clip_adaptive(self, parameters: List[torch.Tensor], total_norm: float):
        """Adaptive gradient clipping based on gradient norm history."""
        if len(self.history) < 10:
            # Use fixed clipping until we have enough history
            self._clip_by_norm(parameters, total_norm)
            return
        
        # Calculate adaptive threshold
        threshold = np.percentile(self.history, self.percentile)
        threshold = max(self.min_norm, min(self.max_norm, threshold))
        
        # Apply clipping
        clip_coef = threshold / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    
    def _clip_by_percentile(self, parameters: List[torch.Tensor], total_norm: float):
        """Clip gradients using percentile of recent norms."""
        if len(self.history) < self.window_size:
            return  # No clipping until we have enough history
        
        recent_history = self.history[-self.window_size:]
        threshold = np.percentile(recent_history, self.percentile)
        
        clip_coef = threshold / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get gradient clipping statistics."""
        if not self.history:
            return {}
        
        return {
            'mean_grad_norm': np.mean(self.history),
            'std_grad_norm': np.std(self.history),
            'max_grad_norm': np.max(self.history),
            'min_grad_norm': np.min(self.history),
            'recent_grad_norm': self.history[-1] if self.history else 0.0,
            'clipping_frequency': sum(1 for norm in self.history[-100:] 
                                    if norm > getattr(self, 'max_norm', float('inf'))) / min(100, len(self.history)),
        }


class GradientNoiseInjector:
    """
    Inject noise into gradients for improved generalization.
    
    Implements gradient noise injection techniques that can help
    escape local minima and improve model robustness.
    """
    
    def __init__(self, noise_type: str = "gaussian", **kwargs):
        """
        Initialize gradient noise injector.
        
        Args:
            noise_type: Type of noise ('gaussian', 'uniform', 'annealed')
            **kwargs: Noise-specific parameters
        """
        self.noise_type = noise_type
        self.step_count = 0
        
        if noise_type == "gaussian":
            self.std = kwargs.get('std', 0.01)
            
        elif noise_type == "uniform":
            self.range = kwargs.get('range', 0.01)
            
        elif noise_type == "annealed":
            self.initial_std = kwargs.get('initial_std', 0.1)
            self.final_std = kwargs.get('final_std', 0.001)
            self.annealing_steps = kwargs.get('annealing_steps', 10000)
    
    def inject_noise(self, parameters: Union[torch.Tensor, Iterable[torch.Tensor]]):
        """
        Inject noise into gradients.
        
        Args:
            parameters: Model parameters or parameter tensors
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        parameters = [p for p in parameters if p.grad is not None]
        
        for p in parameters:
            if self.noise_type == "gaussian":
                noise = torch.randn_like(p.grad) * self.std
            elif self.noise_type == "uniform":
                noise = (torch.rand_like(p.grad) - 0.5) * 2 * self.range
            elif self.noise_type == "annealed":
                current_std = self._get_annealed_std()
                noise = torch.randn_like(p.grad) * current_std
            
            p.grad.data.add_(noise)
        
        self.step_count += 1
    
    def _get_annealed_std(self) -> float:
        """Calculate annealed standard deviation."""
        if self.step_count >= self.annealing_steps:
            return self.final_std
        
        progress = self.step_count / self.annealing_steps
        return self.initial_std * (1 - progress) + self.final_std * progress


class GradientAccumulator:
    """
    Accumulate gradients over multiple batches for large effective batch sizes.
    
    Useful when memory constraints prevent using large batch sizes directly.
    """
    
    def __init__(self, accumulation_steps: int, normalize: bool = True):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate over
            normalize: Whether to normalize accumulated gradients
        """
        self.accumulation_steps = accumulation_steps
        self.normalize = normalize
        self.current_step = 0
        
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def accumulate_step(self):
        """Increment accumulation step counter."""
        self.current_step = (self.current_step + 1) % self.accumulation_steps
    
    def scale_gradients(self, parameters: Union[torch.Tensor, Iterable[torch.Tensor]]):
        """
        Scale gradients for accumulation.
        
        Args:
            parameters: Model parameters
        """
        if not self.normalize:
            return
        
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        scale_factor = 1.0 / self.accumulation_steps
        
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale_factor)


class GradientAnalyzer:
    """
    Analyze gradients for debugging and monitoring purposes.
    
    Provides detailed statistics and diagnostics for gradient behavior.
    """
    
    def __init__(self, track_layers: bool = True):
        """
        Initialize gradient analyzer.
        
        Args:
            track_layers: Whether to track per-layer statistics
        """
        self.track_layers = track_layers
        self.history = []
        self.layer_history = {}
    
    def analyze_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze gradients of model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary of gradient statistics
        """
        stats = {
            'total_norm': 0.0,
            'total_params': 0,
            'params_with_grad': 0,
            'zero_grad_params': 0,
            'inf_grad_params': 0,
            'nan_grad_params': 0,
            'layer_stats': {},
        }
        
        total_norm_squared = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Parameter-level statistics
                param_norm = grad.norm().item()
                total_norm_squared += param_norm ** 2
                stats['params_with_grad'] += 1
                
                # Check for problematic gradients
                if torch.isnan(grad).any():
                    stats['nan_grad_params'] += 1
                if torch.isinf(grad).any():
                    stats['inf_grad_params'] += 1
                if (grad == 0).all():
                    stats['zero_grad_params'] += 1
                
                # Layer-level statistics
                if self.track_layers:
                    layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else 'root'
                    if layer_name not in stats['layer_stats']:
                        stats['layer_stats'][layer_name] = {
                            'norm': 0.0,
                            'mean': 0.0,
                            'std': 0.0,
                            'min': float('inf'),
                            'max': float('-inf'),
                            'param_count': 0,
                        }
                    
                    layer_stats = stats['layer_stats'][layer_name]
                    layer_stats['norm'] += param_norm ** 2
                    layer_stats['mean'] += grad.mean().item()
                    layer_stats['std'] += grad.std().item()
                    layer_stats['min'] = min(layer_stats['min'], grad.min().item())
                    layer_stats['max'] = max(layer_stats['max'], grad.max().item())
                    layer_stats['param_count'] += param.numel()
            
            stats['total_params'] += param.numel()
        
        # Finalize statistics
        stats['total_norm'] = math.sqrt(total_norm_squared)
        
        # Finalize layer statistics
        for layer_name, layer_stats in stats['layer_stats'].items():
            layer_stats['norm'] = math.sqrt(layer_stats['norm'])
        
        # Store in history
        self.history.append(stats)
        
        return stats
    
    def get_gradient_flow_info(self, model: nn.Module) -> Dict[str, List[float]]:
        """
        Get gradient flow information for visualization.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary mapping layer names to gradient norms
        """
        gradient_flow = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else 'root'
                if layer_name not in gradient_flow:
                    gradient_flow[layer_name] = []
                
                gradient_flow[layer_name].append(param.grad.data.norm().item())
        
        return gradient_flow
    
    def detect_vanishing_gradients(self, threshold: float = 1e-6) -> List[str]:
        """
        Detect layers with vanishing gradients.
        
        Args:
            threshold: Threshold below which gradients are considered vanishing
            
        Returns:
            List of layer names with vanishing gradients
        """
        if not self.history:
            return []
        
        latest_stats = self.history[-1]
        vanishing_layers = []
        
        for layer_name, layer_stats in latest_stats['layer_stats'].items():
            if layer_stats['norm'] < threshold:
                vanishing_layers.append(layer_name)
        
        return vanishing_layers
    
    def detect_exploding_gradients(self, threshold: float = 10.0) -> List[str]:
        """
        Detect layers with exploding gradients.
        
        Args:
            threshold: Threshold above which gradients are considered exploding
            
        Returns:
            List of layer names with exploding gradients
        """
        if not self.history:
            return []
        
        latest_stats = self.history[-1]
        exploding_layers = []
        
        for layer_name, layer_stats in latest_stats['layer_stats'].items():
            if layer_stats['norm'] > threshold:
                exploding_layers.append(layer_name)
        
        return exploding_layers


def apply_gradient_modification(parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
                              clipper: Optional[GradientClipper] = None,
                              noise_injector: Optional[GradientNoiseInjector] = None,
                              accumulator: Optional[GradientAccumulator] = None) -> float:
    """
    Apply multiple gradient modifications in sequence.
    
    Args:
        parameters: Model parameters
        clipper: Optional gradient clipper
        noise_injector: Optional noise injector
        accumulator: Optional gradient accumulator
        
    Returns:
        Total gradient norm before modifications
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    total_norm = 0.0
    
    # Calculate initial norm if clipper is provided
    if clipper:
        total_norm = clipper.clip_gradients(parameters)
    
    # Inject noise if provided
    if noise_injector:
        noise_injector.inject_noise(parameters)
    
    # Scale for accumulation if provided
    if accumulator:
        accumulator.scale_gradients(parameters)
    
    return total_norm