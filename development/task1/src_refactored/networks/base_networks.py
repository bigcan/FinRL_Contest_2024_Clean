"""
Base network components and utilities for the FinRL Contest 2024 framework.

This module provides foundational network classes and utility functions
that are shared across different network architectures.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from torch import Tensor

TEN = Tensor


class QNetBase(nn.Module):
    """
    Base class for Q-networks with state/value normalization.
    
    Provides common functionality for:
    - State normalization/denormalization
    - Value normalization/denormalization  
    - Exploration rate management
    - Device-aware parameter management
    """
    
    def __init__(self, state_dim: int, action_dim: int, explore_rate: float = 0.125):
        super().__init__()
        self.explore_rate = explore_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State normalization parameters (running statistics)
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        
        # Value normalization parameters (for target scaling)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)
    
    def state_norm(self, state: TEN) -> TEN:
        """
        Normalize state using running statistics.
        
        Args:
            state: Input state tensor
            
        Returns:
            Normalized state tensor
        """
        # Ensure normalization parameters are on same device as input
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / (state_std + 1e-8)  # Add epsilon for numerical stability
    
    def value_re_norm(self, value: TEN) -> TEN:
        """
        Denormalize value using stored statistics.
        
        Args:
            value: Normalized value tensor
            
        Returns:
            Denormalized value tensor
        """
        value_avg = self.value_avg.to(value.device)
        value_std = self.value_std.to(value.device)
        return value * value_std + value_avg
    
    def update_normalization_stats(self, states: TEN, values: Optional[TEN] = None, tau: float = 0.01):
        """
        Update running normalization statistics.
        
        Args:
            states: Batch of states for updating state statistics
            values: Batch of values for updating value statistics (optional)
            tau: Update rate (EMA coefficient)
        """
        with torch.no_grad():
            # Update state statistics
            batch_state_mean = states.mean(dim=0)
            batch_state_std = states.std(dim=0)
            
            self.state_avg.data = (1 - tau) * self.state_avg.data + tau * batch_state_mean
            self.state_std.data = (1 - tau) * self.state_std.data + tau * batch_state_std + 1e-4
            
            # Update value statistics if provided
            if values is not None:
                batch_value_mean = values.mean()
                batch_value_std = values.std()
                
                self.value_avg.data = (1 - tau) * self.value_avg.data + tau * batch_value_mean
                self.value_std.data = (1 - tau) * self.value_std.data + tau * batch_value_std + 1e-4


def build_mlp(dims: List[int], 
              activation: Optional[nn.Module] = None, 
              if_raw_out: bool = True,
              dropout_rate: float = 0.0,
              batch_norm: bool = False) -> nn.Sequential:
    """
    Build Multi-Layer Perceptron (MLP) network.
    
    Args:
        dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function class (default: ReLU)
        if_raw_out: If True, remove activation from output layer
        dropout_rate: Dropout rate (0.0 = no dropout)
        batch_norm: Whether to use batch normalization
        
    Returns:
        Sequential network
    """
    if activation is None:
        activation = nn.ReLU
    
    if len(dims) < 2:
        raise ValueError("dims must contain at least 2 elements (input and output dimensions)")
    
    layers = []
    
    for i in range(len(dims) - 1):
        # Linear layer
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        # Skip activation/normalization/dropout for output layer if if_raw_out is True
        if i == len(dims) - 2 and if_raw_out:
            break
            
        # Batch normalization (before activation)
        if batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        
        # Activation function
        layers.append(activation())
        
        # Dropout (after activation)
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
    
    return nn.Sequential(*layers)


def layer_init_with_orthogonal(layer: nn.Module, std: float = 1.0, bias_const: float = 1e-6):
    """
    Initialize layer weights with orthogonal initialization.
    
    Args:
        layer: Neural network layer to initialize
        std: Standard deviation for orthogonal initialization
        bias_const: Constant value for bias initialization
    """
    if hasattr(layer, 'weight'):
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)


def layer_init_with_xavier(layer: nn.Module, gain: float = 1.0):
    """
    Initialize layer weights with Xavier/Glorot initialization.
    
    Args:
        layer: Neural network layer to initialize
        gain: Scaling factor for Xavier initialization
    """
    if hasattr(layer, 'weight'):
        torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0.0)


def layer_init_with_kaiming(layer: nn.Module, mode: str = 'fan_in', nonlinearity: str = 'relu'):
    """
    Initialize layer weights with Kaiming/He initialization.
    
    Args:
        layer: Neural network layer to initialize
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', etc.)
    """
    if hasattr(layer, 'weight'):
        torch.nn.init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0.0)


class NetworkUtils:
    """Utility class for common network operations."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total number of trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def freeze_parameters(model: nn.Module):
        """Freeze all parameters in model."""
        for param in model.parameters():
            param.requires_grad = False
    
    @staticmethod
    def unfreeze_parameters(model: nn.Module):
        """Unfreeze all parameters in model."""
        for param in model.parameters():
            param.requires_grad = True
    
    @staticmethod
    def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
        """
        Soft update target network parameters.
        
        Args:
            target_net: Target network to update
            source_net: Source network to copy from
            tau: Update rate (0 = no update, 1 = full copy)
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )
    
    @staticmethod
    def hard_update(target_net: nn.Module, source_net: nn.Module):
        """
        Hard update (full copy) target network parameters.
        
        Args:
            target_net: Target network to update
            source_net: Source network to copy from
        """
        target_net.load_state_dict(source_net.state_dict())
    
    @staticmethod
    def get_device_from_parameters(model: nn.Module) -> torch.device:
        """Get device of model parameters."""
        return next(model.parameters()).device
    
    @staticmethod
    def move_to_device(model: nn.Module, device: torch.device) -> nn.Module:
        """Move model to specified device."""
        return model.to(device)