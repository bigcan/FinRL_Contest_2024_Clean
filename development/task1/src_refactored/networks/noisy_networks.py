"""
Noisy network architectures for the FinRL Contest 2024 framework.

This module provides NoisyNet implementations that replace epsilon-greedy
exploration with learned parameter noise. Includes:
- NoisyLinear layer with factorized Gaussian noise
- Noisy Twin Q-networks
- Noisy Dueling Q-networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List
from torch import Tensor

from .base_networks import QNetBase, build_mlp, layer_init_with_orthogonal

TEN = Tensor


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer implementing NoisyNet for parameter space exploration.
    
    Replaces epsilon-greedy exploration with learned parameter noise:
    - Factorized Gaussian noise for efficiency
    - Separate noise for weights and biases
    - Noise scales are learned parameters
    
    Reference: "Noisy Networks for Exploration" (Fortunato et al., 2017)
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable mean parameters for weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        
        # Learnable mean parameters for biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Buffers for noise (not learnable, reset each forward pass)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize learnable parameters using uniform distribution."""
        mu_range = 1 / math.sqrt(self.in_features)
        
        # Initialize means with uniform distribution
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        # Initialize noise scales
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Generate new noise for weights and biases using factorized approach."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Use outer product for factorized noise (more efficient than independent noise)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> Tensor:
        """
        Create scaled noise using factorized Gaussian approach.
        
        Args:
            size: Dimension of noise vector
            
        Returns:
            Scaled noise tensor
        """
        x = torch.randn(size, device=self.weight_mu.device)
        # Apply sign(x) * sqrt(|x|) transformation for better noise properties
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass with noisy parameters.
        
        Args:
            input: Input tensor [batch_size, in_features]
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        if self.training:
            # Use noisy parameters during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use mean parameters during evaluation (no noise)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class QNetTwinNoisy(QNetBase):
    """
    Twin Q-Network with Noisy Linear layers for exploration.
    
    Replaces epsilon-greedy exploration with parameter noise:
    - Regular linear layers for feature extraction
    - Noisy linear layers for final Q-value computation
    - Automatic noise reset for exploration
    """
    
    def __init__(self, dims: List[int], state_dim: int, action_dim: int,
                 noise_std_init: float = 0.5, activation: nn.Module = nn.ReLU):
        super().__init__(state_dim=state_dim, action_dim=action_dim, explore_rate=0.0)  # No epsilon needed
        
        self.noise_std_init = noise_std_init
        
        # Build networks with noisy output layers
        self.net1 = self._build_noisy_network(dims, state_dim, action_dim, activation)
        self.net2 = self._build_noisy_network(dims, state_dim, action_dim, activation)
    
    def _build_noisy_network(self, dims: List[int], state_dim: int, 
                           action_dim: int, activation: nn.Module) -> nn.Sequential:
        """
        Build network with noisy output layer.
        
        Args:
            dims: Hidden layer dimensions
            state_dim: Input state dimension
            action_dim: Output action dimension
            activation: Activation function
            
        Returns:
            Sequential network with noisy output
        """
        layers = []
        
        # Input layer (regular linear)
        layers.append(nn.Linear(state_dim, dims[0]))
        layers.append(activation())
        
        # Hidden layers (regular linear)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        
        # Output layer (noisy for exploration)
        layers.append(NoisyLinear(dims[-1], action_dim, std_init=self.noise_std_init))
        
        return nn.Sequential(*layers)
    
    def forward(self, state: TEN) -> TEN:
        """
        Forward pass returning Q-values from first noisy network.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Q-values from first network [batch_size, action_dim]
        """
        state = self.state_norm(state)
        return self.net1(state)
    
    def get_q1_q2(self, state: TEN) -> Tuple[TEN, TEN]:
        """
        Get Q-values from both noisy networks.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (q1_values, q2_values) each [batch_size, action_dim]
        """
        state = self.state_norm(state)
        q1_values = self.net1(state)
        q2_values = self.net2(state)
        return q1_values, q2_values
    
    def get_action(self, state: TEN) -> TEN:
        """
        Get action using noisy networks (no epsilon-greedy needed).
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Selected actions [batch_size, 1]
        """
        q1_values, q2_values = self.get_q1_q2(state)
        # Use minimum Q-values for action selection (conservative)
        q_values = torch.min(q1_values, q2_values)
        action = q_values.argmax(dim=1, keepdim=True)
        return action
    
    def reset_noise(self):
        """Reset noise in all noisy layers of both networks."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class QNetTwinDuelNoisy(QNetBase):
    """
    Twin Dueling Q-Network with Noisy Linear layers.
    
    Combines Dueling architecture with NoisyNet exploration:
    - Shared feature extraction (regular layers)
    - Separate value and advantage streams
    - Noisy output layers for exploration
    - Dueling aggregation formula
    """
    
    def __init__(self, dims: List[int], state_dim: int, action_dim: int,
                 noise_std_init: float = 0.5, activation: nn.Module = nn.ReLU):
        super().__init__(state_dim=state_dim, action_dim=action_dim, explore_rate=0.0)  # No epsilon needed
        
        self.noise_std_init = noise_std_init
        
        # Shared feature extractor (regular layers)
        self.feature_net = self._build_feature_network(dims, state_dim, activation)
        
        # Dueling streams for first network
        self.advantage1_net = self._build_advantage_network(dims[-1], action_dim)
        self.value1_net = self._build_value_network(dims[-1])
        
        # Dueling streams for second network
        self.advantage2_net = self._build_advantage_network(dims[-1], action_dim)
        self.value2_net = self._build_value_network(dims[-1])
    
    def _build_feature_network(self, dims: List[int], state_dim: int, 
                             activation: nn.Module) -> nn.Sequential:
        """Build shared feature extraction network (regular layers)."""
        layers = []
        layers.append(nn.Linear(state_dim, dims[0]))
        layers.append(activation())
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        
        return nn.Sequential(*layers)
    
    def _build_advantage_network(self, feature_dim: int, action_dim: int) -> nn.Sequential:
        """Build advantage stream with noisy output layer."""
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            NoisyLinear(feature_dim // 2, action_dim, std_init=self.noise_std_init)
        )
    
    def _build_value_network(self, feature_dim: int) -> nn.Sequential:
        """Build value stream (regular layers since it's not action-dependent)."""
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def _dueling_forward(self, state: TEN, advantage_net: nn.Sequential, 
                        value_net: nn.Sequential) -> TEN:
        """
        Forward pass through dueling architecture.
        
        Args:
            state: Normalized input state
            advantage_net: Advantage stream network
            value_net: Value stream network
            
        Returns:
            Q-values from dueling combination
        """
        features = self.feature_net(state)
        
        advantage = advantage_net(features)  # [batch_size, action_dim]
        value = value_net(features)         # [batch_size, 1]
        
        # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
    def forward(self, state: TEN) -> TEN:
        """
        Forward pass returning Q-values from first dueling network.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Q-values from first network [batch_size, action_dim]
        """
        state = self.state_norm(state)
        return self._dueling_forward(state, self.advantage1_net, self.value1_net)
    
    def get_q1_q2(self, state: TEN) -> Tuple[TEN, TEN]:
        """
        Get Q-values from both noisy dueling networks.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (q1_values, q2_values) each [batch_size, action_dim]
        """
        state = self.state_norm(state)
        
        q1_values = self._dueling_forward(state, self.advantage1_net, self.value1_net)
        q2_values = self._dueling_forward(state, self.advantage2_net, self.value2_net)
        
        return q1_values, q2_values
    
    def get_action(self, state: TEN) -> TEN:
        """
        Get action using noisy dueling networks.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Selected actions [batch_size, 1]
        """
        q1_values, q2_values = self.get_q1_q2(state)
        # Use minimum Q-values for conservative action selection
        q_values = torch.min(q1_values, q2_values)
        action = q_values.argmax(dim=1, keepdim=True)
        return action
    
    def reset_noise(self):
        """Reset noise in all noisy layers of both networks."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def get_value_advantage_estimates(self, state: TEN) -> Tuple[TEN, TEN, TEN, TEN]:
        """
        Get separate value and advantage estimates from both networks.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (value1, advantage1, value2, advantage2)
        """
        state = self.state_norm(state)
        features = self.feature_net(state)
        
        value1 = self.value1_net(features)
        advantage1 = self.advantage1_net(features)
        value2 = self.value2_net(features)
        advantage2 = self.advantage2_net(features)
        
        return value1, advantage1, value2, advantage2