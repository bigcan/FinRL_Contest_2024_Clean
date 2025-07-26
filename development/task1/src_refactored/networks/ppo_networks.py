"""
PPO (Proximal Policy Optimization) network architectures for the FinRL Contest 2024 framework.

This module provides PPO-specific network implementations including:
- Discrete action actor networks
- Value/advantage critic networks  
- Combined actor-critic architectures
- Support for shared and separate network designs
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from torch import Tensor

from .base_networks import build_mlp, layer_init_with_orthogonal

TEN = Tensor


class ActorDiscretePPO(nn.Module):
    """
    Discrete action actor network for PPO.
    
    Outputs action probabilities for discrete action spaces using softmax policy.
    Features:
    - State normalization for stable learning
    - Configurable network architecture
    - Numerical stability for probability calculations
    - Small initialization for stable policy gradients
    """
    
    def __init__(self, dims: List[int], state_dim: int, action_dim: int,
                 activation: nn.Module = nn.ReLU, dropout_rate: float = 0.0,
                 init_std: float = 0.01):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.init_std = init_std
        
        # State normalization parameters (running statistics)
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        
        # Policy network
        self.net = build_mlp(
            dims=[state_dim, *dims, action_dim],
            activation=activation,
            dropout_rate=dropout_rate,
            if_raw_out=True  # No activation on output layer
        )
        
        # Initialize output layer with small weights for stable policy learning
        layer_init_with_orthogonal(self.net[-1], std=init_std)
        
        # Softmax for probability computation
        self.softmax = nn.Softmax(dim=-1)
    
    def state_norm(self, state: TEN) -> TEN:
        """
        Normalize state using running statistics.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Normalized state tensor
        """
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / (state_std + 1e-8)
    
    def forward(self, state: TEN) -> TEN:
        """
        Forward pass returning action probabilities.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        state = self.state_norm(state)
        logits = self.net(state)
        action_probs = self.softmax(logits)
        
        # Add small epsilon to prevent log(0) and ensure probability sum is 1
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        return action_probs
    
    def get_action_log_prob(self, state: TEN, action: TEN) -> TEN:
        """
        Get log probability of given actions.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            action: Action indices [batch_size, 1]
            
        Returns:
            Log probabilities [batch_size, 1]
        """
        action_probs = self.forward(state)
        action_log_probs = torch.log(action_probs)
        return action_log_probs.gather(1, action.long())
    
    def get_entropy(self, state: TEN) -> TEN:
        """
        Calculate policy entropy for exploration bonus.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Entropy values [batch_size]
        """
        action_probs = self.forward(state)
        entropy = -(action_probs * torch.log(action_probs)).sum(dim=-1)
        return entropy
    
    def sample_action(self, state: TEN) -> Tuple[TEN, TEN]:
        """
        Sample action from policy and return action with log probability.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (actions, log_probs)
        """
        action_probs = self.forward(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.unsqueeze(-1), log_prob.unsqueeze(-1)
    
    def update_normalization_stats(self, states: TEN, tau: float = 0.01):
        """
        Update running normalization statistics.
        
        Args:
            states: Batch of states for updating statistics
            tau: Update rate (EMA coefficient)
        """
        with torch.no_grad():
            batch_state_mean = states.mean(dim=0)
            batch_state_std = states.std(dim=0)
            
            self.state_avg.data = (1 - tau) * self.state_avg.data + tau * batch_state_mean
            self.state_std.data = (1 - tau) * self.state_std.data + tau * batch_state_std + 1e-4


class CriticAdv(nn.Module):
    """
    Advantage critic network for PPO.
    
    Estimates state values V(s) for advantage computation:
    Advantage A(s,a) = Q(s,a) - V(s) ≈ R + γV(s') - V(s)
    
    Features:
    - State and value normalization
    - Configurable network architecture
    - Proper initialization for value learning
    """
    
    def __init__(self, dims: List[int], state_dim: int, output_dim: int = 1,
                 activation: nn.Module = nn.ReLU, dropout_rate: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        # State normalization parameters (running statistics)
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        
        # Value normalization parameters
        self.value_avg = nn.Parameter(torch.zeros((output_dim,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((output_dim,)), requires_grad=False)
        
        # Value network
        self.net = build_mlp(
            dims=[state_dim, *dims, output_dim],
            activation=activation,
            dropout_rate=dropout_rate,
            if_raw_out=True
        )
        
        # Initialize output layer for value learning
        layer_init_with_orthogonal(self.net[-1], std=1.0)
    
    def state_norm(self, state: TEN) -> TEN:
        """
        Normalize state using running statistics.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Normalized state tensor
        """
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / (state_std + 1e-8)
    
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
    
    def forward(self, state: TEN) -> TEN:
        """
        Forward pass returning state values.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            State values [batch_size, output_dim]
        """
        state = self.state_norm(state)
        value = self.net(state)
        value = self.value_re_norm(value)
        return value
    
    def update_normalization_stats(self, states: TEN, values: TEN, tau: float = 0.01):
        """
        Update running normalization statistics.
        
        Args:
            states: Batch of states for updating statistics
            values: Batch of values for updating statistics
            tau: Update rate (EMA coefficient)
        """
        with torch.no_grad():
            # Update state statistics
            batch_state_mean = states.mean(dim=0)
            batch_state_std = states.std(dim=0)
            
            self.state_avg.data = (1 - tau) * self.state_avg.data + tau * batch_state_mean
            self.state_std.data = (1 - tau) * self.state_std.data + tau * batch_state_std + 1e-4
            
            # Update value statistics
            batch_value_mean = values.mean(dim=0)
            batch_value_std = values.std(dim=0)
            
            self.value_avg.data = (1 - tau) * self.value_avg.data + tau * batch_value_mean
            self.value_std.data = (1 - tau) * self.value_std.data + tau * batch_value_std + 1e-4


class ActorCriticPPO(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    Shares feature extraction between policy and value networks for efficiency:
    - Shared trunk for feature extraction
    - Separate heads for policy and value
    - Reduced parameter count and faster training
    
    Features:
    - Shared feature extraction
    - Separate policy and value heads
    - Coordinated normalization
    - Efficient computation
    """
    
    def __init__(self, dims: List[int], state_dim: int, action_dim: int,
                 activation: nn.Module = nn.ReLU, dropout_rate: float = 0.0,
                 policy_init_std: float = 0.01, value_init_std: float = 1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State normalization parameters (shared)
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        
        # Shared feature extraction network
        if len(dims) < 2:
            raise ValueError("dims must have at least 2 elements for shared network")
        
        self.shared_net = build_mlp(
            dims=[state_dim, *dims[:-1]],
            activation=activation,
            dropout_rate=dropout_rate,
            if_raw_out=False  # Keep activation for shared features
        )
        
        # Policy head (actor)
        self.policy_head = nn.Linear(dims[-2], action_dim)
        layer_init_with_orthogonal(self.policy_head, std=policy_init_std)
        
        # Value head (critic)
        self.value_head = nn.Linear(dims[-2], 1)
        layer_init_with_orthogonal(self.value_head, std=value_init_std)
        
        # Softmax for policy probabilities
        self.softmax = nn.Softmax(dim=-1)
    
    def state_norm(self, state: TEN) -> TEN:
        """
        Normalize state using running statistics.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Normalized state tensor
        """
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / (state_std + 1e-8)
    
    def forward(self, state: TEN) -> Tuple[TEN, TEN]:
        """
        Forward pass returning both action probabilities and state values.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (action_probs, state_values)
        """
        state = self.state_norm(state)
        features = self.shared_net(state)
        
        # Policy output
        policy_logits = self.policy_head(features)
        action_probs = self.softmax(policy_logits)
        action_probs = action_probs + 1e-8  # Prevent log(0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        # Value output
        state_values = self.value_head(features)
        
        return action_probs, state_values
    
    def get_action_probs(self, state: TEN) -> TEN:
        """
        Get only action probabilities (policy evaluation).
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        action_probs, _ = self.forward(state)
        return action_probs
    
    def get_state_values(self, state: TEN) -> TEN:
        """
        Get only state values (value evaluation).
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            State values [batch_size, 1]
        """
        _, state_values = self.forward(state)
        return state_values
    
    def get_action_log_prob_and_value(self, state: TEN, action: TEN) -> Tuple[TEN, TEN, TEN]:
        """
        Get action log probabilities, entropy, and values efficiently.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            action: Action indices [batch_size, 1]
            
        Returns:
            Tuple of (log_probs, entropy, values)
        """
        action_probs, state_values = self.forward(state)
        
        # Log probabilities of taken actions
        action_log_probs = torch.log(action_probs).gather(1, action.long())
        
        # Policy entropy
        entropy = -(action_probs * torch.log(action_probs)).sum(dim=-1, keepdim=True)
        
        return action_log_probs, entropy, state_values
    
    def sample_action(self, state: TEN) -> Tuple[TEN, TEN, TEN]:
        """
        Sample action and return action, log probability, and value.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (actions, log_probs, values)
        """
        action_probs, state_values = self.forward(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.unsqueeze(-1), log_prob.unsqueeze(-1), state_values
    
    def update_normalization_stats(self, states: TEN, tau: float = 0.01):
        """
        Update running state normalization statistics.
        
        Args:
            states: Batch of states for updating statistics
            tau: Update rate (EMA coefficient)
        """
        with torch.no_grad():
            batch_state_mean = states.mean(dim=0)
            batch_state_std = states.std(dim=0)
            
            self.state_avg.data = (1 - tau) * self.state_avg.data + tau * batch_state_mean
            self.state_std.data = (1 - tau) * self.state_std.data + tau * batch_state_std + 1e-4