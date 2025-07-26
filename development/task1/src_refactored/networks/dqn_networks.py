"""
DQN network architectures for the FinRL Contest 2024 framework.

This module provides various DQN network implementations including:
- Standard Double DQN (Twin Q-networks)
- Dueling DQN with advantage and value streams
- Support for both discrete action spaces
"""

import torch
import torch.nn as nn
from typing import Tuple, List
from torch import Tensor

from .base_networks import QNetBase, build_mlp, layer_init_with_orthogonal

TEN = Tensor


class QNetTwin(QNetBase):
    """
    Twin Q-Network for Double DQN implementation.
    
    Uses two separate Q-networks to reduce overestimation bias:
    - Network 1: Used for action selection and training
    - Network 2: Used for evaluation in Double DQN
    
    Features:
    - Shared state encoder
    - Separate Q-value heads
    - Epsilon-greedy exploration
    - State and value normalization
    """
    
    def __init__(self, dims: List[int], state_dim: int, action_dim: int, 
                 activation: nn.Module = nn.ReLU, dropout_rate: float = 0.0):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        
        # Shared state encoder
        self.net_state = build_mlp(
            dims=[state_dim, *dims], 
            activation=activation,
            dropout_rate=dropout_rate,
            if_raw_out=False
        )
        
        # Twin Q-value heads
        self.net_q1 = build_mlp(dims=[dims[-1], action_dim], if_raw_out=True)
        self.net_q2 = build_mlp(dims=[dims[-1], action_dim], if_raw_out=True)
        
        # Softmax for action probability (used in some exploration strategies)
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize output layers with smaller std for stability
        layer_init_with_orthogonal(self.net_q1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_q2[-1], std=0.1)
    
    def forward(self, state: TEN) -> TEN:
        """
        Forward pass returning Q-values from first network.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Q-values from first network [batch_size, action_dim]
        """
        state = self.state_norm(state)
        state_encoding = self.net_state(state)
        q_values = self.net_q1(state_encoding)
        return q_values
    
    def get_q1_q2(self, state: TEN) -> Tuple[TEN, TEN]:
        """
        Get Q-values from both networks.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (q1_values, q2_values) each [batch_size, action_dim]
        """
        state = self.state_norm(state)
        state_encoding = self.net_state(state)
        
        q1_values = self.net_q1(state_encoding)
        q2_values = self.net_q2(state_encoding)
        
        # Apply value denormalization
        q1_values = self.value_re_norm(q1_values)
        q2_values = self.value_re_norm(q2_values)
        
        return q1_values, q2_values
    
    def get_action(self, state: TEN) -> TEN:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Selected actions [batch_size, 1]
        """
        state = self.state_norm(state)
        state_encoding = self.net_state(state)
        q_values = self.net_q1(state_encoding)
        
        batch_size = state.shape[0]
        
        if self.training and self.explore_rate > torch.rand(1).item():
            # Random exploration
            action = torch.randint(
                self.action_dim, 
                size=(batch_size, 1), 
                device=state.device
            )
        else:
            # Greedy action selection
            action = q_values.argmax(dim=1, keepdim=True)
        
        return action
    
    def get_action_probabilities(self, state: TEN, temperature: float = 1.0) -> TEN:
        """
        Get action probabilities using softmax policy.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            temperature: Temperature for softmax (higher = more random)
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        q_values = self.forward(state)
        action_probs = torch.softmax(q_values / temperature, dim=1)
        return action_probs


class QNetTwinDuel(QNetBase):
    """
    Twin Dueling Q-Network implementation.
    
    Combines Double DQN with Dueling architecture:
    - Separates state value V(s) and action advantage A(s,a)
    - Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    - Reduces variance and improves learning stability
    
    Features:
    - Shared state encoder
    - Separate value and advantage streams for each network
    - Dueling aggregation formula
    - State and value normalization
    """
    
    def __init__(self, dims: List[int], state_dim: int, action_dim: int,
                 activation: nn.Module = nn.ReLU, dropout_rate: float = 0.0):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        
        # Shared state encoder
        self.net_state = build_mlp(
            dims=[state_dim, *dims],
            activation=activation,
            dropout_rate=dropout_rate,
            if_raw_out=False
        )
        
        # First network: Value and Advantage streams
        self.net_value1 = build_mlp(dims=[dims[-1], 1], if_raw_out=True)  # State value V(s)
        self.net_advantage1 = build_mlp(dims=[dims[-1], action_dim], if_raw_out=True)  # Advantage A(s,a)
        
        # Second network: Value and Advantage streams  
        self.net_value2 = build_mlp(dims=[dims[-1], 1], if_raw_out=True)
        self.net_advantage2 = build_mlp(dims=[dims[-1], action_dim], if_raw_out=True)
        
        # Softmax for action probabilities
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize output layers
        layer_init_with_orthogonal(self.net_value1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_advantage1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_value2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_advantage2[-1], std=0.1)
    
    def _dueling_forward(self, state_encoding: TEN, value_net: nn.Module, 
                        advantage_net: nn.Module) -> TEN:
        """
        Forward pass through dueling architecture.
        
        Args:
            state_encoding: Encoded state features
            value_net: Value stream network
            advantage_net: Advantage stream network
            
        Returns:
            Q-values from dueling combination
        """
        # Get value and advantage estimates
        value = value_net(state_encoding)  # [batch_size, 1]
        advantage = advantage_net(state_encoding)  # [batch_size, action_dim]
        
        # Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        # Subtracting mean advantage ensures identifiability
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
        state_encoding = self.net_state(state)
        q_values = self._dueling_forward(state_encoding, self.net_value1, self.net_advantage1)
        q_values = self.value_re_norm(q_values)
        return q_values
    
    def get_q1_q2(self, state: TEN) -> Tuple[TEN, TEN]:
        """
        Get Q-values from both dueling networks.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (q1_values, q2_values) each [batch_size, action_dim]
        """
        state = self.state_norm(state)
        state_encoding = self.net_state(state)
        
        # First dueling network
        q1_values = self._dueling_forward(state_encoding, self.net_value1, self.net_advantage1)
        q1_values = self.value_re_norm(q1_values)
        
        # Second dueling network
        q2_values = self._dueling_forward(state_encoding, self.net_value2, self.net_advantage2)
        q2_values = self.value_re_norm(q2_values)
        
        return q1_values, q2_values
    
    def get_action(self, state: TEN) -> TEN:
        """
        Get action using epsilon-greedy policy on first network.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Selected actions [batch_size, 1]
        """
        state = self.state_norm(state)
        state_encoding = self.net_state(state)
        q_values = self._dueling_forward(state_encoding, self.net_value1, self.net_advantage1)
        
        batch_size = state.shape[0]
        
        if self.training and self.explore_rate > torch.rand(1).item():
            # Random exploration
            action = torch.randint(
                self.action_dim,
                size=(batch_size, 1),
                device=state.device
            )
        else:
            # Greedy action selection
            action = q_values.argmax(dim=1, keepdim=True)
        
        return action
    
    def get_value_advantage_estimates(self, state: TEN) -> Tuple[TEN, TEN, TEN, TEN]:
        """
        Get separate value and advantage estimates from both networks.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (value1, advantage1, value2, advantage2)
        """
        state = self.state_norm(state)
        state_encoding = self.net_state(state)
        
        value1 = self.net_value1(state_encoding)
        advantage1 = self.net_advantage1(state_encoding)
        value2 = self.net_value2(state_encoding)
        advantage2 = self.net_advantage2(state_encoding)
        
        return value1, advantage1, value2, advantage2
    
    def get_state_values(self, state: TEN) -> Tuple[TEN, TEN]:
        """
        Get state values from both networks (useful for analysis).
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (value1, value2) each [batch_size, 1]
        """
        state = self.state_norm(state)
        state_encoding = self.net_state(state)
        
        value1 = self.net_value1(state_encoding)
        value2 = self.net_value2(state_encoding)
        
        return value1, value2