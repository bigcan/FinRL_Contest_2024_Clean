import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from torch import Tensor

TEN = Tensor


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for NoisyNet-based exploration.
    Replaces epsilon-greedy exploration with learned parameter noise.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters for weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        
        # Learnable parameters for biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Buffers for noise (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize learnable parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Generate new noise for weights and biases"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product to create correlated noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> Tensor:
        """Scale noise using factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with noisy weights and biases"""
        if self.training:
            # Use noisy parameters during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use mean parameters during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class QNetTwinNoisy(nn.Module):
    """Twin Q-Network with Noisy Linear layers for exploration"""
    
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_rate = 0.125  # Not used in noisy networks, kept for compatibility
        
        # State normalization parameters
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

        # Build networks with noisy layers
        self.net1 = self._build_noisy_network(net_dims, state_dim, action_dim)
        self.net2 = self._build_noisy_network(net_dims, state_dim, action_dim)
        
    def _build_noisy_network(self, net_dims: [int], state_dim: int, action_dim: int) -> nn.Sequential:
        """Build network with noisy linear layers"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim, net_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers (regular linear)
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer (noisy for exploration)
        layers.append(NoisyLinear(net_dims[-1], action_dim))
        
        return nn.Sequential(*layers)

    def state_norm(self, state: TEN) -> TEN:
        """Normalize state"""
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / state_std

    def forward(self, state: TEN) -> TEN:
        """Forward pass returning Q-values from first network"""
        state = self.state_norm(state)
        return self.net1(state)

    def get_q1_q2(self, state: TEN) -> Tuple[TEN, TEN]:
        """Get Q-values from both networks"""
        state = self.state_norm(state)
        return self.net1(state), self.net2(state)

    def get_action(self, state: TEN) -> TEN:
        """Get action using noisy networks (no epsilon-greedy needed)"""
        state = self.state_norm(state)
        action = torch.min(*self.get_q1_q2(state)).argmax(dim=1, keepdim=True)
        return action
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class QNetTwinDuelNoisy(nn.Module):
    """Twin Dueling Q-Network with Noisy Linear layers"""
    
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_rate = 0.125  # Not used in noisy networks
        
        # State normalization parameters
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

        # Shared feature extractor
        self.feature_net = self._build_feature_network(net_dims, state_dim)
        
        # Dueling streams for both networks
        self.advantage1_net = self._build_advantage_network(net_dims[-1], action_dim)
        self.value1_net = self._build_value_network(net_dims[-1])
        
        self.advantage2_net = self._build_advantage_network(net_dims[-1], action_dim)
        self.value2_net = self._build_value_network(net_dims[-1])
        
    def _build_feature_network(self, net_dims: [int], state_dim: int) -> nn.Sequential:
        """Build shared feature extraction network"""
        layers = []
        layers.append(nn.Linear(state_dim, net_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))
            layers.append(nn.ReLU())
            
        return nn.Sequential(*layers)
    
    def _build_advantage_network(self, feature_dim: int, action_dim: int) -> nn.Sequential:
        """Build advantage stream with noisy output"""
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            NoisyLinear(feature_dim // 2, action_dim)
        )
    
    def _build_value_network(self, feature_dim: int) -> nn.Sequential:
        """Build value stream"""
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )

    def state_norm(self, state: TEN) -> TEN:
        """Normalize state"""
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / state_std

    def _dueling_forward(self, state: TEN, advantage_net: nn.Sequential, value_net: nn.Sequential) -> TEN:
        """Forward pass for dueling architecture"""
        state = self.state_norm(state)
        features = self.feature_net(state)
        
        advantage = advantage_net(features)
        value = value_net(features)
        
        # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def forward(self, state: TEN) -> TEN:
        """Forward pass returning Q-values from first network"""
        return self._dueling_forward(state, self.advantage1_net, self.value1_net)

    def get_q1_q2(self, state: TEN) -> Tuple[TEN, TEN]:
        """Get Q-values from both dueling networks"""
        q1 = self._dueling_forward(state, self.advantage1_net, self.value1_net)
        q2 = self._dueling_forward(state, self.advantage2_net, self.value2_net)
        return q1, q2

    def get_action(self, state: TEN) -> TEN:
        """Get action using noisy dueling networks"""
        action = torch.min(*self.get_q1_q2(state)).argmax(dim=1, keepdim=True)
        return action
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()