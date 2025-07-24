"""
Enhanced Neural Network Architectures for 16-Feature State Space
Improved architectures optimized for enhanced feature dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TEN = torch.Tensor

class QNetEnhanced(nn.Module):
    """Enhanced Q-Network with improved architecture for 16+ features"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default architecture scaled to state dimension
        if hidden_dims is None:
            # Scale network size based on state dimension
            base_size = max(128, state_dim * 8)
            hidden_dims = [base_size, base_size // 2, base_size // 4]
        
        self.hidden_dims = hidden_dims
        
        # State normalization
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)
        
        # Feature extraction layers
        self.feature_extractor = self._build_feature_extractor()
        
        # Value and advantage streams
        self.value_stream = self._build_value_stream()
        self.advantage_stream = self._build_advantage_stream()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_feature_extractor(self):
        """Build feature extraction layers"""
        layers = []
        prev_dim = self.state_dim
        
        for hidden_dim in self.hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def _build_value_stream(self):
        """Build value function stream"""
        feature_dim = self.hidden_dims[-2] if len(self.hidden_dims) > 1 else self.state_dim
        return nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1], 1)
        )
    
    def _build_advantage_stream(self):
        """Build advantage function stream"""
        feature_dim = self.hidden_dims[-2] if len(self.hidden_dims) > 1 else self.state_dim
        return nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1], self.action_dim)
        )
    
    def _initialize_weights(self):
        """Initialize network weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def state_norm(self, state: TEN) -> TEN:
        """Normalize state input"""
        return (state - self.state_avg) / (self.state_std + 1e-8)
    
    def value_re_norm(self, value: TEN) -> TEN:
        """Denormalize value output"""
        return value * self.value_std + self.value_avg
    
    def forward(self, state):
        """Forward pass through the network"""
        # Normalize state
        state = self.state_norm(state)
        
        # Feature extraction
        features = self.feature_extractor(state)
        features = self.dropout(features)
        
        # Dueling architecture: V(s) + A(s,a) - mean(A(s))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return self.value_re_norm(q_values)
    
    def get_action(self, state):
        """Select action using epsilon-greedy policy"""
        q_values = self.forward(state)
        
        if self.explore_rate < torch.rand(1):
            # Greedy action
            action = q_values.argmax(dim=1, keepdim=True)
        else:
            # Random action
            action = torch.randint(self.action_dim, size=(state.shape[0], 1), device=state.device)
        
        return action

class QNetTwinEnhanced(nn.Module):
    """Enhanced Twin Q-Network for Double DQN with improved architecture"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default architecture scaled to state dimension
        if hidden_dims is None:
            base_size = max(128, state_dim * 8)
            hidden_dims = [base_size, base_size // 2, base_size // 4]
        
        # State normalization
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)
        
        # Shared feature extractor
        self.feature_extractor = self._build_feature_extractor(hidden_dims)
        
        # Twin Q-networks
        feature_dim = hidden_dims[-2] if len(hidden_dims) > 1 else state_dim
        self.q_net1 = self._build_q_network(feature_dim, hidden_dims[-1])
        self.q_net2 = self._build_q_network(feature_dim, hidden_dims[-1])
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_feature_extractor(self, hidden_dims):
        """Build shared feature extraction layers"""
        layers = []
        prev_dim = self.state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def _build_q_network(self, input_dim, hidden_dim):
        """Build individual Q-network"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.action_dim)
        )
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-8)
    
    def value_re_norm(self, value: TEN) -> TEN:
        return value * self.value_std + self.value_avg
    
    def forward(self, state):
        """Forward pass using first Q-network"""
        state = self.state_norm(state)
        features = self.feature_extractor(state)
        q_values = self.q_net1(features)
        return self.value_re_norm(q_values)
    
    def get_q1_q2(self, state):
        """Get Q-values from both networks"""
        state = self.state_norm(state)
        features = self.feature_extractor(state)
        
        q1 = self.value_re_norm(self.q_net1(features))
        q2 = self.value_re_norm(self.q_net2(features))
        
        return q1, q2
    
    def get_action(self, state):
        """Select action using epsilon-greedy policy"""
        q_values = self.forward(state)
        
        if self.explore_rate < torch.rand(1):
            action = q_values.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1), device=state.device)
        
        return action

class QNetAttention(nn.Module):
    """Q-Network with attention mechanism for feature importance"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None, n_heads: int = 4):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_heads = n_heads
        
        if hidden_dims is None:
            base_size = max(128, state_dim * 8)
            hidden_dims = [base_size, base_size // 2]
        
        # State normalization
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)
        
        # Input projection
        self.input_projection = nn.Linear(state_dim, hidden_dims[0])
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature processing
        self.feature_layers = nn.Sequential(
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[1], action_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / (self.state_std + 1e-8)
    
    def value_re_norm(self, value: TEN) -> TEN:
        return value * self.value_std + self.value_avg
    
    def forward(self, state):
        """Forward pass with attention mechanism"""
        # Normalize state
        state = self.state_norm(state)
        
        # Project to hidden dimension
        x = self.input_projection(state)  # [batch, hidden_dim]
        
        # Add sequence dimension for attention (treat features as sequence)
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Self-attention
        attended, attention_weights = self.attention(x, x, x)
        attended = attended.squeeze(1)  # [batch, hidden_dim]
        
        # Feature processing
        features = self.feature_layers(attended)
        
        # Output Q-values
        q_values = self.output_layer(features)
        
        return self.value_re_norm(q_values)
    
    def get_action(self, state):
        """Select action using epsilon-greedy policy"""
        q_values = self.forward(state)
        
        if self.explore_rate < torch.rand(1):
            action = q_values.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1), device=state.device)
        
        return action

def get_enhanced_network_configs():
    """Get recommended network configurations for different state dimensions"""
    configs = {
        '16_features': {
            'hidden_dims': [256, 128, 64],
            'description': 'Optimized for 16 enhanced features',
            'dropout': 0.1
        },
        '12_features': {
            'hidden_dims': [192, 96, 48],
            'description': 'Optimized for 12 selected features',
            'dropout': 0.1
        },
        '8_features': {
            'hidden_dims': [128, 64, 32],
            'description': 'Optimized for 8 top features',
            'dropout': 0.05
        },
        'attention': {
            'hidden_dims': [256, 128],
            'n_heads': 4,
            'description': 'Attention-based architecture',
            'dropout': 0.1
        }
    }
    return configs

def create_enhanced_network(network_type: str, state_dim: int, action_dim: int, config_name: str = None):
    """Factory function to create enhanced networks"""
    configs = get_enhanced_network_configs()
    
    # Auto-select config based on state dimension if not specified
    if config_name is None:
        if state_dim >= 16:
            config_name = '16_features'
        elif state_dim >= 12:
            config_name = '12_features'
        else:
            config_name = '8_features'
    
    config = configs[config_name]
    
    if network_type.lower() == 'enhanced':
        return QNetEnhanced(state_dim, action_dim, config['hidden_dims'])
    elif network_type.lower() == 'twin':
        return QNetTwinEnhanced(state_dim, action_dim, config['hidden_dims'])
    elif network_type.lower() == 'attention':
        return QNetAttention(state_dim, action_dim, config.get('hidden_dims'), config.get('n_heads', 4))
    else:
        raise ValueError(f"Unknown network type: {network_type}")

# Backward compatibility functions
def build_mlp(dims: list, activation=None, if_raw_out: bool = True) -> nn.Sequential:
    """Build MLP with improved initialization"""
    if activation is None:
        activation = nn.ReLU
    
    layers = []
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        # Improved initialization
        nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(linear.bias, 0.01)
        
        layers.append(linear)
        if i < len(dims) - 2 or not if_raw_out:
            layers.append(activation())
    
    return nn.Sequential(*layers)

def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    """Improved layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)