import torch
import torch.nn as nn

TEN = torch.Tensor


class QNetBase(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        # Ensure state_avg and state_std are on the same device as state
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / state_std

    def value_re_norm(self, value: TEN) -> TEN:
        return value * self.value_std + self.value_avg


class QNetTwin(QNetBase):  # Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        return q_val  # one group of Q values

    def get_q1_q2(self, state):
        # CRITICAL FIX: Ensure state is on correct device before neural network operations
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            state = state.to(next(self.parameters()).device)
            
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val1 = self.net_val1(s_enc)  # q value 1
        q_val1 = self.value_re_norm(q_val1)
        q_val2 = self.net_val2(s_enc)  # q value 2
        q_val2 = self.value_re_norm(q_val2)
        return q_val1, q_val2  # two groups of Q values

    def get_action(self, state):
        # CRITICAL FIX: Ensure state is on correct device before neural network operations
        # This method is called during episode transitions and must validate device consistency
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            state = state.to(next(self.parameters()).device)
            
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            # a_prob = self.soft_max(q_val)
            # action = torch.multinomial(a_prob, num_samples=1)
            action = torch.randint(self.action_dim, size=(state.shape[0], 1), device=state.device)
        return action


class QNetTwinDuel(QNetBase):  # D3QN: Dueling Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        # Fixed architecture: advantage per action, value per state
        self.net_adv1 = build_mlp(dims=[dims[-1], action_dim])  # advantage per action 1
        self.net_val1 = build_mlp(dims=[dims[-1], 1])  # state value 1  
        self.net_adv2 = build_mlp(dims=[dims[-1], action_dim])  # advantage per action 2
        self.net_val2 = build_mlp(dims=[dims[-1], 1])  # state value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        # Corrected dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_adv = self.net_adv1(s_enc)  # advantage per action [batch_size, action_dim]
        q_val = self.net_val1(s_enc)  # state value [batch_size, 1]
        value = q_val + (q_adv - q_adv.mean(dim=1, keepdim=True))  # dueling Q value
        value = self.value_re_norm(value)
        return value

    def get_q1_q2(self, state):
        # CRITICAL FIX: Ensure state is on correct device before neural network operations
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            state = state.to(next(self.parameters()).device)
            
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        
        # Stream 1: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_adv1 = self.net_adv1(s_enc)  # advantage per action 1 [batch_size, action_dim]
        q_val1 = self.net_val1(s_enc)  # state value 1 [batch_size, 1]
        
        # Shape validation for debugging
        assert q_adv1.shape[-1] == self.action_dim, f"Advantage shape mismatch: expected [*, {self.action_dim}], got {q_adv1.shape}"
        assert q_val1.shape[-1] == 1, f"Value shape mismatch: expected [*, 1], got {q_val1.shape}"
        assert q_adv1.shape[0] == q_val1.shape[0], f"Batch size mismatch: adv={q_adv1.shape[0]}, val={q_val1.shape[0]}"
        
        q_duel1 = q_val1 + (q_adv1 - q_adv1.mean(dim=1, keepdim=True))
        q_duel1 = self.value_re_norm(q_duel1)

        # Stream 2: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_adv2 = self.net_adv2(s_enc)  # advantage per action 2 [batch_size, action_dim]
        q_val2 = self.net_val2(s_enc)  # state value 2 [batch_size, 1]
        
        # Shape validation for debugging
        assert q_adv2.shape[-1] == self.action_dim, f"Advantage2 shape mismatch: expected [*, {self.action_dim}], got {q_adv2.shape}"
        assert q_val2.shape[-1] == 1, f"Value2 shape mismatch: expected [*, 1], got {q_val2.shape}"
        assert q_adv2.shape[0] == q_val2.shape[0], f"Batch size mismatch: adv2={q_adv2.shape[0]}, val2={q_val2.shape[0]}"
        
        q_duel2 = q_val2 + (q_adv2 - q_adv2.mean(dim=1, keepdim=True))
        q_duel2 = self.value_re_norm(q_duel2)
        return q_duel1, q_duel2  # two dueling Q values

    def get_action(self, state):
        # CRITICAL FIX: Ensure state is on correct device before neural network operations
        # This method is called during episode transitions and must validate device consistency
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            state = state.to(next(self.parameters()).device)
            
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        # Use the corrected dueling formula for action selection
        q_adv = self.net_adv1(s_enc)  # advantage per action [batch_size, action_dim]
        q_val = self.net_val1(s_enc)  # state value [batch_size, 1]
        q_values = q_val + (q_adv - q_adv.mean(dim=1, keepdim=True))  # dueling Q values
        
        if self.explore_rate < torch.rand(1):
            action = q_values.argmax(dim=1, keepdim=True)
        else:
            # a_prob = self.soft_max(q_values)
            # action = torch.multinomial(a_prob, num_samples=1)
            action = torch.randint(self.action_dim, size=(state.shape[0], 1), device=state.device)
        return action


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class ActorDiscretePPO(nn.Module):
    """
    Discrete action actor network for PPO
    Outputs action probabilities for discrete action space
    """
    
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State normalization
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        
        # Policy network
        self.net = build_mlp(dims=[state_dim, *dims, action_dim], activation=nn.ReLU, if_raw_out=True)
        
        # Initialize output layer with smaller weights for stable learning
        layer_init_with_orthogonal(self.net[-1], std=0.01)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def state_norm(self, state: TEN) -> TEN:
        """Normalize state"""
        # Ensure state_avg and state_std are on the same device as state
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / state_std
    
    def forward(self, state: TEN) -> TEN:
        """Forward pass returning action probabilities"""
        state = self.state_norm(state)
        logits = self.net(state)
        action_probs = self.softmax(logits)
        
        # Add small epsilon to prevent log(0)
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        return action_probs
    
    def get_action_log_prob(self, state: TEN, action: TEN) -> TEN:
        """Get log probability of given action"""
        action_probs = self.forward(state)
        action_log_probs = torch.log(action_probs)
        return action_log_probs.gather(1, action.long())


class CriticAdv(nn.Module):
    """
    Advantage critic network for PPO
    Estimates state values for advantage computation
    """
    
    def __init__(self, dims: [int], state_dim: int, output_dim: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        # State normalization  
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        
        # Value normalization
        self.value_avg = nn.Parameter(torch.zeros((output_dim,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((output_dim,)), requires_grad=False)
        
        # Value network
        self.net = build_mlp(dims=[state_dim, *dims, output_dim], activation=nn.ReLU, if_raw_out=True)
        
        # Initialize output layer
        layer_init_with_orthogonal(self.net[-1], std=1.0)
    
    def state_norm(self, state: TEN) -> TEN:
        """Normalize state"""
        # Ensure state_avg and state_std are on the same device as state
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / state_std
    
    def value_re_norm(self, value: TEN) -> TEN:
        """Denormalize value"""
        return value * self.value_std + self.value_avg
    
    def forward(self, state: TEN) -> TEN:
        """Forward pass returning state values"""
        state = self.state_norm(state)
        value = self.net(state)
        value = self.value_re_norm(value)
        return value


class ActorCriticPPO(nn.Module):
    """
    Combined Actor-Critic network for PPO
    Shares feature extraction between policy and value networks
    """
    
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State normalization
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        
        # Shared feature extraction
        self.shared_net = build_mlp(dims=[state_dim, *dims[:-1]], activation=nn.ReLU, if_raw_out=False)
        
        # Policy head
        self.policy_head = nn.Linear(dims[-2], action_dim)
        layer_init_with_orthogonal(self.policy_head, std=0.01)
        
        # Value head  
        self.value_head = nn.Linear(dims[-2], 1)
        layer_init_with_orthogonal(self.value_head, std=1.0)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def state_norm(self, state: TEN) -> TEN:
        """Normalize state"""
        # Ensure state_avg and state_std are on the same device as state
        state_avg = self.state_avg.to(state.device)
        state_std = self.state_std.to(state.device)
        return (state - state_avg) / state_std
    
    def forward(self, state: TEN) -> tuple:
        """Forward pass returning both action probabilities and state values"""
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
        """Get only action probabilities"""
        action_probs, _ = self.forward(state)
        return action_probs
    
    def get_state_values(self, state: TEN) -> TEN:
        """Get only state values"""
        _, state_values = self.forward(state)
        return state_values
