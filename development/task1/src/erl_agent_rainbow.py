#!/usr/bin/env python3
"""
Rainbow DQN Agent - State-of-the-Art DQN with 6 Key Improvements
Combines: DQN + Double DQN + Dueling DQN + Prioritized Replay + Noisy Networks + N-Step Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from collections import deque
import math

from erl_net import build_mlp, layer_init_with_orthogonal
from erl_replay_buffer import ReplayBuffer


class NoisyLinear(nn.Module):
    """
    Noisy Network layer for parameter space exploration
    Replaces epsilon-greedy exploration with learnable noise
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not parameters)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise for both weights and biases"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product for weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using sign(x) * sqrt(|x|)"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters"""
        if self.training:
            # Use noisy parameters
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use mean parameters for evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class RainbowQNet(nn.Module):
    """
    Rainbow DQN Network with Dueling Architecture and Noisy Layers
    Combines distributional Q-learning with dueling network architecture
    """
    
    def __init__(self, 
                 dims: List[int], 
                 state_dim: int, 
                 action_dim: int,
                 n_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 use_noisy: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_noisy = use_noisy
        
        # State normalization
        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        
        # Value normalization
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)
        
        # Shared feature extraction
        self.feature_net = build_mlp(dims=[state_dim, *dims[:-1]], activation=nn.ReLU, if_raw_out=False)
        
        # Dueling architecture
        if use_noisy:
            # Noisy layers for exploration
            self.value_head = NoisyLinear(dims[-2], n_atoms)
            self.advantage_head = NoisyLinear(dims[-2], action_dim * n_atoms)
        else:
            # Standard linear layers
            self.value_head = nn.Linear(dims[-2], n_atoms)
            self.advantage_head = nn.Linear(dims[-2], action_dim * n_atoms)
            
            # Initialize layers
            layer_init_with_orthogonal(self.value_head, std=0.1)
            layer_init_with_orthogonal(self.advantage_head, std=0.1)
        
        # Support for distributional RL
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
        
        print(f"🌈 Rainbow Q-Network initialized:")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
        print(f"   Atoms: {n_atoms}, Support: [{v_min:.1f}, {v_max:.1f}]")
        print(f"   Noisy networks: {use_noisy}")
        print(f"   Architecture: {dims}")
    
    def state_norm(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state"""
        return (state - self.state_avg) / self.state_std
    
    def value_re_norm(self, value: torch.Tensor) -> torch.Tensor:
        """Denormalize value"""
        return value * self.value_std + self.value_avg
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.use_noisy:
            self.value_head.reset_noise()
            self.advantage_head.reset_noise()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-value distributions
        
        Returns:
            Q-distributions of shape (batch_size, action_dim, n_atoms)
        """
        # Normalize state
        state = self.state_norm(state)
        
        # Extract features
        features = self.feature_net(state)
        
        # Dueling architecture
        value_dist = self.value_head(features)  # (batch, n_atoms)
        advantage_dist = self.advantage_head(features)  # (batch, action_dim * n_atoms)
        
        # Reshape advantage
        advantage_dist = advantage_dist.view(-1, self.action_dim, self.n_atoms)
        
        # Dueling combination
        value_dist = value_dist.unsqueeze(1).expand_as(advantage_dist)
        advantage_mean = advantage_dist.mean(dim=1, keepdim=True)
        q_dist = value_dist + advantage_dist - advantage_mean
        
        # Apply softmax to get probability distributions
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get expected Q-values from distributions"""
        q_dist = self.forward(state)
        q_values = torch.sum(q_dist * self.support, dim=-1)
        return self.value_re_norm(q_values)
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get greedy action (no exploration needed with noisy networks)"""
        q_values = self.get_q_values(state)
        return q_values.argmax(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples experiences based on TD-error priority
    """
    
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000):
        """
        Initialize prioritized replay buffer
        
        Args:
            max_size: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            gpu_id: GPU device ID
            alpha: Prioritization exponent
            beta_start: Initial importance sampling weight
            beta_frames: Frames over which to anneal beta to 1.0
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")
        
        # Storage
        self.state = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.action = torch.empty((max_size, action_dim), dtype=torch.int64, device=self.device)
        self.reward = torch.empty((max_size,), dtype=torch.float32, device=self.device)
        self.done = torch.empty((max_size,), dtype=torch.bool, device=self.device)
        self.next_state = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        
        # Priority storage
        self.priorities = torch.zeros((max_size,), dtype=torch.float32, device=self.device)
        self.max_priority = 1.0
        
        # Buffer management
        self.p = 0
        self.now_len = 0
        self.if_full = False
        
        print(f"📦 Prioritized Replay Buffer initialized:")
        print(f"   Max size: {max_size}")
        print(f"   Alpha: {alpha}, Beta: {beta_start} → 1.0")
        print(f"   Device: {self.device}")
    
    def get_beta(self) -> float:
        """Get current beta value (annealed from beta_start to 1.0)"""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def update(self, buffer_items: List[torch.Tensor], priorities: torch.Tensor = None) -> None:
        """
        Update buffer with new experiences
        
        Args:
            buffer_items: [states, actions, rewards, dones, next_states]
            priorities: Optional priority values for new experiences
        """
        states, actions, rewards, dones, next_states = buffer_items
        
        # Ensure correct device and shape
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device)
        
        batch_size = states.shape[0]
        
        # Handle priorities
        if priorities is None:
            # New experiences get maximum priority
            priorities = torch.full((batch_size,), self.max_priority, device=self.device)
        
        # Add to buffer
        for i in range(batch_size):
            idx = self.p
            
            self.state[idx] = states[i]
            self.action[idx] = actions[i]
            self.reward[idx] = rewards[i]
            self.done[idx] = dones[i]
            self.next_state[idx] = next_states[i]
            self.priorities[idx] = priorities[i]
            
            self.p = (self.p + 1) % self.max_size
            if not self.if_full and self.p == 0:
                self.if_full = True
        
        self.now_len = min(self.now_len + batch_size, self.max_size)
        self.frame += batch_size
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritized sampling
        
        Returns:
            Tuple of (states, actions, rewards, dones, next_states, indices, weights)
        """
        # Get sampling probabilities
        priorities = self.priorities[:self.now_len]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Get importance sampling weights
        beta = self.get_beta()
        weights = (self.now_len * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize by max weight
        
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
            self.next_state[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """Update priorities for given indices"""
        priorities = priorities.clamp(min=1e-6)  # Avoid zero priorities
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())


class AgentRainbow:
    """
    Rainbow DQN Agent combining 6 key improvements:
    1. DQN - Basic deep Q-learning
    2. Double DQN - Reduced overestimation bias
    3. Dueling DQN - Separate value and advantage streams
    4. Prioritized Replay - Sample important transitions more
    5. Noisy Networks - Learnable exploration
    6. N-Step Learning - Multi-step bootstrapping
    """
    
    def __init__(self,
                 net_dims: tuple,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 args=None):
        """
        Initialize Rainbow DQN agent
        
        Args:
            net_dims: Network architecture
            state_dim: State space dimension
            action_dim: Action space dimension
            gpu_id: GPU device ID
            args: Training configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_dims = net_dims
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")
        
        # Rainbow hyperparameters
        self.n_step = getattr(args, 'rainbow_n_step', 3)  # N-step learning
        self.n_atoms = getattr(args, 'rainbow_n_atoms', 51)  # Distributional atoms
        self.v_min = getattr(args, 'rainbow_v_min', -10.0)  # Distribution support
        self.v_max = getattr(args, 'rainbow_v_max', 10.0)
        self.use_noisy = getattr(args, 'rainbow_use_noisy', True)  # Noisy networks
        self.use_prioritized = getattr(args, 'rainbow_use_prioritized', True)  # Prioritized replay
        
        # Standard RL hyperparameters
        self.learning_rate = getattr(args, 'learning_rate', 1e-4)
        self.gamma = getattr(args, 'gamma', 0.995)
        self.batch_size = getattr(args, 'batch_size', 512)
        self.target_update_freq = getattr(args, 'target_update_freq', 4)
        self.soft_update_tau = getattr(args, 'soft_update_tau', 0.005)
        
        # Networks
        self.act = RainbowQNet(
            net_dims, state_dim, action_dim,
            n_atoms=self.n_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            use_noisy=self.use_noisy
        ).to(self.device)
        
        self.act_target = RainbowQNet(
            net_dims, state_dim, action_dim,
            n_atoms=self.n_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            use_noisy=self.use_noisy
        ).to(self.device)
        
        # Copy parameters to target network
        self.act_target.load_state_dict(self.act.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        
        # N-step storage
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # Training metrics
        self.training_step = 0
        self.losses = deque(maxlen=100)
        
        # State tracking
        self.last_state = None
        
        print(f"🌈 Rainbow DQN Agent initialized:")
        print(f"   Device: {self.device}")
        print(f"   N-step: {self.n_step}")
        print(f"   Atoms: {self.n_atoms}")
        print(f"   Noisy networks: {self.use_noisy}")
        print(f"   Prioritized replay: {self.use_prioritized}")
        print(f"   Architecture: {net_dims}")
    
    def act(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for action selection"""
        return self.act.get_q_values(state)
    
    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Select actions (no epsilon needed with noisy networks)"""
        with torch.no_grad():
            if self.use_noisy:
                # Noisy networks provide exploration
                return self.act.get_action(state)
            else:
                # Fall back to epsilon-greedy
                if torch.rand(1) < 0.1:  # 10% exploration
                    return torch.randint(0, self.action_dim, (state.shape[0], 1), device=self.device)
                else:
                    return self.act.get_action(state)
    
    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> List[torch.Tensor]:
        """
        Collect experience from environment with N-step returns
        """
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        
        # Get initial state
        if self.last_state is None:
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = self.last_state
        
        # Collect experience
        for t in range(horizon_len):
            if if_random:
                # Random exploration
                action = torch.randint(0, self.action_dim, (env.num_envs, 1), device=self.device)
            else:
                # Policy actions
                action = self.select_actions(state)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if not isinstance(done, torch.Tensor):
                done = torch.tensor(done, dtype=torch.bool, device=self.device)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            # Update state (reset if done)
            if done.any():
                state = env.reset()
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                # Keep non-done states
                state = torch.where(done.unsqueeze(-1), state, next_state)
            else:
                state = next_state
        
        self.last_state = state.detach()
        
        # Stack and flatten
        states = torch.stack(states).reshape(-1, self.state_dim)
        actions = torch.stack(actions).reshape(-1, 1)
        rewards = torch.stack(rewards).reshape(-1)
        dones = torch.stack(dones).reshape(-1)
        next_states = torch.stack(next_states).reshape(-1, self.state_dim)
        
        # Apply N-step returns
        if self.n_step > 1:
            states, actions, rewards, dones, next_states = self._compute_n_step_returns(
                states, actions, rewards, dones, next_states
            )
        
        return [states, actions, rewards, dones, next_states]
    
    def _compute_n_step_returns(self, states, actions, rewards, dones, next_states):
        """Compute N-step returns for improved learning"""
        n_step_states = []
        n_step_actions = []
        n_step_rewards = []
        n_step_dones = []
        n_step_next_states = []
        
        batch_size = states.shape[0]
        
        for i in range(batch_size - self.n_step + 1):
            # N-step reward
            n_step_reward = 0
            n_step_done = False
            
            for j in range(self.n_step):
                n_step_reward += (self.gamma ** j) * rewards[i + j]
                if dones[i + j]:
                    n_step_done = True
                    n_step_next_state = next_states[i + j]
                    break
            else:
                n_step_next_state = next_states[i + self.n_step - 1]
            
            n_step_states.append(states[i])
            n_step_actions.append(actions[i])
            n_step_rewards.append(n_step_reward)
            n_step_dones.append(n_step_done)
            n_step_next_states.append(n_step_next_state)
        
        return (
            torch.stack(n_step_states),
            torch.stack(n_step_actions),
            torch.tensor(n_step_rewards, device=self.device),
            torch.tensor(n_step_dones, device=self.device),
            torch.stack(n_step_next_states)
        )
    
    def update_net(self, buffer) -> Tuple[float, float]:
        """
        Update Rainbow networks using distributional loss
        
        Returns:
            Tuple of (distributional_loss, q_value_mean)
        """
        self.training_step += 1
        
        # Reset noise in noisy networks
        if self.use_noisy:
            self.act.reset_noise()
            self.act_target.reset_noise()
        
        # Sample from buffer
        if hasattr(buffer, 'sample'):
            # Prioritized replay buffer
            batch_data = buffer.sample(self.batch_size)
            states, actions, rewards, dones, next_states, indices, weights = batch_data
        else:
            # Standard replay buffer
            indices = torch.randint(0, buffer.now_len, (self.batch_size,), device=self.device)
            states = buffer.state[indices]
            actions = buffer.action[indices].long()
            rewards = buffer.reward[indices]
            dones = buffer.done[indices]
            next_states = buffer.next_state[indices] if hasattr(buffer, 'next_state') else torch.roll(states, -1, dims=0)
            weights = torch.ones_like(rewards)
        
        # Current Q-distributions
        current_q_dist = self.act(states)
        current_q_dist = current_q_dist.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)).squeeze(1)
        
        # Next Q-distributions (Double DQN)
        with torch.no_grad():
            next_q_values = self.act.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)
            next_q_dist = self.act_target(next_states)
            next_q_dist = next_q_dist.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)).squeeze(1)
            
            # Compute target distribution
            target_support = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * self.act.support.unsqueeze(0) * (~dones).unsqueeze(1)
            target_support = target_support.clamp(self.v_min, self.v_max)
            
            # Distribute target support
            b = (target_support - self.v_min) / ((self.v_max - self.v_min) / (self.n_atoms - 1))
            l = b.floor().long()
            u = b.ceil().long()
            
            # Distribute probability mass
            target_q_dist = torch.zeros_like(next_q_dist)
            target_q_dist.scatter_add_(1, l, next_q_dist * (u - b))
            target_q_dist.scatter_add_(1, u, next_q_dist * (b - l))
        
        # Distributional loss (KL divergence)
        loss = -(target_q_dist * torch.log(current_q_dist + 1e-8)).sum(dim=1)
        
        # Apply importance sampling weights
        loss = (loss * weights).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.act.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities in prioritized replay
        if hasattr(buffer, 'update_priorities') and hasattr(buffer, 'sample'):
            with torch.no_grad():
                td_errors = (target_q_dist * torch.log(target_q_dist / (current_q_dist + 1e-8) + 1e-8)).sum(dim=1)
                buffer.update_priorities(indices, td_errors.abs() + 1e-6)
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self._soft_update_target()
        
        # Track metrics
        self.losses.append(loss.item())
        q_mean = current_q_dist.sum(dim=1).mean().item()
        
        return loss.item(), q_mean
    
    def _soft_update_target(self):
        """Soft update of target network"""
        for param, target_param in zip(self.act.parameters(), self.act_target.parameters()):
            target_param.data.copy_(self.soft_update_tau * param.data + (1 - self.soft_update_tau) * target_param.data)
    
    def save_or_load_agent(self, cwd: str, if_save: bool):
        """Save or load agent networks"""
        act_path = f"{cwd}/actor_rainbow.pth"
        
        if if_save:
            torch.save(self.act.state_dict(), act_path)
            print(f"✅ Rainbow agent saved to {cwd}")
        else:
            try:
                self.act.load_state_dict(torch.load(act_path, map_location=self.device))
                self.act_target.load_state_dict(self.act.state_dict())
                print(f"✅ Rainbow agent loaded from {cwd}")
            except FileNotFoundError:
                print(f"⚠️ Rainbow checkpoint not found at {cwd}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        return {
            'distributional_loss': np.mean(list(self.losses)) if self.losses else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_steps': self.training_step,
            'n_step': self.n_step,
            'atoms': self.n_atoms
        }


def test_rainbow_agent():
    """Test Rainbow DQN agent"""
    
    print("🧪 Testing Rainbow DQN Agent")
    print("=" * 60)
    
    # Mock configuration
    class MockArgs:
        learning_rate = 1e-4
        gamma = 0.995
        batch_size = 256
        rainbow_n_step = 3
        rainbow_n_atoms = 51
        rainbow_v_min = -10.0
        rainbow_v_max = 10.0
        rainbow_use_noisy = True
        rainbow_use_prioritized = True
        target_update_freq = 4
        soft_update_tau = 0.005
    
    # Create agent
    agent = AgentRainbow(
        net_dims=(128, 64, 32),
        state_dim=8,
        action_dim=3,
        gpu_id=-1,  # CPU for testing
        args=MockArgs()
    )
    
    # Test action selection
    test_states = torch.randn(16, 8)
    q_values = agent.act(test_states)
    actions = agent.select_actions(test_states)
    
    print(f"\n🎯 Action Selection Test:")
    print(f"   Input shape: {test_states.shape}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Q-value range: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")
    print(f"   Action range: [{actions.min().item()}, {actions.max().item()}]")
    
    # Test expected Q-values from distributions  
    q_expected = agent.act.get_q_values(test_states)
    print(f"   Expected Q-values shape: {q_expected.shape}")
    print(f"   Expected Q-value range: [{q_expected.min().item():.3f}, {q_expected.max().item():.3f}]")
    
    # Test distributional output
    q_dist = agent.act.forward(test_states)
    print(f"\n📊 Distributional Output:")
    print(f"   Q-distribution shape: {q_dist.shape}")
    print(f"   Distribution sum (should be ~1): {q_dist.sum(dim=-1).mean().item():.3f}")
    
    # Test training stats
    stats = agent.get_training_stats()
    print(f"\n📈 Training Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Rainbow DQN agent test completed successfully!")


if __name__ == "__main__":
    test_rainbow_agent()