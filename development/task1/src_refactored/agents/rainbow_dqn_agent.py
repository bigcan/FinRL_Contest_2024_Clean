"""
Rainbow DQN agent implementation for the FinRL Contest 2024 framework.

This module provides the Rainbow DQN agent that combines multiple DQN improvements:
- Double DQN for reduced overestimation
- Dueling networks for better value estimation
- Prioritized Experience Replay for sample efficiency
- Multi-step learning for better temporal credit assignment
- Noisy networks for exploration
"""

from typing import Optional, Tuple
import torch
from torch import Tensor

from .base_dqn_agent import BaseDQNAgent
from ..core.types import TrainingStats
from ..config import RainbowDQNConfig
from ..replay import PrioritizedReplayBuffer


class RainbowDQNAgent(BaseDQNAgent):
    """
    Rainbow DQN Agent - combines multiple DQN enhancements.
    
    Integrates the following techniques:
    1. Double DQN - reduced overestimation bias
    2. Dueling Networks - separate value and advantage estimation
    3. Prioritized Experience Replay - sample important experiences more
    4. Multi-step learning - better temporal credit assignment
    5. Noisy Networks - parameter space exploration
    
    This represents the state-of-the-art in value-based RL.
    """
    
    def __init__(self, 
                 config: Optional[RainbowDQNConfig] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize Rainbow DQN agent.
        
        Args:
            config: Agent configuration (will create default if None)
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Computing device
            **kwargs: Additional configuration parameters
        """
        # Create default config if not provided
        if config is None:
            config = RainbowDQNConfig(**kwargs)
        
        # Ensure agent type is set correctly
        config.agent_type = "AgentRainbowDQN"
        
        # Store Rainbow-specific parameters
        self.n_step = config.n_step
        self.noise_std_init = getattr(config, 'noise_std_init', 0.5)
        self.per_alpha = config.per_alpha
        self.per_beta = config.per_beta
        self.per_beta_annealing_steps = config.per_beta_annealing_steps
        
        # Override exploration rate (not used with noisy networks)
        config.explore_rate = 0.0
        
        super().__init__(config, state_dim, action_dim, device)
    
    def _build_networks(self):
        """Build noisy dueling twin Q-networks for Rainbow."""
        from ..networks import QNetTwinDuelNoisy
        
        # Create online network (combines dueling + noisy)
        self.online_network = QNetTwinDuelNoisy(
            dims=self.config.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            noise_std_init=self.noise_std_init
        ).to(self.device)
        
        # Create target network (copy of online network)
        from copy import deepcopy
        self.target_network = deepcopy(self.online_network)
        
        # Set exploration rate (not used but kept for compatibility)
        self.online_network.explore_rate = 0.0
        
        print(f"Built Rainbow DQN networks: {self.config.net_dims} dims, "
              f"{self.state_dim} state_dim, {self.action_dim} action_dim, "
              f"n_step={self.n_step}, noise_std={self.noise_std_init}")
    
    def _build_replay_buffer(self):
        """Build prioritized replay buffer for Rainbow."""
        # Calculate buffer size
        buffer_size = getattr(self.config, 'buffer_size', 100000)
        
        self.replay_buffer = PrioritizedReplayBuffer(
            max_size=buffer_size,
            state_dim=self.state_dim,
            action_dim=1,  # Discrete actions
            device=self.device,
            num_sequences=getattr(self.config, 'num_sequences', 1),
            alpha=self.per_alpha,
            beta=self.per_beta,
            beta_annealing_steps=self.per_beta_annealing_steps
        )
        
        print(f"Built Prioritized Replay Buffer for Rainbow: size={buffer_size}, "
              f"alpha={self.per_alpha}, beta={self.per_beta}")
    
    def select_action(self, state, deterministic: bool = False):
        """
        Select action using noisy dueling networks (Rainbow exploration).
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Selected action
        """
        if isinstance(state, (list, tuple)):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure correct shape [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        # Reset noise before action selection (for exploration)
        if not deterministic and self.training:
            self.online_network.reset_noise()
        
        with torch.no_grad():
            # Use noisy dueling networks for action selection
            q1_values, q2_values = self.online_network.get_q1_q2(state)
            q_values = torch.min(q1_values, q2_values)
            action = q_values.argmax(dim=1, keepdim=True)
        
        return action.squeeze().cpu().numpy() if action.shape[0] == 1 else action.cpu().numpy()
    
    def _update_networks(self) -> Tuple[float, float]:
        """
        Update networks with multi-step learning and PER.
        
        Returns:
            Tuple of (critic_loss, average_q_value)
        """
        # Reset noise before each update step
        self.online_network.reset_noise()
        self.target_network.reset_noise()
        
        # Sample batch from prioritized replay buffer
        (states, actions, rewards, undones, next_states, 
         indices, importance_weights) = self.replay_buffer.sample(self.batch_size)
        
        # Calculate target Q-values using Double DQN with multi-step learning
        with torch.no_grad():
            # Use online network for action selection
            next_q1_online, next_q2_online = self.online_network.get_q1_q2(next_states)
            next_actions = torch.min(next_q1_online, next_q2_online).argmax(dim=1, keepdim=True)
            
            # Use target network for Q-value evaluation
            next_q1_target, next_q2_target = self.target_network.get_q1_q2(next_states)
            next_q_values = torch.min(next_q1_target, next_q2_target).gather(1, next_actions).squeeze(1)
            
            # Multi-step discount factor
            gamma_n = self.gamma ** self.n_step
            
            # Calculate target Q-values with multi-step returns
            target_q_values = rewards + undones * gamma_n * next_q_values
        
        # Get current Q-values
        current_q1, current_q2 = self.online_network.get_q1_q2(states)
        current_q1_values = current_q1.gather(1, actions.long()).squeeze(1)
        current_q2_values = current_q2.gather(1, actions.long()).squeeze(1)
        
        # Calculate TD errors for priority updates
        td_errors_q1 = torch.abs(current_q1_values - target_q_values)
        td_errors_q2 = torch.abs(current_q2_values - target_q_values)
        td_errors = td_errors_q1 + td_errors_q2
        
        # Apply importance sampling weights to losses
        weighted_loss_q1 = (self.criterion(current_q1_values, target_q_values) * importance_weights).mean()
        weighted_loss_q2 = (self.criterion(current_q2_values, target_q_values) * importance_weights).mean()
        critic_loss = weighted_loss_q1 + weighted_loss_q2
        
        # Optimize
        self.optimizer.zero_grad()
        critic_loss.backward()
        
        # Get performance metric for adaptive scheduling
        performance = -critic_loss.item()  # Higher is better
        self.optimizer.step(performance=performance)
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        return critic_loss.item(), current_q1_values.mean().item()
    
    def update(self, batch_data: Tuple[Tensor, ...]) -> TrainingStats:
        """
        Update agent with Rainbow enhancements.
        
        Args:
            batch_data: Tuple of (states, actions, rewards, dones)
            
        Returns:
            Training statistics including Rainbow metrics
        """
        # Reset noise before each batch update
        if hasattr(self.online_network, 'reset_noise'):
            self.online_network.reset_noise()
        
        # Call parent update method
        stats = super().update(batch_data)
        
        # Add Rainbow-specific statistics
        if hasattr(self.replay_buffer, 'get_priority_stats'):
            priority_stats = self.replay_buffer.get_priority_stats()
            stats.additional_metrics.update({
                'rainbow_per_beta': priority_stats.get('current_beta', self.per_beta),
                'rainbow_max_priority': priority_stats.get('max_priority', 0.0),
                'rainbow_mean_priority': priority_stats.get('mean_priority', 0.0),
                'rainbow_n_step': self.n_step,
                'rainbow_noise_std': self.noise_std_init,
            })
        
        return stats
    
    def get_value_advantage_estimates(self, state):
        """
        Get separate value and advantage estimates from Rainbow networks.
        
        Args:
            state: Input state
            
        Returns:
            Tuple of (value1, advantage1, value2, advantage2)
        """
        if isinstance(state, (list, tuple)):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        # Reset noise for consistent evaluation
        if hasattr(self.online_network, 'reset_noise'):
            self.online_network.reset_noise()
        
        with torch.no_grad():
            return self.online_network.get_value_advantage_estimates(state)
    
    def get_training_info(self) -> dict:
        """Get training information including Rainbow statistics."""
        info = super().get_training_info()
        
        # Add Rainbow-specific information
        if hasattr(self.replay_buffer, 'get_priority_stats'):
            priority_stats = self.replay_buffer.get_priority_stats()
            info.update({
                'rainbow_n_step': self.n_step,
                'rainbow_noise_std': self.noise_std_init,
                'rainbow_per_alpha': self.per_alpha,
                'rainbow_per_beta': priority_stats.get('current_beta', self.per_beta),
                'rainbow_max_priority': priority_stats.get('max_priority', 0.0),
                'rainbow_mean_priority': priority_stats.get('mean_priority', 0.0),
            })
        
        return info
    
    def get_algorithm_info(self) -> dict:
        """Get algorithm-specific information."""
        return {
            'algorithm': 'Rainbow DQN',
            'description': 'State-of-the-art DQN combining multiple enhancements',
            'features': [
                'Double DQN - reduced overestimation bias',
                'Dueling Networks - value/advantage separation',
                'Prioritized Experience Replay - sample efficiency',
                'Multi-step learning - temporal credit assignment',
                'Noisy Networks - parameter space exploration',
                'Target network soft updates',
                'Importance sampling correction'
            ],
            'network_type': 'QNetTwinDuelNoisy',
            'replay_type': 'Prioritized',
            'exploration_type': 'NoisyNet',
            'rainbow_config': {
                'n_step': self.n_step,
                'noise_std_init': self.noise_std_init,
                'per_alpha': self.per_alpha,
                'per_beta': self.per_beta,
                'per_beta_annealing_steps': self.per_beta_annealing_steps,
            }
        }
    
    def set_rainbow_parameters(self, 
                              n_step: Optional[int] = None,
                              noise_std: Optional[float] = None,
                              per_alpha: Optional[float] = None,
                              per_beta: Optional[float] = None,
                              max_priority: Optional[float] = None):
        """
        Update Rainbow parameters during training.
        
        Args:
            n_step: Multi-step learning parameter
            noise_std: Noise standard deviation
            per_alpha: PER prioritization exponent
            per_beta: PER importance sampling correction
            max_priority: Maximum priority for new experiences
        """
        if n_step is not None:
            self.n_step = n_step
        
        if noise_std is not None:
            self.noise_std_init = noise_std
        
        if per_alpha is not None:
            self.per_alpha = per_alpha
            if hasattr(self.replay_buffer, 'alpha'):
                self.replay_buffer.alpha = per_alpha
        
        if per_beta is not None:
            self.per_beta = per_beta
            if hasattr(self.replay_buffer, 'beta'):
                self.replay_buffer.beta = per_beta
        
        if max_priority is not None and hasattr(self.replay_buffer, 'set_max_priority'):
            self.replay_buffer.set_max_priority(max_priority)
    
    def reset_priorities(self):
        """Reset all priorities in the replay buffer to maximum."""
        if hasattr(self.replay_buffer, 'reset_priorities'):
            self.replay_buffer.reset_priorities()
    
    def save_checkpoint(self, filepath: str):
        """Save Rainbow agent checkpoint."""
        checkpoint = {
            'config': self.config.to_dict(),
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'last_state': self.last_state,
            'explore_rate': self.explore_rate,
            'n_step': self.n_step,
            'noise_std_init': self.noise_std_init,
            'per_alpha': self.per_alpha,
            'per_beta': self.per_beta,
        }
        
        # Save replay buffer (includes priority tree state)
        import os
        replay_buffer_path = filepath.replace('.pth', '_rainbow_buffer')
        os.makedirs(replay_buffer_path, exist_ok=True)
        self.replay_buffer.save_buffer(replay_buffer_path)
        
        torch.save(checkpoint, filepath)
        print(f"Rainbow DQN checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load Rainbow agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint.get('training_step', 0)
        self.last_state = checkpoint.get('last_state')
        self.explore_rate = checkpoint.get('explore_rate', self.explore_rate)
        self.n_step = checkpoint.get('n_step', self.n_step)
        self.noise_std_init = checkpoint.get('noise_std_init', self.noise_std_init)
        self.per_alpha = checkpoint.get('per_alpha', self.per_alpha)
        self.per_beta = checkpoint.get('per_beta', self.per_beta)
        
        # Load replay buffer (includes priority tree state)
        try:
            import os
            replay_buffer_path = filepath.replace('.pth', '_rainbow_buffer')
            self.replay_buffer.load_buffer(replay_buffer_path)
        except (FileNotFoundError, KeyError):
            print("Warning: Could not load Rainbow replay buffer from checkpoint")
        
        print(f"Rainbow DQN checkpoint loaded from {filepath}")
    
    def __repr__(self) -> str:
        return (f"RainbowDQNAgent("
               f"state_dim={self.state_dim}, "
               f"action_dim={self.action_dim}, "
               f"n_step={self.n_step}, "
               f"noise_std={self.noise_std_init}, "
               f"per_alpha={self.per_alpha}, "
               f"device={self.device}, "
               f"training_step={self.training_step})")