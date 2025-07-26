"""
Prioritized DQN agent implementation for the FinRL Contest 2024 framework.

This module provides DQN agents that use Prioritized Experience Replay (PER)
for improved sample efficiency and learning stability.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor

from .base_dqn_agent import BaseDQNAgent
from ..core.types import TrainingStats
from ..config import PrioritizedDQNConfig
from ..replay import PrioritizedReplayBuffer


class PrioritizedDQNAgent(BaseDQNAgent):
    """
    Double DQN with Prioritized Experience Replay.
    
    Implements Prioritized Experience Replay (PER) to sample more important
    experiences more frequently, leading to improved sample efficiency.
    
    Features:
    - Prioritized replay buffer with sum tree
    - Importance sampling weight correction
    - TD error-based priority updates
    - Beta annealing for bias correction
    """
    
    def __init__(self, 
                 config: Optional[PrioritizedDQNConfig] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize Prioritized DQN agent.
        
        Args:
            config: Agent configuration (will create default if None)
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Computing device
            **kwargs: Additional configuration parameters
        """
        # Create default config if not provided
        if config is None:
            config = PrioritizedDQNConfig(**kwargs)
        
        # Ensure agent type is set correctly
        config.agent_type = "AgentPrioritizedDQN"
        
        # Store PER parameters
        self.per_alpha = config.per_alpha
        self.per_beta = config.per_beta
        self.per_beta_annealing_steps = config.per_beta_annealing_steps
        
        super().__init__(config, state_dim, action_dim, device)
    
    def _build_networks(self):
        """Build dueling networks for better value estimation with PER."""
        from ..networks import QNetTwinDuel
        
        # Create online network (use dueling for better PER performance)
        self.online_network = QNetTwinDuel(
            dims=self.config.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        # Create target network (copy of online network)
        from copy import deepcopy
        self.target_network = deepcopy(self.online_network)
        
        # Set exploration rate for online network
        self.online_network.explore_rate = self.explore_rate
        
        print(f"Built Prioritized DQN dueling networks: {self.config.net_dims} dims, "
              f"{self.state_dim} state_dim, {self.action_dim} action_dim")
    
    def _build_replay_buffer(self):
        """Build prioritized replay buffer."""
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
        
        print(f"Built Prioritized Replay Buffer: size={buffer_size}, "
              f"alpha={self.per_alpha}, beta={self.per_beta}")
    
    def _update_networks(self) -> Tuple[float, float]:
        """
        Update networks with prioritized sampling and importance weights.
        
        Returns:
            Tuple of (critic_loss, average_q_value)
        """
        # Sample batch from prioritized replay buffer
        (states, actions, rewards, undones, next_states, 
         indices, importance_weights) = self.replay_buffer.sample(self.batch_size)
        
        # Calculate target Q-values using Double DQN
        with torch.no_grad():
            # Use online network for action selection
            next_q1_online, next_q2_online = self.online_network.get_q1_q2(next_states)
            next_actions = torch.min(next_q1_online, next_q2_online).argmax(dim=1, keepdim=True)
            
            # Use target network for Q-value evaluation
            next_q1_target, next_q2_target = self.target_network.get_q1_q2(next_states)
            next_q_values = torch.min(next_q1_target, next_q2_target).gather(1, next_actions).squeeze(1)
            
            # Calculate target Q-values
            target_q_values = rewards + undones * self.gamma * next_q_values
        
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
        Update agent with batch of experiences using PER.
        
        Args:
            batch_data: Tuple of (states, actions, rewards, dones)
            
        Returns:
            Training statistics including PER metrics
        """
        # Call parent update method
        stats = super().update(batch_data)
        
        # Add PER-specific statistics
        if hasattr(self.replay_buffer, 'get_priority_stats'):
            priority_stats = self.replay_buffer.get_priority_stats()
            stats.additional_metrics.update({
                'per_beta': priority_stats.get('current_beta', self.per_beta),
                'max_priority': priority_stats.get('max_priority', 0.0),
                'mean_priority': priority_stats.get('mean_priority', 0.0),
                'total_priority': priority_stats.get('total_priority', 0.0),
            })
        
        return stats
    
    def get_training_info(self) -> dict:
        """Get training information including PER statistics."""
        info = super().get_training_info()
        
        # Add PER-specific information
        if hasattr(self.replay_buffer, 'get_priority_stats'):
            priority_stats = self.replay_buffer.get_priority_stats()
            info.update({
                'per_alpha': self.per_alpha,
                'per_beta': priority_stats.get('current_beta', self.per_beta),
                'per_max_priority': priority_stats.get('max_priority', 0.0),
                'per_mean_priority': priority_stats.get('mean_priority', 0.0),
                'per_priority_std': priority_stats.get('priority_std', 0.0),
            })
        
        return info
    
    def get_algorithm_info(self) -> dict:
        """Get algorithm-specific information."""
        return {
            'algorithm': 'Prioritized Double DQN',
            'description': 'Double DQN with Prioritized Experience Replay for improved sample efficiency',
            'features': [
                'Prioritized Experience Replay (PER)',
                'Importance sampling weight correction',
                'TD error-based priority updates',
                'Beta annealing for bias correction',
                'Dueling network architecture',
                'Twin Q-networks for reduced overestimation',
                'Target network soft updates',
                'Epsilon-greedy exploration'
            ],
            'network_type': 'QNetTwinDuel',
            'replay_type': 'Prioritized',
            'exploration_type': 'Epsilon-greedy',
            'per_config': {
                'alpha': self.per_alpha,
                'beta': self.per_beta,
                'beta_annealing_steps': self.per_beta_annealing_steps,
            }
        }
    
    def set_per_parameters(self, alpha: Optional[float] = None, 
                          beta: Optional[float] = None,
                          max_priority: Optional[float] = None):
        """
        Update PER parameters during training.
        
        Args:
            alpha: Prioritization exponent
            beta: Importance sampling correction
            max_priority: Maximum priority for new experiences
        """
        if alpha is not None:
            self.per_alpha = alpha
            if hasattr(self.replay_buffer, 'alpha'):
                self.replay_buffer.alpha = alpha
        
        if beta is not None:
            self.per_beta = beta
            if hasattr(self.replay_buffer, 'beta'):
                self.replay_buffer.beta = beta
        
        if max_priority is not None and hasattr(self.replay_buffer, 'set_max_priority'):
            self.replay_buffer.set_max_priority(max_priority)
    
    def reset_priorities(self):
        """Reset all priorities in the replay buffer to maximum."""
        if hasattr(self.replay_buffer, 'reset_priorities'):
            self.replay_buffer.reset_priorities()
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint including PER state."""
        checkpoint = {
            'config': self.config.to_dict(),
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'last_state': self.last_state,
            'explore_rate': self.explore_rate,
            'per_alpha': self.per_alpha,
            'per_beta': self.per_beta,
        }
        
        # Save replay buffer (includes priority tree state)
        import os
        replay_buffer_path = filepath.replace('.pth', '_buffer')
        os.makedirs(replay_buffer_path, exist_ok=True)
        self.replay_buffer.save_buffer(replay_buffer_path)
        
        torch.save(checkpoint, filepath)
        print(f"Prioritized DQN checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint including PER state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint.get('training_step', 0)
        self.last_state = checkpoint.get('last_state')
        self.explore_rate = checkpoint.get('explore_rate', self.explore_rate)
        self.per_alpha = checkpoint.get('per_alpha', self.per_alpha)
        self.per_beta = checkpoint.get('per_beta', self.per_beta)
        
        # Load replay buffer (includes priority tree state)
        try:
            import os
            replay_buffer_path = filepath.replace('.pth', '_buffer')
            self.replay_buffer.load_buffer(replay_buffer_path)
        except (FileNotFoundError, KeyError):
            print("Warning: Could not load prioritized replay buffer from checkpoint")
        
        print(f"Prioritized DQN checkpoint loaded from {filepath}")
    
    def __repr__(self) -> str:
        return (f"PrioritizedDQNAgent("
               f"state_dim={self.state_dim}, "
               f"action_dim={self.action_dim}, "
               f"per_alpha={self.per_alpha}, "
               f"per_beta={self.per_beta}, "
               f"device={self.device}, "
               f"training_step={self.training_step})")