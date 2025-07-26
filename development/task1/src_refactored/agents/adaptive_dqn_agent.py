"""
Adaptive DQN agent implementation for the FinRL Contest 2024 framework.

This module provides DQN agents with advanced adaptive optimization features
including adaptive learning rates, gradient clipping, and performance monitoring.
"""

from typing import Optional, Dict, Any
import torch

from .base_dqn_agent import BaseDQNAgent
from ..core.types import TrainingStats
from ..config import AdaptiveDQNConfig
from ..optimization import create_optimizer_suite


class AdaptiveDQNAgent(BaseDQNAgent):
    """
    Double DQN with Adaptive Learning Rate Scheduling and Advanced Optimization.
    
    Features advanced optimization techniques:
    - Adaptive learning rate scheduling (cosine annealing, plateau reduction, etc.)
    - Adaptive gradient clipping based on gradient norm history
    - Performance-based parameter adjustments
    - Comprehensive training monitoring
    
    This agent is designed for maximum training stability and efficiency.
    """
    
    def __init__(self, 
                 config: Optional[AdaptiveDQNConfig] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize Adaptive DQN agent.
        
        Args:
            config: Agent configuration (will create default if None)
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Computing device
            **kwargs: Additional configuration parameters
        """
        # Create default config if not provided
        if config is None:
            config = AdaptiveDQNConfig(**kwargs)
        
        # Ensure agent type is set correctly
        config.agent_type = "AgentAdaptiveDQN"
        
        # Store adaptive optimization parameters
        self.lr_strategy = config.lr_strategy
        self.adaptive_grad_clip = config.adaptive_grad_clip
        self.lr_T_max = getattr(config, 'lr_T_max', 10000)
        self.lr_patience = getattr(config, 'lr_patience', 100)
        self.lr_factor = getattr(config, 'lr_factor', 0.8)
        self.lr_min = getattr(config, 'lr_min', 1e-8)
        
        super().__init__(config, state_dim, action_dim, device)
        
        # Initialize performance tracking
        self.performance_history = []
        self.training_metrics = {}
    
    def _build_networks(self):
        """Build dueling networks for better value estimation."""
        from ..networks import QNetTwinDuel
        
        # Create online network (use dueling for better performance)
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
        
        print(f"Built Adaptive DQN dueling networks: {self.config.net_dims} dims, "
              f"{self.state_dim} state_dim, {self.action_dim} action_dim, "
              f"lr_strategy={self.lr_strategy}")
    
    def _build_optimizers(self):
        """Build adaptive optimizer with advanced scheduling."""
        # Create optimizer suite with adaptive features
        self.optimizer = create_optimizer_suite(
            self.online_network,
            optimizer_type="adamw",
            lr=self.config.learning_rate,
            scheduler_strategy=self.lr_strategy,
            use_gradient_clipping=True,
            grad_clip_norm=self.clip_grad_norm,
            adaptive_grad_clip=self.adaptive_grad_clip,
            T_max=self.lr_T_max,
            patience=self.lr_patience,
            factor=self.lr_factor,
            eta_min=self.lr_min,
            weight_decay=1e-3,
        )
        
        print(f"Built adaptive optimizer: {self.lr_strategy} scheduling, "
              f"adaptive_grad_clip={self.adaptive_grad_clip}")
    
    def update(self, batch_data) -> TrainingStats:
        """
        Update agent with adaptive optimization and performance tracking.
        
        Args:
            batch_data: Tuple of (states, actions, rewards, dones)
            
        Returns:
            Training statistics with adaptive metrics
        """
        # Call parent update method
        stats = super().update(batch_data)
        
        # Track performance for adaptive adjustments
        self.performance_history.append(stats.critic_loss)
        if len(self.performance_history) > 1000:  # Keep window manageable
            self.performance_history.pop(0)
        
        # Add adaptive optimization statistics
        if hasattr(self.optimizer, 'get_lr'):
            current_lr = self.optimizer.get_lr()
            stats.learning_rate = current_lr
        
        if hasattr(self.optimizer, 'get_grad_statistics'):
            grad_stats = self.optimizer.get_grad_statistics()
            stats.additional_metrics.update({
                'adaptive_grad_norm': grad_stats.get('current_grad_norm', 0.0),
                'mean_grad_norm': grad_stats.get('mean_grad_norm', 0.0),
                'adaptive_lr': self.optimizer.get_lr(),
                'lr_strategy': self.lr_strategy,
            })
        
        # Update training metrics
        self._update_training_metrics(stats)
        
        return stats
    
    def _update_training_metrics(self, stats: TrainingStats):
        """Update comprehensive training metrics."""
        step = self.training_step
        
        # Store key metrics
        self.training_metrics[step] = {
            'critic_loss': stats.critic_loss,
            'q_value': stats.q_value,
            'learning_rate': stats.learning_rate,
            'exploration_rate': stats.exploration_rate,
        }
        
        # Add gradient statistics if available
        if hasattr(self.optimizer, 'get_grad_statistics'):
            grad_stats = self.optimizer.get_grad_statistics()
            self.training_metrics[step].update(grad_stats)
    
    def get_adaptive_statistics(self) -> Dict[str, Any]:
        """Get adaptive optimization statistics."""
        stats = {}
        
        # Learning rate information
        if hasattr(self.optimizer, 'get_lr'):
            stats['current_lr'] = self.optimizer.get_lr()
            stats['initial_lr'] = self.config.learning_rate
            stats['lr_strategy'] = self.lr_strategy
        
        # Gradient statistics
        if hasattr(self.optimizer, 'get_grad_statistics'):
            grad_stats = self.optimizer.get_grad_statistics()
            stats.update(grad_stats)
        
        # Performance statistics
        if self.performance_history:
            import numpy as np
            stats.update({
                'mean_performance': np.mean(self.performance_history[-100:]),
                'std_performance': np.std(self.performance_history[-100:]),
                'performance_trend': (
                    np.mean(self.performance_history[-50:]) - 
                    np.mean(self.performance_history[-100:-50])
                    if len(self.performance_history) >= 100 else 0.0
                ),
            })
        
        # Optimization configuration
        stats.update({
            'adaptive_grad_clip': self.adaptive_grad_clip,
            'lr_T_max': self.lr_T_max,
            'lr_patience': self.lr_patience,
            'lr_factor': self.lr_factor,
        })
        
        return stats
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        info = super().get_training_info()
        
        # Add adaptive-specific information
        adaptive_stats = self.get_adaptive_statistics()
        info.update({
            'adaptive_lr_strategy': self.lr_strategy,
            'adaptive_grad_clip': self.adaptive_grad_clip,
            'current_lr': adaptive_stats.get('current_lr', self.config.learning_rate),
            'performance_trend': adaptive_stats.get('performance_trend', 0.0),
            'mean_grad_norm': adaptive_stats.get('mean_grad_norm', 0.0),
        })
        
        return info
    
    def get_algorithm_info(self) -> dict:
        """Get algorithm-specific information."""
        return {
            'algorithm': 'Adaptive Double DQN',
            'description': 'Double DQN with adaptive learning rate scheduling and optimization',
            'features': [
                f'Adaptive learning rate scheduling ({self.lr_strategy})',
                'Adaptive gradient clipping' if self.adaptive_grad_clip else 'Standard gradient clipping',
                'Performance-based parameter adjustments',
                'Comprehensive training monitoring',
                'Dueling network architecture',
                'Twin Q-networks for reduced overestimation',
                'Target network soft updates',
                'Uniform experience replay',
                'Epsilon-greedy exploration'
            ],
            'network_type': 'QNetTwinDuel',
            'replay_type': 'Uniform',
            'exploration_type': 'Epsilon-greedy',
            'optimization_config': {
                'lr_strategy': self.lr_strategy,
                'adaptive_grad_clip': self.adaptive_grad_clip,
                'lr_T_max': self.lr_T_max,
                'lr_patience': self.lr_patience,
                'lr_factor': self.lr_factor,
            }
        }
    
    def adjust_learning_rate(self, new_lr: float):
        """Manually adjust learning rate."""
        if hasattr(self.optimizer, 'set_lr'):
            self.optimizer.set_lr(new_lr)
            print(f"Learning rate adjusted to {new_lr}")
        else:
            print("Warning: Cannot adjust learning rate for this optimizer")
    
    def get_performance_summary(self, window: int = 100) -> Dict[str, float]:
        """
        Get performance summary over recent training.
        
        Args:
            window: Number of recent steps to analyze
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.performance_history:
            return {}
        
        import numpy as np
        recent_performance = self.performance_history[-window:]
        
        return {
            'mean_loss': np.mean(recent_performance),
            'std_loss': np.std(recent_performance),
            'min_loss': np.min(recent_performance),
            'max_loss': np.max(recent_performance),
            'loss_trend': (
                np.mean(recent_performance[-window//2:]) - 
                np.mean(recent_performance[:window//2])
                if len(recent_performance) >= window else 0.0
            ),
        }
    
    def reset_adaptive_statistics(self):
        """Reset adaptive optimization statistics."""
        self.performance_history = []
        self.training_metrics = {}
        
        # Reset optimizer statistics if available
        if hasattr(self.optimizer, 'reset_statistics'):
            self.optimizer.reset_statistics()
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint including adaptive state."""
        checkpoint = {
            'config': self.config.to_dict(),
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'last_state': self.last_state,
            'explore_rate': self.explore_rate,
            'lr_strategy': self.lr_strategy,
            'adaptive_grad_clip': self.adaptive_grad_clip,
            'performance_history': self.performance_history,
            'training_metrics': self.training_metrics,
        }
        
        # Save replay buffer
        import os
        replay_buffer_path = filepath.replace('.pth', '_adaptive_buffer')
        os.makedirs(replay_buffer_path, exist_ok=True)
        self.replay_buffer.save_buffer(replay_buffer_path)
        
        torch.save(checkpoint, filepath)
        print(f"Adaptive DQN checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint including adaptive state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint.get('training_step', 0)
        self.last_state = checkpoint.get('last_state')
        self.explore_rate = checkpoint.get('explore_rate', self.explore_rate)
        self.lr_strategy = checkpoint.get('lr_strategy', self.lr_strategy)
        self.adaptive_grad_clip = checkpoint.get('adaptive_grad_clip', self.adaptive_grad_clip)
        self.performance_history = checkpoint.get('performance_history', [])
        self.training_metrics = checkpoint.get('training_metrics', {})
        
        # Load replay buffer
        try:
            import os
            replay_buffer_path = filepath.replace('.pth', '_adaptive_buffer')
            self.replay_buffer.load_buffer(replay_buffer_path)
        except (FileNotFoundError, KeyError):
            print("Warning: Could not load adaptive replay buffer from checkpoint")
        
        print(f"Adaptive DQN checkpoint loaded from {filepath}")
    
    def __repr__(self) -> str:
        return (f"AdaptiveDQNAgent("
               f"state_dim={self.state_dim}, "
               f"action_dim={self.action_dim}, "
               f"lr_strategy={self.lr_strategy}, "
               f"adaptive_grad_clip={self.adaptive_grad_clip}, "
               f"device={self.device}, "
               f"training_step={self.training_step})")