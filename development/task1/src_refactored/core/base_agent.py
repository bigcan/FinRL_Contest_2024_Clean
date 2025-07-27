"""
Base agent implementation providing common functionality for all RL agents.
"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy

from .interfaces import (
    AgentProtocol, NetworkProtocol, ReplayBufferProtocol, 
    ExplorationProtocol, OptimizerProtocol
)
from .types import (
    StateType, ActionType, BatchExperience, TrainingStats, 
    AgentConfig, TensorType
)


class BaseAgent(ABC):
    """
    Abstract base class for RL agents providing common functionality.
    
    This class implements the AgentProtocol and provides:
    - Common initialization patterns
    - Network management (main and target networks)
    - Save/load functionality
    - Training/evaluation mode switching
    - Basic statistics tracking
    """
    
    def __init__(self, config: AgentConfig, state_dim: int = None, action_dim: int = None, device: Optional[torch.device] = None):
        """
        Initialize base agent.
        
        Args:
            config: Agent configuration
            state_dim: State space dimensionality (optional, can be in config)
            action_dim: Action space dimensionality (optional, can be in config)
            device: PyTorch device (auto-detected if None)
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set dimensions from parameters or config
        self.state_dim = state_dim or getattr(config, 'state_dim', None)
        self.action_dim = action_dim or getattr(config, 'action_dim', None)
        
        # Training mode
        self.training = True
        self.training_mode = True
        
        # Core components (to be set by subclasses)
        self.network: Optional[NetworkProtocol] = None
        self.target_network: Optional[NetworkProtocol] = None
        self.replay_buffer: Optional[ReplayBufferProtocol] = None
        self.exploration: Optional[ExplorationProtocol] = None
        self.optimizer: Optional[OptimizerProtocol] = None
        
        # Training state
        self.training_mode = True
        self.step_count = 0
        self.episode_count = 0
        
        # Statistics tracking
        self.stats_history = []
        self.last_stats: Optional[TrainingStats] = None
        
        # Network update frequency
        self.target_update_freq = getattr(config, 'target_update_freq', 1000)
        self.soft_update_tau = getattr(config, 'soft_update_tau', 0.005)
        
        print(f"Initialized {self.__class__.__name__} on device: {self.device}")
    
    @abstractmethod
    def _build_networks(self) -> None:
        """Build main and target networks. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _compute_loss(self, batch: BatchExperience) -> Tuple[TensorType, Dict[str, float]]:
        """
        Compute training loss for given batch.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        pass
    
    def select_action(self, state: StateType, training: bool = True) -> ActionType:
        """
        Select action given current state.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            q_values = self.network.get_q_values(state)
            
            if training and self.exploration:
                action = self.exploration.select_action(q_values, self.step_count)
            else:
                # Greedy action selection
                action = q_values.argmax(dim=1)
        
        return action.squeeze().item() if action.numel() == 1 else action.squeeze()
    
    def update(self, batch: BatchExperience) -> TrainingStats:
        """
        Update agent with a batch of experiences.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Training statistics
        """
        if not self.training_mode:
            raise RuntimeError("Cannot update agent in evaluation mode")
        
        # Compute loss and metrics
        loss, metrics = self._compute_loss(batch)
        
        # Optimization step
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(loss, metrics.get('performance'))
        else:
            # Fallback to basic optimization
            self.network.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            # Assuming network has an optimizer attribute
            if hasattr(self.network, 'optimizer'):
                self.network.optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        # Update exploration strategy
        if self.exploration:
            self.exploration.update(self.step_count)
        
        # Create training statistics
        stats = TrainingStats(
            step=self.step_count,
            episode_reward=metrics.get('episode_reward', 0.0),
            critic_loss=loss.item(),
            actor_loss=metrics.get('actor_loss'),
            learning_rate=self.optimizer.get_lr() if self.optimizer else None,
            gradient_norm=self.optimizer.get_grad_norm() if self.optimizer else None,
            exploration_rate=metrics.get('exploration_rate')
        )
        
        self.last_stats = stats
        self.stats_history.append(stats)
        self.step_count += 1
        
        return stats
    
    def _update_target_network(self) -> None:
        """Update target network using hard or soft updates."""
        if not self.target_network:
            return
        
        if self.step_count % self.target_update_freq == 0:
            # Hard update
            self.target_network.load_state_dict(self.network.state_dict())
        else:
            # Soft update
            self._soft_update_target_network()
    
    def _soft_update_target_network(self) -> None:
        """Perform soft update of target network."""
        if not self.target_network:
            return
        
        for target_param, main_param in zip(
            self.target_network.parameters(), 
            self.network.parameters()
        ):
            target_param.data.copy_(
                self.soft_update_tau * main_param.data + 
                (1.0 - self.soft_update_tau) * target_param.data
            )
    
    def save(self, path: str) -> None:
        """
        Save agent state to disk.
        
        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save networks
        if self.network:
            torch.save(self.network.state_dict(), os.path.join(path, "network.pth"))
        if self.target_network:
            torch.save(self.target_network.state_dict(), os.path.join(path, "target_network.pth"))
        
        # Save optimizer state
        if self.optimizer and hasattr(self.optimizer, 'state_dict'):
            torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        
        # Save agent state
        state = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config,
            'stats_history': self.stats_history[-100:]  # Keep last 100 stats
        }
        torch.save(state, os.path.join(path, "agent_state.pth"))
        
        print(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load agent state from disk.
        
        Args:
            path: Directory path to load from
        """
        # Load networks
        network_path = os.path.join(path, "network.pth")
        if os.path.exists(network_path) and self.network:
            self.network.load_state_dict(torch.load(network_path, map_location=self.device))
        
        target_network_path = os.path.join(path, "target_network.pth")
        if os.path.exists(target_network_path) and self.target_network:
            self.target_network.load_state_dict(torch.load(target_network_path, map_location=self.device))
        
        # Load optimizer state
        optimizer_path = os.path.join(path, "optimizer.pth")
        if os.path.exists(optimizer_path) and self.optimizer and hasattr(self.optimizer, 'load_state_dict'):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        
        # Load agent state
        agent_state_path = os.path.join(path, "agent_state.pth")
        if os.path.exists(agent_state_path):
            state = torch.load(agent_state_path, map_location=self.device)
            self.step_count = state.get('step_count', 0)
            self.episode_count = state.get('episode_count', 0)
            self.stats_history = state.get('stats_history', [])
        
        print(f"Agent loaded from {path}")
    
    def set_training_mode(self, training: bool) -> None:
        """
        Set agent to training or evaluation mode.
        
        Args:
            training: Whether to set training mode
        """
        self.training_mode = training
        
        if self.network:
            self.network.train(training)
        if self.target_network:
            self.target_network.train(False)  # Target network always in eval mode
    
    def get_stats_summary(self, last_n: int = 100) -> Dict[str, float]:
        """
        Get summary statistics from recent training.
        
        Args:
            last_n: Number of recent steps to summarize
            
        Returns:
            Dictionary of summary statistics
        """
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-last_n:]
        
        return {
            'mean_critic_loss': sum(s.critic_loss for s in recent_stats) / len(recent_stats),
            'mean_episode_reward': sum(s.episode_reward for s in recent_stats) / len(recent_stats),
            'current_lr': recent_stats[-1].learning_rate or 0.0,
            'current_grad_norm': recent_stats[-1].gradient_norm or 0.0,
            'total_steps': self.step_count,
            'total_episodes': self.episode_count
        }
    
    def reset_exploration(self) -> None:
        """Reset exploration state (e.g., for noisy networks)."""
        if self.exploration:
            self.exploration.reset()
    
    @property
    def is_ready(self) -> bool:
        """Check if agent is ready for training/inference."""
        return (
            self.network is not None and
            self.target_network is not None and
            self.replay_buffer is not None
        )