"""
Base DQN agent implementation for the FinRL Contest 2024 framework.

This module provides the foundational DQN agent class that implements
core DQN functionality using composition over inheritance.
"""

import os
import torch
from typing import Tuple, Optional, Dict, Any, Union
from copy import deepcopy
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from ..core.base_agent import BaseAgent
from ..core.interfaces import AgentProtocol, ReplayBufferProtocol
from ..core.types import StateType, ActionType, TrainingStats
from ..networks import QNetTwin, QNetTwinDuel, NetworkUtils
from ..replay import UniformReplayBuffer, create_buffer
from ..optimization import create_optimizer_suite, get_recommended_optimizer_config
from ..config import BaseAgentConfig


class BaseDQNAgent(BaseAgent):
    """
    Base Double DQN agent with composition-based architecture.
    
    Implements core DQN functionality:
    - Double DQN for reduced overestimation bias
    - Target network soft updates
    - Experience replay
    - Epsilon-greedy exploration
    - Configurable network architectures
    - Flexible optimization strategies
    """
    
    def __init__(self, 
                 config: BaseAgentConfig,
                 state_dim: int,
                 action_dim: int,
                 device: Optional[torch.device] = None):
        """
        Initialize Base DQN agent.
        
        Args:
            config: Agent configuration
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Computing device
        """
        super().__init__(config, state_dim, action_dim, device)
        
        # Agent-specific parameters
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.repeat_times = config.repeat_times
        self.reward_scale = config.reward_scale
        self.clip_grad_norm = config.clip_grad_norm
        self.soft_update_tau = config.soft_update_tau
        self.state_value_tau = config.state_value_tau
        self.explore_rate = config.explore_rate
        
        # Initialize networks
        self._build_networks()
        
        # Initialize optimizers
        self._build_optimizers()
        
        # Initialize replay buffer
        self._build_replay_buffer()
        
        # Loss function
        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
        
        # Training state
        self.last_state = None
        self.training_step = 0
        
        # Attributes for saving/loading
        self.save_attr_names = {
            'online_network', 'target_network', 'optimizer', 
            'replay_buffer', 'training_step', 'last_state'
        }
    
    def _build_networks(self):
        """Build online and target networks."""
        # Determine network class based on agent type
        if "Duel" in self.config.agent_type:
            network_class = QNetTwinDuel
        else:
            network_class = QNetTwin
        
        # Create online network
        self.online_network = network_class(
            dims=self.config.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        # Create target network (copy of online network)
        self.target_network = deepcopy(self.online_network)
        
        # Set exploration rate for online network
        self.online_network.explore_rate = self.explore_rate
        
        # Initialize networks if needed
        self._initialize_networks()
    
    def _initialize_networks(self):
        """Initialize network weights."""
        # Networks are already initialized by their constructors
        # This method can be overridden for custom initialization
        pass
    
    def _build_optimizers(self):
        """Build optimizer with adaptive features."""
        # Get recommended optimizer configuration
        optimizer_config = get_recommended_optimizer_config(
            self.config.agent_type, 
            model_size="medium"
        )
        
        # Override with config parameters
        optimizer_config.update({
            'lr': self.config.learning_rate,
            'grad_clip_norm': self.clip_grad_norm,
        })
        
        # Create optimizer suite
        self.optimizer = create_optimizer_suite(
            self.online_network,
            **optimizer_config
        )
    
    def _build_replay_buffer(self):
        """Build replay buffer."""
        # Calculate buffer size
        buffer_size = getattr(self.config, 'buffer_size', 100000)
        
        # Determine buffer type
        buffer_type = "uniform"  # Base DQN uses uniform sampling
        
        self.replay_buffer = create_buffer(
            buffer_type=buffer_type,
            max_size=buffer_size,
            state_dim=self.state_dim,
            action_dim=1,  # Discrete actions
            device=self.device,
            num_sequences=getattr(self.config, 'num_sequences', 1)
        )
    
    def select_action(self, state: StateType, deterministic: bool = False) -> ActionType:
        """
        Select action using epsilon-greedy policy.
        
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
        
        if deterministic or not self.training:
            # Deterministic action selection
            with torch.no_grad():
                q1_values, q2_values = self.online_network.get_q1_q2(state)
                q_values = torch.min(q1_values, q2_values)
                action = q_values.argmax(dim=1, keepdim=True)
        else:
            # Epsilon-greedy exploration
            if torch.rand(1).item() < self.explore_rate:
                # Random action
                action = torch.randint(
                    self.action_dim, 
                    size=(state.shape[0], 1), 
                    device=self.device
                )
            else:
                # Greedy action
                with torch.no_grad():
                    q1_values, q2_values = self.online_network.get_q1_q2(state)
                    q_values = torch.min(q1_values, q2_values)
                    action = q_values.argmax(dim=1, keepdim=True)
        
        return action.squeeze().cpu().numpy() if action.shape[0] == 1 else action.cpu().numpy()
    
    def update(self, batch_data: Tuple[Tensor, ...]) -> TrainingStats:
        """
        Update agent with batch of experiences.
        
        Args:
            batch_data: Tuple of (states, actions, rewards, dones)
            
        Returns:
            Training statistics
        """
        states, actions, rewards, dones = batch_data
        
        # Store for replay buffer
        self.replay_buffer.update((states, actions, rewards, dones))
        
        # Update normalization statistics
        with torch.no_grad():
            self._update_normalization_stats(states, rewards, dones)
        
        # Perform training updates
        total_critic_loss = 0.0
        total_q_value = 0.0
        
        if self.replay_buffer.is_ready(self.batch_size):
            update_times = int(self.replay_buffer.add_size * self.repeat_times)
            update_times = max(1, update_times)
            
            for _ in range(update_times):
                critic_loss, q_value = self._update_networks()
                total_critic_loss += critic_loss
                total_q_value += q_value
                
                # Soft update target network
                self._soft_update_target()
                
                self.training_step += 1
            
            # Average losses
            total_critic_loss /= update_times
            total_q_value /= update_times
        
        return TrainingStats(
            actor_loss=0.0,  # DQN doesn't have separate actor
            critic_loss=total_critic_loss,
            policy_entropy=0.0,
            q_value=total_q_value,
            learning_rate=self.optimizer.get_lr(),
            exploration_rate=self.explore_rate,
            training_step=self.training_step
        )
    
    def _update_networks(self) -> Tuple[float, float]:
        """
        Update networks with sampled batch.
        
        Returns:
            Tuple of (critic_loss, average_q_value)
        """
        # Sample batch from replay buffer
        states, actions, rewards, undones, next_states = self.replay_buffer.sample(self.batch_size)
        
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
        
        # Calculate loss
        critic_loss = (
            self.criterion(current_q1_values, target_q_values) + 
            self.criterion(current_q2_values, target_q_values)
        )
        
        # Optimize
        self.optimizer.zero_grad()
        critic_loss.backward()
        
        # Get performance metric for adaptive scheduling
        performance = -critic_loss.item()  # Higher is better
        self.optimizer.step(performance=performance)
        
        return critic_loss.item(), current_q1_values.mean().item()
    
    def _soft_update_target(self):
        """Soft update target network."""
        NetworkUtils.soft_update(
            self.target_network, 
            self.online_network, 
            self.soft_update_tau
        )
    
    def _update_normalization_stats(self, states: Tensor, rewards: Tensor, dones: Tensor):
        """Update running normalization statistics."""
        if self.state_value_tau == 0:
            return
        
        # Update state normalization
        self.online_network.update_normalization_stats(
            states.view(-1, self.state_dim),
            tau=self.state_value_tau
        )
        
        # Update target network normalization to match
        self.target_network.state_avg.data.copy_(self.online_network.state_avg.data)
        self.target_network.state_std.data.copy_(self.online_network.state_std.data)
        
        # Update value normalization if applicable
        if hasattr(self.online_network, 'value_avg'):
            # Calculate cumulative returns for value normalization
            undones = 1.0 - dones.float()
            returns = self._calculate_returns(rewards, undones)
            
            # Update value stats
            returns_avg = returns.mean()
            returns_std = returns.std()
            
            tau = self.state_value_tau
            self.online_network.value_avg.data = (
                self.online_network.value_avg.data * (1 - tau) + returns_avg * tau
            )
            self.online_network.value_std.data = (
                self.online_network.value_std.data * (1 - tau) + returns_std * tau + 1e-4
            )
            
            # Update target network to match
            self.target_network.value_avg.data.copy_(self.online_network.value_avg.data)
            self.target_network.value_std.data.copy_(self.online_network.value_std.data)
    
    def _calculate_returns(self, rewards: Tensor, undones: Tensor) -> Tensor:
        """Calculate discounted returns for normalization."""
        returns = torch.zeros_like(rewards)
        horizon_len = rewards.shape[0]
        
        # Simple return calculation (can be improved with actual next state values)
        for t in range(horizon_len - 1, -1, -1):
            if t == horizon_len - 1:
                returns[t] = rewards[t]
            else:
                returns[t] = rewards[t] + undones[t] * self.gamma * returns[t + 1]
        
        return returns
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        checkpoint = {
            'config': self.config.to_dict(),
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'last_state': self.last_state,
            'explore_rate': self.explore_rate,
        }
        
        # Save replay buffer separately if needed
        replay_buffer_path = filepath.replace('.pth', '_buffer.pth')
        self.replay_buffer.save_buffer(os.path.dirname(replay_buffer_path))
        
        torch.save(checkpoint, filepath)
        print(f"Agent checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint.get('training_step', 0)
        self.last_state = checkpoint.get('last_state')
        self.explore_rate = checkpoint.get('explore_rate', self.explore_rate)
        
        # Load replay buffer if exists
        try:
            replay_buffer_dir = os.path.dirname(filepath)
            self.replay_buffer.load_buffer(replay_buffer_dir)
        except (FileNotFoundError, KeyError):
            print("Warning: Could not load replay buffer from checkpoint")
        
        print(f"Agent checkpoint loaded from {filepath}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get current training information."""
        info = super().get_training_info()
        info.update({
            'training_step': self.training_step,
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_utilization': len(self.replay_buffer) / self.replay_buffer.max_size,
            'target_network_updates': self.training_step,
            'exploration_rate': self.explore_rate,
            'gamma': self.gamma,
            'soft_update_tau': self.soft_update_tau,
        })
        return info
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        super().set_training_mode(training)
        self.online_network.train(training)
        self.target_network.train(False)  # Target network always in eval mode
    
    def to_device(self, device: torch.device):
        """Move agent to device."""
        super().to_device(device)
        self.online_network = self.online_network.to(device)
        self.target_network = self.target_network.to(device)
        self.replay_buffer = self.replay_buffer.to_device(device)
        return self
    
    def get_q_values(self, state: StateType) -> Tuple[Tensor, Tensor]:
        """
        Get Q-values for debugging/analysis.
        
        Args:
            state: Input state
            
        Returns:
            Tuple of Q-values from both networks
        """
        if isinstance(state, (list, tuple)):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        with torch.no_grad():
            return self.online_network.get_q1_q2(state)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
               f"state_dim={self.state_dim}, "
               f"action_dim={self.action_dim}, "
               f"device={self.device}, "
               f"training_step={self.training_step})")