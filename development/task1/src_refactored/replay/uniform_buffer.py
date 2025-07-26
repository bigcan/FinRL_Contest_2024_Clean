"""
Uniform replay buffer implementation for the FinRL Contest 2024 framework.

This module provides standard uniform experience replay functionality
with support for vectorized environments and efficient sampling.
"""

import torch
from typing import Tuple, Union
from torch import Tensor

from .base_buffer import BaseReplayBuffer
from ..core.types import Experience


class UniformReplayBuffer(BaseReplayBuffer):
    """
    Standard uniform experience replay buffer.
    
    Implements uniform random sampling from stored experiences.
    Suitable for off-policy algorithms like DQN variants.
    
    Features:
    - Circular buffer with automatic overflow handling
    - Uniform random sampling
    - Vectorized environment support
    - Efficient tensor operations
    - Device-aware storage
    """
    
    def __init__(self, 
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 device: Union[str, torch.device] = "cpu",
                 num_sequences: int = 1,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize uniform replay buffer.
        
        Args:
            max_size: Maximum buffer capacity
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            device: Device for tensor storage
            num_sequences: Number of parallel sequences (for vectorized envs)
            dtype: Data type for tensors
        """
        super().__init__(
            max_size=max_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            num_sequences=num_sequences,
            dtype=dtype
        )
    
    def add(self, experience: Experience) -> None:
        """
        Add single experience to buffer.
        
        Args:
            experience: Experience tuple containing (state, action, reward, next_state, done)
        """
        # Convert single experience to batch format
        state = experience.state.unsqueeze(0) if experience.state.dim() == 1 else experience.state
        action = experience.action.unsqueeze(0) if experience.action.dim() == 0 else experience.action
        reward = experience.reward.unsqueeze(0) if experience.reward.dim() == 0 else experience.reward
        done = experience.done.unsqueeze(0) if experience.done.dim() == 0 else experience.done
        
        # Ensure tensors are on correct device
        state = state.to(self.device, dtype=self.dtype)
        action = action.to(self.device, dtype=self.dtype)
        reward = reward.to(self.device, dtype=self.dtype)
        done = done.to(self.device, dtype=torch.bool)
        
        # Add to buffer using update method
        self.update((state, action, reward, done))
    
    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample batch of experiences uniformly at random.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, undones, next_states)
            
        Raises:
            ValueError: If buffer doesn't have enough samples
        """
        if not self.is_ready(batch_size):
            raise ValueError(f"Buffer has {len(self)} samples, but {batch_size} requested")
        
        # Calculate sample range (exclude last state since we need next_state)
        sample_length = self.current_size - 1
        
        if sample_length <= 0:
            raise ValueError("Buffer needs at least 2 transitions to sample")
        
        # Generate random indices for sampling
        # ids: random indices for (timestep, sequence) pairs
        total_samples = sample_length * self.num_sequences
        sample_ids = torch.randint(
            total_samples, 
            size=(batch_size,), 
            device=self.device,
            requires_grad=False
        )
        
        # Convert flat indices to (timestep, sequence) coordinates
        timestep_ids = torch.fmod(sample_ids, sample_length)  # timestep indices
        sequence_ids = torch.div(sample_ids, sample_length, rounding_mode='floor')  # sequence indices
        
        # Sample transitions
        states = self.states[timestep_ids, sequence_ids]
        actions = self.actions[timestep_ids, sequence_ids]
        rewards = self.rewards[timestep_ids, sequence_ids]
        undones = self.undones[timestep_ids, sequence_ids]
        next_states = self.states[timestep_ids + 1, sequence_ids]
        
        return states, actions, rewards, undones, next_states
    
    def sample_sequential(self, batch_size: int, sequence_length: int) -> Tuple[Tensor, ...]:
        """
        Sample sequential experiences for algorithms that need temporal context.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of sequential tensors with shape [batch_size, sequence_length, ...]
        """
        if not self.is_ready(batch_size + sequence_length):
            raise ValueError(f"Buffer needs at least {batch_size + sequence_length} samples")
        
        # Sample starting indices ensuring we can get full sequences
        max_start_idx = self.current_size - sequence_length
        start_indices = torch.randint(
            max_start_idx * self.num_sequences,
            size=(batch_size,),
            device=self.device
        )
        
        # Convert to (timestep, sequence) coordinates
        timestep_starts = torch.fmod(start_indices, max_start_idx)
        sequence_ids = torch.div(start_indices, max_start_idx, rounding_mode='floor')
        
        # Create sequence indices
        sequence_offsets = torch.arange(sequence_length, device=self.device).unsqueeze(0)
        timestep_indices = timestep_starts.unsqueeze(1) + sequence_offsets
        sequence_indices = sequence_ids.unsqueeze(1).expand(-1, sequence_length)
        
        # Sample sequential data
        states = self.states[timestep_indices, sequence_indices]
        actions = self.actions[timestep_indices, sequence_indices]
        rewards = self.rewards[timestep_indices, sequence_indices]
        undones = self.undones[timestep_indices, sequence_indices]
        next_states = self.states[timestep_indices + 1, sequence_indices]
        
        return states, actions, rewards, undones, next_states
    
    def get_latest_transitions(self, n_transitions: int) -> Tuple[Tensor, ...]:
        """
        Get the most recently added transitions.
        
        Args:
            n_transitions: Number of recent transitions to retrieve
            
        Returns:
            Tuple of recent transitions
        """
        if n_transitions > self.current_size:
            n_transitions = self.current_size
        
        if n_transitions == 0:
            # Return empty tensors with correct shapes
            batch_shape = (0, self.num_sequences)
            return (
                torch.empty(batch_shape + (self.state_dim,), device=self.device, dtype=self.dtype),
                torch.empty(batch_shape + (self.action_dim,), device=self.device, dtype=self.dtype),
                torch.empty(batch_shape, device=self.device, dtype=self.dtype),
                torch.empty(batch_shape, device=self.device, dtype=self.dtype),
                torch.empty(batch_shape + (self.state_dim,), device=self.device, dtype=self.dtype),
            )
        
        # Get indices for most recent transitions
        if self.is_full:
            # Buffer is full, need to handle wrap-around
            start_idx = max(0, self.pointer - n_transitions)
            if start_idx == 0:
                # Wraps around
                end_part_size = self.pointer
                start_part_size = n_transitions - end_part_size
                
                states = torch.cat([
                    self.states[-start_part_size:],
                    self.states[:end_part_size]
                ], dim=0)
                actions = torch.cat([
                    self.actions[-start_part_size:],
                    self.actions[:end_part_size]
                ], dim=0)
                rewards = torch.cat([
                    self.rewards[-start_part_size:],
                    self.rewards[:end_part_size]
                ], dim=0)
                undones = torch.cat([
                    self.undones[-start_part_size:],
                    self.undones[:end_part_size]
                ], dim=0)
            else:
                # No wrap-around needed
                states = self.states[start_idx:self.pointer]
                actions = self.actions[start_idx:self.pointer]
                rewards = self.rewards[start_idx:self.pointer]
                undones = self.undones[start_idx:self.pointer]
        else:
            # Buffer not full, simple indexing
            start_idx = max(0, self.pointer - n_transitions)
            states = self.states[start_idx:self.pointer]
            actions = self.actions[start_idx:self.pointer]
            rewards = self.rewards[start_idx:self.pointer]
            undones = self.undones[start_idx:self.pointer]
        
        # For next_states, we need to handle the last transition specially
        # (it might not have a valid next_state)
        next_states = torch.zeros_like(states)
        if len(states) > 0:
            if len(states) > 1:
                next_states[:-1] = states[1:]
            # For the last state, use the next state from buffer if available
            if self.current_size > 1:
                next_idx = self.pointer % self.current_size
                next_states[-1] = self.states[next_idx]
        
        return states, actions, rewards, undones, next_states
    
    def compute_statistics(self) -> dict:
        """
        Compute buffer statistics for analysis.
        
        Returns:
            Dictionary of buffer statistics
        """
        if self.current_size == 0:
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'episode_length_mean': 0.0,
                'done_ratio': 0.0,
            }
        
        # Get valid rewards and dones
        if self.is_full:
            valid_rewards = self.rewards.flatten()
            valid_dones = self.dones.flatten()
        else:
            valid_rewards = self.rewards[:self.current_size].flatten()
            valid_dones = self.dones[:self.current_size].flatten()
        
        # Compute reward statistics
        reward_stats = {
            'mean_reward': valid_rewards.mean().item(),
            'std_reward': valid_rewards.std().item(),
            'min_reward': valid_rewards.min().item(),
            'max_reward': valid_rewards.max().item(),
        }
        
        # Compute episode statistics
        done_ratio = valid_dones.float().mean().item()
        episode_length_mean = 1.0 / (done_ratio + 1e-8)  # Approximate
        
        episode_stats = {
            'done_ratio': done_ratio,
            'episode_length_mean': episode_length_mean,
        }
        
        return {**reward_stats, **episode_stats}
    
    def get_buffer_info(self) -> str:
        """Get human-readable buffer information."""
        stats = self.get_stats()
        return (f"UniformReplayBuffer: {stats['current_size']}/{stats['max_size']} "
               f"({stats['utilization']:.1%} full) on {stats['device']}")
    
    def __repr__(self) -> str:
        return self.get_buffer_info()