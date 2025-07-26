"""
Prioritized Experience Replay buffer implementation for the FinRL Contest 2024 framework.

This module provides Prioritized Experience Replay (PER) functionality
with sum-tree data structure for efficient priority-based sampling.
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional
from torch import Tensor

from .base_buffer import BaseReplayBuffer
from ..core.types import Experience


class SumTree:
    """
    Binary tree data structure for efficient prioritized sampling.
    
    Implements a sum tree where:
    - Leaf nodes store priority values
    - Internal nodes store sum of their children
    - Root stores total sum of all priorities
    
    Allows O(log n) priority updates and sampling.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize sum tree.
        
        Args:
            capacity: Maximum number of leaf nodes
        """
        self.capacity = capacity
        # Tree has 2*capacity-1 nodes (capacity leaves + capacity-1 internal nodes)
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
    
    def update(self, tree_idx: int, priority: float):
        """
        Update priority of leaf node and propagate change up the tree.
        
        Args:
            tree_idx: Index of leaf node in tree
            priority: New priority value
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def add(self, priority: float, data_idx: int):
        """
        Add new priority value.
        
        Args:
            priority: Priority value to add
            data_idx: Data index (not used in sum tree, but kept for interface consistency)
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    def get_leaf(self, v: float) -> Tuple[int, float, int]:
        """
        Get leaf node corresponding to cumulative sum value.
        
        Args:
            v: Cumulative sum value to search for
            
        Returns:
            Tuple of (leaf_index, priority_value, data_index)
        """
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # Reached leaf node
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx
    
    @property
    def total_priority(self) -> float:
        """Get total sum of all priorities."""
        return self.tree[0]
    
    def get_priorities(self, indices: np.ndarray) -> np.ndarray:
        """
        Get priorities for given data indices.
        
        Args:
            indices: Array of data indices
            
        Returns:
            Array of corresponding priorities
        """
        tree_indices = indices + self.capacity - 1
        return self.tree[tree_indices]


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Implements priority-based sampling where experiences with higher
    temporal difference (TD) errors are sampled more frequently.
    
    Features:
    - Sum tree for efficient priority sampling
    - Importance sampling weights correction
    - Beta annealing for bias correction
    - Configurable alpha for prioritization strength
    
    Reference: "Prioritized Experience Replay" (Schaul et al., 2015)
    """
    
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 device: Union[str, torch.device] = "cpu",
                 num_sequences: int = 1,
                 dtype: torch.dtype = torch.float32,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_annealing_steps: int = 100000,
                 epsilon: float = 1e-6,
                 max_priority: float = 1.0):
        """
        Initialize prioritized replay buffer.
        
        Args:
            max_size: Maximum buffer capacity
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            device: Device for tensor storage
            num_sequences: Number of parallel sequences
            dtype: Data type for tensors
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling initial value (0 = no correction, 1 = full correction)
            beta_annealing_steps: Steps to anneal beta to 1.0
            epsilon: Small constant to ensure non-zero priorities
            max_priority: Maximum priority value for new experiences
        """
        super().__init__(
            max_size=max_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            num_sequences=num_sequences,
            dtype=dtype
        )
        
        # PER hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.beta_initial = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon
        self.max_priority = max_priority
        
        # Sum tree for priority storage
        self.sum_tree = SumTree(capacity=max_size * num_sequences)
        
        # Step counter for beta annealing
        self.step_count = 0
    
    def add(self, experience: Experience) -> None:
        """
        Add single experience with maximum priority.
        
        Args:
            experience: Experience tuple to add
        """
        # Convert to batch format and use update method
        state = experience.state.unsqueeze(0) if experience.state.dim() == 1 else experience.state
        action = experience.action.unsqueeze(0) if experience.action.dim() == 0 else experience.action
        reward = experience.reward.unsqueeze(0) if experience.reward.dim() == 0 else experience.reward
        done = experience.done.unsqueeze(0) if experience.done.dim() == 0 else experience.done
        
        # Ensure tensors are on correct device
        state = state.to(self.device, dtype=self.dtype)
        action = action.to(self.device, dtype=self.dtype)
        reward = reward.to(self.device, dtype=self.dtype)
        done = done.to(self.device, dtype=torch.bool)
        
        self.update((state, action, reward, done))
    
    def update(self, items: Tuple[Tensor, ...]) -> None:
        """
        Update buffer with batch of experiences, assigning maximum priority.
        
        Args:
            items: Tuple of (states, actions, rewards, dones) tensors
        """
        # Store old size for priority assignment
        old_size = self.current_size
        
        # Update buffer using parent method
        super().update(items)
        
        # Assign maximum priority to new experiences
        num_new_experiences = self.add_size * self.num_sequences
        for i in range(num_new_experiences):
            data_idx = (old_size + i // self.num_sequences) % self.max_size
            self.sum_tree.add(self.max_priority ** self.alpha, data_idx)
    
    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample batch of experiences with priority-based sampling.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, undones, next_states, indices, weights)
            where indices are for priority updates and weights are importance sampling weights
        """
        if not self.is_ready(batch_size):
            raise ValueError(f"Buffer has {len(self)} samples, but {batch_size} requested")
        
        # Calculate current beta with annealing
        current_beta = min(1.0, self.beta_initial + 
                          (1.0 - self.beta_initial) * self.step_count / self.beta_annealing_steps)
        self.step_count += 1
        
        # Sample from sum tree
        indices = []
        priorities = []
        segment = self.sum_tree.total_priority / batch_size
        
        for i in range(batch_size):
            # Sample value from segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            # Get corresponding leaf
            tree_idx, priority, data_idx = self.sum_tree.get_leaf(s)
            indices.append(data_idx)
            priorities.append(priority)
        
        indices = np.array(indices)
        priorities = np.array(priorities)
        
        # Convert data indices to (timestep, sequence) coordinates
        sample_length = self.current_size - 1
        if sample_length <= 0:
            raise ValueError("Buffer needs at least 2 transitions to sample")
        
        # Map data indices to valid sampling range
        valid_indices = indices % sample_length
        sequence_indices = np.random.randint(0, self.num_sequences, size=batch_size)
        
        # Convert to tensors
        timestep_indices = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
        sequence_indices = torch.tensor(sequence_indices, device=self.device, dtype=torch.long)
        
        # Sample transitions
        states = self.states[timestep_indices, sequence_indices]
        actions = self.actions[timestep_indices, sequence_indices]
        rewards = self.rewards[timestep_indices, sequence_indices]
        undones = self.undones[timestep_indices, sequence_indices]
        next_states = self.states[timestep_indices + 1, sequence_indices]
        
        # Calculate importance sampling weights
        if self.sum_tree.total_priority > 0:
            # Normalize priorities
            sampling_probs = priorities / self.sum_tree.total_priority
            # Calculate importance sampling weights
            is_weights = np.power(sampling_probs * len(self), -current_beta)
            # Normalize weights
            max_weight = np.max(is_weights)
            is_weights = is_weights / max_weight
        else:
            is_weights = np.ones(batch_size)
        
        # Convert to tensors
        indices_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        weights_tensor = torch.tensor(is_weights, device=self.device, dtype=self.dtype)
        
        return states, actions, rewards, undones, next_states, indices_tensor, weights_tensor
    
    def update_priorities(self, indices: Tensor, priorities: Tensor) -> None:
        """
        Update priorities for given experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values (TD errors)
        """
        # Convert to numpy for sum tree operations
        indices_np = indices.detach().cpu().numpy()
        priorities_np = priorities.detach().cpu().numpy()
        
        # Clip priorities and add epsilon for numerical stability
        priorities_np = np.abs(priorities_np) + self.epsilon
        priorities_np = np.minimum(priorities_np, self.max_priority)
        
        # Update priorities in sum tree
        for idx, priority in zip(indices_np, priorities_np):
            tree_idx = idx + self.sum_tree.capacity - 1
            self.sum_tree.update(tree_idx, priority ** self.alpha)
    
    def get_max_priority(self) -> float:
        """Get current maximum priority in buffer."""
        if self.sum_tree.total_priority > 0:
            return np.max(self.sum_tree.tree[self.sum_tree.capacity-1:])
        return self.max_priority
    
    def set_max_priority(self, max_priority: float) -> None:
        """Set maximum priority for new experiences."""
        self.max_priority = max_priority
    
    def get_priority_stats(self) -> dict:
        """Get priority statistics for analysis."""
        if self.current_size == 0:
            return {
                'total_priority': 0.0,
                'mean_priority': 0.0,
                'max_priority': 0.0,
                'min_priority': 0.0,
                'priority_std': 0.0,
            }
        
        # Get priorities for current data
        valid_size = min(self.current_size, self.max_size)
        priorities = self.sum_tree.tree[self.sum_tree.capacity-1:self.sum_tree.capacity-1+valid_size]
        priorities = priorities[priorities > 0]  # Filter out empty slots
        
        if len(priorities) == 0:
            return {
                'total_priority': 0.0,
                'mean_priority': 0.0,
                'max_priority': 0.0,
                'min_priority': 0.0,
                'priority_std': 0.0,
            }
        
        return {
            'total_priority': self.sum_tree.total_priority,
            'mean_priority': np.mean(priorities),
            'max_priority': np.max(priorities),
            'min_priority': np.min(priorities),
            'priority_std': np.std(priorities),
            'current_beta': min(1.0, self.beta_initial + 
                              (1.0 - self.beta_initial) * self.step_count / self.beta_annealing_steps),
        }
    
    def reset_priorities(self) -> None:
        """Reset all priorities to maximum value."""
        for i in range(self.current_size):
            tree_idx = i + self.sum_tree.capacity - 1
            self.sum_tree.update(tree_idx, self.max_priority ** self.alpha)
    
    def clear(self) -> None:
        """Clear buffer and reset all priorities."""
        super().clear()
        self.sum_tree = SumTree(capacity=self.max_size * self.num_sequences)
        self.step_count = 0
    
    def get_buffer_info(self) -> str:
        """Get human-readable buffer information."""
        stats = self.get_stats()
        priority_stats = self.get_priority_stats()
        return (f"PrioritizedReplayBuffer: {stats['current_size']}/{stats['max_size']} "
               f"({stats['utilization']:.1%} full), "
               f"α={self.alpha:.2f}, β={priority_stats.get('current_beta', self.beta):.2f}, "
               f"max_p={priority_stats['max_priority']:.3f} on {stats['device']}")
    
    def __repr__(self) -> str:
        return self.get_buffer_info()