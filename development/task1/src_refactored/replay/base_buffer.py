"""
Base replay buffer components and interfaces for the FinRL Contest 2024 framework.

This module provides foundational replay buffer classes and utilities
that are shared across different buffer implementations.
"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict, Any
from torch import Tensor

from ..core.types import StateType, ActionType, RewardType, DoneType, Experience


class BaseReplayBuffer(ABC):
    """
    Abstract base class for replay buffers.
    
    Defines the interface that all replay buffer implementations must follow.
    Provides common functionality for device management, state tracking, and
    buffer operations.
    """
    
    def __init__(self, 
                 max_size: int,
                 state_dim: int, 
                 action_dim: int,
                 device: Union[str, torch.device] = "cpu",
                 num_sequences: int = 1,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize base replay buffer.
        
        Args:
            max_size: Maximum buffer capacity
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space  
            device: Device for tensor storage
            num_sequences: Number of parallel sequences (for vectorized envs)
            dtype: Data type for tensors
        """
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_sequences = num_sequences
        self.dtype = dtype
        
        # Device management
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        # Buffer state tracking
        self.pointer = 0          # Current insertion pointer
        self.is_full = False      # Whether buffer has been filled once
        self.current_size = 0     # Current number of valid entries
        self.add_size = 0         # Size of last addition
        self.add_item = None      # Last added item (for normalization)
        
        # Initialize storage tensors
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage tensors for states, actions, rewards, and dones."""
        self.states = torch.empty(
            (self.max_size, self.num_sequences, self.state_dim),
            dtype=self.dtype,
            device=self.device
        )
        self.actions = torch.empty(
            (self.max_size, self.num_sequences, self.action_dim),
            dtype=self.dtype,
            device=self.device
        )
        self.rewards = torch.empty(
            (self.max_size, self.num_sequences),
            dtype=self.dtype,
            device=self.device
        )
        self.dones = torch.empty(
            (self.max_size, self.num_sequences),
            dtype=torch.bool,
            device=self.device
        )
        
        # Store undones (1.0 - dones) for convenience in computation
        self.undones = torch.empty(
            (self.max_size, self.num_sequences),
            dtype=self.dtype,
            device=self.device
        )
    
    @abstractmethod
    def add(self, experience: Experience) -> None:
        """
        Add experience to buffer.
        
        Args:
            experience: Experience tuple to add
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        """
        Sample batch of experiences from buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of sampled tensors
        """
        pass
    
    def update(self, items: Tuple[Tensor, ...]) -> None:
        """
        Update buffer with batch of experiences.
        
        Args:
            items: Tuple of (states, actions, rewards, dones) tensors
        """
        self.add_item = items
        states, actions, rewards, dones = items
        
        # Validate input shapes
        self._validate_input_shapes(states, actions, rewards, dones)
        
        # Convert dones to undones
        undones = 1.0 - dones.float()
        
        # Calculate addition size
        self.add_size = rewards.shape[0]
        
        # Calculate new pointer position
        new_pointer = self.pointer + self.add_size
        
        if new_pointer > self.max_size:
            # Handle buffer overflow with circular wrapping
            self._handle_overflow(states, actions, rewards, undones, dones, new_pointer)
        else:
            # Simple insertion
            self._insert_data(states, actions, rewards, undones, dones, 
                            self.pointer, new_pointer)
        
        # Update buffer state
        self.pointer = new_pointer % self.max_size
        self.current_size = self.max_size if self.is_full else self.pointer
    
    def _validate_input_shapes(self, states: Tensor, actions: Tensor, 
                             rewards: Tensor, dones: Tensor) -> None:
        """Validate that input tensors have correct shapes."""
        expected_shapes = [
            (states, (None, self.num_sequences, self.state_dim)),
            (actions, (None, self.num_sequences, self.action_dim)),
            (rewards, (None, self.num_sequences)),
            (dones, (None, self.num_sequences)),
        ]
        
        for tensor, expected_shape in expected_shapes:
            actual_shape = tensor.shape
            if len(actual_shape) != len(expected_shape):
                raise ValueError(f"Shape mismatch: expected {expected_shape}, got {actual_shape}")
            
            for i, (actual, expected) in enumerate(zip(actual_shape[1:], expected_shape[1:])):
                if expected is not None and actual != expected:
                    raise ValueError(f"Shape mismatch at dimension {i+1}: expected {expected}, got {actual}")
    
    def _handle_overflow(self, states: Tensor, actions: Tensor, rewards: Tensor, 
                        undones: Tensor, dones: Tensor, new_pointer: int) -> None:
        """Handle buffer overflow with circular wrapping."""
        self.is_full = True
        
        # Calculate split points
        first_part_size = self.max_size - self.pointer
        second_part_size = self.add_size - first_part_size
        
        # Insert first part (to end of buffer)
        if first_part_size > 0:
            self._insert_data(
                states[:first_part_size], actions[:first_part_size],
                rewards[:first_part_size], undones[:first_part_size], dones[:first_part_size],
                self.pointer, self.max_size
            )
        
        # Insert second part (from beginning of buffer)
        if second_part_size > 0:
            self._insert_data(
                states[first_part_size:], actions[first_part_size:],
                rewards[first_part_size:], undones[first_part_size:], dones[first_part_size:],
                0, second_part_size
            )
    
    def _insert_data(self, states: Tensor, actions: Tensor, rewards: Tensor,
                    undones: Tensor, dones: Tensor, start_idx: int, end_idx: int) -> None:
        """Insert data into buffer at specified indices."""
        self.states[start_idx:end_idx] = states
        self.actions[start_idx:end_idx] = actions
        self.rewards[start_idx:end_idx] = rewards
        self.undones[start_idx:end_idx] = undones
        self.dones[start_idx:end_idx] = dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.current_size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for the requested batch size."""
        return len(self) >= batch_size
    
    def clear(self) -> None:
        """Clear buffer by resetting pointers and flags."""
        self.pointer = 0
        self.is_full = False
        self.current_size = 0
        self.add_size = 0
        self.add_item = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'current_size': self.current_size,
            'max_size': self.max_size,
            'utilization': self.current_size / self.max_size,
            'is_full': self.is_full,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }
    
    def save_buffer(self, save_path: str) -> None:
        """
        Save buffer contents to disk.
        
        Args:
            save_path: Directory to save buffer files
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Get valid data range
        if self.current_size == self.pointer:
            # Buffer not wrapped
            valid_states = self.states[:self.current_size]
            valid_actions = self.actions[:self.current_size]
            valid_rewards = self.rewards[:self.current_size]
            valid_undones = self.undones[:self.current_size]
        else:
            # Buffer wrapped, concatenate end and beginning
            valid_states = torch.cat([
                self.states[self.pointer:self.current_size],
                self.states[:self.pointer]
            ], dim=0)
            valid_actions = torch.cat([
                self.actions[self.pointer:self.current_size],
                self.actions[:self.pointer]
            ], dim=0)
            valid_rewards = torch.cat([
                self.rewards[self.pointer:self.current_size],
                self.rewards[:self.pointer]
            ], dim=0)
            valid_undones = torch.cat([
                self.undones[self.pointer:self.current_size],
                self.undones[:self.pointer]
            ], dim=0)
        
        # Save tensors
        items = [
            (valid_states, "states"),
            (valid_actions, "actions"), 
            (valid_rewards, "rewards"),
            (valid_undones, "undones"),
        ]
        
        for tensor, name in items:
            file_path = os.path.join(save_path, f"replay_buffer_{name}.pth")
            torch.save(tensor, file_path)
            print(f"Saved buffer {name} to {file_path}")
        
        # Save metadata
        metadata = {
            'max_size': self.max_size,
            'current_size': self.current_size,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'num_sequences': self.num_sequences,
            'is_full': self.is_full,
            'pointer': self.pointer,
        }
        metadata_path = os.path.join(save_path, "buffer_metadata.pth")
        torch.save(metadata, metadata_path)
        print(f"Saved buffer metadata to {metadata_path}")
    
    def load_buffer(self, load_path: str) -> None:
        """
        Load buffer contents from disk.
        
        Args:
            load_path: Directory containing buffer files
        """
        # Check if all required files exist
        required_files = ["states", "actions", "rewards", "undones"]
        file_paths = [os.path.join(load_path, f"replay_buffer_{name}.pth") for name in required_files]
        metadata_path = os.path.join(load_path, "buffer_metadata.pth")
        
        if not all(os.path.isfile(path) for path in file_paths + [metadata_path]):
            raise FileNotFoundError(f"Missing buffer files in {load_path}")
        
        # Load metadata
        metadata = torch.load(metadata_path, map_location=self.device)
        
        # Validate metadata compatibility
        if (metadata['state_dim'] != self.state_dim or 
            metadata['action_dim'] != self.action_dim or
            metadata['num_sequences'] != self.num_sequences):
            raise ValueError("Buffer metadata incompatible with current buffer configuration")
        
        # Load data
        for i, name in enumerate(required_files):
            tensor = torch.load(file_paths[i], map_location=self.device)
            size = tensor.shape[0]
            
            if name == "states":
                self.states[:size] = tensor
            elif name == "actions":
                self.actions[:size] = tensor
            elif name == "rewards":
                self.rewards[:size] = tensor
            elif name == "undones":
                self.undones[:size] = tensor
                
            print(f"Loaded buffer {name} from {file_paths[i]}")
        
        # Restore buffer state
        self.current_size = metadata['current_size']
        self.pointer = metadata['pointer']
        self.is_full = metadata['is_full']
        
        print(f"Buffer loaded successfully: {self.current_size}/{self.max_size} entries")
    
    def to_device(self, device: Union[str, torch.device]) -> 'BaseReplayBuffer':
        """
        Move buffer to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        if device != self.device:
            self.device = device
            self.states = self.states.to(device)
            self.actions = self.actions.to(device)
            self.rewards = self.rewards.to(device)
            self.undones = self.undones.to(device)
            self.dones = self.dones.to(device)
        
        return self