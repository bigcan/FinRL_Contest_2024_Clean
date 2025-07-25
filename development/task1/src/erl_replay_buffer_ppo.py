#!/usr/bin/env python3
"""
Enhanced Replay Buffer with PPO Support
Supports both DQN (off-policy) and PPO (on-policy) data storage
"""

import torch
from typing import Tuple, List, Optional
from torch import Tensor


class PPOReplayBuffer:
    """
    On-policy replay buffer for PPO
    Stores trajectories with log probabilities and value estimates
    """
    
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1):
        """
        Initialize PPO replay buffer
        
        Args:
            max_size: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension  
            gpu_id: GPU device ID
            num_seqs: Number of parallel sequences/environments
        """
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_seqs = num_seqs
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")
        
        # Buffer pointers
        self.p = 0  # Current pointer
        self.if_full = False
        self.now_len = 0
        
        # Core trajectory data
        self.state = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.action = torch.empty((max_size, 1), dtype=torch.int64, device=self.device)
        self.reward = torch.empty((max_size,), dtype=torch.float32, device=self.device)
        self.done = torch.empty((max_size,), dtype=torch.bool, device=self.device)
        self.next_state = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        
        # PPO-specific data
        self.log_prob = torch.empty((max_size,), dtype=torch.float32, device=self.device)
        self.value = torch.empty((max_size,), dtype=torch.float32, device=self.device)
        self.advantage = torch.empty((max_size,), dtype=torch.float32, device=self.device)
        self.returns = torch.empty((max_size,), dtype=torch.float32, device=self.device)
        
        print(f"📦 PPO Replay Buffer initialized:")
        print(f"   Max size: {max_size}")
        print(f"   State dim: {state_dim}")
        print(f"   Device: {self.device}")
        print(f"   Sequences: {num_seqs}")
    
    def update(self, buffer_items: List[Tensor]) -> None:
        """
        Update buffer with new trajectory data
        
        Args:
            buffer_items: List of [states, actions, rewards, dones, next_states, log_probs, values]
        """
        # Unpack data
        states, actions, rewards, dones, next_states = buffer_items[:5]
        log_probs = buffer_items[5] if len(buffer_items) > 5 else torch.zeros_like(rewards)
        values = buffer_items[6] if len(buffer_items) > 6 else torch.zeros_like(rewards)
        
        # Ensure correct device
        states = states.to(self.device)
        actions = actions.to(self.device) 
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device)
        log_probs = log_probs.to(self.device)
        values = values.to(self.device)
        
        # Get batch size
        add_size = states.shape[0]
        
        # Handle buffer overflow
        if self.p + add_size > self.max_size:
            # Split data to handle wrap-around
            first_part = self.max_size - self.p
            second_part = add_size - first_part
            
            # First part (to end of buffer)
            self.state[self.p:self.p + first_part] = states[:first_part]
            self.action[self.p:self.p + first_part] = actions[:first_part]
            self.reward[self.p:self.p + first_part] = rewards[:first_part]
            self.done[self.p:self.p + first_part] = dones[:first_part]
            self.next_state[self.p:self.p + first_part] = next_states[:first_part]
            self.log_prob[self.p:self.p + first_part] = log_probs[:first_part]
            self.value[self.p:self.p + first_part] = values[:first_part]
            
            # Second part (from beginning of buffer)
            if second_part > 0:
                self.state[:second_part] = states[first_part:]
                self.action[:second_part] = actions[first_part:]
                self.reward[:second_part] = rewards[first_part:]
                self.done[:second_part] = dones[first_part:]
                self.next_state[:second_part] = next_states[first_part:]
                self.log_prob[:second_part] = log_probs[first_part:]
                self.value[:second_part] = values[first_part:]
            
            self.p = second_part
            self.if_full = True
        else:
            # Normal case - no wrap-around
            self.state[self.p:self.p + add_size] = states
            self.action[self.p:self.p + add_size] = actions
            self.reward[self.p:self.p + add_size] = rewards
            self.done[self.p:self.p + add_size] = dones
            self.next_state[self.p:self.p + add_size] = next_states
            self.log_prob[self.p:self.p + add_size] = log_probs
            self.value[self.p:self.p + add_size] = values
            
            self.p += add_size
        
        # Update current length
        self.now_len = self.max_size if self.if_full else self.p
    
    def compute_gae_advantages(self, 
                             gamma: float = 0.995,
                             gae_lambda: float = 0.95) -> None:
        """
        Compute Generalized Advantage Estimation (GAE) for stored trajectories
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        advantages = torch.zeros(self.now_len, device=self.device)
        returns = torch.zeros(self.now_len, device=self.device)
        
        # Compute advantages backward through time
        gae = 0
        for t in reversed(range(self.now_len)):
            if t == self.now_len - 1:
                # Last step - no next state
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = self.value[t + 1]
                next_non_terminal = 1.0 - self.done[t].float()
            
            # TD error
            delta = self.reward[t] + gamma * next_value * next_non_terminal - self.value[t]
            
            # GAE computation
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + self.value[:self.now_len]
        
        # Normalize advantages
        if self.now_len > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store computed values
        self.advantage[:self.now_len] = advantages
        self.returns[:self.now_len] = returns
    
    def get_batch(self, batch_size: int) -> Tuple[Tensor, ...]:
        """
        Get random batch of data for training
        
        Args:
            batch_size: Size of batch to return
            
        Returns:
            Tuple of tensors (states, actions, rewards, dones, next_states, 
                           log_probs, values, advantages, returns)
        """
        # Random indices
        indices = torch.randint(0, self.now_len, size=(batch_size,), device=self.device)
        
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
            self.next_state[indices],
            self.log_prob[indices],
            self.value[indices],
            self.advantage[indices],
            self.returns[indices]
        )
    
    def get_all_data(self) -> Tuple[Tensor, ...]:
        """
        Get all stored data (for on-policy training)
        
        Returns:
            Tuple of all tensors up to current length
        """
        return (
            self.state[:self.now_len],
            self.action[:self.now_len],
            self.reward[:self.now_len],
            self.done[:self.now_len],
            self.next_state[:self.now_len],
            self.log_prob[:self.now_len],
            self.value[:self.now_len],
            self.advantage[:self.now_len],
            self.returns[:self.now_len]
        )
    
    def clear(self) -> None:
        """Clear buffer for new trajectory collection"""
        self.p = 0
        self.if_full = False
        self.now_len = 0
    
    def print_stats(self) -> None:
        """Print buffer statistics"""
        if self.now_len > 0:
            print(f"📊 PPO Buffer Stats:")
            print(f"   Current length: {self.now_len}/{self.max_size}")
            print(f"   Average reward: {self.reward[:self.now_len].mean().item():.4f}")
            print(f"   Average value: {self.value[:self.now_len].mean().item():.4f}")
            print(f"   Average advantage: {self.advantage[:self.now_len].mean().item():.4f}")
            print(f"   Done episodes: {self.done[:self.now_len].sum().item()}")
        else:
            print(f"📊 PPO Buffer: Empty")


def test_ppo_buffer():
    """Test PPO replay buffer functionality"""
    
    print("🧪 Testing PPO Replay Buffer")
    print("=" * 50)
    
    # Create buffer
    buffer = PPOReplayBuffer(
        max_size=1000,
        state_dim=8,
        action_dim=1,
        gpu_id=-1,  # CPU
        num_seqs=4
    )
    
    # Generate sample data
    batch_size = 100
    states = torch.randn(batch_size, 8)
    actions = torch.randint(0, 3, (batch_size, 1))
    rewards = torch.randn(batch_size) * 0.1
    dones = torch.rand(batch_size) < 0.1  # 10% done rate
    next_states = torch.randn(batch_size, 8)
    log_probs = torch.randn(batch_size) * 0.1
    values = torch.randn(batch_size)
    
    # Update buffer
    buffer_items = [states, actions, rewards, dones, next_states, log_probs, values]
    buffer.update(buffer_items)
    
    print(f"\n✅ Buffer updated with {batch_size} samples")
    buffer.print_stats()
    
    # Compute GAE
    buffer.compute_gae_advantages()
    print(f"\n✅ GAE advantages computed")
    buffer.print_stats()
    
    # Test batch sampling
    batch_data = buffer.get_batch(32)
    print(f"\n✅ Batch sampling test:")
    print(f"   Batch shapes: {[x.shape for x in batch_data]}")
    
    # Test clearing
    buffer.clear()
    print(f"\n✅ Buffer cleared")
    buffer.print_stats()
    
    print(f"\n✅ PPO buffer test completed successfully!")


if __name__ == "__main__":
    test_ppo_buffer()