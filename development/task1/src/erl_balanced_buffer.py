"""
Balanced Experience Replay Buffer
Addresses conservative trading by maintaining balanced action distributions in replay
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque
from erl_replay_buffer import ReplayBuffer


class BalancedReplayBuffer(ReplayBuffer):
    """
    Experience replay buffer that maintains balanced action distributions
    to prevent conservative convergence during training
    """
    
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
                 balance_ratio: float = 0.3,
                 action_bins: int = 3):
        """
        Initialize balanced replay buffer
        
        Args:
            max_size: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            gpu_id: GPU device ID
            num_seqs: Number of sequences
            balance_ratio: Target ratio for each action type (1/3 for 3 actions)
            action_bins: Number of action categories (buy/hold/sell = 3)
        """
        super().__init__(max_size, state_dim, action_dim, gpu_id, num_seqs)
        
        self.balance_ratio = balance_ratio
        self.action_bins = action_bins
        
        # Track indices by action type for balanced sampling
        self.action_indices = {i: deque(maxlen=max_size//action_bins) 
                              for i in range(action_bins)}
        self.action_counts = {i: 0 for i in range(action_bins)}
        
        # Statistics tracking
        self.sampling_stats = {
            'total_samples': 0,
            'balanced_samples': 0,
            'action_distribution': {i: 0 for i in range(action_bins)}
        }
        
    def update(self, items: Tuple[torch.Tensor, ...]) -> None:
        """
        Update buffer with new transitions, tracking action types
        
        Args:
            items: (states, actions, rewards, undones)
        """
        states, actions, rewards, undones = items
        
        # Convert to CPU for processing
        actions_cpu = actions.cpu().numpy()
        
        # Get the actual buffer size before update
        old_size = self.cur_size
        
        # Call parent update
        super().update(items)
        
        # Track new indices by action type
        new_size = self.cur_size
        
        if new_size > old_size:
            # Process new items
            num_new = new_size - old_size
            new_indices = np.arange(old_size, new_size) % self.max_size
            
            # Flatten actions for categorization
            flat_actions = actions_cpu.flatten()[-num_new:]
            
            for i, (idx, action) in enumerate(zip(new_indices, flat_actions)):
                action_bin = int(action)  # Assuming discrete actions 0,1,2
                if 0 <= action_bin < self.action_bins:
                    self.action_indices[action_bin].append(idx)
                    self.action_counts[action_bin] += 1
                    
    def sample_balanced(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with balanced action distribution
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Balanced batch of (states, actions, rewards, undones, next_states)
        """
        if self.cur_size < batch_size:
            return self.sample(batch_size)
            
        # Calculate samples per action type
        samples_per_action = batch_size // self.action_bins
        remainder = batch_size % self.action_bins
        
        # Collect indices for balanced sampling
        balanced_indices = []
        
        for action_type in range(self.action_bins):
            available_indices = list(self.action_indices[action_type])
            
            if len(available_indices) == 0:
                continue
                
            # Sample from this action type
            n_samples = samples_per_action
            if action_type < remainder:
                n_samples += 1
                
            if len(available_indices) >= n_samples:
                sampled = np.random.choice(available_indices, n_samples, replace=False)
            else:
                # Not enough unique samples, allow replacement
                sampled = np.random.choice(available_indices, n_samples, replace=True)
                
            balanced_indices.extend(sampled)
            
            # Update statistics
            self.sampling_stats['action_distribution'][action_type] += n_samples
            
        # If we don't have enough balanced samples, fill with random
        if len(balanced_indices) < batch_size:
            n_random = batch_size - len(balanced_indices)
            random_indices = np.random.randint(self.cur_size, size=n_random)
            balanced_indices.extend(random_indices)
            
        # Shuffle indices
        np.random.shuffle(balanced_indices)
        indices = torch.tensor(balanced_indices[:batch_size], device=self.device)
        
        # Sample using indices
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        undones = self.undones[indices]
        
        # Calculate next states
        next_indices = (indices + 1) % self.max_size
        next_states = self.states[next_indices]
        
        # Update statistics
        self.sampling_stats['total_samples'] += batch_size
        self.sampling_stats['balanced_samples'] += len(balanced_indices)
        
        return states, actions, rewards, undones, next_states
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with optional balancing
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Batch of (states, actions, rewards, undones, next_states)
        """
        # Use balanced sampling 70% of the time
        if np.random.random() < 0.7 and self.cur_size >= batch_size * 2:
            return self.sample_balanced(batch_size)
        else:
            return super().sample(batch_size)
            
    def get_action_distribution(self) -> Dict[int, float]:
        """Get current action distribution in buffer"""
        total = sum(self.action_counts.values())
        if total == 0:
            return {i: 0.0 for i in range(self.action_bins)}
            
        return {i: count/total for i, count in self.action_counts.items()}
        
    def get_sampling_stats(self) -> Dict:
        """Get sampling statistics"""
        stats = self.sampling_stats.copy()
        stats['current_action_distribution'] = self.get_action_distribution()
        stats['balance_effectiveness'] = (
            stats['balanced_samples'] / max(1, stats['total_samples'])
        )
        return stats
        
    def reset_stats(self):
        """Reset sampling statistics"""
        self.sampling_stats = {
            'total_samples': 0,
            'balanced_samples': 0,
            'action_distribution': {i: 0 for i in range(self.action_bins)}
        }


class PrioritizedBalancedBuffer(BalancedReplayBuffer):
    """
    Combines prioritized experience replay with action balancing
    """
    
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 balance_ratio: float = 0.3):
        """
        Initialize prioritized balanced buffer
        
        Args:
            max_size: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            gpu_id: GPU device ID
            num_seqs: Number of sequences
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            balance_ratio: Target ratio for each action type
        """
        super().__init__(max_size, state_dim, action_dim, gpu_id, 
                        num_seqs, balance_ratio)
        
        self.alpha = alpha
        self.beta = beta
        
        # Priority tree
        self.priorities = torch.ones(max_size, device=self.device)
        self.max_priority = 1.0
        
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for given indices"""
        self.priorities[indices] = priorities.pow(self.alpha)
        self.max_priority = self.priorities[:self.cur_size].max()
        
    def sample_prioritized_balanced(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample with both prioritization and action balancing
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Batch with importance weights
        """
        if self.cur_size < batch_size:
            return self.sample(batch_size)
            
        # Get priorities for current buffer
        priorities = self.priorities[:self.cur_size]
        
        # Calculate sampling probabilities
        probs = priorities / priorities.sum()
        
        # Adjust probabilities for action balancing
        action_weights = torch.ones_like(probs)
        action_dist = self.get_action_distribution()
        
        for action_type, current_ratio in action_dist.items():
            if current_ratio > 0:
                # Reduce weight for over-represented actions
                weight_adjustment = self.balance_ratio / max(current_ratio, 0.01)
                weight_adjustment = np.clip(weight_adjustment, 0.5, 2.0)
                
                # Apply adjustment to relevant indices
                action_mask = (self.actions[:self.cur_size, 0] == action_type)
                action_weights[action_mask] *= weight_adjustment
                
        # Combine priorities with action balancing
        adjusted_probs = (probs * action_weights) / (probs * action_weights).sum()
        
        # Sample indices
        indices = torch.multinomial(adjusted_probs, batch_size, replacement=True)
        
        # Calculate importance sampling weights
        weights = (self.cur_size * adjusted_probs[indices]).pow(-self.beta)
        weights = weights / weights.max()
        
        # Get batch
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        undones = self.undones[indices]
        
        # Next states
        next_indices = (indices + 1) % self.max_size
        next_states = self.states[next_indices]
        
        return states, actions, rewards, undones, next_states, indices, weights


# Testing
if __name__ == "__main__":
    print("🧪 Testing Balanced Replay Buffer")
    print("=" * 50)
    
    # Create buffer
    buffer = BalancedReplayBuffer(
        max_size=10000,
        state_dim=10,
        action_dim=1,
        gpu_id=-1,  # CPU
        action_bins=3
    )
    
    # Simulate conservative trading data (mostly hold actions)
    print("\n📊 Adding conservative trading data:")
    for i in range(1000):
        state = torch.randn(1, 10)
        
        # 80% hold, 10% buy, 10% sell (conservative)
        action_probs = [0.1, 0.8, 0.1]
        action = torch.tensor([[np.random.choice(3, p=action_probs)]])
        
        reward = torch.randn(1)
        done = torch.tensor([False])
        
        buffer.update((state, action, reward, done))
        
    # Check action distribution
    action_dist = buffer.get_action_distribution()
    print(f"Buffer action distribution: {action_dist}")
    
    # Sample balanced batch
    print("\n📊 Sampling balanced batches:")
    for i in range(5):
        batch = buffer.sample_balanced(32)
        sampled_actions = batch[1].cpu().numpy().flatten()
        
        action_counts = {i: 0 for i in range(3)}
        for a in sampled_actions:
            action_counts[int(a)] += 1
            
        print(f"Batch {i+1} action distribution: {action_counts}")
        
    # Get statistics
    stats = buffer.get_sampling_stats()
    print(f"\n📊 Sampling statistics:")
    print(f"   Balance effectiveness: {stats['balance_effectiveness']:.2%}")
    print(f"   Action sampling distribution: {stats['action_distribution']}")