import numpy as np
import torch
from typing import Tuple
from torch import Tensor


class SumTree:
    """
    Binary tree data structure for efficient prioritized sampling.
    Each leaf stores a priority value, and each internal node stores the sum of its children.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree stored as array
        self.data_pointer = 0
        
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index based on priority sum"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def total(self) -> float:
        """Return total priority sum"""
        return self.tree[0]
        
    def add(self, priority: float, data):
        """Add new data with priority"""
        idx = self.data_pointer + self.capacity - 1
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
    def update(self, idx: int, priority: float):
        """Update priority of existing data"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        
    def get(self, s: float) -> Tuple[int, float]:
        """Get data index and priority for given priority sum"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer implementing the PER algorithm.
    Samples transitions with probability proportional to their TD error.
    """
    
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_annealing_steps: int = 100000,
                 epsilon: float = 1e-6):
        
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        
        # PER hyperparameters
        self.alpha = alpha  # How much prioritization is used (0 = uniform, 1 = full prioritization)
        self.beta = beta    # Importance sampling weight (0 = no correction, 1 = full correction)
        self.beta_start = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon  # Small constant to prevent zero probabilities
        self.max_priority = 1.0
        
        # Buffer storage
        self.states = torch.zeros((max_size, num_seqs, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((max_size, num_seqs, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((max_size, num_seqs), dtype=torch.float32, device=self.device)
        self.undones = torch.zeros((max_size, num_seqs), dtype=torch.float32, device=self.device)
        
        # Priority trees (one per sequence/environment)
        self.sum_trees = [SumTree(max_size) for _ in range(num_seqs)]
        
        # Buffer management
        self.p = 0  # Current pointer
        self.cur_size = 0
        self.if_full = False
        self.add_size = 0
        self.add_item = None
        self.step_count = 0
        
    def update_beta(self):
        """Anneal beta from initial value to 1.0 over training"""
        self.step_count += 1
        fraction = min(self.step_count / self.beta_annealing_steps, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)
        
    def update(self, items: Tuple[Tensor, ...]):
        """Add new experiences to buffer with maximum priority"""
        self.add_item = items
        states, actions, rewards, undones = items
        self.add_size = rewards.shape[0]
        
        p = self.p + self.add_size
        
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size
            
            # Update buffer
            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
            
            # Update priorities for new experiences
            for seq_idx in range(self.num_seqs):
                for i in range(self.add_size):
                    buffer_idx = (p0 + i) % self.max_size
                    self.sum_trees[seq_idx].update(buffer_idx, self.max_priority ** self.alpha)
        else:
            # Update buffer
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
            
            # Update priorities for new experiences
            for seq_idx in range(self.num_seqs):
                for i in range(self.add_size):
                    buffer_idx = self.p + i
                    self.sum_trees[seq_idx].add(self.max_priority ** self.alpha, buffer_idx)
        
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p
        
    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample batch with prioritized sampling and return importance sampling weights
        
        Returns:
            states, actions, rewards, undones, next_states, indices, weights
        """
        # CRITICAL FIX: Reserve space for next_state access
        sample_len = max(1, self.cur_size - 2)  # Ensures ids0 + 1 < cur_size
        if sample_len <= 0 or self.cur_size < 2:
            # Fallback to uniform sampling if buffer is too small
            return self._uniform_sample(batch_size)
            
        indices = []
        priorities = []
        
        # Sample from each environment proportionally
        samples_per_seq = batch_size // self.num_seqs
        remaining_samples = batch_size % self.num_seqs
        
        for seq_idx in range(self.num_seqs):
            seq_samples = samples_per_seq + (1 if seq_idx < remaining_samples else 0)
            total_priority = self.sum_trees[seq_idx].total()
            
            if total_priority <= 0:
                # Fallback to uniform sampling for this sequence
                seq_indices = torch.randint(0, sample_len, (seq_samples,))
                seq_priorities = [1.0] * seq_samples
            else:
                seq_indices = []
                seq_priorities = []
                
                segment_size = total_priority / seq_samples
                for i in range(seq_samples):
                    a = segment_size * i
                    b = segment_size * (i + 1)
                    s = np.random.uniform(a, b)
                    
                    idx, priority = self.sum_trees[seq_idx].get(s)
                    seq_indices.append(idx)
                    seq_priorities.append(priority)
                
                seq_indices = torch.tensor(seq_indices, dtype=torch.long)
                
            # Convert to global indices
            global_indices = seq_idx * sample_len + seq_indices
            indices.extend(global_indices.tolist())
            priorities.extend(seq_priorities)
        
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        priorities = torch.tensor(priorities, dtype=torch.float32, device=self.device)
        
        # Convert to buffer indices
        ids0 = torch.fmod(indices, sample_len)
        ids1 = torch.div(indices, sample_len, rounding_mode='floor')
        
        # Calculate importance sampling weights
        sampling_probabilities = priorities / torch.sum(priorities)
        weights = (batch_size * sampling_probabilities) ** (-self.beta)
        weights = weights / torch.max(weights)  # Normalize by max weight
        
        return (
            self.states[ids0, ids1],      # states
            self.actions[ids0, ids1],     # actions
            self.rewards[ids0, ids1],     # rewards
            self.undones[ids0, ids1],     # undones
            self.states[ids0 + 1, ids1],  # next_states
            indices,                      # indices for priority updates
            weights                       # importance sampling weights
        )
    
    def _uniform_sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fallback uniform sampling when buffer is small or priorities are invalid"""
        # CRITICAL FIX: Reserve space for next_state access
        sample_len = max(1, self.cur_size - 2)
        
        ids = torch.randint(sample_len * self.num_seqs, size=(batch_size,), device=self.device)
        ids0 = torch.fmod(ids, sample_len)
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')
        
        weights = torch.ones(batch_size, dtype=torch.float32, device=self.device)
        
        return (
            self.states[ids0, ids1],
            self.actions[ids0, ids1],
            self.rewards[ids0, ids1],
            self.undones[ids0, ids1],
            self.states[ids0 + 1, ids1],
            ids,
            weights
        )
    
    def update_priorities(self, indices: Tensor, priorities: Tensor):
        """Update priorities for sampled experiences"""
        priorities = priorities.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()
        
        # CRITICAL FIX: Use same sample_len as in sampling
        sample_len = max(1, self.cur_size - 2)
        
        for i, priority in zip(indices, priorities):
            seq_idx = i // sample_len
            buffer_idx = i % sample_len
            
            if 0 <= seq_idx < self.num_seqs and 0 <= buffer_idx < sample_len:
                # Clamp priority and add epsilon
                priority = max(abs(priority) + self.epsilon, self.epsilon)
                self.max_priority = max(self.max_priority, priority)
                
                # Update priority in sum tree
                self.sum_trees[seq_idx].update(buffer_idx, priority ** self.alpha)
                
        # Anneal beta
        self.update_beta()
    
    def save_or_load_history(self, cwd: str, if_save: bool):
        """Save or load buffer history (without priority trees for simplicity)"""
        item_names = [
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.undones, "undones"),
        ]
        
        if if_save:
            for item, name in item_names:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = torch.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/per_buffer_{name}.pth"
                print(f"| PER buffer.save_or_load_history(): Save {file_path}")
                torch.save(buf_item, file_path)
        
        elif all([os.path.isfile(f"{cwd}/per_buffer_{name}.pth") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/per_buffer_{name}.pth"
                print(f"| PER buffer.save_or_load_history(): Load {file_path}")
                buf_item = torch.load(file_path, map_location=self.device)
                
                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
                
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = self.p = max_sizes[0]
            self.if_full = self.cur_size == self.max_size
            
            # Reinitialize priority trees with uniform priorities
            for seq_idx in range(self.num_seqs):
                for i in range(self.cur_size):
                    self.sum_trees[seq_idx].add(1.0, i)