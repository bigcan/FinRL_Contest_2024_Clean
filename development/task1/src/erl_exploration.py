"""
Enhanced Exploration Strategies for DRL Agents
Addresses the conservative trading problem through adaptive exploration
"""

import numpy as np
import torch
from typing import Dict, Optional, Union, List
from collections import deque
from abc import ABC, abstractmethod


class ExplorationStrategy(ABC):
    """Base class for exploration strategies"""
    
    @abstractmethod
    def get_exploration_rate(self, step: int) -> float:
        """Get current exploration rate"""
        pass
        
    @abstractmethod
    def should_explore(self) -> bool:
        """Determine if agent should explore"""
        pass


class AdaptiveEpsilonGreedy(ExplorationStrategy):
    """
    Adaptive epsilon-greedy exploration with dynamic adjustments
    based on action diversity and performance
    """
    
    def __init__(self,
                 initial_epsilon: float = 0.3,
                 min_epsilon: float = 0.01,
                 decay_rate: float = 0.995,
                 warmup_steps: int = 5000,
                 diversity_threshold: float = 0.5,
                 performance_window: int = 100):
        """
        Initialize adaptive epsilon-greedy exploration
        
        Args:
            initial_epsilon: Starting exploration rate
            min_epsilon: Minimum exploration rate
            decay_rate: Exponential decay rate
            warmup_steps: Steps before decay starts
            diversity_threshold: Minimum action diversity required
            performance_window: Window for performance tracking
        """
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.diversity_threshold = diversity_threshold
        self.performance_window = performance_window
        
        # State tracking
        self.current_epsilon = initial_epsilon
        self.total_steps = 0
        self.action_history = deque(maxlen=performance_window * 2)
        self.reward_history = deque(maxlen=performance_window)
        
    def get_exploration_rate(self, step: Optional[int] = None) -> float:
        """Get current exploration rate with adaptive adjustments"""
        if step is None:
            step = self.total_steps
            
        # Warmup phase
        if step < self.warmup_steps:
            return self.initial_epsilon
            
        # Base decay
        base_epsilon = max(self.min_epsilon, 
                          self.initial_epsilon * (self.decay_rate ** (step - self.warmup_steps)))
        
        # Adaptive adjustment based on action diversity
        diversity_adjustment = self._calculate_diversity_adjustment()
        
        # Performance-based adjustment
        performance_adjustment = self._calculate_performance_adjustment()
        
        # Combine adjustments
        adjusted_epsilon = base_epsilon * (1 + diversity_adjustment + performance_adjustment)
        
        # Ensure bounds
        self.current_epsilon = np.clip(adjusted_epsilon, self.min_epsilon, 1.0)
        return self.current_epsilon
        
    def _calculate_diversity_adjustment(self) -> float:
        """Calculate adjustment based on action diversity"""
        if len(self.action_history) < 50:
            return 0.0
            
        recent_actions = list(self.action_history)[-100:]
        unique_actions = len(set(recent_actions))
        max_actions = len(set(self.action_history))  # Estimate from all history
        
        if max_actions == 0:
            return 0.0
            
        diversity_ratio = unique_actions / max(3, max_actions)  # Assume at least 3 actions
        
        if diversity_ratio < self.diversity_threshold:
            # Low diversity: increase exploration
            return (self.diversity_threshold - diversity_ratio) * 2.0
        else:
            # Good diversity: can reduce exploration slightly
            return -0.1
            
    def _calculate_performance_adjustment(self) -> float:
        """Calculate adjustment based on recent performance"""
        if len(self.reward_history) < 20:
            return 0.0
            
        recent_rewards = list(self.reward_history)[-20:]
        
        # Check if performance is stagnating
        reward_variance = np.var(recent_rewards)
        if reward_variance < 1e-6:  # Very low variance suggests stagnation
            return 0.5  # Boost exploration
            
        # Check if performance is declining
        if len(recent_rewards) >= 10:
            first_half = np.mean(recent_rewards[:10])
            second_half = np.mean(recent_rewards[10:])
            if second_half < first_half * 0.8:  # 20% decline
                return 0.3  # Moderate exploration boost
                
        return 0.0
        
    def should_explore(self) -> bool:
        """Probabilistic exploration decision"""
        return np.random.random() < self.current_epsilon
        
    def update(self, action: int, reward: float):
        """Update exploration strategy with new experience"""
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.total_steps += 1
        
        # Update epsilon
        self.get_exploration_rate()


class CyclicalExploration(ExplorationStrategy):
    """
    Cyclical exploration strategy that periodically boosts exploration
    to prevent local optima convergence
    """
    
    def __init__(self,
                 base_epsilon: float = 0.05,
                 peak_epsilon: float = 0.3,
                 cycle_length: int = 10000,
                 min_epsilon: float = 0.01):
        """
        Initialize cyclical exploration
        
        Args:
            base_epsilon: Base exploration rate
            peak_epsilon: Peak exploration rate during cycles
            cycle_length: Steps per exploration cycle
            min_epsilon: Minimum exploration rate
        """
        self.base_epsilon = base_epsilon
        self.peak_epsilon = peak_epsilon
        self.cycle_length = cycle_length
        self.min_epsilon = min_epsilon
        self.total_steps = 0
        
    def get_exploration_rate(self, step: Optional[int] = None) -> float:
        """Get exploration rate with cyclical pattern"""
        if step is None:
            step = self.total_steps
            
        # Cyclical component
        cycle_position = (step % self.cycle_length) / self.cycle_length
        cycle_value = np.sin(2 * np.pi * cycle_position) * 0.5 + 0.5  # 0 to 1
        
        # Interpolate between base and peak
        epsilon = self.base_epsilon + (self.peak_epsilon - self.base_epsilon) * cycle_value
        
        # Decay over time
        decay_factor = 0.9999 ** step
        epsilon *= decay_factor
        
        return max(self.min_epsilon, epsilon)
        
    def should_explore(self) -> bool:
        """Probabilistic exploration decision"""
        return np.random.random() < self.get_exploration_rate()
        
    def update(self, action: int, reward: float):
        """Update step counter"""
        self.total_steps += 1


class ActionMaskingExploration:
    """
    Exploration through action masking to prevent excessive conservatism
    """
    
    def __init__(self,
                 conservatism_threshold: float = 0.8,
                 mask_probability: float = 0.5,
                 window_size: int = 100):
        """
        Initialize action masking exploration
        
        Args:
            conservatism_threshold: Threshold for detecting conservative behavior
            mask_probability: Probability of masking conservative action
            window_size: Window for tracking action history
        """
        self.conservatism_threshold = conservatism_threshold
        self.mask_probability = mask_probability
        self.window_size = window_size
        self.action_history = deque(maxlen=window_size)
        
    def get_action_mask(self, num_actions: int, hold_action: int = 1) -> torch.Tensor:
        """
        Get action mask that may prevent conservative actions
        
        Args:
            num_actions: Total number of actions
            hold_action: Index of hold/conservative action
            
        Returns:
            Boolean mask tensor (True = allowed, False = masked)
        """
        # Default: all actions allowed
        mask = torch.ones(num_actions, dtype=torch.bool)
        
        if len(self.action_history) < 20:
            return mask
            
        # Check recent conservatism
        recent_actions = list(self.action_history)[-50:]
        hold_ratio = recent_actions.count(hold_action) / len(recent_actions)
        
        if hold_ratio > self.conservatism_threshold:
            # Probabilistically mask hold action
            if np.random.random() < self.mask_probability:
                mask[hold_action] = False
                
        return mask
        
    def update(self, action: int):
        """Update action history"""
        self.action_history.append(action)


class ExplorationOrchestrator:
    """
    Orchestrates multiple exploration strategies for robust exploration
    """
    
    def __init__(self,
                 strategies: Optional[List[ExplorationStrategy]] = None,
                 action_dim: int = 3):
        """
        Initialize exploration orchestrator
        
        Args:
            strategies: List of exploration strategies to use
            action_dim: Number of possible actions
        """
        self.action_dim = action_dim
        
        if strategies is None:
            # Default strategies
            self.strategies = [
                AdaptiveEpsilonGreedy(initial_epsilon=0.3, min_epsilon=0.01),
                CyclicalExploration(base_epsilon=0.05, peak_epsilon=0.2)
            ]
        else:
            self.strategies = strategies
            
        self.action_masking = ActionMaskingExploration()
        self.exploration_stats = {
            'forced_explorations': 0,
            'masked_actions': 0,
            'total_steps': 0
        }
        
    def should_explore(self) -> bool:
        """Determine if exploration should occur using any strategy"""
        for strategy in self.strategies:
            if strategy.should_explore():
                self.exploration_stats['forced_explorations'] += 1
                return True
        return False
        
    def get_exploration_rate(self) -> float:
        """Get maximum exploration rate across strategies"""
        return max(strategy.get_exploration_rate() for strategy in self.strategies)
        
    def get_masked_action(self, 
                         q_values: torch.Tensor, 
                         temperature: float = 1.0) -> torch.Tensor:
        """
        Get action with potential masking and exploration
        
        Args:
            q_values: Q-values for each action
            temperature: Temperature for softmax exploration
            
        Returns:
            Selected action tensor
        """
        batch_size = q_values.shape[0]
        
        # Get action mask
        mask = self.action_masking.get_action_mask(self.action_dim)
        
        # Apply mask to Q-values
        masked_q_values = q_values.clone()
        masked_q_values[:, ~mask] = -float('inf')
        
        # Check if we should explore
        if self.should_explore():
            # Exploration: sample from valid actions
            valid_actions = torch.where(mask)[0]
            if len(valid_actions) > 0:
                actions = valid_actions[torch.randint(len(valid_actions), (batch_size,))]
            else:
                actions = torch.randint(self.action_dim, (batch_size,))
        else:
            # Exploitation with temperature
            if temperature > 0:
                probs = torch.softmax(masked_q_values / temperature, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
            else:
                actions = masked_q_values.argmax(dim=-1)
                
        return actions
        
    def update(self, action: int, reward: float):
        """Update all strategies with new experience"""
        for strategy in self.strategies:
            strategy.update(action, reward)
        self.action_masking.update(action)
        self.exploration_stats['total_steps'] += 1
        
    def get_stats(self) -> Dict[str, float]:
        """Get exploration statistics"""
        stats = self.exploration_stats.copy()
        stats['current_exploration_rate'] = self.get_exploration_rate()
        stats['exploration_ratio'] = (stats['forced_explorations'] / 
                                     max(1, stats['total_steps']))
        return stats


# Testing
if __name__ == "__main__":
    print("🧪 Testing Exploration Strategies")
    print("=" * 50)
    
    # Test adaptive epsilon-greedy
    adaptive = AdaptiveEpsilonGreedy(initial_epsilon=0.3, min_epsilon=0.01)
    
    print("\n📊 Adaptive Epsilon-Greedy:")
    for step in [0, 1000, 5000, 10000, 20000]:
        epsilon = adaptive.get_exploration_rate(step)
        print(f"   Step {step:5d}: ε = {epsilon:.4f}")
        
    # Simulate low diversity scenario
    print("\n📊 Low Diversity Adjustment:")
    for _ in range(100):
        adaptive.update(action=1, reward=0.1)  # Only hold actions
    epsilon = adaptive.get_exploration_rate()
    print(f"   After 100 hold actions: ε = {epsilon:.4f}")
    
    # Test cyclical exploration
    print("\n📊 Cyclical Exploration:")
    cyclical = CyclicalExploration(base_epsilon=0.05, peak_epsilon=0.3, cycle_length=100)
    
    for step in range(0, 200, 20):
        epsilon = cyclical.get_exploration_rate(step)
        print(f"   Step {step:3d}: ε = {epsilon:.4f}")
        
    # Test orchestrator
    print("\n📊 Exploration Orchestrator:")
    orchestrator = ExplorationOrchestrator(action_dim=3)
    
    # Simulate trading
    for i in range(100):
        should_explore = orchestrator.should_explore()
        action = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])  # Conservative
        orchestrator.update(action, reward=0.1)
        
    stats = orchestrator.get_stats()
    print(f"   Exploration rate: {stats['current_exploration_rate']:.4f}")
    print(f"   Forced explorations: {stats['forced_explorations']}")
    print(f"   Exploration ratio: {stats['exploration_ratio']:.4f}")