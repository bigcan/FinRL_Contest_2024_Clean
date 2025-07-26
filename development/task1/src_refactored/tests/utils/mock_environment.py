"""
Mock environment for testing the FinRL Contest 2024 framework.

This module provides a simplified trading environment that mimics the
interface of the real trading simulator for testing purposes.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, List, Optional
from collections import deque


class MockEnvironment:
    """
    Mock trading environment for testing agents and ensembles.
    
    Provides a simplified version of the trading environment interface
    with predictable behavior for unit testing.
    """
    
    def __init__(self, 
                 state_dim: int = 10,
                 action_dim: int = 3,
                 max_steps: int = 100,
                 seed: Optional[int] = None):
        """
        Initialize mock environment.
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Number of possible actions
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        
        if seed is not None:
            np.random.seed(seed)
        
        # Environment state
        self.current_step = 0
        self.current_state = None
        self.last_transition = None
        self.episode_reward = 0.0
        
        # Market simulation parameters
        self.price_trend = 0.0  # Upward/downward trend
        self.volatility = 0.1
        self.transaction_cost = 0.001
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.position = 0.0  # Current position (-1 to 1)
        self.cash = 1.0  # Starting cash
        self.portfolio_value = 1.0
        
        # History for state construction
        self.price_history = deque(maxlen=state_dim)
        self.volume_history = deque(maxlen=state_dim)
        self.return_history = deque(maxlen=state_dim)
        
        print(f"MockEnvironment initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.episode_reward = 0.0
        self.position = 0.0
        self.cash = 1.0
        self.portfolio_value = 1.0
        
        # Reset market parameters
        self.price_trend = np.random.uniform(-0.01, 0.01)
        self.volatility = np.random.uniform(0.05, 0.15)
        
        # Initialize histories
        self.price_history.clear()
        self.volume_history.clear()
        self.return_history.clear()
        
        # Fill initial history
        for _ in range(self.state_dim):
            self.price_history.append(np.random.uniform(0.9, 1.1))
            self.volume_history.append(np.random.uniform(0.5, 1.5))
            self.return_history.append(np.random.uniform(-0.02, 0.02))
        
        self.current_state = self._construct_state()
        return self.current_state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        self.current_step += 1
        
        # Store previous state for transition
        prev_state = self.current_state.copy()
        prev_portfolio_value = self.portfolio_value
        
        # Simulate market movement
        price_change = self._simulate_price_change()
        new_price = self.price_history[-1] * (1 + price_change)
        new_volume = np.random.uniform(0.5, 1.5)
        
        # Update histories
        self.price_history.append(new_price)
        self.volume_history.append(new_volume)
        self.return_history.append(price_change)
        
        # Execute action
        reward = self._execute_action(action, price_change)
        
        # Update state
        self.current_state = self._construct_state()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Store transition for agent updates
        self.last_transition = (prev_state, action, reward, done, self.current_state.copy())
        
        # Episode reward tracking
        self.episode_reward += reward
        
        # Info dictionary
        info = {
            'step': self.current_step,
            'position': self.position,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'price_change': price_change,
            'episode_reward': self.episode_reward,
            'action_taken': action
        }
        
        return self.current_state.copy(), reward, done, info
    
    def _simulate_price_change(self) -> float:
        """Simulate realistic price movements."""
        # Random walk with trend and volatility
        random_component = np.random.normal(0, self.volatility)
        trend_component = self.price_trend
        
        # Add some mean reversion
        current_price = self.price_history[-1]
        mean_reversion = -0.01 * (current_price - 1.0)
        
        return trend_component + random_component + mean_reversion
    
    def _execute_action(self, action: int, price_change: float) -> float:
        """
        Execute trading action and calculate reward.
        
        Args:
            action: Trading action (0=hold, 1=buy, 2=sell)
            price_change: Price change for this step
            
        Returns:
            Reward for this action
        """
        prev_portfolio_value = self.portfolio_value
        
        # Calculate current portfolio value
        self.portfolio_value = self.cash + self.position * self.price_history[-1]
        
        # Execute action
        if action == 1:  # Buy
            if self.position < 1.0 and self.cash > 0:
                # Buy as much as possible
                shares_to_buy = min(self.cash / self.price_history[-1], 1.0 - self.position)
                cost = shares_to_buy * self.price_history[-1] * (1 + self.transaction_cost)
                
                if cost <= self.cash:
                    self.position += shares_to_buy
                    self.cash -= cost
        
        elif action == 2:  # Sell
            if self.position > -1.0:
                # Sell as much as possible
                shares_to_sell = min(self.position + 1.0, 1.0)
                proceeds = shares_to_sell * self.price_history[-1] * (1 - self.transaction_cost)
                
                self.position -= shares_to_sell
                self.cash += proceeds
        
        # Action 0 (hold) requires no execution
        
        # Recalculate portfolio value after action
        self.portfolio_value = self.cash + self.position * self.price_history[-1]
        
        # Calculate reward as portfolio value change
        reward = self.portfolio_value - prev_portfolio_value
        
        # Add penalty for excessive trading
        if action != 0:
            reward -= self.transaction_cost * 0.1
        
        return reward
    
    def _construct_state(self) -> np.ndarray:
        """
        Construct state vector from market history and portfolio state.
        
        Returns:
            State vector of length state_dim
        """
        state_components = []
        
        # Price features (normalized)
        if len(self.price_history) >= 2:
            recent_prices = np.array(list(self.price_history)[-5:])
            price_features = [
                recent_prices[-1] / recent_prices[0],  # Price ratio
                np.std(recent_prices) / np.mean(recent_prices),  # Volatility
            ]
        else:
            price_features = [1.0, 0.1]
        
        state_components.extend(price_features)
        
        # Return features
        if len(self.return_history) >= 2:
            recent_returns = np.array(list(self.return_history)[-5:])
            return_features = [
                np.mean(recent_returns),  # Mean return
                np.std(recent_returns),   # Return volatility
            ]
        else:
            return_features = [0.0, 0.1]
        
        state_components.extend(return_features)
        
        # Portfolio features
        portfolio_features = [
            self.position,  # Current position
            self.cash,      # Available cash
            self.portfolio_value,  # Total portfolio value
        ]
        state_components.extend(portfolio_features)
        
        # Technical indicators (simplified)
        if len(self.price_history) >= 5:
            prices = np.array(list(self.price_history)[-5:])
            sma = np.mean(prices)
            current_price = prices[-1]
            technical_features = [
                (current_price - sma) / sma,  # Price vs SMA
            ]
        else:
            technical_features = [0.0]
        
        state_components.extend(technical_features)
        
        # Pad or truncate to exact state_dim
        while len(state_components) < self.state_dim:
            state_components.append(0.0)
        
        state_components = state_components[:self.state_dim]
        
        return np.array(state_components, dtype=np.float32)
    
    def get_last_transition(self) -> Optional[Tuple[np.ndarray, int, float, bool, np.ndarray]]:
        """
        Get the last transition for agent training.
        
        Returns:
            Tuple of (state, action, reward, done, next_state) or None
        """
        return self.last_transition
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the current environment state.
        
        Args:
            mode: Rendering mode
            
        Returns:
            String representation if mode='ansi', None otherwise
        """
        if mode == 'ansi':
            return (f"Step: {self.current_step}, "
                   f"Position: {self.position:.3f}, "
                   f"Cash: {self.cash:.3f}, "
                   f"Portfolio: {self.portfolio_value:.3f}, "
                   f"Price: {self.price_history[-1] if self.price_history else 0:.3f}")
        
        elif mode == 'human':
            print(f"MockEnvironment State:")
            print(f"  Step: {self.current_step}/{self.max_steps}")
            print(f"  Position: {self.position:.3f}")
            print(f"  Cash: {self.cash:.3f}")
            print(f"  Portfolio Value: {self.portfolio_value:.3f}")
            print(f"  Current Price: {self.price_history[-1] if self.price_history else 0:.3f}")
            print(f"  Episode Reward: {self.episode_reward:.3f}")
    
    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    @property
    def observation_space(self):
        """Mock observation space for compatibility."""
        return type('MockSpace', (), {'shape': (self.state_dim,)})()
    
    @property 
    def action_space(self):
        """Mock action space for compatibility."""
        return type('MockSpace', (), {'n': self.action_dim})()


class MockVectorizedEnvironment:
    """
    Mock vectorized environment for testing ensemble training.
    
    Simulates multiple parallel environments for batch training.
    """
    
    def __init__(self, 
                 num_envs: int = 4,
                 state_dim: int = 10,
                 action_dim: int = 3,
                 max_steps: int = 100,
                 seed: Optional[int] = None):
        """
        Initialize vectorized mock environment.
        
        Args:
            num_envs: Number of parallel environments
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            max_steps: Maximum steps per episode
            seed: Base random seed
        """
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        
        # Create individual environments
        self.envs = []
        for i in range(num_envs):
            env_seed = seed + i if seed is not None else None
            env = MockEnvironment(state_dim, action_dim, max_steps, env_seed)
            self.envs.append(env)
        
        print(f"MockVectorizedEnvironment initialized with {num_envs} environments")
    
    def reset(self) -> np.ndarray:
        """Reset all environments."""
        states = []
        for env in self.envs:
            state = env.reset()
            states.append(state)
        return np.array(states)
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments with given actions."""
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        
        states = []
        rewards = []
        dones = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.array(states), np.array(rewards), np.array(dones), infos
    
    def get_last_transitions(self) -> List[Optional[Tuple]]:
        """Get last transitions from all environments."""
        return [env.get_last_transition() for env in self.envs]
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()