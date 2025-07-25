"""
Simple test environment for meta-learning integration tests
"""

import numpy as np
import torch


class TestTradingEnvironment:
    """
    Simple mock trading environment for testing
    """
    
    def __init__(self, state_dim=50, action_dim=3, max_steps=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        
        # Environment state
        self.current_step = 0
        self.current_price = 100.0
        self.current_volume = 1000.0
        self.price_history = [100.0]
        
        # For compatibility with existing code
        self.if_discrete = True
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_price = 100.0 + np.random.randn() * 5
        self.current_volume = 1000.0 + np.random.randn() * 100
        self.price_history = [self.current_price]
        
        # Return initial state
        state = self._get_state()
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        self.current_step += 1
        
        # Simulate price movement based on action
        if action == 0:  # Sell
            price_change = np.random.normal(-0.001, 0.01)  # Slightly negative bias
            reward = -price_change  # Profit from selling before price drops
        elif action == 2:  # Buy
            price_change = np.random.normal(0.001, 0.01)   # Slightly positive bias
            reward = price_change   # Profit from buying before price rises
        else:  # Hold
            price_change = np.random.normal(0, 0.01)
            reward = 0.0
        
        # Update price
        self.current_price *= (1 + price_change)
        self.current_volume = max(100, self.current_volume + np.random.randn() * 50)
        self.price_history.append(self.current_price)
        
        # Get next state
        next_state = self._get_state()
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Info
        info = {
            'price': self.current_price,
            'volume': self.current_volume,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Generate state representation"""
        # Create a state with market features
        state = np.zeros(self.state_dim)
        
        # Price-based features
        if len(self.price_history) >= 2:
            recent_prices = self.price_history[-20:] if len(self.price_history) >= 20 else self.price_history
            
            # Normalized prices (last 10 features)
            for i, price in enumerate(recent_prices[-10:]):
                if i < 10:
                    state[i] = (price - 100.0) / 100.0  # Normalized around starting price
            
            # Returns (next 10 features)
            returns = np.diff(recent_prices) / recent_prices[:-1]
            for i, ret in enumerate(returns[-10:]):
                if i < 10:
                    state[10 + i] = ret * 100  # Scale returns
            
            # Technical indicators (next 20 features)
            if len(recent_prices) >= 5:
                # Simple moving averages
                sma_5 = np.mean(recent_prices[-5:])
                sma_10 = np.mean(recent_prices[-10:]) if len(recent_prices) >= 10 else sma_5
                
                state[20] = (self.current_price - sma_5) / sma_5
                state[21] = (self.current_price - sma_10) / sma_10
                
                # Volatility
                if len(returns) > 0:
                    state[22] = np.std(returns[-10:]) * 100
                
                # Volume features
                state[23] = (self.current_volume - 1000.0) / 1000.0
        
        # Random market features (remaining features)
        for i in range(24, self.state_dim):
            state[i] = np.random.randn() * 0.1
        
        return state
    
    def close(self):
        """Close environment"""
        pass


def create_test_environment(state_dim=50, action_dim=3, max_steps=1000):
    """Factory function to create test environment"""
    return TestTradingEnvironment(state_dim, action_dim, max_steps)