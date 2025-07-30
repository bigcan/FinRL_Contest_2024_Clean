"""
Trading environment for HPO optimization
Compatible with the feature data and reward system
"""

import numpy as np
import torch as th
from typing import Tuple, Dict, Optional, Any

class LOBTradingEnvironment:
    """
    Limit Order Book (LOB) trading environment
    """
    
    def __init__(self, 
                 data: np.ndarray,
                 max_position: int = 3,
                 lookback_window: int = 10,
                 step_gap: int = 1,
                 delay_step: int = 1,
                 transaction_cost: float = 0.001,
                 slippage: float = 5e-5,
                 max_holding_time: int = 3600,
                 initial_asset: float = 1e6):
        """
        Initialize LOB trading environment
        
        Args:
            data: Feature data array (time_steps, features)
            max_position: Maximum position size
            lookback_window: Lookback window for state
            step_gap: Gap between steps
            delay_step: Delay for order execution
            transaction_cost: Transaction cost ratio
            slippage: Slippage factor
            max_holding_time: Maximum holding time in steps
            initial_asset: Initial asset value
        """
        self.data = data
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.step_gap = step_gap
        self.delay_step = delay_step
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_holding_time = max_holding_time
        self.initial_asset = initial_asset
        
        # Environment properties
        self.state_dim = data.shape[1] if len(data.shape) > 1 else 1
        self.action_dim = 3  # Buy, Hold, Sell
        self.if_discrete = True
        self.max_step = len(data) - lookback_window - 10
        
        # Price simulation (since we don't have actual prices in features)
        self._generate_price_series()
        
        # Reset environment
        self.reset()
        
    def _generate_price_series(self):
        """Generate synthetic price series based on features"""
        # Use first feature as proxy for price movement
        if self.data.shape[1] > 0:
            # Normalize and scale first feature
            price_feature = self.data[:, 0]
            price_feature = (price_feature - np.mean(price_feature)) / (np.std(price_feature) + 1e-8)
            
            # Generate cumulative price series
            returns = price_feature * 0.001  # Small returns
            self.prices = 100 * np.exp(np.cumsum(returns))
        else:
            # Fallback: random walk
            returns = np.random.normal(0, 0.001, len(self.data))
            self.prices = 100 * np.exp(np.cumsum(returns))
            
        # Generate volumes (synthetic)
        self.volumes = np.random.lognormal(10, 0.5, len(self.data))
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.position = 0
        self.entry_price = 0
        self.holding_time = 0
        
        # Financial metrics
        self.current_asset = self.initial_asset
        self.total_value = self.initial_asset
        self.cash = self.initial_asset
        
        # History tracking
        self.position_history = [0]
        self.asset_history = [self.initial_asset]
        self.action_history = []
        
        # For reward calculation
        self.initial_total_value = self.initial_asset
        self.previous_total_value = self.initial_asset
        self.current_price = self.prices[self.current_step]
        self.previous_price = self.prices[self.current_step - 1]
        
        # Trade tracking
        self.trades_completed = 0
        self.winning_trades = 0
        self.total_pnl = 0
        
        return self._get_state()
        
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Get recent features
        state = self.data[self.current_step].copy()
        
        # Add position information
        position_feature = self.position / self.max_position
        holding_time_feature = min(self.holding_time / 100, 1.0)
        
        # Add price momentum
        if self.current_step > 0:
            price_momentum = (self.current_price - self.previous_price) / self.previous_price
        else:
            price_momentum = 0
            
        # Combine features
        extended_state = np.concatenate([
            state,
            [position_feature, holding_time_feature, price_momentum]
        ])
        
        return extended_state.astype(np.float32)
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute trading action
        
        Args:
            action: 0=sell, 1=hold, 2=buy
            
        Returns:
            next_state, reward, done, info
        """
        # Store previous values
        self.previous_position = self.position
        self.previous_total_value = self.total_value
        self.previous_price = self.current_price
        
        # Execute action
        self._execute_action(action)
        
        # Update step
        self.current_step += self.step_gap
        if self.current_step >= len(self.prices) - 1:
            self.current_step = len(self.prices) - 1
            
        # Update price
        self.current_price = self.prices[self.current_step]
        self.current_volume = self.volumes[self.current_step]
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Calculate reward (basic - will be overridden by reward calculator)
        reward = (self.total_value - self.previous_total_value) / self.previous_total_value
        
        # Update holding time
        if self.position != 0:
            self.holding_time += 1
        else:
            self.holding_time = 0
            
        # Check if done
        done = (self.current_step >= self.max_step) or (self.total_value < self.initial_asset * 0.5)
        
        # Prepare info
        info = {
            'current_price': self.current_price,
            'position': self.position,
            'total_value': self.total_value,
            'cash': self.cash,
            'holding_time': self.holding_time,
            'trades_completed': self.trades_completed,
            'current_step': self.current_step
        }
        
        # Check for completed trades
        if self.previous_position != 0 and self.position == 0:
            info['trade_completed'] = True
            info['trade_return'] = self._calculate_last_trade_return()
        else:
            info['trade_completed'] = False
            
        # Get next state
        next_state = self._get_state()
        
        # Store history
        self.position_history.append(self.position)
        self.asset_history.append(self.total_value)
        self.action_history.append(action)
        
        return next_state, reward, done, info
        
    def _execute_action(self, action: int):
        """Execute trading action with transaction costs"""
        if action == 0 and self.position > 0:  # Sell
            # Calculate proceeds
            sell_price = self.current_price * (1 - self.slippage)
            proceeds = self.position * sell_price * (1 - self.transaction_cost)
            
            # Update cash and position
            self.cash += proceeds
            self.position = 0
            
            # Track trade
            self.trades_completed += 1
            trade_pnl = proceeds - (self.entry_price * self.position)
            if trade_pnl > 0:
                self.winning_trades += 1
            self.total_pnl += trade_pnl
            
        elif action == 2 and self.position < self.max_position:  # Buy
            # Calculate cost
            buy_price = self.current_price * (1 + self.slippage)
            position_size = min(1, self.max_position - self.position)
            cost = position_size * buy_price * (1 + self.transaction_cost)
            
            # Check if we have enough cash
            if self.cash >= cost:
                self.cash -= cost
                self.position += position_size
                
                # Track entry price (weighted average)
                if self.position == position_size:
                    self.entry_price = buy_price
                else:
                    total_cost = self.entry_price * (self.position - position_size) + buy_price * position_size
                    self.entry_price = total_cost / self.position
                    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        position_value = self.position * self.current_price
        self.total_value = self.cash + position_value
        self.current_total_value = self.total_value  # For compatibility
        
    def _calculate_last_trade_return(self) -> float:
        """Calculate return of last completed trade"""
        if self.entry_price > 0:
            return (self.current_price - self.entry_price) / self.entry_price
        return 0.0
        
    def get_metrics(self) -> Dict[str, float]:
        """Get trading performance metrics"""
        total_return = (self.total_value - self.initial_asset) / self.initial_asset
        
        if self.trades_completed > 0:
            win_rate = self.winning_trades / self.trades_completed
            avg_trade = self.total_pnl / self.trades_completed
        else:
            win_rate = 0
            avg_trade = 0
            
        # Calculate Sharpe ratio
        if len(self.asset_history) > 2:
            returns = np.diff(self.asset_history) / self.asset_history[:-1]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 86400)
            else:
                sharpe = 0
        else:
            sharpe = 0
            
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'trades_completed': self.trades_completed,
            'avg_trade': avg_trade,
            'final_value': self.total_value
        }


# Alias for compatibility
LOBEnvironment = LOBTradingEnvironment