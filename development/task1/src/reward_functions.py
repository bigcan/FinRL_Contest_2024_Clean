"""
Enhanced Reward Functions for Profitable Trading
Addresses the core profitability issues in the FinRL Contest 2024 model
"""

import torch as th
import numpy as np
from typing import Tuple, Optional
from collections import deque


class RewardCalculator:
    """
    Advanced reward calculation system with multiple risk-adjusted variants
    Solves the core profitability issues of the original model
    """
    
    def __init__(self, reward_type: str = "sharpe_adjusted", 
                 lookback_window: int = 100,
                 risk_free_rate: float = 0.02,
                 transaction_cost_penalty: float = 0.001,
                 max_drawdown_penalty: float = 2.0,
                 device: str = "cpu"):
        """
        Initialize reward calculator with multiple variants
        
        Args:
            reward_type: "sharpe_adjusted", "transaction_cost_adjusted", "multi_objective", "simple"
            lookback_window: Period for calculating volatility and Sharpe ratios
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            transaction_cost_penalty: Penalty factor for transaction costs
            max_drawdown_penalty: Penalty multiplier for drawdown periods
            device: PyTorch device for calculations
        """
        self.reward_type = reward_type
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate / (252 * 24 * 60)  # Convert annual to per-minute
        self.transaction_cost_penalty = transaction_cost_penalty
        self.max_drawdown_penalty = max_drawdown_penalty
        self.device = device
        
        # Historical tracking for risk calculations
        self.asset_history = deque(maxlen=lookback_window)
        self.return_history = deque(maxlen=lookback_window)
        self.drawdown_history = deque(maxlen=lookback_window)
        
        # Performance tracking
        self.total_transaction_costs = 0.0
        self.peak_asset_value = 1e6  # Starting capital
        self.current_drawdown = 0.0
        
        print(f"🎯 RewardCalculator initialized with '{reward_type}' method")
        print(f"   Lookback window: {lookback_window}")
        print(f"   Risk-free rate: {risk_free_rate:.4f} (annual)")
        print(f"   Transaction cost penalty: {transaction_cost_penalty}")
    
    def calculate_reward(self, 
                        old_asset: th.Tensor, 
                        new_asset: th.Tensor, 
                        action_int: th.Tensor, 
                        mid_price: th.Tensor,
                        slippage: float) -> th.Tensor:
        """
        Calculate risk-adjusted reward based on selected method
        
        Args:
            old_asset: Previous total asset value
            new_asset: Current total asset value  
            action_int: Action taken (-1: sell, 0: hold, 1: buy)
            mid_price: Current mid price
            slippage: Transaction slippage rate
            
        Returns:
            Risk-adjusted reward tensor
        """
        
        if self.reward_type == "simple":
            return self._simple_reward(old_asset, new_asset)
        elif self.reward_type == "transaction_cost_adjusted":
            return self._transaction_cost_adjusted_reward(old_asset, new_asset, action_int, mid_price, slippage)
        elif self.reward_type == "sharpe_adjusted":
            return self._sharpe_adjusted_reward(old_asset, new_asset, action_int, mid_price, slippage)
        elif self.reward_type == "multi_objective":
            return self._multi_objective_reward(old_asset, new_asset, action_int, mid_price, slippage)
        elif self.reward_type == "profit_maximizing":
            return self._profit_maximizing_reward(old_asset, new_asset, action_int, mid_price, slippage)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _simple_reward(self, old_asset: th.Tensor, new_asset: th.Tensor) -> th.Tensor:
        """Original simple reward (baseline for comparison)"""
        return new_asset - old_asset
    
    def _transaction_cost_adjusted_reward(self, 
                                        old_asset: th.Tensor, 
                                        new_asset: th.Tensor, 
                                        action_int: th.Tensor, 
                                        mid_price: th.Tensor,
                                        slippage: float) -> th.Tensor:
        """
        Reward adjusted for transaction costs and slippage
        Penalizes excessive trading and accounts for real trading costs
        """
        # Basic return
        raw_return = new_asset - old_asset
        
        # Calculate transaction costs
        trade_occurred = action_int.abs() > 0
        transaction_volume = action_int.abs() * mid_price
        
        # Transaction cost = slippage + fixed cost penalty
        transaction_cost = th.where(
            trade_occurred,
            transaction_volume * (slippage + self.transaction_cost_penalty),
            th.zeros_like(transaction_volume)
        )
        
        # Update total costs for tracking
        self.total_transaction_costs += transaction_cost.sum().item()
        
        # Adjusted reward
        reward = raw_return - transaction_cost
        
        return reward
    
    def _sharpe_adjusted_reward(self, 
                              old_asset: th.Tensor, 
                              new_asset: th.Tensor, 
                              action_int: th.Tensor, 
                              mid_price: th.Tensor,
                              slippage: float) -> th.Tensor:
        """
        Sharpe ratio-based reward that promotes risk-adjusted returns
        Encourages consistent performance over volatile high returns
        """
        # Get transaction-cost adjusted return
        base_reward = self._transaction_cost_adjusted_reward(old_asset, new_asset, action_int, mid_price, slippage)
        
        # Calculate return rate
        return_rate = base_reward / old_asset
        
        # Update return history
        self.return_history.append(return_rate.mean().item())
        
        # Calculate rolling Sharpe ratio if we have enough history
        if len(self.return_history) >= 10:  # Minimum history for meaningful calculation
            returns_array = np.array(list(self.return_history))
            excess_returns = returns_array - self.risk_free_rate
            
            if np.std(returns_array) > 1e-8:  # Avoid division by zero
                sharpe_ratio = np.mean(excess_returns) / np.std(returns_array)
                
                # Scale Sharpe ratio to reasonable reward magnitude
                sharpe_bonus = th.tensor(sharpe_ratio * 0.01, device=self.device, dtype=base_reward.dtype)
                reward = base_reward + sharpe_bonus
            else:
                reward = base_reward
        else:
            reward = base_reward
        
        return reward
    
    def _multi_objective_reward(self, 
                              old_asset: th.Tensor, 
                              new_asset: th.Tensor, 
                              action_int: th.Tensor, 
                              mid_price: th.Tensor,
                              slippage: float) -> th.Tensor:
        """
        Multi-objective reward combining returns, Sharpe ratio, and drawdown control
        Most sophisticated approach balancing profitability and risk
        """
        # Base components
        raw_return = new_asset - old_asset
        transaction_cost_reward = self._transaction_cost_adjusted_reward(old_asset, new_asset, action_int, mid_price, slippage)
        
        # Update asset history for drawdown calculation
        current_asset = new_asset.mean().item()
        self.asset_history.append(current_asset)
        
        # Update peak and calculate drawdown
        self.peak_asset_value = max(self.peak_asset_value, current_asset)
        self.current_drawdown = (self.peak_asset_value - current_asset) / self.peak_asset_value
        self.drawdown_history.append(self.current_drawdown)
        
        # Calculate drawdown penalty
        if self.current_drawdown > 0.01:  # 1% drawdown threshold
            drawdown_penalty = th.tensor(
                self.current_drawdown * self.max_drawdown_penalty * current_asset,
                device=self.device, 
                dtype=raw_return.dtype
            )
        else:
            drawdown_penalty = th.zeros_like(raw_return)
        
        # Sharpe component (if enough history)
        sharpe_bonus = th.zeros_like(raw_return)
        if len(self.return_history) >= 20:
            returns_array = np.array(list(self.return_history))
            if np.std(returns_array) > 1e-8:
                sharpe_ratio = np.mean(returns_array - self.risk_free_rate) / np.std(returns_array)
                sharpe_bonus = th.tensor(sharpe_ratio * 0.005, device=self.device, dtype=raw_return.dtype)
        
        # Multi-objective combination
        # α * returns + β * sharpe - γ * drawdown - δ * transaction_costs
        alpha, beta, gamma = 1.0, 0.5, 1.0
        
        reward = (alpha * raw_return + 
                 beta * sharpe_bonus - 
                 gamma * drawdown_penalty - 
                 (raw_return - transaction_cost_reward))  # Transaction cost component
        
        # Update return history
        return_rate = (raw_return / old_asset).mean().item()
        self.return_history.append(return_rate)
        
        return reward
    
    def _profit_maximizing_reward(self,
                                old_asset: th.Tensor,
                                new_asset: th.Tensor,
                                action_int: th.Tensor,
                                mid_price: th.Tensor,
                                slippage: float) -> th.Tensor:
        """
        Aggressive profit-maximizing reward function
        Optimized for achieving Sharpe ratio > 1.0
        """
        # Base profit/loss
        raw_return = new_asset - old_asset
        
        # Amplify profitable signals by 10x
        profit_amplifier = 10.0
        amplified_return = raw_return * profit_amplifier
        
        # Momentum bonus - reward consistent profitable trades
        current_asset = new_asset.mean().item()
        self.asset_history.append(current_asset)
        
        momentum_bonus = th.zeros_like(raw_return)
        if len(self.asset_history) >= 5:
            recent_returns = []
            for i in range(1, min(6, len(self.asset_history))):
                ret = (self.asset_history[-i] - self.asset_history[-i-1]) / self.asset_history[-i-1]
                recent_returns.append(ret)
            
            # Bonus for consistent positive returns
            if len(recent_returns) >= 3 and all(r > 0 for r in recent_returns[-3:]):
                momentum_bonus = th.tensor(1000.0, device=self.device, dtype=raw_return.dtype)
        
        # Position holding bonus - encourage maintaining profitable positions
        position_bonus = th.zeros_like(raw_return)
        if raw_return.item() > 0 and action_int.abs().sum().item() == 0:  # Holding profitable position
            position_bonus = raw_return * 2.0  # Double reward for holding winners
        
        # Reduced transaction cost penalty (encourage more trading)
        transaction_cost = th.zeros_like(raw_return)
        if action_int.abs().sum().item() > 0:  # Only if trading occurred
            cost = mid_price * slippage * self.transaction_cost_penalty * 0.5  # Half penalty
            transaction_cost = cost.expand_as(raw_return)
        
        # Volatility-adjusted scaling (reward higher vol periods more)
        vol_adjustment = 1.0
        if len(self.return_history) >= 10:
            returns_array = np.array(list(self.return_history)[-10:])
            current_vol = np.std(returns_array)
            if current_vol > 0:
                vol_adjustment = min(3.0, 1.0 + current_vol * 100)  # Scale with volatility
        
        # Combine all components
        total_reward = (amplified_return + 
                       momentum_bonus + 
                       position_bonus - 
                       transaction_cost) * vol_adjustment
        
        # Update tracking
        if len(self.asset_history) >= 2:
            current_return = (current_asset - self.asset_history[-2]) / self.asset_history[-2]
            self.return_history.append(current_return)
        
        return total_reward
    
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics for monitoring"""
        metrics = {
            "total_transaction_costs": self.total_transaction_costs,
            "current_drawdown": self.current_drawdown,
            "peak_asset_value": self.peak_asset_value,
            "return_history_length": len(self.return_history)
        }
        
        if len(self.return_history) >= 10:
            returns_array = np.array(list(self.return_history))
            metrics.update({
                "mean_return": np.mean(returns_array),
                "return_volatility": np.std(returns_array),
                "sharpe_ratio": np.mean(returns_array - self.risk_free_rate) / max(np.std(returns_array), 1e-8),
                "max_drawdown_period": max(self.drawdown_history) if self.drawdown_history else 0.0
            })
        
        return metrics
    
    def reset(self):
        """Reset historical tracking for new episode"""
        self.asset_history.clear()
        self.return_history.clear() 
        self.drawdown_history.clear()
        self.total_transaction_costs = 0.0
        self.peak_asset_value = 1e6
        self.current_drawdown = 0.0


def create_reward_calculator(reward_type: str = "multi_objective", 
                           lookback_window: int = 100,
                           device: str = "cpu") -> RewardCalculator:
    """
    Factory function to create reward calculator
    
    Args:
        reward_type: "simple", "transaction_cost_adjusted", "sharpe_adjusted", "multi_objective"
        lookback_window: History window for calculations
        device: PyTorch device
        
    Returns:
        Configured RewardCalculator instance
    """
    return RewardCalculator(
        reward_type=reward_type,
        lookback_window=lookback_window,
        device=device
    )


# Example usage and testing
if __name__ == "__main__":
    # Test different reward functions
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    print("🧪 Testing Reward Functions")
    print("=" * 50)
    
    # Test each reward type
    reward_types = ["simple", "transaction_cost_adjusted", "sharpe_adjusted", "multi_objective"]
    
    for reward_type in reward_types:
        print(f"\n🎯 Testing {reward_type} reward:")
        calc = create_reward_calculator(reward_type, device=device)
        
        # Simulate some trading
        old_asset = th.tensor([1000000.0], device=device)
        new_asset = th.tensor([1001000.0], device=device)  # $1000 gain
        action_int = th.tensor([1], device=device)  # Buy action
        mid_price = th.tensor([50000.0], device=device)  # BTC price
        slippage = 7e-7
        
        reward = calc.calculate_reward(old_asset, new_asset, action_int, mid_price, slippage)
        print(f"   Reward: {reward.item():.2f}")
        
        # Get metrics
        metrics = calc.get_performance_metrics()
        print(f"   Transaction costs: ${metrics['total_transaction_costs']:.2f}")
        print(f"   Current drawdown: {metrics['current_drawdown']:.4f}")