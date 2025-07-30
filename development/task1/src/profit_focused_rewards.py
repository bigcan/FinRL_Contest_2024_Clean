"""
Profit-Focused Reward Functions for FinRL Contest 2024
Addresses the core issue: Agent minimizes losses but doesn't generate profits

Key Components:
1. Amplified profit rewards (3x multiplier for positive returns)
2. Trade completion bonuses
3. Opportunity cost penalties for excessive holding
4. Market regime-aware adjustments
"""

import torch as th
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
import logging

class ProfitFocusedRewardCalculator:
    """
    Implements aggressive profit-seeking reward function
    Designed to transition from loss minimization to profit maximization
    """
    
    def __init__(self,
                 profit_amplifier: float = 3.0,
                 loss_multiplier: float = 1.0,
                 trade_completion_bonus: float = 0.02,
                 opportunity_cost_penalty: float = 0.001,
                 momentum_window: int = 10,
                 regime_sensitivity: bool = True,
                 profit_speed_enabled: bool = True,
                 max_speed_multiplier: float = 5.0,
                 speed_decay_rate: float = 0.02,
                 min_holding_time: int = 5,
                 device: str = "cpu"):
        """
        Initialize profit-focused reward calculator
        
        Args:
            profit_amplifier: Multiplier for positive returns (default 3x)
            loss_multiplier: Multiplier for negative returns (default 1x)
            trade_completion_bonus: Bonus for closing profitable trades
            opportunity_cost_penalty: Penalty per step for holding positions
            momentum_window: Window for momentum calculation
            regime_sensitivity: Whether to adjust rewards based on market regime
            profit_speed_enabled: Enable profit speed multiplier
            max_speed_multiplier: Maximum multiplier for ultra-fast profits
            speed_decay_rate: Exponential decay rate for speed multiplier
            min_holding_time: Minimum holding time to avoid noise trades
            device: PyTorch device
        """
        self.profit_amplifier = profit_amplifier
        self.loss_multiplier = loss_multiplier
        self.trade_completion_bonus = trade_completion_bonus
        self.opportunity_cost_penalty = opportunity_cost_penalty
        self.momentum_window = momentum_window
        self.regime_sensitivity = regime_sensitivity
        self.profit_speed_enabled = profit_speed_enabled
        self.max_speed_multiplier = max_speed_multiplier
        self.speed_decay_rate = speed_decay_rate
        self.min_holding_time = min_holding_time
        self.device = device
        
        # State tracking
        self.position_entry_price = None
        self.position_holding_time = 0
        self.recent_returns = deque(maxlen=momentum_window)
        self.completed_trades = []
        self.total_profit = 0.0
        
        # Performance metrics
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        
        self.logger = logging.getLogger(__name__)
        
    def calculate_reward(self,
                        action: int,
                        current_price: float,
                        previous_price: float,
                        position: int,
                        previous_position: int,
                        market_regime: Optional[str] = None) -> float:
        """
        Calculate profit-focused reward
        
        Args:
            action: Trading action (0=sell, 1=hold, 2=buy)
            current_price: Current market price
            previous_price: Previous market price
            position: Current position after action
            previous_position: Position before action
            market_regime: Optional market regime for adjustments
            
        Returns:
            Reward value (amplified for profits)
        """
        reward = 0.0
        
        # Calculate price return
        price_return = (current_price - previous_price) / previous_price
        
        # Track returns for momentum
        self.recent_returns.append(price_return)
        
        # 1. Position-based P&L (Core Component)
        position_pnl = previous_position * price_return
        
        # Apply asymmetric amplification
        if position_pnl > 0:
            # AMPLIFY PROFITS - This is the key change
            base_reward = position_pnl * self.profit_amplifier
            
            # Additional momentum bonus for consecutive wins
            if self._is_momentum_positive():
                momentum_bonus = position_pnl * 0.5  # 50% extra for momentum
                reward += momentum_bonus
        else:
            # Normal penalty for losses
            base_reward = position_pnl * self.loss_multiplier
        
        reward += base_reward
        
        # 2. Trade Completion Bonus
        if self._is_trade_completed(position, previous_position):
            trade_return = self._calculate_trade_return(current_price)
            
            if trade_return > 0:
                # NEW: Apply profit speed multiplier
                speed_multiplier = self.calculate_profit_speed_multiplier(self.position_holding_time)
                
                # Amplify the base reward with speed multiplier
                # This stacks with the profit amplifier for compound effect
                # Example: 1% profit in 10s = 1% × 3 (profit amp) × 4.1 (speed) = 12.3% reward
                if speed_multiplier > 1.0:
                    # Apply speed bonus to the entire profit component
                    profit_component = base_reward * (speed_multiplier - 1.0)
                    reward += profit_component
                    
                    # Log exceptional speed trades
                    if speed_multiplier > 3.0:
                        self.logger.debug(f"Fast profit! Return: {trade_return:.4f}, "
                                        f"Time: {self.position_holding_time}s, "
                                        f"Speed multiplier: {speed_multiplier:.2f}x")
                
                # Standard completion bonus
                completion_bonus = self.trade_completion_bonus * (1 + trade_return)
                reward += completion_bonus
                
                # Track winning trade
                self.winning_trades += 1
                self.largest_win = max(self.largest_win, trade_return)
            else:
                # Small penalty for losing trade (encourage cutting losses)
                # No speed bonus for losses
                reward += trade_return * 0.5
                
                # Track losing trade
                self.losing_trades += 1
                self.largest_loss = min(self.largest_loss, trade_return)
            
            self.total_trades += 1
            self.completed_trades.append(trade_return)
            
            # Reset position tracking
            self.position_entry_price = None
            self.position_holding_time = 0
            
        # 3. Position Entry Tracking
        elif self._is_new_position(position, previous_position):
            self.position_entry_price = current_price
            self.position_holding_time = 0
            
        # 4. Opportunity Cost Penalty
        if position != 0:
            self.position_holding_time += 1
            
            # Escalating penalty for holding too long
            holding_penalty = self.opportunity_cost_penalty * min(self.position_holding_time / 100, 2.0)
            reward -= holding_penalty
            
            # Additional penalty for holding losing positions
            if self.position_entry_price is not None:
                unrealized_pnl = (current_price - self.position_entry_price) / self.position_entry_price
                if unrealized_pnl < -0.02:  # More than 2% loss
                    reward -= abs(unrealized_pnl) * 0.1  # Encourage cutting losses
                    
        # 5. Action Encouragement
        # Reward decisive actions over holding
        if action != 1:  # Not holding
            action_bonus = 0.001
            
            # Extra bonus for counter-trend actions (buy low, sell high)
            if (action == 2 and price_return < 0) or (action == 0 and price_return > 0):
                action_bonus *= 2
                
            reward += action_bonus
            
        # 6. Market Regime Adjustments
        if self.regime_sensitivity and market_regime:
            regime_multipliers = {
                "trending": 1.2,    # Amplify rewards in trends
                "volatile": 0.8,    # Reduce rewards in volatile markets
                "ranging": 1.0      # Normal rewards in ranging markets
            }
            regime_mult = regime_multipliers.get(market_regime, 1.0)
            reward *= regime_mult
            
        # 7. Risk-Adjusted Performance Bonus
        if len(self.completed_trades) >= 5:
            win_rate = self.winning_trades / max(self.total_trades, 1)
            if win_rate > 0.6:  # More than 60% win rate
                performance_bonus = 0.01 * win_rate
                reward += performance_bonus
        
        return reward
    
    def calculate_profit_speed_multiplier(self, holding_time: int) -> float:
        """
        Calculate speed multiplier for profits based on holding time
        Faster profits get higher multipliers
        
        Args:
            holding_time: Time in steps (seconds) position was held
            
        Returns:
            Speed multiplier (1.0 to max_speed_multiplier)
        """
        if not self.profit_speed_enabled:
            return 1.0
            
        # Ignore very short holds (likely noise)
        if holding_time < self.min_holding_time:
            return 1.0
            
        # Exponential decay function
        # Fast profits (10s) → high multiplier, slow profits (300s+) → low multiplier
        multiplier = self.max_speed_multiplier * np.exp(-self.speed_decay_rate * holding_time)
        
        # Ensure minimum multiplier of 1.0
        return max(multiplier, 1.0)
    
    def _is_momentum_positive(self) -> bool:
        """Check if recent returns show positive momentum"""
        if len(self.recent_returns) < 3:
            return False
        
        recent = list(self.recent_returns)[-3:]
        return sum(1 for r in recent if r > 0) >= 2
    
    def _is_trade_completed(self, position: int, previous_position: int) -> bool:
        """Check if a trade was just completed"""
        return previous_position != 0 and position == 0
    
    def _is_new_position(self, position: int, previous_position: int) -> bool:
        """Check if a new position was just opened"""
        return previous_position == 0 and position != 0
    
    def _calculate_trade_return(self, exit_price: float) -> float:
        """Calculate return for completed trade"""
        if self.position_entry_price is None:
            return 0.0
        
        return (exit_price - self.position_entry_price) / self.position_entry_price
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        total_trades = max(self.total_trades, 1)
        
        metrics = {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / total_trades,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_trade_return": np.mean(self.completed_trades) if self.completed_trades else 0.0,
            "total_profit": sum(self.completed_trades),
            "current_holding_time": self.position_holding_time,
            "profit_factor": abs(sum(t for t in self.completed_trades if t > 0) / 
                                min(sum(t for t in self.completed_trades if t < 0), -0.0001))
                                if self.completed_trades else 1.0
        }
        
        return metrics
    
    def reset(self):
        """Reset calculator for new episode"""
        self.position_entry_price = None
        self.position_holding_time = 0
        self.recent_returns.clear()
        self.completed_trades = []
        self.total_profit = 0.0
        
        # Reset metrics
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.largest_win = 0.0
        self.largest_loss = 0.0


class MetaRewardCalculator:
    """
    Parameterized reward calculator for HPO optimization
    Allows systematic search for optimal reward weights
    """
    
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 device: str = "cpu"):
        """
        Initialize meta reward calculator with customizable weights
        
        Args:
            weights: Dictionary of component weights
            device: PyTorch device
        """
        self.device = device
        
        # Default weights if not provided
        self.weights = weights or {
            "profit_amplifier": 3.0,
            "loss_multiplier": 1.0,
            "trade_completion_bonus": 0.02,
            "opportunity_cost_penalty": 0.001,
            "momentum_bonus": 0.5,
            "action_bonus": 0.001,
            "holding_penalty_escalation": 0.01,
            "loss_cutting_penalty": 0.1,
            "win_rate_bonus": 0.01
        }
        
        # Initialize profit-focused calculator with weights
        self.profit_calculator = ProfitFocusedRewardCalculator(
            profit_amplifier=self.weights["profit_amplifier"],
            loss_multiplier=self.weights["loss_multiplier"],
            trade_completion_bonus=self.weights["trade_completion_bonus"],
            opportunity_cost_penalty=self.weights["opportunity_cost_penalty"],
            device=device
        )
        
    def calculate_reward(self, **kwargs) -> float:
        """Calculate reward using weighted components"""
        return self.profit_calculator.calculate_reward(**kwargs)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics with weight info"""
        metrics = self.profit_calculator.get_metrics()
        metrics.update({f"weight_{k}": v for k, v in self.weights.items()})
        return metrics
    
    def reset(self):
        """Reset calculator"""
        self.profit_calculator.reset()


def create_profit_focused_calculator(
    calculator_type: str = "profit_focused",
    weights: Optional[Dict[str, float]] = None,
    device: str = "cpu") -> ProfitFocusedRewardCalculator:
    """
    Factory function to create profit-focused reward calculator
    
    Args:
        calculator_type: Type of calculator ("profit_focused" or "meta")
        weights: Optional weight dictionary for meta calculator
        device: PyTorch device
        
    Returns:
        Configured reward calculator
    """
    if calculator_type == "meta":
        return MetaRewardCalculator(weights=weights, device=device)
    else:
        return ProfitFocusedRewardCalculator(device=device)


# Integration function for existing codebase
def integrate_profit_rewards(existing_reward_calc):
    """
    Integrate profit-focused rewards into existing reward calculator
    
    This function modifies the existing RewardCalculator to use
    profit-focused calculations
    """
    # Add profit calculator as a component
    existing_reward_calc.profit_calculator = ProfitFocusedRewardCalculator(
        device=existing_reward_calc.device
    )
    
    # Override the calculate_reward method
    original_calculate = existing_reward_calc.calculate_reward
    
    def enhanced_calculate_reward(old_asset, new_asset, action_int, mid_price, slippage):
        """Enhanced reward calculation with profit focus"""
        
        # Get original reward first
        original_reward = original_calculate(old_asset, new_asset, action_int, mid_price, slippage)
        
        # Extract values for profit-focused calculation
        if isinstance(old_asset, th.Tensor):
            old_asset_val = old_asset.item()
            new_asset_val = new_asset.item()
            action = action_int.item() if isinstance(action_int, th.Tensor) else action_int
            current_price = mid_price.item() if isinstance(mid_price, th.Tensor) else mid_price
        else:
            old_asset_val = float(old_asset)
            new_asset_val = float(new_asset)
            action = int(action_int)
            current_price = float(mid_price)
        
        # Estimate previous price from asset change
        asset_ratio = new_asset_val / old_asset_val if old_asset_val > 0 else 1.0
        price_change_estimate = (asset_ratio - 1.0) / 3.0  # Rough estimate assuming position size
        previous_price = current_price / (1 + price_change_estimate)
        
        # Track position state (simplified - in real implementation this would be tracked properly)
        # For now, we'll use action to infer position changes
        if not hasattr(existing_reward_calc, '_position_state'):
            existing_reward_calc._position_state = 0
            
        previous_position = existing_reward_calc._position_state
        
        # Update position based on action
        if action == 0:  # sell
            position = 0
        elif action == 2:  # buy
            position = 1
        else:  # hold
            position = previous_position
            
        existing_reward_calc._position_state = position
        
        # Calculate profit-focused reward
        profit_reward = existing_reward_calc.profit_calculator.calculate_reward(
            action=action,
            current_price=current_price,
            previous_price=previous_price,
            position=position,
            previous_position=previous_position,
            market_regime=None  # Can be enhanced with regime detection
        )
        
        # Blend rewards (can be adjusted)
        blend_factor = 0.7  # 70% profit-focused, 30% original
        final_reward = blend_factor * profit_reward + (1 - blend_factor) * original_reward
        
        return final_reward
    
    # Replace method
    existing_reward_calc.calculate_reward = enhanced_calculate_reward
    
    # Add metrics tracking
    original_get_metrics = existing_reward_calc.get_performance_metrics
    
    def enhanced_get_metrics():
        """Enhanced metrics with profit tracking"""
        original_metrics = original_get_metrics()
        profit_metrics = existing_reward_calc.profit_calculator.get_metrics()
        
        # Combine metrics
        original_metrics.update({f"profit_{k}": v for k, v in profit_metrics.items()})
        return original_metrics
    
    existing_reward_calc.get_performance_metrics = enhanced_get_metrics
    
    return existing_reward_calc