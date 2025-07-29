"""
Enhanced Reward Functions for Profitable Trading
Addresses the core profitability issues in the FinRL Contest 2024 model
"""

import torch as th
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque
from abc import ABC, abstractmethod


class MarketRegimeDetector:
    """
    Detects market regimes (trending, ranging, volatile) for adaptive reward adjustment
    """
    
    def __init__(self, lookback: int = 50, device: str = "cpu"):
        self.lookback = lookback
        self.device = device
        self.price_history = deque(maxlen=lookback * 2)
        self.volatility_history = deque(maxlen=lookback)
        
    def detect_regime(self, price_data: th.Tensor) -> str:
        """
        Detect current market regime based on price patterns
        
        Args:
            price_data: Current price tensor
            
        Returns:
            Market regime: "trending", "ranging", or "volatile"
        """
        # Update price history
        if isinstance(price_data, th.Tensor):
            price_value = price_data.mean().item()
        else:
            price_value = float(price_data)
            
        self.price_history.append(price_value)
        
        if len(self.price_history) < self.lookback:
            return "ranging"  # Default until we have enough history
            
        # Calculate metrics
        prices = np.array(list(self.price_history)[-self.lookback:])
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility
        volatility = np.std(returns)
        self.volatility_history.append(volatility)
        
        # Trend strength (ADX-like calculation)
        price_changes = np.diff(prices)
        positive_changes = np.where(price_changes > 0, price_changes, 0)
        negative_changes = np.where(price_changes < 0, -price_changes, 0)
        
        avg_positive = np.mean(positive_changes)
        avg_negative = np.mean(negative_changes)
        
        if avg_positive + avg_negative > 0:
            directional_index = abs(avg_positive - avg_negative) / (avg_positive + avg_negative)
        else:
            directional_index = 0
            
        # Classify regime
        high_volatility_threshold = np.percentile(list(self.volatility_history), 80) if len(self.volatility_history) > 10 else volatility * 1.5
        
        if volatility > high_volatility_threshold:
            return "volatile"
        elif directional_index > 0.3:  # Strong directional movement
            return "trending"
        else:
            return "ranging"
            
    def get_regime_multiplier(self, regime: str, context: str = "conservatism") -> float:
        """
        Get regime-specific multiplier for various reward components
        
        Args:
            regime: Current market regime
            context: Context for multiplier ("conservatism", "risk", "exploration")
            
        Returns:
            Multiplier value
        """
        multipliers = {
            "conservatism": {
                "trending": 1.5,    # Higher penalty for holding during trends
                "volatile": 0.7,    # Lower penalty during high volatility
                "ranging": 1.0      # Normal penalty in ranging markets
            },
            "risk": {
                "trending": 0.8,    # Lower risk penalty in trends
                "volatile": 1.5,    # Higher risk penalty in volatile markets
                "ranging": 1.0      # Normal risk penalty
            },
            "exploration": {
                "trending": 0.8,    # Less exploration needed in clear trends
                "volatile": 1.2,    # More exploration in volatile markets
                "ranging": 1.0      # Normal exploration
            }
        }
        
        return multipliers.get(context, {}).get(regime, 1.0)


class DynamicConservatismPenalty:
    """
    Implements dynamic conservatism penalties that adapt based on trading activity and market conditions
    """
    
    def __init__(self, 
                 base_penalty_weight: float = 0.1,
                 activity_threshold: float = 0.3,
                 escalation_rate: float = 1.5,
                 regime_sensitivity: float = 1.0,
                 device: str = "cpu"):
        """
        Initialize dynamic conservatism penalty system
        
        Args:
            base_penalty_weight: Base penalty for excessive holding
            activity_threshold: Minimum desired trading activity (30% = 0.3)
            escalation_rate: Rate at which penalty escalates over time
            regime_sensitivity: Sensitivity to market regime changes
            device: PyTorch device
        """
        self.base_penalty_weight = base_penalty_weight
        self.activity_threshold = activity_threshold
        self.escalation_rate = escalation_rate
        self.regime_sensitivity = regime_sensitivity
        self.device = device
        
        # Tracking
        self.action_window = deque(maxlen=100)
        self.penalty_escalation = 1.0
        self.consecutive_conservative_periods = 0
        
    def calculate_penalty(self, 
                         recent_actions: th.Tensor,
                         market_regime: str,
                         regime_multiplier: float = 1.0) -> th.Tensor:
        """
        Calculate dynamic conservatism penalty
        
        Args:
            recent_actions: Recent action history (0=sell, 1=hold, 2=buy)
            market_regime: Current market regime
            regime_multiplier: Market regime adjustment factor
            
        Returns:
            Penalty value (to be subtracted from reward)
        """
        # Update action tracking
        if isinstance(recent_actions, th.Tensor):
            actions_list = recent_actions.cpu().numpy().tolist()
            if isinstance(actions_list, list):
                self.action_window.extend(actions_list)
            else:
                self.action_window.append(actions_list)
        
        if len(self.action_window) < 20:  # Need minimum history
            return th.tensor(0.0, device=self.device)
            
        # Calculate activity metrics
        recent_window = list(self.action_window)[-50:]  # Last 50 actions
        hold_ratio = recent_window.count(1) / len(recent_window)
        buy_ratio = recent_window.count(2) / len(recent_window)
        sell_ratio = recent_window.count(0) / len(recent_window)
        
        # Check for extreme conservatism
        activity_ratio = 1.0 - hold_ratio
        is_conservative = activity_ratio < self.activity_threshold
        
        # Escalate penalty for persistent conservatism
        if is_conservative:
            self.consecutive_conservative_periods += 1
            self.penalty_escalation = min(3.0, 1.0 + (self.consecutive_conservative_periods * 0.1))
        else:
            self.consecutive_conservative_periods = max(0, self.consecutive_conservative_periods - 1)
            self.penalty_escalation = max(1.0, self.penalty_escalation * 0.95)
            
        # Base penalty calculation
        conservatism_factor = max(0, self.activity_threshold - activity_ratio)
        base_penalty = conservatism_factor * self.base_penalty_weight * self.penalty_escalation
        
        # Apply regime-based adjustments
        regime_adjusted_penalty = base_penalty * regime_multiplier * self.regime_sensitivity
        
        # Additional penalty for zero trading (no buys at all)
        if buy_ratio == 0 and len(recent_window) >= 50:
            no_buy_penalty = 0.2 * self.penalty_escalation  # Significant penalty for never buying
            regime_adjusted_penalty += no_buy_penalty
            
        # Balance penalty (penalize extreme imbalance between buy/sell)
        action_imbalance = abs(buy_ratio - sell_ratio)
        if action_imbalance > 0.3:  # More than 30% imbalance
            balance_penalty = action_imbalance * 0.05 * self.penalty_escalation
            regime_adjusted_penalty += balance_penalty
            
        return th.tensor(regime_adjusted_penalty, device=self.device, dtype=th.float32)
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current conservatism metrics"""
        if len(self.action_window) < 20:
            return {
                "hold_ratio": 0.0,
                "buy_ratio": 0.0,
                "sell_ratio": 0.0,
                "activity_ratio": 0.0,
                "penalty_escalation": self.penalty_escalation,
                "consecutive_conservative_periods": self.consecutive_conservative_periods
            }
            
        recent_window = list(self.action_window)[-50:]
        hold_ratio = recent_window.count(1) / len(recent_window)
        buy_ratio = recent_window.count(2) / len(recent_window)
        sell_ratio = recent_window.count(0) / len(recent_window)
        
        return {
            "hold_ratio": hold_ratio,
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "activity_ratio": 1.0 - hold_ratio,
            "penalty_escalation": self.penalty_escalation,
            "consecutive_conservative_periods": self.consecutive_conservative_periods
        }


class ActionDiversityTracker:
    """
    Tracks and rewards action diversity to prevent conservative convergence
    """
    
    def __init__(self, target_entropy: float = 0.8, device: str = "cpu"):
        """
        Initialize action diversity tracker
        
        Args:
            target_entropy: Target entropy for action distribution (0-1)
            device: PyTorch device
        """
        self.target_entropy = target_entropy
        self.device = device
        self.action_counts = {0: 0, 1: 0, 2: 0}  # sell, hold, buy
        self.total_actions = 0
        
    def update(self, action: int):
        """Update action counts"""
        if action in self.action_counts:
            self.action_counts[action] += 1
            self.total_actions += 1
            
    def calculate_diversity_reward(self) -> th.Tensor:
        """
        Calculate reward based on action distribution entropy
        
        Returns:
            Diversity reward (positive for good diversity)
        """
        if self.total_actions < 10:
            return th.tensor(0.0, device=self.device)
            
        # Calculate action probabilities
        probs = np.array([self.action_counts[i] / self.total_actions for i in range(3)])
        
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs)) / np.log(3)  # Normalized to [0, 1]
        
        # Reward based on entropy vs target
        diversity_reward = (entropy - 0.5) * 0.1  # Positive if entropy > 0.5
        
        # Bonus for balanced distribution
        if min(probs) > 0.15:  # All actions used reasonably
            diversity_reward += 0.05
            
        return th.tensor(diversity_reward, device=self.device, dtype=th.float32)
        
    def reset_window(self, keep_ratio: float = 0.5):
        """Partially reset counts to maintain some history"""
        for key in self.action_counts:
            self.action_counts[key] = int(self.action_counts[key] * keep_ratio)
        self.total_actions = sum(self.action_counts.values())


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
                 device: str = "cpu",
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize reward calculator with multiple variants
        
        Args:
            reward_type: "sharpe_adjusted", "transaction_cost_adjusted", "multi_objective", 
                        "adaptive_multi_objective", "simple"
            lookback_window: Period for calculating volatility and Sharpe ratios
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            transaction_cost_penalty: Penalty factor for transaction costs
            max_drawdown_penalty: Penalty multiplier for drawdown periods
            device: PyTorch device for calculations
            reward_weights: Custom weights for multi-objective rewards
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
        
        # Default reward weights for multi-objective
        self.reward_weights = reward_weights or {
            'risk_adjusted_return_weight': 0.7,
            'conservatism_penalty_weight': 0.2,
            'action_diversity_weight': 0.15,
            'transaction_cost_weight': 0.5,
            'conservatism_escalation_rate': 1.5,
            'activity_threshold': 0.3,
            'regime_sensitivity': 1.0
        }
        
        # Initialize adaptive components for adaptive_multi_objective
        if reward_type == "adaptive_multi_objective":
            self.market_regime_detector = MarketRegimeDetector(lookback=50, device=device)
            self.conservatism_penalty = DynamicConservatismPenalty(
                base_penalty_weight=self.reward_weights.get('conservatism_penalty_weight', 0.2),
                activity_threshold=self.reward_weights.get('activity_threshold', 0.3),
                escalation_rate=self.reward_weights.get('conservatism_escalation_rate', 1.5),
                regime_sensitivity=self.reward_weights.get('regime_sensitivity', 1.0),
                device=device
            )
            self.diversity_tracker = ActionDiversityTracker(target_entropy=0.8, device=device)
        
        print(f"🎯 RewardCalculator initialized with '{reward_type}' method")
        print(f"   Lookback window: {lookback_window}")
        print(f"   Risk-free rate: {risk_free_rate:.4f} (annual)")
        print(f"   Transaction cost penalty: {transaction_cost_penalty}")
        if reward_type in ["multi_objective", "adaptive_multi_objective"]:
            print(f"   Using custom reward weights: {list(self.reward_weights.keys())}")
    
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
        elif self.reward_type == "adaptive_multi_objective":
            return self._adaptive_multi_objective_reward(old_asset, new_asset, action_int, mid_price, slippage)
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
        
        # CRITICAL FIX: Prevent division by zero and NaN propagation
        old_asset_safe = th.clamp(old_asset.abs(), min=1e-8)  # Ensure non-zero denominator
        return_rate = base_reward / old_asset_safe
        
        # CRITICAL FIX: Check for non-finite values before updating history
        return_rate_item = return_rate.mean().item()
        if not (th.isfinite(return_rate).all() and np.isfinite(return_rate_item)):
            print(f"⚠️ Non-finite return rate detected: {return_rate_item}, using fallback")
            return_rate_item = 0.0
        
        # Update return history
        self.return_history.append(return_rate_item)
        
        # Calculate rolling Sharpe ratio if we have enough history
        if len(self.return_history) >= 10:  # Minimum history for meaningful calculation
            returns_array = np.array(list(self.return_history))
            
            # CRITICAL FIX: Remove non-finite values from returns array
            returns_array = returns_array[np.isfinite(returns_array)]
            
            if len(returns_array) >= 5:  # Need minimum valid data points
                excess_returns = returns_array - self.risk_free_rate
                returns_std = np.std(returns_array)
                
                # CRITICAL FIX: Robust standard deviation check
                if returns_std > 1e-8 and np.isfinite(returns_std):  # Avoid division by zero
                    sharpe_ratio = np.mean(excess_returns) / returns_std
                    
                    # CRITICAL FIX: Validate Sharpe ratio before using
                    if np.isfinite(sharpe_ratio) and abs(sharpe_ratio) < 100:  # Reasonable bounds
                        sharpe_bonus = th.tensor(sharpe_ratio * 0.01, device=self.device, dtype=base_reward.dtype)
                        reward = base_reward + sharpe_bonus
                    else:
                        reward = base_reward
                else:
                    reward = base_reward
            else:
                reward = base_reward
        else:
            reward = base_reward
        
        # CRITICAL FIX: Final safety check on reward
        if not th.isfinite(reward).all():
            print(f"⚠️ Non-finite reward detected, falling back to base reward")
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
        
        # CRITICAL FIX: Validate asset values before processing
        current_asset = new_asset.mean().item()
        if not np.isfinite(current_asset) or current_asset <= 0:
            print(f"⚠️ Invalid asset value detected: {current_asset}, using fallback")
            current_asset = max(1e6, self.peak_asset_value * 0.99)  # Fallback to reasonable value
            
        self.asset_history.append(current_asset)
        
        # Update peak and calculate drawdown with safety checks
        if current_asset > self.peak_asset_value:
            self.peak_asset_value = current_asset
            
        # CRITICAL FIX: Safe drawdown calculation
        if self.peak_asset_value > 0:
            self.current_drawdown = max(0.0, (self.peak_asset_value - current_asset) / self.peak_asset_value)
        else:
            self.current_drawdown = 0.0
            
        # CRITICAL FIX: Validate drawdown before using
        if not np.isfinite(self.current_drawdown):
            print(f"⚠️ Invalid drawdown detected, resetting to 0")
            self.current_drawdown = 0.0
            
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
            
            # CRITICAL FIX: Remove non-finite values from returns array
            returns_array = returns_array[np.isfinite(returns_array)]
            
            if len(returns_array) >= 10:  # Need minimum valid data points
                returns_std = np.std(returns_array)
                if returns_std > 1e-8 and np.isfinite(returns_std):
                    sharpe_ratio = np.mean(returns_array - self.risk_free_rate) / returns_std
                    
                    # CRITICAL FIX: Validate Sharpe ratio
                    if np.isfinite(sharpe_ratio) and abs(sharpe_ratio) < 100:
                        sharpe_bonus = th.tensor(sharpe_ratio * 0.005, device=self.device, dtype=raw_return.dtype)
        
        # Multi-objective combination
        # α * returns + β * sharpe - γ * drawdown - δ * transaction_costs
        alpha, beta, gamma = 1.0, 0.5, 1.0
        
        reward = (alpha * raw_return + 
                 beta * sharpe_bonus - 
                 gamma * drawdown_penalty - 
                 (raw_return - transaction_cost_reward))  # Transaction cost component
        
        # Update return history with safety checks
        old_asset_safe = th.clamp(old_asset.abs(), min=1e-8)  # Prevent division by zero
        return_rate = (raw_return / old_asset_safe).mean().item()
        
        # CRITICAL FIX: Validate return rate before adding to history
        if np.isfinite(return_rate) and abs(return_rate) < 10:  # Reasonable bounds
            self.return_history.append(return_rate)
        else:
            print(f"⚠️ Invalid return rate detected: {return_rate}, using 0.0")
            self.return_history.append(0.0)
        
        # CRITICAL FIX: Final safety check on reward
        if not th.isfinite(reward).all():
            print(f"⚠️ Non-finite multi-objective reward detected, using raw return")
            reward = raw_return
            
        return reward
    
    def _adaptive_multi_objective_reward(self,
                                       old_asset: th.Tensor,
                                       new_asset: th.Tensor,
                                       action_int: th.Tensor,
                                       mid_price: th.Tensor,
                                       slippage: float) -> th.Tensor:
        """
        Adaptive multi-objective reward with dynamic conservatism penalties and market regime awareness
        Most sophisticated approach that adapts to market conditions and agent behavior
        """
        # Base components
        raw_return = new_asset - old_asset
        
        # CRITICAL FIX: Safe return rate calculation
        old_asset_safe = th.clamp(old_asset.abs(), min=1e-8)  # Prevent division by zero
        return_rate = (raw_return / old_asset_safe).mean().item()
        
        # CRITICAL FIX: Validate return rate before using
        if not np.isfinite(return_rate) or abs(return_rate) > 10:
            print(f"⚠️ Invalid return rate detected: {return_rate}, using fallback")
            return_rate = 0.0
        
        # CRITICAL FIX: Validate asset values before processing
        current_asset = new_asset.mean().item()
        if not np.isfinite(current_asset) or current_asset <= 0:
            print(f"⚠️ Invalid asset value detected: {current_asset}, using fallback")
            current_asset = max(1e6, self.peak_asset_value * 0.99)  # Fallback to reasonable value
        
        # Update histories
        self.asset_history.append(current_asset)
        self.return_history.append(return_rate)
        
        # 1. Market regime detection with safety
        try:
            market_regime = self.market_regime_detector.detect_regime(mid_price)
            regime_multiplier = self.market_regime_detector.get_regime_multiplier(market_regime, "conservatism")
        except Exception as e:
            print(f"⚠️ Market regime detection failed: {e}, using defaults")
            market_regime = "ranging"
            regime_multiplier = 1.0
        
        # 2. Transaction cost component
        trade_occurred = action_int.abs() > 0
        transaction_volume = action_int.abs() * mid_price
        transaction_cost = th.where(
            trade_occurred,
            transaction_volume * (slippage + self.transaction_cost_penalty),
            th.zeros_like(transaction_volume)
        )
        
        # CRITICAL FIX: Validate transaction cost before updating
        transaction_cost_sum = transaction_cost.sum().item()
        if np.isfinite(transaction_cost_sum) and transaction_cost_sum >= 0:
            self.total_transaction_costs += transaction_cost_sum
        
        # 3. Risk-adjusted return component (Sharpe-like)
        sharpe_component = th.zeros_like(raw_return)
        if len(self.return_history) >= 20:
            returns_array = np.array(list(self.return_history))
            
            # CRITICAL FIX: Remove non-finite values from returns array
            returns_array = returns_array[np.isfinite(returns_array)]
            
            if len(returns_array) >= 10:  # Need minimum valid data points
                returns_std = np.std(returns_array)
                if returns_std > 1e-8 and np.isfinite(returns_std):
                    sharpe_ratio = np.mean(returns_array - self.risk_free_rate) / returns_std
                    
                    # CRITICAL FIX: Validate Sharpe ratio
                    if np.isfinite(sharpe_ratio) and abs(sharpe_ratio) < 100:
                        sharpe_component = th.tensor(
                            sharpe_ratio * 0.01 * self.reward_weights['risk_adjusted_return_weight'],
                            device=self.device, 
                            dtype=raw_return.dtype
                        )
        
        # 4. Drawdown control with safety checks
        if current_asset > self.peak_asset_value:
            self.peak_asset_value = current_asset
            
        # CRITICAL FIX: Safe drawdown calculation
        if self.peak_asset_value > 0:
            self.current_drawdown = max(0.0, (self.peak_asset_value - current_asset) / self.peak_asset_value)
        else:
            self.current_drawdown = 0.0
            
        # CRITICAL FIX: Validate drawdown before using
        if not np.isfinite(self.current_drawdown):
            print(f"⚠️ Invalid drawdown detected, resetting to 0")
            self.current_drawdown = 0.0
            
        self.drawdown_history.append(self.current_drawdown)
        
        drawdown_penalty = th.zeros_like(raw_return)
        if self.current_drawdown > 0.02:  # 2% drawdown threshold
            drawdown_penalty = th.tensor(
                self.current_drawdown * self.max_drawdown_penalty * current_asset,
                device=self.device, 
                dtype=raw_return.dtype
            )
        
        # 5. Dynamic conservatism penalty with error handling
        try:
            # Convert action_int to 0-2 range for tracking
            action_for_tracking = action_int + 1  # -1,0,1 -> 0,1,2
            conservatism_penalty = self.conservatism_penalty.calculate_penalty(
                recent_actions=action_for_tracking,
                market_regime=market_regime,
                regime_multiplier=regime_multiplier
            )
        except Exception as e:
            print(f"⚠️ Conservatism penalty calculation failed: {e}, using zero penalty")
            conservatism_penalty = th.tensor(0.0, device=self.device, dtype=raw_return.dtype)
        
        # 6. Action diversity reward with error handling
        try:
            action_for_update = action_for_tracking.item() if action_for_tracking.numel() == 1 else action_for_tracking[0].item()
            if 0 <= action_for_update <= 2:  # Valid action range
                self.diversity_tracker.update(int(action_for_update))
            diversity_reward = self.diversity_tracker.calculate_diversity_reward()
            
            # Partially reset diversity tracker periodically
            if self.diversity_tracker.total_actions > 200:
                self.diversity_tracker.reset_window(keep_ratio=0.7)
        except Exception as e:
            print(f"⚠️ Diversity tracking failed: {e}, using zero diversity reward")
            diversity_reward = th.tensor(0.0, device=self.device, dtype=raw_return.dtype)
        
        # 7. Combine all components with adaptive weights
        weights = self.reward_weights
        
        # Base return with risk adjustment
        risk_adjusted_return = raw_return * weights.get('risk_adjusted_return_weight', 0.7) + sharpe_component
        
        # Apply all modifiers
        total_reward = (
            risk_adjusted_return
            + diversity_reward * weights.get('action_diversity_weight', 0.15)
            - conservatism_penalty * weights.get('conservatism_penalty_weight', 0.2)
            - transaction_cost * weights.get('transaction_cost_weight', 0.5)
            - drawdown_penalty
        )
        
        # Market regime-specific adjustments with safety checks
        try:
            action_sum = action_int.abs().sum().item()
            if np.isfinite(action_sum):
                if market_regime == "trending" and action_sum > 0:  # Reward trading in trends
                    total_reward *= 1.2
                elif market_regime == "volatile" and action_sum == 0:  # Penalize holding in volatility
                    total_reward *= 0.8
        except Exception as e:
            print(f"⚠️ Regime adjustment failed: {e}, skipping adjustment")
        
        # CRITICAL FIX: Final safety check on total reward
        if not th.isfinite(total_reward).all():
            print(f"⚠️ Non-finite adaptive reward detected, using raw return")
            total_reward = raw_return
            
        return total_reward
    
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
        
        # Add adaptive components metrics if available
        if hasattr(self, 'conservatism_penalty'):
            conservatism_metrics = self.conservatism_penalty.get_metrics()
            metrics.update({
                f"conservatism_{k}": v for k, v in conservatism_metrics.items()
            })
            
        if hasattr(self, 'market_regime_detector') and len(self.market_regime_detector.price_history) > 0:
            current_regime = self.market_regime_detector.detect_regime(
                th.tensor(self.market_regime_detector.price_history[-1])
            )
            metrics["current_market_regime"] = current_regime
            
        if hasattr(self, 'diversity_tracker'):
            metrics["action_diversity_score"] = self.diversity_tracker.calculate_diversity_reward().item()
        
        return metrics
    
    def reset(self):
        """Reset historical tracking for new episode"""
        self.asset_history.clear()
        self.return_history.clear() 
        self.drawdown_history.clear()
        self.total_transaction_costs = 0.0
        self.peak_asset_value = 1e6
        self.current_drawdown = 0.0
        
        # Reset adaptive components if available
        if hasattr(self, 'conservatism_penalty'):
            self.conservatism_penalty.action_window.clear()
            self.conservatism_penalty.penalty_escalation = 1.0
            self.conservatism_penalty.consecutive_conservative_periods = 0
            
        if hasattr(self, 'market_regime_detector'):
            self.market_regime_detector.price_history.clear()
            self.market_regime_detector.volatility_history.clear()
            
        if hasattr(self, 'diversity_tracker'):
            self.diversity_tracker.action_counts = {0: 0, 1: 0, 2: 0}
            self.diversity_tracker.total_actions = 0


def create_reward_calculator(reward_type: str = "multi_objective", 
                           lookback_window: int = 100,
                           device: str = "cpu",
                           reward_weights: Optional[Dict[str, float]] = None) -> RewardCalculator:
    """
    Factory function to create reward calculator
    
    Args:
        reward_type: "simple", "transaction_cost_adjusted", "sharpe_adjusted", 
                    "multi_objective", "adaptive_multi_objective"
        lookback_window: History window for calculations
        device: PyTorch device
        reward_weights: Custom weights for multi-objective rewards
        
    Returns:
        Configured RewardCalculator instance
    """
    return RewardCalculator(
        reward_type=reward_type,
        lookback_window=lookback_window,
        device=device,
        reward_weights=reward_weights
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