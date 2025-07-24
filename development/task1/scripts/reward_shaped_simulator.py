"""
Reward-Shaped Trading Simulator

Extends the existing trading simulator with configurable reward shaping
to encourage more active and intelligent trading behavior.
"""

import torch as th
import numpy as np
from trade_simulator import EvalTradeSimulator
from reward_shaping_config import RewardShapingConfig


class RewardShapedEvalTradeSimulator(EvalTradeSimulator):
    def __init__(self, reward_config=None, **kwargs):
        # Extract reward_config from kwargs if passed through env_args
        if 'reward_config' in kwargs:
            reward_config = kwargs.pop('reward_config')
        
        super().__init__(**kwargs)
        self.reward_config = reward_config or RewardShapingConfig.balanced_config()
        
        # Track additional state for reward calculations
        self.price_history = []
        self.action_history = []
        self.position_hold_time = th.zeros((self.num_sims,), dtype=th.int32, device=self.device)
        self.total_trades = th.zeros((self.num_sims,), dtype=th.int32, device=self.device)
        
        # For logging reward components
        self.reward_breakdown = {
            'base_reward': [],
            'activity_bonus': [],
            'opportunity_cost': [],
            'timing_bonus': [],
            'risk_adjustment': [],
            'position_management': []
        }
        
        print(f"RewardShapedEvalTradeSimulator initialized with config: {self.reward_config.get_summary()}")

    def _step(self, action, _if_random=True):
        # Get the basic step results from parent class
        old_cash = self.cash
        old_asset = self.asset
        old_position = self.position
        
        # Call parent step to get base reward and update state
        state, base_reward, done, info_dict = super()._step(action, _if_random)
        
        # Current state after step
        new_cash = self.cash
        new_asset = self.asset
        new_position = self.position
        current_price = self.price_ary[self.step_i, 2].to(self.device)
        
        # Update tracking variables
        self.price_history.append(current_price.cpu().item())
        if len(self.price_history) > 10:  # Keep only recent history
            self.price_history.pop(0)
            
        action_squeezed = action.squeeze(1).to(self.device)
        action_int = action_squeezed - 1  # Convert to {-1, 0, 1}
        self.action_history.append(action_int.cpu().numpy())
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # Update position hold time
        position_changed = (new_position != old_position).any()
        if position_changed:
            self.position_hold_time.fill_(0)
        else:
            self.position_hold_time += 1
        
        # Track total trades
        trade_mask = action_int.ne(0)
        self.total_trades += trade_mask.int()
        
        # Calculate reward shaping components
        shaped_reward = self._calculate_shaped_reward(
            base_reward, action_int, old_position, new_position, 
            old_cash, new_cash, current_price
        )
        
        return state, shaped_reward, done, info_dict
    
    def _calculate_shaped_reward(self, base_reward, action_int, old_position, new_position, 
                                old_cash, new_cash, current_price):
        """Calculate the full shaped reward with multiple components"""
        config = self.reward_config
        
        # Component 1: Base reward (profit/loss)
        base_component = base_reward * config.base_reward_weight
        
        # Component 2: Activity bonus/penalty
        activity_component = self._calculate_activity_reward(action_int)
        
        # Component 3: Opportunity cost
        opportunity_component = self._calculate_opportunity_cost(
            action_int, old_position, current_price
        )
        
        # Component 4: Market timing bonus
        timing_component = self._calculate_timing_bonus(
            action_int, current_price
        )
        
        # Component 5: Risk adjustment
        risk_component = self._calculate_risk_adjustment(
            base_reward, action_int
        )
        
        # Component 6: Position management
        position_component = self._calculate_position_management_reward(
            old_position, new_position, new_cash
        )
        
        # Combine all components
        total_reward = (base_component + activity_component + opportunity_component + 
                       timing_component + risk_component + position_component)
        
        # Log components for analysis
        self.reward_breakdown['base_reward'].append(base_component.cpu().mean().item())
        self.reward_breakdown['activity_bonus'].append(activity_component.cpu().mean().item())
        self.reward_breakdown['opportunity_cost'].append(opportunity_component.cpu().mean().item())
        self.reward_breakdown['timing_bonus'].append(timing_component.cpu().mean().item())
        self.reward_breakdown['risk_adjustment'].append(risk_component.cpu().mean().item())
        self.reward_breakdown['position_management'].append(position_component.cpu().mean().item())
        
        return total_reward
    
    def _calculate_activity_reward(self, action_int):
        """Reward trading activity, penalize excessive holding"""
        config = self.reward_config
        
        # Bonus for making trades
        trade_mask = action_int.ne(0)
        activity_reward = th.where(
            trade_mask, 
            th.tensor(config.trade_bonus, device=self.device),
            th.tensor(config.hold_penalty, device=self.device)
        )
        
        return activity_reward * config.activity_bonus_weight
    
    def _calculate_opportunity_cost(self, action_int, position, current_price):
        """Penalize missed opportunities based on recent price momentum"""
        config = self.reward_config
        
        if len(self.price_history) < config.momentum_lookback:
            return th.zeros_like(action_int, dtype=th.float32)
        
        # Calculate recent price momentum
        recent_prices = self.price_history[-config.momentum_lookback:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        opportunity_cost = th.zeros_like(action_int, dtype=th.float32)
        
        # Handle single environment case (flatten tensors)
        if action_int.dim() > 0 and action_int.shape[0] == 1:
            action_int_scalar = action_int[0]
            position_scalar = position[0] if position.dim() > 0 else position
            
            # Penalize holding cash during uptrends
            if price_change > 0.001:  # Price increasing
                if position_scalar.eq(0):  # No BTC position
                    opportunity_cost[0] -= config.cash_opportunity_penalty * price_change
            
            # Penalize holding BTC during downtrends  
            elif price_change < -0.001:  # Price decreasing
                if position_scalar.gt(0):  # Has BTC position
                    opportunity_cost[0] -= config.btc_opportunity_penalty * abs(price_change)
        else:
            # Handle vectorized case
            # Penalize holding cash during uptrends
            if price_change > 0.001:  # Price increasing
                cash_holders = position.eq(0)  # Those with no BTC position
                opportunity_cost[cash_holders] -= config.cash_opportunity_penalty * price_change
            
            # Penalize holding BTC during downtrends  
            elif price_change < -0.001:  # Price decreasing
                btc_holders = position.gt(0)  # Those with BTC position
                opportunity_cost[btc_holders] -= config.btc_opportunity_penalty * abs(price_change)
        
        return opportunity_cost * config.opportunity_cost_weight
    
    def _calculate_timing_bonus(self, action_int, current_price):
        """Reward good market timing based on short-term price movements"""
        config = self.reward_config
        
        if len(self.price_history) < config.timing_lookback + 1:
            return th.zeros_like(action_int, dtype=th.float32)
        
        # Look at recent price trend to evaluate timing
        recent_trend = 0
        if len(self.price_history) >= 2:
            recent_trend = self.price_history[-1] - self.price_history[-2]
        
        timing_bonus = th.zeros_like(action_int, dtype=th.float32)
        
        # Reward buying before price increases
        buy_mask = action_int.gt(0)
        if recent_trend > 0:  # Price just increased
            timing_bonus[buy_mask] += config.good_timing_bonus
        
        # Reward selling before price decreases
        sell_mask = action_int.lt(0)
        if recent_trend < 0:  # Price just decreased
            timing_bonus[sell_mask] += config.good_timing_bonus
        
        return timing_bonus * config.timing_bonus_weight
    
    def _calculate_risk_adjustment(self, base_reward, action_int):
        """Adjust rewards based on risk characteristics"""
        config = self.reward_config
        
        # Penalize very volatile trades (large absolute rewards)
        volatility_penalty = th.abs(base_reward) * config.volatility_penalty
        risk_adjustment = -volatility_penalty
        
        return risk_adjustment * config.risk_adjustment_weight
    
    def _calculate_position_management_reward(self, old_position, new_position, cash):
        """Reward good position management practices"""
        config = self.reward_config
        
        position_reward = th.zeros_like(old_position, dtype=th.float32)
        
        # Penalize holding positions too long without taking profits
        long_hold_mask = self.position_hold_time.gt(config.max_hold_penalty)
        position_reward[long_hold_mask] -= config.max_hold_penalty * 0.01
        
        # Penalize overconcentration (having all money in one asset)
        total_value = cash + new_position * self.price_ary[self.step_i, 2].to(self.device)
        cash_ratio = cash / total_value
        overconcentrated = th.logical_or(cash_ratio.lt(0.1), cash_ratio.gt(0.9))
        position_reward[overconcentrated] -= config.overconcentration_penalty
        
        return position_reward * config.position_management_weight
    
    def get_reward_analysis(self):
        """Return analysis of reward components"""
        if not self.reward_breakdown['base_reward']:
            return "No reward data available yet"
        
        analysis = {}
        for component, values in self.reward_breakdown.items():
            if values:
                analysis[component] = {
                    'mean': np.mean(values),
                    'sum': np.sum(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        return analysis
    
    def print_reward_summary(self):
        """Print a summary of reward components"""
        analysis = self.get_reward_analysis()
        if isinstance(analysis, str):
            print(analysis)
            return
        
        print("\n=== REWARD SHAPING ANALYSIS ===")
        total_reward = sum(comp['sum'] for comp in analysis.values())
        
        for component, stats in analysis.items():
            contribution_pct = (stats['sum'] / total_reward * 100) if total_reward != 0 else 0
            print(f"{component:20s}: Sum={stats['sum']:8.4f} ({contribution_pct:5.1f}%) "
                  f"Mean={stats['mean']:7.4f} Std={stats['std']:7.4f}")
        
        print(f"{'Total Reward':20s}: {total_reward:8.4f}")
        print(f"{'Average per step':20s}: {total_reward/analysis['base_reward']['count']:8.4f}")