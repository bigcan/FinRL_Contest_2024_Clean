"""
Simplified Reward-Shaped Trading Simulator

A simpler version focused on single environment evaluation with proper tensor handling.
"""

import torch as th
import numpy as np
from trade_simulator import EvalTradeSimulator
from reward_shaping_config import RewardShapingConfig


class SimpleRewardShapedEvalTradeSimulator(EvalTradeSimulator):
    def __init__(self, reward_config=None, **kwargs):
        # Extract reward_config from kwargs if passed through env_args
        if 'reward_config' in kwargs:
            reward_config = kwargs.pop('reward_config')
        
        super().__init__(**kwargs)
        self.reward_config = reward_config or RewardShapingConfig.balanced_config()
        
        # Track additional state for reward calculations (simplified for single env)
        self.price_history = []
        self.action_history = []
        self.steps_since_last_trade = 0
        
        # For logging reward components
        self.reward_breakdown = {
            'base_reward': [],
            'activity_bonus': [],
            'opportunity_cost': [],
            'timing_bonus': []
        }
        
        print(f"SimpleRewardShapedEvalTradeSimulator initialized")
        print(f"Reward config: {self.reward_config.get_summary()}")

    def _step(self, action, _if_random=True):
        # Call parent step to get base results
        state, base_reward, done, info_dict = super()._step(action, _if_random)
        
        # Get current price for reward shaping
        current_price = self.price_ary[self.step_i, 2].to(self.device)
        self.price_history.append(current_price.cpu().item())
        if len(self.price_history) > 10:  # Keep recent history
            self.price_history.pop(0)
        
        # Convert action to scalar for easier handling
        action_squeezed = action.squeeze(1).to(self.device)
        action_int = action_squeezed - 1  # Convert to {-1, 0, 1}
        
        # Track if trade was made
        trade_made = action_int.ne(0).any().item()
        if trade_made:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1
        
        # Calculate shaped reward
        shaped_reward = self._calculate_simple_shaped_reward(base_reward, action_int, current_price)
        
        return state, shaped_reward, done, info_dict
    
    def _calculate_simple_shaped_reward(self, base_reward, action_int, current_price):
        """Calculate shaped reward with simplified components"""
        config = self.reward_config
        
        # Component 1: Base reward (unchanged)
        base_component = base_reward * config.base_reward_weight
        
        # Component 2: Activity bonus (encourage trading)
        activity_component = self._get_activity_bonus(action_int)
        
        # Component 3: Opportunity cost (penalize inactivity during trends)
        opportunity_component = self._get_opportunity_penalty()
        
        # Component 4: Timing bonus (reward good market timing)
        timing_component = self._get_timing_bonus(action_int)
        
        # Combine components
        total_reward = base_component + activity_component + opportunity_component + timing_component
        
        # Log for analysis
        self.reward_breakdown['base_reward'].append(base_component.cpu().mean().item())
        self.reward_breakdown['activity_bonus'].append(activity_component.cpu().mean().item())
        self.reward_breakdown['opportunity_cost'].append(opportunity_component.cpu().mean().item())
        self.reward_breakdown['timing_bonus'].append(timing_component.cpu().mean().item())
        
        return total_reward
    
    def _get_activity_bonus(self, action_int):
        """Simple activity bonus: reward trades, small penalty for holds"""
        config = self.reward_config
        
        # Check if any action is non-zero (trade)
        is_trade = action_int.ne(0).any()
        
        if is_trade:
            bonus = th.tensor(config.trade_bonus, device=self.device, dtype=th.float32)
        else:
            # Small penalty for holding, larger if held too long
            hold_penalty = config.hold_penalty
            if self.steps_since_last_trade > 20:  # Long hold penalty
                hold_penalty *= 2
            bonus = th.tensor(hold_penalty, device=self.device, dtype=th.float32)
        
        return bonus * config.activity_bonus_weight
    
    def _get_opportunity_penalty(self):
        """Penalize missing opportunities during price trends"""
        config = self.reward_config
        
        if len(self.price_history) < config.momentum_lookback:
            return th.tensor(0.0, device=self.device, dtype=th.float32)
        
        # Calculate recent momentum
        recent_prices = self.price_history[-config.momentum_lookback:]
        price_change_pct = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        penalty = 0.0
        
        # Penalty for not trading during significant price movements
        if abs(price_change_pct) > 0.002:  # Significant price movement
            if self.steps_since_last_trade > 10:  # Haven't traded recently
                penalty = -abs(price_change_pct) * config.cash_opportunity_penalty
        
        return th.tensor(penalty, device=self.device, dtype=th.float32) * config.opportunity_cost_weight
    
    def _get_timing_bonus(self, action_int):
        """Reward good timing based on recent price movements"""
        config = self.reward_config
        
        if len(self.price_history) < 2:
            return th.tensor(0.0, device=self.device, dtype=th.float32)
        
        # Recent price change
        recent_change = self.price_history[-1] - self.price_history[-2]
        
        bonus = 0.0
        
        # Reward buying before/during price increases
        if action_int.gt(0).any() and recent_change > 0:
            bonus = config.good_timing_bonus * (recent_change / self.price_history[-2])
        
        # Reward selling before/during price decreases  
        elif action_int.lt(0).any() and recent_change < 0:
            bonus = config.good_timing_bonus * abs(recent_change / self.price_history[-2])
        
        return th.tensor(bonus, device=self.device, dtype=th.float32) * config.timing_bonus_weight
    
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
            print(f"{component:15s}: Sum={stats['sum']:8.4f} ({contribution_pct:5.1f}%) "
                  f"Mean={stats['mean']:7.4f} Std={stats['std']:7.4f}")
        
        print(f"{'Total Reward':15s}: {total_reward:8.4f}")
        print(f"{'Steps since trade':15s}: {self.steps_since_last_trade}")
        print(f"{'Price history len':15s}: {len(self.price_history)}")
        if len(self.price_history) >= 2:
            recent_change_pct = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2] * 100
            print(f"{'Recent price chg':15s}: {recent_change_pct:+6.3f}%")