"""
Reward-Shaped Training Simulator

A training environment that implements comprehensive reward shaping to encourage
balanced profit maximization and trading activity during agent training.
"""

import torch as th
import numpy as np
from trade_simulator import TradeSimulator
from training_reward_config import TrainingRewardConfig


class RewardShapedTrainingSimulator(TradeSimulator):
    def __init__(self, reward_config=None, training_step_tracker=None, **kwargs):
        # Extract reward_config from kwargs if passed through env_args
        if 'reward_config' in kwargs:
            reward_config = kwargs.pop('reward_config')
        if 'training_step_tracker' in kwargs:
            training_step_tracker = kwargs.pop('training_step_tracker')
        
        super().__init__(**kwargs)
        self.reward_config = reward_config or TrainingRewardConfig.balanced_training_config()
        
        # Training step tracker for curriculum learning
        self.training_step_tracker = training_step_tracker or {'step': 0}
        
        # Enhanced tracking for reward shaping
        self.price_history = []
        self.action_history = []
        self.position_hold_time = th.zeros((self.num_sims,), dtype=th.int32, device=self.device)
        self.last_trade_step = th.zeros((self.num_sims,), dtype=th.int32, device=self.device)
        self.episode_drawdown = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        self.episode_peak = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        
        # Action diversity tracking for exploration bonus
        self.recent_actions = []
        self.action_counts = th.zeros((self.num_sims, 3), dtype=th.int32, device=self.device)  # [sell, hold, buy]
        
        # Reward component tracking
        self.reward_components = {
            'base_reward': [],
            'activity_bonus': [],
            'opportunity_cost': [],
            'timing_bonus': [],
            'risk_adjustment': [],
            'position_management': [],
            'diversity_bonus': []
        }
        
        print(f"RewardShapedTrainingSimulator initialized with {self.num_sims} environments")
        if hasattr(self.reward_config, 'print_training_config'):
            self.reward_config.print_training_config()

    def _step(self, action, _if_random=True):
        # Call parent step to get base results
        state, base_reward, done, info_dict = super()._step(action, _if_random)
        
        # Update tracking variables
        self._update_tracking_variables(action, base_reward)
        
        # Calculate comprehensive shaped reward
        shaped_reward = self._calculate_comprehensive_shaped_reward(base_reward, action)
        
        # Update training step for curriculum learning
        self.training_step_tracker['step'] += 1
        
        return state, shaped_reward, done, info_dict
    
    def _update_tracking_variables(self, action, base_reward):
        """Update all tracking variables needed for reward shaping"""
        # Update price history
        current_price = self.price_ary[self.step_i, 2]
        self.price_history.append(current_price.cpu().mean().item())
        if len(self.price_history) > 20:  # Keep reasonable history
            self.price_history.pop(0)
        
        # Track actions
        action_squeezed = action.squeeze(1).to(self.device)
        action_int = action_squeezed - 1  # Convert to {-1, 0, 1}
        self.action_history.append(action_int.cpu())
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # Update action counts for diversity tracking
        for i in range(self.num_sims):
            if i < len(action_int):  # Ensure we don't go out of bounds
                action_idx = action_int[i] + 1  # +1 to convert {-1,0,1} to {0,1,2}
                if 0 <= action_idx <= 2:  # Ensure valid action index
                    self.action_counts[i, action_idx] += 1
        
        # Update position hold time
        position_changed = (action_int != 0)
        self.position_hold_time[position_changed] = 0
        self.position_hold_time[~position_changed] += 1
        
        # Update last trade step
        self.last_trade_step[position_changed] = 0
        self.last_trade_step[~position_changed] += 1
        
        # Update drawdown tracking
        current_asset = self.asset
        self.episode_peak = th.maximum(self.episode_peak, current_asset)
        self.episode_drawdown = (self.episode_peak - current_asset) / self.episode_peak
        
    def _calculate_comprehensive_shaped_reward(self, base_reward, action):
        """Calculate the full shaped reward with all components"""
        config = self.reward_config
        training_step = self.training_step_tracker['step']
        
        # Get curriculum multipliers if enabled
        if config.curriculum_learning:
            activity_mult, opportunity_mult, timing_mult = config.get_curriculum_multipliers(training_step)
        else:
            activity_mult = opportunity_mult = timing_mult = 1.0
        
        # Component 1: Base reward (profit/loss) - always primary
        base_component = base_reward * config.base_reward_weight
        
        # Component 2: Activity incentives
        activity_component = self._calculate_activity_rewards(action) * activity_mult
        
        # Component 3: Opportunity cost penalties
        opportunity_component = self._calculate_opportunity_costs(action) * opportunity_mult
        
        # Component 4: Market timing rewards
        timing_component = self._calculate_timing_rewards(action) * timing_mult
        
        # Component 5: Risk management adjustments
        risk_component = self._calculate_risk_adjustments(base_reward, action)
        
        # Component 6: Position management rewards
        position_component = self._calculate_position_management_rewards()
        
        # Component 7: Diversity and exploration bonuses
        diversity_component = self._calculate_diversity_bonuses(action)
        
        # Combine all components
        total_reward = (base_component + activity_component + opportunity_component + 
                       timing_component + risk_component + position_component + diversity_component)
        
        # Log components for analysis (sample every 100 steps to avoid memory issues)
        if training_step % 100 == 0:
            self._log_reward_components(base_component, activity_component, opportunity_component,
                                      timing_component, risk_component, position_component, diversity_component)
        
        return total_reward
    
    def _calculate_activity_rewards(self, action):
        """Calculate rewards for trading activity"""
        config = self.reward_config
        action_squeezed = action.squeeze(1).to(self.device)
        action_int = action_squeezed - 1
        
        # Base activity rewards
        trade_mask = action_int.ne(0)
        activity_reward = th.where(trade_mask, 
                                 th.tensor(config.trade_bonus, device=self.device),
                                 th.tensor(config.hold_penalty, device=self.device))
        
        # Excessive hold penalty
        excessive_hold_mask = self.position_hold_time > config.excessive_hold_threshold
        activity_reward[excessive_hold_mask] += config.excessive_hold_penalty
        
        return activity_reward * config.activity_bonus_weight
    
    def _calculate_opportunity_costs(self, action):
        """Calculate opportunity cost penalties for missing market movements"""
        config = self.reward_config
        
        if len(self.price_history) < config.momentum_lookback:
            return th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        
        # Calculate price momentum
        recent_prices = self.price_history[-config.momentum_lookback:]
        price_change_pct = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        opportunity_cost = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        
        # Only apply penalties for significant movements
        if abs(price_change_pct) > config.min_movement_threshold:
            # Penalty for not participating in trends
            inactive_mask = self.last_trade_step > 20  # Haven't traded recently
            
            if price_change_pct > 0:  # Uptrend - penalize not buying
                cash_heavy_mask = self.position.le(0)  # No long position
                penalty_mask = inactive_mask & cash_heavy_mask
                opportunity_cost[penalty_mask] -= config.cash_opportunity_penalty * price_change_pct
                
            else:  # Downtrend - penalize not selling
                btc_heavy_mask = self.position.gt(0)  # Has long position
                penalty_mask = inactive_mask & btc_heavy_mask
                opportunity_cost[penalty_mask] -= config.btc_opportunity_penalty * abs(price_change_pct)
        
        return opportunity_cost * config.opportunity_cost_weight
    
    def _calculate_timing_rewards(self, action):
        """Calculate rewards for good market timing"""
        config = self.reward_config
        
        if len(self.price_history) < config.timing_lookback + 1:
            return th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        
        # Calculate recent price momentum
        lookback_change = (self.price_history[-1] - self.price_history[-config.timing_lookback]) / self.price_history[-config.timing_lookback]
        
        action_squeezed = action.squeeze(1).to(self.device)
        action_int = action_squeezed - 1
        
        timing_reward = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        
        # Reward buying before/during uptrends
        buy_mask = action_int.gt(0)
        if lookback_change > 0:
            timing_reward[buy_mask] += config.good_timing_bonus * lookback_change
        else:
            timing_reward[buy_mask] += config.bad_timing_penalty * abs(lookback_change)
        
        # Reward selling before/during downtrends
        sell_mask = action_int.lt(0)
        if lookback_change < 0:
            timing_reward[sell_mask] += config.good_timing_bonus * abs(lookback_change)
        else:
            timing_reward[sell_mask] += config.bad_timing_penalty * lookback_change
        
        return timing_reward * config.timing_bonus_weight
    
    def _calculate_risk_adjustments(self, base_reward, action):
        """Calculate risk-based reward adjustments"""
        config = self.reward_config
        
        # Volatility penalty for very large rewards/losses
        volatility_penalty = th.abs(base_reward) * config.volatility_penalty
        
        # Drawdown penalty
        drawdown_penalty = th.zeros_like(base_reward)
        large_drawdown_mask = self.episode_drawdown > config.drawdown_threshold
        drawdown_penalty[large_drawdown_mask] = -config.max_drawdown_penalty * self.episode_drawdown[large_drawdown_mask]
        
        risk_adjustment = -volatility_penalty + drawdown_penalty
        return risk_adjustment * config.risk_adjustment_weight
    
    def _calculate_position_management_rewards(self):
        """Calculate rewards for good position management"""
        config = self.reward_config
        
        position_reward = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        
        # Penalty for excessive holding without reason
        long_hold_mask = self.position_hold_time > config.max_hold_penalty
        position_reward[long_hold_mask] -= config.max_hold_penalty * 0.01
        
        # Penalty for overconcentration
        # Note: This is simplified - in a real implementation, you'd calculate actual portfolio weights
        extreme_position_mask = th.abs(self.position) >= self.max_position
        position_reward[extreme_position_mask] -= config.overconcentration_penalty
        
        # Bonus for balanced position management (simplified)
        balanced_mask = (th.abs(self.position) > 0) & (th.abs(self.position) < self.max_position)
        position_reward[balanced_mask] += config.diversification_bonus
        
        return position_reward * config.position_management_weight
    
    def _calculate_diversity_bonuses(self, action):
        """Calculate bonuses for action diversity and exploration"""
        config = self.reward_config
        
        diversity_bonus = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        
        # Action diversity bonus
        if len(self.action_history) >= 5:
            for i in range(self.num_sims):
                try:
                    # Check if we have valid action history for this environment
                    recent_actions = []
                    for act in self.action_history[-5:]:
                        if i < len(act):
                            recent_actions.append(act[i].item())
                    
                    if len(recent_actions) >= 3:  # Need at least 3 actions to judge diversity
                        unique_actions = len(set(recent_actions))
                        if unique_actions >= 2:  # Used at least 2 different actions recently
                            diversity_bonus[i] += config.action_diversity_bonus
                except (IndexError, RuntimeError):
                    # Skip this environment if there's an indexing issue
                    continue
        
        return diversity_bonus * config.diversity_bonus_weight
    
    def _log_reward_components(self, base, activity, opportunity, timing, risk, position, diversity):
        """Log reward components for analysis"""
        self.reward_components['base_reward'].append(base.cpu().mean().item())
        self.reward_components['activity_bonus'].append(activity.cpu().mean().item())
        self.reward_components['opportunity_cost'].append(opportunity.cpu().mean().item())
        self.reward_components['timing_bonus'].append(timing.cpu().mean().item())
        self.reward_components['risk_adjustment'].append(risk.cpu().mean().item())
        self.reward_components['position_management'].append(position.cpu().mean().item())
        self.reward_components['diversity_bonus'].append(diversity.cpu().mean().item())
        
        # Keep only recent history to avoid memory issues
        max_history = 1000
        for key in self.reward_components:
            if len(self.reward_components[key]) > max_history:
                self.reward_components[key] = self.reward_components[key][-max_history:]
    
    def get_training_reward_analysis(self):
        """Get comprehensive analysis of reward components during training"""
        if not self.reward_components['base_reward']:
            return "No training reward data available yet"
        
        analysis = {}
        for component, values in self.reward_components.items():
            if values:
                analysis[component] = {
                    'mean': np.mean(values),
                    'sum': np.sum(values),
                    'std': np.std(values),
                    'count': len(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Add training-specific metrics
        training_step = self.training_step_tracker['step']
        if self.reward_config.curriculum_learning:
            activity_mult, opportunity_mult, timing_mult = self.reward_config.get_curriculum_multipliers(training_step)
            analysis['curriculum_multipliers'] = {
                'activity': activity_mult,
                'opportunity': opportunity_mult,
                'timing': timing_mult,
                'training_step': training_step
            }
        
        return analysis
    
    def print_training_reward_summary(self):
        """Print comprehensive training reward analysis"""
        analysis = self.get_training_reward_analysis()
        if isinstance(analysis, str):
            print(analysis)
            return
        
        print("\n=== TRAINING REWARD SHAPING ANALYSIS ===")
        total_reward = sum(comp['sum'] for comp in analysis.values() if isinstance(comp, dict) and 'sum' in comp)
        
        print("Component Analysis:")
        for component, stats in analysis.items():
            if isinstance(stats, dict) and 'sum' in stats:
                contribution_pct = (stats['sum'] / total_reward * 100) if total_reward != 0 else 0
                print(f"  {component:18s}: Sum={stats['sum']:8.2f} ({contribution_pct:5.1f}%) "
                      f"Mean={stats['mean']:7.4f} Range=[{stats['min']:6.3f}, {stats['max']:6.3f}]")
        
        if 'curriculum_multipliers' in analysis:
            mult = analysis['curriculum_multipliers']
            print(f"\nCurriculum Learning (Step {mult['training_step']}):")
            print(f"  Activity multiplier: {mult['activity']:.3f}")
            print(f"  Opportunity multiplier: {mult['opportunity']:.3f}")
            print(f"  Timing multiplier: {mult['timing']:.3f}")
        
        print(f"\nTotal Reward Sum: {total_reward:.2f}")
        print(f"Average per logged step: {total_reward/len(analysis['base_reward']):.4f}")
        print("="*55)