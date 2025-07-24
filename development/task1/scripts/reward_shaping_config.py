"""
Reward Shaping Configuration for Trading Agents

This module defines configurable reward shaping to encourage more active trading
while maintaining focus on profitability.
"""

class RewardShapingConfig:
    def __init__(self):
        # Base reward weight (profit/loss)
        self.base_reward_weight = 1.0
        
        # Activity incentives
        self.activity_bonus_weight = 0.01  # Small bonus for making trades
        self.trade_bonus = 0.5  # Bonus per trade (scaled by weight)
        self.hold_penalty = -0.1  # Small penalty for holding (scaled by weight)
        
        # Opportunity cost penalties
        self.opportunity_cost_weight = 0.005
        self.momentum_lookback = 3  # Steps to look back for momentum calculation
        self.cash_opportunity_penalty = 1.0  # Penalty for holding cash during uptrends
        self.btc_opportunity_penalty = 1.0  # Penalty for holding BTC during downtrends
        
        # Market timing bonuses
        self.timing_bonus_weight = 0.008
        self.timing_lookback = 2  # Steps to look forward for timing calculation
        self.good_timing_bonus = 1.0  # Bonus for good market timing
        
        # Risk adjustment
        self.risk_adjustment_weight = 0.003
        self.volatility_penalty = 0.5  # Penalty for high volatility trades
        
        # Position management
        self.position_management_weight = 0.002
        self.max_hold_penalty = 50  # Steps before holding penalty increases
        self.overconcentration_penalty = 0.5  # Penalty for putting all money in one asset

    def get_summary(self):
        """Return a summary of current reward shaping settings"""
        return {
            'base_reward_weight': self.base_reward_weight,
            'activity_bonus_weight': self.activity_bonus_weight,
            'opportunity_cost_weight': self.opportunity_cost_weight,
            'timing_bonus_weight': self.timing_bonus_weight,
            'risk_adjustment_weight': self.risk_adjustment_weight,
            'position_management_weight': self.position_management_weight
        }
    
    @classmethod
    def conservative_config(cls):
        """Configuration with minimal reward shaping"""
        config = cls()
        config.activity_bonus_weight = 0.002
        config.opportunity_cost_weight = 0.001
        config.timing_bonus_weight = 0.002
        config.risk_adjustment_weight = 0.001
        config.position_management_weight = 0.001
        return config
    
    @classmethod
    def aggressive_config(cls):
        """Configuration with stronger trading incentives"""
        config = cls()
        config.activity_bonus_weight = 0.02
        config.opportunity_cost_weight = 0.01
        config.timing_bonus_weight = 0.015
        config.risk_adjustment_weight = 0.005
        config.position_management_weight = 0.005
        return config
    
    @classmethod
    def balanced_config(cls):
        """Balanced configuration (default)"""
        return cls()  # Uses default values