"""
Training-Specific Reward Shaping Configuration

Optimized reward shaping specifically designed for training agents that balance
profit maximization with appropriate trading activity.
"""

from reward_shaping_config import RewardShapingConfig
import numpy as np


class TrainingRewardConfig(RewardShapingConfig):
    def __init__(self):
        super().__init__()
        
        # Training-specific settings (more aggressive than evaluation)
        # Base reward weight remains primary
        self.base_reward_weight = 1.0
        
        # Activity incentives - stronger for training to overcome exploration
        self.activity_bonus_weight = 0.05  # Higher than evaluation (was 0.01)
        self.trade_bonus = 1.0  # Reward per successful trade
        self.hold_penalty = -0.05  # Small penalty for holding
        self.excessive_hold_penalty = -0.2  # Larger penalty for long holds
        self.excessive_hold_threshold = 100  # Steps before excessive penalty
        
        # Opportunity cost - encourage market participation
        self.opportunity_cost_weight = 0.02  # Higher than evaluation (was 0.005)
        self.momentum_lookback = 5  # Look at more history
        self.cash_opportunity_penalty = 2.0  # Stronger penalty for missing uptrends
        self.btc_opportunity_penalty = 2.0  # Stronger penalty for holding during downtrends
        self.min_movement_threshold = 0.001  # Minimum price movement to trigger penalties
        
        # Market timing - reward good decision making
        self.timing_bonus_weight = 0.03  # Much higher than evaluation (was 0.008)
        self.timing_lookback = 3  # Steps to look back for timing
        self.good_timing_bonus = 2.0  # Substantial bonus for good timing
        self.bad_timing_penalty = -1.0  # Penalty for bad timing
        
        # Risk management - prevent reckless trading
        self.risk_adjustment_weight = 0.01  # Higher than evaluation (was 0.003)
        self.volatility_penalty = 0.3  # Penalty for very volatile performance
        self.max_drawdown_penalty = 2.0  # Penalty for large drawdowns
        self.drawdown_threshold = 0.05  # 5% drawdown threshold
        
        # Position management - encourage balanced approach
        self.position_management_weight = 0.008  # Higher than evaluation (was 0.002)
        self.max_hold_penalty = 50  # Shorter hold time before penalty
        self.overconcentration_penalty = 1.0  # Penalty for extreme positions
        self.diversification_bonus = 0.5  # Bonus for balanced positions
        
        # Training curriculum settings
        self.curriculum_learning = True
        self.initial_activity_weight_multiplier = 3.0  # Start with higher activity incentives
        self.final_activity_weight_multiplier = 1.0  # Gradually reduce to normal levels
        self.curriculum_steps = 50000  # Steps over which to reduce activity multiplier
        
        # Experience replay diversity
        self.diversity_bonus_weight = 0.01
        self.action_diversity_bonus = 0.5  # Bonus for using different actions
        self.exploration_bonus = 0.3  # Bonus during exploration phases
        
    def get_curriculum_multipliers(self, training_step):
        """Get curriculum learning multipliers based on training step"""
        if not self.curriculum_learning or training_step >= self.curriculum_steps:
            return 1.0, 1.0, 1.0  # Normal weights
        
        # Linear decay from initial to final multipliers
        progress = training_step / self.curriculum_steps
        activity_mult = (self.initial_activity_weight_multiplier * (1 - progress) + 
                        self.final_activity_weight_multiplier * progress)
        
        # Gradually increase opportunity cost awareness
        opportunity_mult = 0.5 + 0.5 * progress  # Start at 50%, reach 100%
        
        # Timing bonus starts lower, increases as agent learns
        timing_mult = 0.3 + 0.7 * progress  # Start at 30%, reach 100%
        
        return activity_mult, opportunity_mult, timing_mult
    
    @classmethod
    def conservative_training_config(cls):
        """Configuration for conservative training approach"""
        config = cls()
        config.activity_bonus_weight = 0.02
        config.opportunity_cost_weight = 0.01
        config.timing_bonus_weight = 0.015
        config.risk_adjustment_weight = 0.015  # Higher risk adjustment
        config.position_management_weight = 0.01
        return config
    
    @classmethod
    def aggressive_training_config(cls):
        """Configuration for aggressive training approach"""
        config = cls()
        config.activity_bonus_weight = 0.08
        config.opportunity_cost_weight = 0.04
        config.timing_bonus_weight = 0.05
        config.risk_adjustment_weight = 0.005  # Lower risk adjustment
        config.position_management_weight = 0.005
        config.trade_bonus = 2.0
        config.good_timing_bonus = 3.0
        return config
    
    @classmethod
    def ultra_aggressive_training_config(cls):
        """Configuration for ultra-aggressive training approach"""
        config = cls()
        
        # RADICAL: Reduce base reward to let activity dominate
        config.base_reward_weight = 0.1  # Much smaller base reward
        
        # Much higher activity incentives
        config.activity_bonus_weight = 1.0  # Very high activity weight
        config.opportunity_cost_weight = 0.3
        config.timing_bonus_weight = 0.5
        config.risk_adjustment_weight = 0.001  # Minimal risk adjustment
        config.position_management_weight = 0.002
        
        # Stronger individual bonuses
        config.trade_bonus = 10.0  # Extremely high reward per trade
        config.hold_penalty = -0.5  # Strong penalty for holding
        config.excessive_hold_penalty = -2.0  # Very strong penalty for long holds
        config.excessive_hold_threshold = 20  # Even shorter threshold
        
        # More aggressive opportunity costs
        config.cash_opportunity_penalty = 10.0
        config.btc_opportunity_penalty = 10.0
        config.min_movement_threshold = 0.0001  # Very low threshold
        
        # Higher timing bonuses
        config.good_timing_bonus = 8.0
        config.bad_timing_penalty = -3.0
        
        # Curriculum learning with very high initial multiplier
        config.initial_activity_weight_multiplier = 10.0  # Start extremely high
        config.curriculum_steps = 20000  # Shorter curriculum
        
        return config
    
    @classmethod
    def balanced_training_config(cls):
        """Balanced training configuration (default)"""
        return cls()
    
    @classmethod
    def curriculum_config(cls):
        """Configuration with strong curriculum learning"""
        config = cls()
        config.curriculum_learning = True
        config.initial_activity_weight_multiplier = 5.0  # Very high initially
        config.curriculum_steps = 100000  # Longer curriculum
        return config
    
    def get_training_summary(self):
        """Get a summary of training reward configuration"""
        summary = self.get_summary()
        summary.update({
            'curriculum_learning': self.curriculum_learning,
            'initial_activity_multiplier': self.initial_activity_weight_multiplier,
            'curriculum_steps': self.curriculum_steps,
            'excessive_hold_threshold': self.excessive_hold_threshold,
            'min_movement_threshold': self.min_movement_threshold,
            'drawdown_threshold': self.drawdown_threshold
        })
        return summary
    
    def print_training_config(self):
        """Print detailed training configuration"""
        print("=== TRAINING REWARD SHAPING CONFIGURATION ===")
        config = self.get_training_summary()
        
        print("\nCore Weights:")
        print(f"  Base reward weight: {config['base_reward_weight']}")
        print(f"  Activity bonus weight: {config['activity_bonus_weight']}")
        print(f"  Opportunity cost weight: {config['opportunity_cost_weight']}")
        print(f"  Timing bonus weight: {config['timing_bonus_weight']}")
        print(f"  Risk adjustment weight: {config['risk_adjustment_weight']}")
        print(f"  Position management weight: {config['position_management_weight']}")
        
        print("\nTraining Features:")
        print(f"  Curriculum learning: {config['curriculum_learning']}")
        print(f"  Initial activity multiplier: {config['initial_activity_multiplier']}")
        print(f"  Curriculum duration: {config['curriculum_steps']} steps")
        print(f"  Excessive hold threshold: {config['excessive_hold_threshold']} steps")
        print(f"  Min movement threshold: {config['min_movement_threshold']:.1%}")
        print(f"  Drawdown penalty threshold: {config['drawdown_threshold']:.1%}")
        
        total_max_weight = (config['base_reward_weight'] + 
                           config['activity_bonus_weight'] * config['initial_activity_multiplier'] +
                           config['opportunity_cost_weight'] + 
                           config['timing_bonus_weight'] +
                           config['risk_adjustment_weight'] + 
                           config['position_management_weight'])
        
        print(f"\nTotal maximum weight: {total_max_weight:.3f}")
        print(f"Base reward contribution: {config['base_reward_weight']/total_max_weight:.1%}")
        print("="*50)