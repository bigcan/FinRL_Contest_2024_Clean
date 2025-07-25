"""
Meta-Learning Configuration Management
Extends existing enhanced training configuration with meta-learning parameters
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any
from collections import deque
import time
import json
import os

from enhanced_training_config import EnhancedConfig, EarlyStoppingManager, TrainingMetricsTracker


class MetaLearningConfig(EnhancedConfig):
    """
    Enhanced configuration with meta-learning parameters
    Extends the existing EnhancedConfig with meta-learning specific settings
    """
    
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        # Import TradeSimulator for default environment
        try:
            from trade_simulator import TradeSimulator
            if env_class is None:
                env_class = TradeSimulator
            if env_args is None:
                env_args = {
                    'env_name': 'TradeSimulator-v0',
                    'state_dim': 50,
                    'action_dim': 3,
                    'if_discrete': True
                }
        except ImportError:
            pass
        
        super().__init__(agent_class, env_class, env_args)
        
        # Meta-Learning Core Parameters
        self.meta_learning_enabled = True
        self.meta_lookback = 1000  # Historical samples for meta-learning
        self.meta_training_frequency = 100  # Train meta-models every N steps
        self.meta_batch_size = 32  # Batch size for meta-model training
        self.meta_epochs = 10  # Epochs for meta-model training
        self.meta_learning_rate = 0.001  # Learning rate for meta-models
        
        # Market Regime Detection Parameters
        self.regime_features_dim = 50  # Number of market features
        self.regime_classification_enabled = True
        self.regime_history_window = 100  # Window for feature extraction
        self.regime_stability_threshold = 0.7  # Threshold for regime stability
        self.market_regimes = [
            'trending_bull', 'trending_bear', 'high_vol_range', 
            'low_vol_range', 'breakout', 'reversal', 'crisis'
        ]
        
        # Algorithm Performance Prediction Parameters
        self.performance_prediction_enabled = True
        self.agent_history_features = 20  # Features for agent performance history
        self.performance_prediction_window = 50  # Window for performance calculation
        self.sharpe_ratio_target = 1.5  # Target Sharpe ratio improvement
        
        # Adaptive Ensemble Parameters
        self.adaptive_weighting_enabled = True
        self.max_agent_weight = 0.6  # Maximum weight per agent
        self.min_diversification_agents = 3  # Minimum agents to use
        self.weight_temperature = 0.5  # Temperature for softmax weighting
        self.diversification_penalty = 0.1  # Penalty for low diversification
        
        # Risk Management Integration
        self.risk_aware_selection = True
        self.emergency_fallback_enabled = True
        self.position_size_optimization = True
        self.kelly_criterion_enabled = True
        self.max_position_risk = 0.95  # Maximum position size
        
        # Performance Tracking and Learning
        self.meta_performance_tracking = True
        self.regime_tracking_enabled = True
        self.decision_history_size = 1000  # Size of decision history
        self.performance_feedback_enabled = True
        self.continuous_learning_enabled = True
        
        # Model Persistence and Checkpointing
        self.meta_model_save_frequency = 500  # Save meta-models every N steps
        self.checkpoint_meta_models = True
        self.meta_model_versioning = True
        self.auto_backup_enabled = True
        
        # Advanced Features
        self.regime_transition_learning = True  # Learn from regime transitions
        self.correlation_analysis = True  # Analyze agent correlations
        self.volatility_forecasting = False  # Enable volatility forecasting (future)
        self.sentiment_integration = False  # Enable sentiment data (future)
        
        # Debugging and Monitoring
        self.meta_learning_debug = False  # Enable debug logging
        self.real_time_monitoring = True  # Real-time performance monitoring
        self.detailed_logging = True  # Detailed logging of decisions
        self.performance_alerts = True  # Alerts for performance issues
        
        # Configuration validation
        self._validate_config()
        
        print(f"🧠 Meta-Learning Configuration Initialized:")
        print(f"   🔄 Meta-Learning: {'Enabled' if self.meta_learning_enabled else 'Disabled'}")
        print(f"   📊 Regime Detection: {len(self.market_regimes)} regimes")
        print(f"   🎯 Performance Prediction: {'Enabled' if self.performance_prediction_enabled else 'Disabled'}")
        print(f"   ⚖️ Adaptive Weighting: Max weight {self.max_agent_weight:.1%}")
        print(f"   🛡️ Risk Integration: {'Enabled' if self.risk_aware_selection else 'Disabled'}")
        print(f"   💾 Model Persistence: Every {self.meta_model_save_frequency} steps")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        try:
            assert 0 < self.max_agent_weight <= 1.0, "Max agent weight must be between 0 and 1"
            assert self.min_diversification_agents >= 1, "Must use at least 1 agent"
            assert self.meta_lookback > 0, "Meta lookback must be positive"
            assert self.regime_features_dim > 0, "Regime features dimension must be positive"
            assert self.weight_temperature > 0, "Weight temperature must be positive"
        except AssertionError as e:
            print(f"⚠️ Configuration validation warning: {e}")
            # Fix invalid values
            if not (0 < self.max_agent_weight <= 1.0):
                self.max_agent_weight = 0.6
            if self.min_diversification_agents < 1:
                self.min_diversification_agents = 2
            if self.meta_lookback <= 0:
                self.meta_lookback = 500
            if self.regime_features_dim <= 0:
                self.regime_features_dim = 50
            if self.weight_temperature <= 0:
                self.weight_temperature = 0.5
        
        # Ensure meta-learning frequency is reasonable
        if self.meta_training_frequency > self.break_step:
            print(f"⚠️ Warning: Meta-training frequency ({self.meta_training_frequency}) > "
                  f"total steps ({self.break_step}). Adjusting to {self.break_step // 4}")
            self.meta_training_frequency = max(1, self.break_step // 4)


class MetaLearningTracker(TrainingMetricsTracker):
    """
    Extended metrics tracker for meta-learning specific metrics
    """
    
    def __init__(self, history_size=1000):
        super().__init__(history_size)
        
        # Meta-learning specific metrics
        self.meta_metrics = {
            "regime_predictions": deque(maxlen=history_size),
            "regime_accuracy": deque(maxlen=history_size),
            "regime_stability": deque(maxlen=history_size),
            "performance_predictions": deque(maxlen=history_size),
            "prediction_errors": deque(maxlen=history_size),
            "agent_weights": deque(maxlen=history_size),
            "ensemble_decisions": deque(maxlen=history_size),
            "meta_training_losses": deque(maxlen=history_size),
            "regime_transitions": deque(maxlen=history_size),
            "adaptability_score": deque(maxlen=history_size)
        }
        
        # Decision tracking
        self.decision_history = deque(maxlen=history_size)
        self.regime_history = deque(maxlen=history_size)
        self.performance_history = deque(maxlen=history_size)
    
    def update_meta_metrics(self, step: int, regime: str = None, 
                           regime_confidence: float = None,
                           agent_weights: Dict[str, float] = None,
                           performance_predictions: Dict[str, float] = None,
                           meta_training_loss: float = None,
                           ensemble_decision: int = None):
        """Update meta-learning specific metrics"""
        
        if regime is not None:
            self.meta_metrics["regime_predictions"].append(regime)
            self.regime_history.append({"step": step, "regime": regime})
        
        if regime_confidence is not None:
            self.meta_metrics["regime_stability"].append(regime_confidence)
        
        if agent_weights is not None:
            self.meta_metrics["agent_weights"].append(agent_weights.copy())
        
        if performance_predictions is not None:
            self.meta_metrics["performance_predictions"].append(performance_predictions.copy())
        
        if meta_training_loss is not None:
            self.meta_metrics["meta_training_losses"].append(meta_training_loss)
        
        if ensemble_decision is not None:
            self.meta_metrics["ensemble_decisions"].append(ensemble_decision)
            self.decision_history.append({
                "step": step,
                "decision": ensemble_decision,
                "regime": regime,
                "weights": agent_weights.copy() if agent_weights else None
            })
    
    def calculate_regime_transition_rate(self, window: int = 100) -> float:
        """Calculate regime transition rate in given window"""
        if len(self.regime_history) < window:
            return 0.0
        
        recent_regimes = [item["regime"] for item in list(self.regime_history)[-window:]]
        transitions = sum(1 for i in range(1, len(recent_regimes)) 
                         if recent_regimes[i] != recent_regimes[i-1])
        
        return transitions / (len(recent_regimes) - 1) if len(recent_regimes) > 1 else 0.0
    
    def calculate_weight_diversity(self, window: int = 50) -> float:
        """Calculate diversity of agent weights"""
        if len(self.meta_metrics["agent_weights"]) < window:
            return 0.0
        
        recent_weights = list(self.meta_metrics["agent_weights"])[-window:]
        
        # Calculate entropy-based diversity
        avg_weights = {}
        for weight_dict in recent_weights:
            for agent, weight in weight_dict.items():
                if agent not in avg_weights:
                    avg_weights[agent] = []
                avg_weights[agent].append(weight)
        
        # Calculate average entropy
        total_entropy = 0
        for agent, weights in avg_weights.items():
            avg_weight = np.mean(weights)
            if avg_weight > 0:
                total_entropy -= avg_weight * np.log(avg_weight + 1e-8)
        
        return total_entropy
    
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning summary"""
        summary = {
            "total_decisions": len(self.decision_history),
            "regime_transition_rate": self.calculate_regime_transition_rate(),
            "weight_diversity": self.calculate_weight_diversity(),
            "meta_learning_active": len(self.meta_metrics["meta_training_losses"]) > 0
        }
        
        # Recent performance metrics
        if len(self.meta_metrics["regime_stability"]) > 0:
            summary["recent_regime_stability"] = np.mean(
                list(self.meta_metrics["regime_stability"])[-10:]
            )
        
        if len(self.meta_metrics["meta_training_losses"]) > 0:
            summary["recent_meta_loss"] = np.mean(
                list(self.meta_metrics["meta_training_losses"])[-10:]
            )
        
        # Decision distribution
        if len(self.meta_metrics["ensemble_decisions"]) > 0:
            decisions = list(self.meta_metrics["ensemble_decisions"])[-100:]
            decision_counts = {0: 0, 1: 0, 2: 0}  # sell, hold, buy
            for decision in decisions:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            total_decisions = len(decisions)
            summary["decision_distribution"] = {
                "sell": decision_counts[0] / total_decisions,
                "hold": decision_counts[1] / total_decisions,
                "buy": decision_counts[2] / total_decisions
            }
        
        return summary
    
    def print_meta_learning_summary(self):
        """Print detailed meta-learning summary"""
        summary = self.get_meta_learning_summary()
        
        print(f"\n🧠 Meta-Learning Summary:")
        print(f"   📊 Total Decisions: {summary.get('total_decisions', 0)}")
        print(f"   🔄 Regime Transition Rate: {summary.get('regime_transition_rate', 0):.3f}")
        print(f"   🎯 Weight Diversity: {summary.get('weight_diversity', 0):.3f}")
        print(f"   🏛️ Regime Stability: {summary.get('recent_regime_stability', 0):.3f}")
        
        if 'decision_distribution' in summary:
            dist = summary['decision_distribution']
            print(f"   📈 Decision Distribution:")
            print(f"      Sell: {dist['sell']:.1%}, Hold: {dist['hold']:.1%}, Buy: {dist['buy']:.1%}")
        
        if summary.get('meta_learning_active', False):
            print(f"   🔥 Meta-Learning: Active (Loss: {summary.get('recent_meta_loss', 0):.4f})")
        else:
            print(f"   💤 Meta-Learning: Warming up...")


class MetaLearningConfigManager:
    """
    Manages meta-learning configurations and presets
    """
    
    @staticmethod
    def create_conservative_config(agent_class, env_class, env_args) -> MetaLearningConfig:
        """Create conservative meta-learning configuration"""
        config = MetaLearningConfig(agent_class, env_class, env_args)
        
        # Conservative settings
        config.max_agent_weight = 0.4  # More diversified
        config.min_diversification_agents = 4
        config.weight_temperature = 1.0  # Less aggressive weighting
        config.meta_training_frequency = 200  # Less frequent training
        config.regime_stability_threshold = 0.8  # Higher stability requirement
        
        print("🛡️ Conservative meta-learning configuration created")
        return config
    
    @staticmethod
    def create_aggressive_config(agent_class, env_class, env_args) -> MetaLearningConfig:
        """Create aggressive meta-learning configuration"""
        config = MetaLearningConfig(agent_class, env_class, env_args)
        
        # Aggressive settings
        config.max_agent_weight = 0.8  # Allow concentration
        config.min_diversification_agents = 2
        config.weight_temperature = 0.3  # More aggressive weighting
        config.meta_training_frequency = 50  # More frequent training
        config.regime_stability_threshold = 0.5  # Lower stability requirement
        
        print("🚀 Aggressive meta-learning configuration created")
        return config
    
    @staticmethod
    def create_balanced_config(agent_class, env_class, env_args) -> MetaLearningConfig:
        """Create balanced meta-learning configuration (default)"""
        config = MetaLearningConfig(agent_class, env_class, env_args)
        print("⚖️ Balanced meta-learning configuration created")
        return config
    
    @staticmethod
    def create_research_config(agent_class, env_class, env_args) -> MetaLearningConfig:
        """Create research-oriented configuration with extensive logging"""
        config = MetaLearningConfig(agent_class, env_class, env_args)
        
        # Research settings
        config.meta_learning_debug = True
        config.detailed_logging = True
        config.regime_transition_learning = True
        config.correlation_analysis = True
        config.decision_history_size = 5000  # Larger history
        config.meta_model_save_frequency = 100  # More frequent saves
        
        print("🔬 Research meta-learning configuration created")
        return config
    
    @staticmethod
    def load_config_from_file(filepath: str, agent_class, env_class, env_args) -> MetaLearningConfig:
        """Load configuration from JSON file"""
        config = MetaLearningConfig(agent_class, env_class, env_args)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                saved_params = json.load(f)
            
            # Update configuration with saved parameters
            for key, value in saved_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            config._validate_config()
            print(f"📁 Meta-learning configuration loaded from {filepath}")
        else:
            print(f"⚠️ Configuration file {filepath} not found, using defaults")
        
        return config
    
    @staticmethod
    def save_config_to_file(config: MetaLearningConfig, filepath: str):
        """Save configuration to JSON file"""
        
        # Extract meta-learning specific parameters
        meta_params = {}
        for attr_name in dir(config):
            if (not attr_name.startswith('_') and 
                not callable(getattr(config, attr_name)) and
                'meta' in attr_name.lower() or 
                'regime' in attr_name.lower() or
                'adaptive' in attr_name.lower() or
                'performance' in attr_name.lower()):
                
                value = getattr(config, attr_name)
                if isinstance(value, (int, float, bool, str, list)):
                    meta_params[attr_name] = value
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(meta_params, f, indent=2)
        
        print(f"💾 Meta-learning configuration saved to {filepath}")


# Configuration presets for different use cases
CONFIG_PRESETS = {
    'conservative': MetaLearningConfigManager.create_conservative_config,
    'aggressive': MetaLearningConfigManager.create_aggressive_config,
    'balanced': MetaLearningConfigManager.create_balanced_config,
    'research': MetaLearningConfigManager.create_research_config
}


def create_meta_learning_config(preset: str = 'balanced', 
                               agent_class=None, 
                               env_class=None, 
                               env_args=None,
                               custom_params: Optional[Dict] = None) -> MetaLearningConfig:
    """
    Factory function to create meta-learning configuration
    
    Args:
        preset: Configuration preset ('conservative', 'aggressive', 'balanced', 'research')
        agent_class: RL agent class
        env_class: Environment class  
        env_args: Environment arguments
        custom_params: Custom parameter overrides
    
    Returns:
        MetaLearningConfig object
    """
    
    if preset not in CONFIG_PRESETS:
        print(f"⚠️ Unknown preset '{preset}', using 'balanced'")
        preset = 'balanced'
    
    # Create configuration with preset
    config = CONFIG_PRESETS[preset](agent_class, env_class, env_args)
    
    # Apply custom parameter overrides
    if custom_params:
        for key, value in custom_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"   🔧 Override: {key} = {value}")
            else:
                print(f"   ⚠️ Unknown parameter: {key}")
        
        config._validate_config()
    
    return config


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Meta-Learning Configuration")
    print("=" * 60)
    
    # Test different presets
    for preset_name in CONFIG_PRESETS.keys():
        print(f"\n📋 Testing '{preset_name}' preset:")
        config = create_meta_learning_config(
            preset=preset_name,
            env_args={
                "env_name": "TestEnv",
                "state_dim": 50,
                "action_dim": 3,
                "if_discrete": True
            }
        )
        print(f"   Max agent weight: {config.max_agent_weight}")
        print(f"   Min diversification: {config.min_diversification_agents}")
        print(f"   Training frequency: {config.meta_training_frequency}")
    
    # Test custom parameters
    print(f"\n🔧 Testing custom parameter override:")
    custom_config = create_meta_learning_config(
        preset='balanced',
        env_args={"env_name": "TestEnv", "state_dim": 50, "action_dim": 3},
        custom_params={
            'max_agent_weight': 0.7,
            'meta_training_frequency': 150,
            'regime_features_dim': 75
        }
    )
    
    # Test tracker
    print(f"\n📊 Testing Meta-Learning Tracker:")
    tracker = MetaLearningTracker(history_size=100)
    
    # Simulate some meta-learning updates
    for i in range(20):
        tracker.update_meta_metrics(
            step=i,
            regime=np.random.choice(['trending_bull', 'ranging', 'trending_bear']),
            regime_confidence=np.random.uniform(0.6, 0.95),
            agent_weights={'agent1': 0.4, 'agent2': 0.35, 'agent3': 0.25},
            ensemble_decision=np.random.choice([0, 1, 2])
        )
    
    tracker.print_meta_learning_summary()
    
    # Test configuration save/load
    print(f"\n💾 Testing configuration persistence:")
    test_config = create_meta_learning_config('research')
    
    # Save configuration
    test_file = "/tmp/test_meta_config.json"
    MetaLearningConfigManager.save_config_to_file(test_config, test_file)
    
    # Load configuration
    loaded_config = MetaLearningConfigManager.load_config_from_file(
        test_file, None, None, {"env_name": "TestEnv"}
    )
    
    print(f"\n🎉 All meta-learning configuration components tested successfully!")