"""
Advanced Market Regime Detection and Integration
Phase 4 of the profitability enhancement plan
"""

import numpy as np
import torch as th
from collections import deque
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enhanced market regime classifications"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    CHOPPY = "choppy"

@dataclass
class RegimeMetrics:
    """Comprehensive metrics for regime detection"""
    trend_strength: float
    volatility: float
    momentum: float
    volume_ratio: float
    price_acceleration: float
    market_efficiency: float
    regime_stability: float
    
class AdvancedMarketRegimeDetector:
    """
    Sophisticated market regime detection with multiple indicators
    and regime-specific trading parameters
    """
    
    def __init__(self, 
                 short_lookback: int = 20,
                 medium_lookback: int = 50, 
                 long_lookback: int = 100,
                 volume_lookback: int = 30,
                 device: str = "cpu"):
        """
        Initialize advanced regime detector
        
        Args:
            short_lookback: Short-term analysis window
            medium_lookback: Medium-term analysis window
            long_lookback: Long-term trend window
            volume_lookback: Volume analysis window
            device: PyTorch device
        """
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.long_lookback = long_lookback
        self.volume_lookback = volume_lookback
        self.device = device
        
        # Price and volume history
        self.price_history = deque(maxlen=long_lookback * 2)
        self.volume_history = deque(maxlen=volume_lookback * 2)
        self.spread_history = deque(maxlen=medium_lookback)
        
        # Regime history for stability tracking
        self.regime_history = deque(maxlen=20)
        self.regime_confidence_history = deque(maxlen=20)
        
        # Technical indicators cache
        self.sma_short = deque(maxlen=medium_lookback)
        self.sma_medium = deque(maxlen=medium_lookback)
        self.sma_long = deque(maxlen=medium_lookback)
        
        # Metrics history
        self.volatility_history = deque(maxlen=medium_lookback)
        self.momentum_history = deque(maxlen=medium_lookback)
        
        logger.info(f"Initialized AdvancedMarketRegimeDetector with lookbacks: {short_lookback}/{medium_lookback}/{long_lookback}")
        
    def update(self, price: float, volume: float = 1.0, spread: float = 0.0):
        """Update detector with new market data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        if spread > 0:
            self.spread_history.append(spread)
            
    def calculate_metrics(self) -> Optional[RegimeMetrics]:
        """Calculate comprehensive market metrics"""
        if len(self.price_history) < self.long_lookback:
            return None
            
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        
        # Price returns
        returns_short = np.diff(prices[-self.short_lookback:]) / prices[-self.short_lookback:-1]
        returns_medium = np.diff(prices[-self.medium_lookback:]) / prices[-self.medium_lookback:-1]
        returns_long = np.diff(prices[-self.long_lookback:]) / prices[-self.long_lookback:-1]
        
        # 1. Trend Strength (using multiple timeframes)
        sma_short = np.mean(prices[-self.short_lookback:])
        sma_medium = np.mean(prices[-self.medium_lookback:])
        sma_long = np.mean(prices[-self.long_lookback:])
        
        self.sma_short.append(sma_short)
        self.sma_medium.append(sma_medium)
        self.sma_long.append(sma_long)
        
        # Trend alignment score
        current_price = prices[-1]
        trend_score = 0.0
        if current_price > sma_short > sma_medium > sma_long:
            trend_score = 1.0  # Perfect uptrend alignment
        elif current_price < sma_short < sma_medium < sma_long:
            trend_score = -1.0  # Perfect downtrend alignment
        else:
            # Partial alignment
            trend_score = (
                0.4 * np.sign(current_price - sma_short) +
                0.3 * np.sign(sma_short - sma_medium) +
                0.3 * np.sign(sma_medium - sma_long)
            )
            
        # ADX-like trend strength
        positive_moves = np.maximum(np.diff(prices[-self.medium_lookback:]), 0)
        negative_moves = np.maximum(-np.diff(prices[-self.medium_lookback:]), 0)
        
        avg_positive = np.mean(positive_moves)
        avg_negative = np.mean(negative_moves)
        
        if avg_positive + avg_negative > 0:
            directional_strength = abs(avg_positive - avg_negative) / (avg_positive + avg_negative)
        else:
            directional_strength = 0
            
        trend_strength = trend_score * directional_strength
        
        # 2. Volatility (multi-timeframe)
        vol_short = np.std(returns_short) * np.sqrt(252 * 86400)  # Annualized
        vol_medium = np.std(returns_medium) * np.sqrt(252 * 86400)
        vol_long = np.std(returns_long) * np.sqrt(252 * 86400)
        
        # Volatility regime change
        volatility = vol_short
        self.volatility_history.append(volatility)
        
        # 3. Momentum (rate of change)
        momentum_short = (prices[-1] - prices[-self.short_lookback]) / prices[-self.short_lookback]
        momentum_medium = (prices[-1] - prices[-self.medium_lookback]) / prices[-self.medium_lookback]
        
        # Momentum acceleration
        if len(self.momentum_history) >= 10:
            recent_momentum = list(self.momentum_history)[-10:]
            momentum_acceleration = np.polyfit(range(len(recent_momentum)), recent_momentum, 1)[0]
        else:
            momentum_acceleration = 0
            
        momentum = momentum_short
        self.momentum_history.append(momentum)
        
        # 4. Volume analysis
        if len(volumes) >= self.volume_lookback:
            recent_volume = np.mean(volumes[-10:])
            avg_volume = np.mean(volumes[-self.volume_lookback:])
            volume_ratio = recent_volume / (avg_volume + 1e-8)
        else:
            volume_ratio = 1.0
            
        # 5. Price acceleration (second derivative)
        if len(prices) >= 10:
            price_velocity = np.gradient(prices[-10:])
            price_acceleration = np.gradient(price_velocity)[-1]
        else:
            price_acceleration = 0
            
        # 6. Market efficiency (trend to volatility ratio)
        if volatility > 0:
            efficiency_ratio = abs(momentum_medium) / volatility
        else:
            efficiency_ratio = 0
            
        # 7. Regime stability
        if len(self.regime_history) >= 5:
            recent_regimes = list(self.regime_history)[-5:]
            regime_changes = sum(1 for i in range(1, len(recent_regimes)) 
                               if recent_regimes[i] != recent_regimes[i-1])
            regime_stability = 1.0 - (regime_changes / len(recent_regimes))
        else:
            regime_stability = 0.5
            
        return RegimeMetrics(
            trend_strength=trend_strength,
            volatility=volatility,
            momentum=momentum,
            volume_ratio=volume_ratio,
            price_acceleration=price_acceleration,
            market_efficiency=efficiency_ratio,
            regime_stability=regime_stability
        )
        
    def classify_regime(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on metrics
        
        Returns:
            Regime classification and confidence score
        """
        # Volatility percentiles
        if len(self.volatility_history) >= 20:
            vol_percentile = np.percentile(list(self.volatility_history), 
                                         [20, 50, 80])
            is_high_vol = metrics.volatility > vol_percentile[2]
            is_low_vol = metrics.volatility < vol_percentile[0]
        else:
            is_high_vol = metrics.volatility > 0.02
            is_low_vol = metrics.volatility < 0.01
            
        # Trend classification
        strong_trend_threshold = 0.6
        weak_trend_threshold = 0.3
        
        # Check for breakout/breakdown conditions
        if abs(metrics.price_acceleration) > 0.001 and metrics.volume_ratio > 1.5:
            if metrics.momentum > 0.01:
                return MarketRegime.BREAKOUT, 0.8
            elif metrics.momentum < -0.01:
                return MarketRegime.BREAKDOWN, 0.8
                
        # Trending markets
        if abs(metrics.trend_strength) > strong_trend_threshold:
            if metrics.trend_strength > 0:
                return MarketRegime.STRONG_UPTREND, min(abs(metrics.trend_strength), 1.0)
            else:
                return MarketRegime.STRONG_DOWNTREND, min(abs(metrics.trend_strength), 1.0)
                
        elif abs(metrics.trend_strength) > weak_trend_threshold:
            if metrics.trend_strength > 0:
                return MarketRegime.WEAK_UPTREND, 0.6
            else:
                return MarketRegime.WEAK_DOWNTREND, 0.6
                
        # Ranging markets
        elif metrics.market_efficiency < 0.3:  # Low efficiency = ranging
            if is_high_vol:
                return MarketRegime.RANGING_HIGH_VOL, 0.7
            elif is_low_vol:
                return MarketRegime.RANGING_LOW_VOL, 0.7
            else:
                return MarketRegime.CHOPPY, 0.5
        
        # Default to choppy
        return MarketRegime.CHOPPY, 0.4
        
    def detect_regime(self, price: float, volume: float = 1.0, spread: float = 0.0) -> Dict[str, any]:
        """
        Main regime detection method
        
        Returns dict with:
            - regime: MarketRegime enum
            - confidence: float (0-1)
            - metrics: RegimeMetrics
            - parameters: regime-specific trading parameters
        """
        # Update data
        self.update(price, volume, spread)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        if metrics is None:
            # Not enough data yet
            return {
                "regime": MarketRegime.RANGING_LOW_VOL,
                "confidence": 0.0,
                "metrics": None,
                "parameters": self.get_regime_parameters(MarketRegime.RANGING_LOW_VOL)
            }
            
        # Classify regime
        regime, confidence = self.classify_regime(metrics)
        
        # Update history
        self.regime_history.append(regime)
        self.regime_confidence_history.append(confidence)
        
        # Get regime-specific parameters
        parameters = self.get_regime_parameters(regime)
        
        return {
            "regime": regime,
            "confidence": confidence,
            "metrics": metrics,
            "parameters": parameters
        }
        
    def get_regime_parameters(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get regime-specific trading parameters
        
        Returns optimized parameters for each regime
        """
        parameters = {
            MarketRegime.STRONG_UPTREND: {
                "position_size_multiplier": 1.5,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "holding_time_multiplier": 1.2,
                "entry_threshold": 0.001,
                "profit_amplifier": 1.3,
                "exploration_rate": 0.05,
                "action_bias": 0.2  # Bias toward buying
            },
            MarketRegime.WEAK_UPTREND: {
                "position_size_multiplier": 1.2,
                "stop_loss": 0.015,
                "take_profit": 0.03,
                "holding_time_multiplier": 1.0,
                "entry_threshold": 0.0015,
                "profit_amplifier": 1.1,
                "exploration_rate": 0.08,
                "action_bias": 0.1
            },
            MarketRegime.STRONG_DOWNTREND: {
                "position_size_multiplier": 1.5,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "holding_time_multiplier": 1.2,
                "entry_threshold": 0.001,
                "profit_amplifier": 1.3,
                "exploration_rate": 0.05,
                "action_bias": -0.2  # Bias toward selling
            },
            MarketRegime.WEAK_DOWNTREND: {
                "position_size_multiplier": 1.2,
                "stop_loss": 0.015,
                "take_profit": 0.03,
                "holding_time_multiplier": 1.0,
                "entry_threshold": 0.0015,
                "profit_amplifier": 1.1,
                "exploration_rate": 0.08,
                "action_bias": -0.1
            },
            MarketRegime.RANGING_HIGH_VOL: {
                "position_size_multiplier": 0.8,
                "stop_loss": 0.025,
                "take_profit": 0.025,
                "holding_time_multiplier": 0.5,
                "entry_threshold": 0.002,
                "profit_amplifier": 1.5,  # Higher reward for catching ranges
                "exploration_rate": 0.12,
                "action_bias": 0.0
            },
            MarketRegime.RANGING_LOW_VOL: {
                "position_size_multiplier": 0.5,
                "stop_loss": 0.01,
                "take_profit": 0.01,
                "holding_time_multiplier": 0.3,
                "entry_threshold": 0.0005,
                "profit_amplifier": 2.0,  # Very high reward for small profits
                "exploration_rate": 0.15,
                "action_bias": 0.0
            },
            MarketRegime.BREAKOUT: {
                "position_size_multiplier": 2.0,
                "stop_loss": 0.03,
                "take_profit": 0.08,
                "holding_time_multiplier": 1.5,
                "entry_threshold": 0.0005,
                "profit_amplifier": 1.5,
                "exploration_rate": 0.03,
                "action_bias": 0.3
            },
            MarketRegime.BREAKDOWN: {
                "position_size_multiplier": 2.0,
                "stop_loss": 0.03,
                "take_profit": 0.08,
                "holding_time_multiplier": 1.5,
                "entry_threshold": 0.0005,
                "profit_amplifier": 1.5,
                "exploration_rate": 0.03,
                "action_bias": -0.3
            },
            MarketRegime.CHOPPY: {
                "position_size_multiplier": 0.7,
                "stop_loss": 0.02,
                "take_profit": 0.02,
                "holding_time_multiplier": 0.7,
                "entry_threshold": 0.0015,
                "profit_amplifier": 1.2,
                "exploration_rate": 0.1,
                "action_bias": 0.0
            }
        }
        
        return parameters.get(regime, parameters[MarketRegime.CHOPPY])
        
    def get_regime_features(self) -> np.ndarray:
        """
        Get regime features for agent state
        
        Returns array of normalized regime features
        """
        if len(self.regime_history) == 0:
            return np.zeros(12)
            
        # Current regime one-hot encoding (9 regimes)
        regime_one_hot = np.zeros(9)
        current_regime = self.regime_history[-1]
        regime_idx = list(MarketRegime).index(current_regime)
        regime_one_hot[regime_idx] = 1.0
        
        # Regime stability and confidence
        if len(self.regime_confidence_history) > 0:
            confidence = self.regime_confidence_history[-1]
        else:
            confidence = 0.0
            
        # Recent metrics
        metrics = self.calculate_metrics()
        if metrics:
            trend_feature = np.clip(metrics.trend_strength, -1, 1)
            volatility_feature = np.clip(metrics.volatility / 0.05, 0, 1)  # Normalize
        else:
            trend_feature = 0.0
            volatility_feature = 0.5
            
        # Combine features
        features = np.concatenate([
            regime_one_hot,  # 9 features
            [confidence],    # 1 feature
            [trend_feature], # 1 feature
            [volatility_feature]  # 1 feature
        ])
        
        return features.astype(np.float32)


class RegimeAwareEnvironment:
    """
    Wrapper to add regime awareness to trading environment
    """
    
    def __init__(self, base_env, regime_detector: AdvancedMarketRegimeDetector):
        """
        Initialize regime-aware environment wrapper
        
        Args:
            base_env: Base trading environment
            regime_detector: Advanced regime detector instance
        """
        self.env = base_env
        self.regime_detector = regime_detector
        
        # Extend state space for regime features
        self.original_state_dim = base_env.state_dim
        self.regime_feature_dim = 12
        self.state_dim = self.original_state_dim + self.regime_feature_dim
        
        # Copy other attributes
        self.action_dim = base_env.action_dim
        self.if_discrete = base_env.if_discrete
        self.max_step = getattr(base_env, 'max_step', 10000)
        
        # Regime-specific adjustments
        self.current_regime_params = None
        
        logger.info(f"Created RegimeAwareEnvironment with state dim: {self.state_dim}")
        
    def reset(self):
        """Reset environment and regime detector"""
        base_state = self.env.reset()
        
        # Reset regime detector
        self.regime_detector = AdvancedMarketRegimeDetector(
            short_lookback=self.regime_detector.short_lookback,
            medium_lookback=self.regime_detector.medium_lookback,
            long_lookback=self.regime_detector.long_lookback,
            device=self.regime_detector.device
        )
        
        # Get initial regime features
        regime_features = self.regime_detector.get_regime_features()
        
        # Combine states
        enhanced_state = np.concatenate([base_state, regime_features])
        
        return enhanced_state
        
    def step(self, action):
        """
        Take environment step with regime awareness
        
        Modifies action based on regime and adjusts rewards
        """
        # Get current price before step
        current_price = getattr(self.env, 'current_price', 100.0)
        current_volume = getattr(self.env, 'current_volume', 1.0)
        
        # Detect current regime
        regime_info = self.regime_detector.detect_regime(current_price, current_volume)
        self.current_regime_params = regime_info["parameters"]
        
        # Adjust action based on regime bias
        if hasattr(self, 'action_bias') and regime_info["confidence"] > 0.6:
            action_bias = self.current_regime_params.get("action_bias", 0.0)
            
            # Apply bias (for discrete actions)
            if self.if_discrete and action_bias != 0:
                # Random bias application
                if np.random.random() < abs(action_bias) * 0.3:  # 30% max bias
                    if action_bias > 0:  # Bias toward buying
                        action = 2  # Buy
                    else:  # Bias toward selling
                        action = 0  # Sell
        
        # Take base environment step
        base_state, base_reward, done, info = self.env.step(action)
        
        # Adjust reward based on regime
        regime_adjusted_reward = self._adjust_reward_for_regime(
            base_reward, 
            regime_info["regime"], 
            regime_info["confidence"],
            info
        )
        
        # Get regime features
        regime_features = self.regime_detector.get_regime_features()
        
        # Combine states
        enhanced_state = np.concatenate([base_state, regime_features])
        
        # Add regime info to info dict
        info["regime"] = regime_info["regime"].value
        info["regime_confidence"] = regime_info["confidence"]
        info["regime_parameters"] = self.current_regime_params
        
        return enhanced_state, regime_adjusted_reward, done, info
        
    def _adjust_reward_for_regime(self, base_reward: float, 
                                 regime: MarketRegime, 
                                 confidence: float,
                                 info: Dict) -> float:
        """Adjust reward based on current market regime"""
        if self.current_regime_params is None:
            return base_reward
            
        # Get profit amplifier for regime
        profit_amp = self.current_regime_params.get("profit_amplifier", 1.0)
        
        # Apply regime-specific adjustments
        adjusted_reward = base_reward
        
        # Amplify profits in appropriate regimes
        if base_reward > 0:
            adjusted_reward *= profit_amp
            
        # Extra rewards for regime-appropriate actions
        if "trade_completed" in info and info["trade_completed"]:
            trade_return = info.get("trade_return", 0)
            
            # Reward trend-following in trending markets
            if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
                if abs(trade_return) > 0.01:  # Significant profit
                    adjusted_reward += 0.01 * confidence
                    
            # Reward quick trades in ranging markets
            elif regime in [MarketRegime.RANGING_HIGH_VOL, MarketRegime.RANGING_LOW_VOL]:
                holding_time = info.get("holding_time", 100)
                if holding_time < 60 and trade_return > 0:  # Quick profit
                    adjusted_reward += 0.005 * confidence
                    
            # Reward catching breakouts
            elif regime in [MarketRegime.BREAKOUT, MarketRegime.BREAKDOWN]:
                if trade_return > 0.02:  # Large profit
                    adjusted_reward += 0.02 * confidence
        
        # Penalize inappropriate holding in volatile regimes
        if regime == MarketRegime.CHOPPY and info.get("position", 0) != 0:
            holding_time = info.get("holding_time", 0)
            if holding_time > 100:
                adjusted_reward -= 0.001 * confidence
                
        return adjusted_reward
        
    def __getattr__(self, name):
        """Forward attribute access to base environment"""
        return getattr(self.env, name)


def create_regime_aware_agent_config(base_config: Dict, regime: MarketRegime) -> Dict:
    """
    Create regime-specific agent configuration
    
    Adjusts hyperparameters based on market regime
    """
    config = base_config.copy()
    
    # Regime-specific adjustments
    adjustments = {
        MarketRegime.STRONG_UPTREND: {
            "learning_rate": 1.2,  # Multiplier
            "explore_rate": 0.7,
            "batch_size": 1.5,
            "clip_grad_norm": 1.2
        },
        MarketRegime.STRONG_DOWNTREND: {
            "learning_rate": 1.2,
            "explore_rate": 0.7,
            "batch_size": 1.5,
            "clip_grad_norm": 1.2
        },
        MarketRegime.RANGING_HIGH_VOL: {
            "learning_rate": 0.8,
            "explore_rate": 1.5,
            "batch_size": 0.8,
            "clip_grad_norm": 0.8
        },
        MarketRegime.RANGING_LOW_VOL: {
            "learning_rate": 0.5,
            "explore_rate": 2.0,
            "batch_size": 0.5,
            "clip_grad_norm": 0.5
        },
        MarketRegime.BREAKOUT: {
            "learning_rate": 1.5,
            "explore_rate": 0.5,
            "batch_size": 2.0,
            "clip_grad_norm": 1.5
        },
        MarketRegime.CHOPPY: {
            "learning_rate": 1.0,
            "explore_rate": 1.0,
            "batch_size": 1.0,
            "clip_grad_norm": 1.0
        }
    }
    
    # Get adjustments for regime
    regime_adj = adjustments.get(regime, adjustments[MarketRegime.CHOPPY])
    
    # Apply adjustments
    if "agent_config" in config:
        agent_cfg = config["agent_config"]
        
        for param, multiplier in regime_adj.items():
            if param in agent_cfg:
                if param in ["learning_rate", "explore_rate", "clip_grad_norm"]:
                    agent_cfg[param] *= multiplier
                elif param == "batch_size":
                    agent_cfg[param] = int(agent_cfg[param] * multiplier)
                    
    return config