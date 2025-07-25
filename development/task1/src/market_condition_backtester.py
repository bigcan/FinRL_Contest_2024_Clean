"""
Market Condition Analyzer for Backtesting
Advanced market regime detection and condition-specific performance analysis
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketCondition:
    """Market condition data structure"""
    regime: str
    volatility_level: str
    trend_direction: str
    momentum: float
    mean_reversion: float
    volume_profile: str
    start_idx: int
    end_idx: int
    duration: int
    confidence: float

@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime"""
    regime_name: str
    num_periods: int
    total_duration: int
    
    # Performance metrics
    avg_return: float
    avg_sharpe: float
    avg_max_drawdown: float
    win_rate: float
    
    # Consistency metrics
    return_std: float
    sharpe_std: float
    best_period_return: float
    worst_period_return: float
    
    # Market characteristics
    avg_volatility: float
    avg_trend_strength: float
    dominant_conditions: List[str]

class AdvancedMarketRegimeDetector:
    """Advanced market regime detection using multiple indicators"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()
        
        # Regime classification thresholds
        self.volatility_thresholds = {
            'low': 0.2,    # 20% annualized
            'medium': 0.4,  # 40% annualized
            'high': 0.8     # 80% annualized
        }
        
        self.trend_thresholds = {
            'weak': 0.05,    # 5% annualized
            'moderate': 0.15, # 15% annualized
            'strong': 0.30    # 30% annualized
        }
        
    def detect_regimes(self, prices: np.ndarray, volumes: np.ndarray = None) -> List[MarketCondition]:
        """Detect market regimes using advanced indicators"""
        
        if len(prices) < self.lookback_period:
            return []
        
        regimes = []
        window_size = self.lookback_period
        step_size = window_size // 4  # 75% overlap
        
        for start_idx in range(0, len(prices) - window_size, step_size):
            end_idx = start_idx + window_size
            
            # Extract window data
            window_prices = prices[start_idx:end_idx]
            window_volumes = volumes[start_idx:end_idx] if volumes is not None else None
            
            # Calculate regime characteristics
            regime_data = self._analyze_market_window(window_prices, window_volumes)
            
            # Create market condition
            condition = MarketCondition(
                regime=regime_data['regime'],
                volatility_level=regime_data['volatility_level'],
                trend_direction=regime_data['trend_direction'],
                momentum=regime_data['momentum'],
                mean_reversion=regime_data['mean_reversion'],
                volume_profile=regime_data['volume_profile'],
                start_idx=start_idx,
                end_idx=end_idx,
                duration=window_size,
                confidence=regime_data['confidence']
            )
            
            regimes.append(condition)
        
        return regimes
    
    def _analyze_market_window(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict[str, Any]:
        """Analyze a single market window"""
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # 1. Volatility Analysis
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        volatility_level = self._classify_volatility(volatility)
        
        # 2. Trend Analysis
        trend = self._calculate_trend_strength(prices)
        trend_direction = self._classify_trend(trend)
        
        # 3. Momentum Analysis
        momentum = self._calculate_momentum(prices)
        
        # 4. Mean Reversion Analysis
        mean_reversion = self._calculate_mean_reversion(prices)
        
        # 5. Volume Analysis
        volume_profile = self._analyze_volume_profile(volumes) if volumes is not None else 'unknown'
        
        # 6. Overall Regime Classification
        regime = self._classify_overall_regime(volatility_level, trend_direction, momentum, mean_reversion)
        
        # 7. Confidence Score
        confidence = self._calculate_confidence_score(volatility, trend, momentum, mean_reversion)
        
        return {
            'regime': regime,
            'volatility_level': volatility_level,
            'trend_direction': trend_direction,
            'momentum': momentum,
            'mean_reversion': mean_reversion,
            'volume_profile': volume_profile,
            'confidence': confidence,
            'raw_volatility': volatility,
            'raw_trend': trend
        }
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < self.volatility_thresholds['low']:
            return 'low'
        elif volatility < self.volatility_thresholds['medium']:
            return 'medium'
        elif volatility < self.volatility_thresholds['high']:
            return 'high'
        else:
            return 'extreme'
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using multiple methods"""
        
        # Linear regression slope
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        # Normalize by price level
        trend_strength = (slope / np.mean(prices)) * len(prices) * 252 / len(prices)  # Annualized
        
        return trend_strength
    
    def _classify_trend(self, trend_strength: float) -> str:
        """Classify trend direction and strength"""
        abs_trend = abs(trend_strength)
        
        if abs_trend < self.trend_thresholds['weak']:
            return 'sideways'
        elif trend_strength > 0:
            if abs_trend < self.trend_thresholds['moderate']:
                return 'weak_bull'
            elif abs_trend < self.trend_thresholds['strong']:
                return 'moderate_bull'
            else:
                return 'strong_bull'
        else:
            if abs_trend < self.trend_thresholds['moderate']:
                return 'weak_bear'
            elif abs_trend < self.trend_thresholds['strong']:
                return 'moderate_bear'
            else:
                return 'strong_bear'
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate momentum indicator"""
        if len(prices) < 20:
            return 0
        
        # Rate of change over different periods
        roc_short = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
        roc_medium = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
        roc_long = (prices[-1] / prices[-50] - 1) if len(prices) >= 50 else 0
        
        # Weighted average momentum
        momentum = 0.5 * roc_short + 0.3 * roc_medium + 0.2 * roc_long
        
        return momentum
    
    def _calculate_mean_reversion(self, prices: np.ndarray) -> float:
        """Calculate mean reversion tendency"""
        if len(prices) < 20:
            return 0
        
        # Hurst exponent approximation
        returns = np.diff(np.log(prices))
        
        # Calculate rescaled range
        def hurst_rs(ts, max_lag=20):
            lags = range(2, min(max_lag, len(ts)//2))
            tau = [np.std(np.cumsum(ts[:lag]) - np.mean(ts[:lag]) * np.arange(1, lag+1)) / np.std(ts[:lag]) / np.sqrt(lag) 
                   for lag in lags]
            
            # Simple approximation
            return np.mean(tau) if tau else 0.5
        
        hurst = hurst_rs(returns)
        
        # Mean reversion score (0.5 = random walk, <0.5 = mean reverting, >0.5 = trending)
        mean_reversion_score = 0.5 - hurst  # Positive = mean reverting
        
        return mean_reversion_score
    
    def _analyze_volume_profile(self, volumes: np.ndarray) -> str:
        """Analyze volume profile"""
        if volumes is None or len(volumes) == 0:
            return 'unknown'
        
        # Volume trend
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        
        # Volume volatility
        volume_cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
        
        # Classify volume profile
        if volume_trend > np.std(volumes) * 0.1:
            return 'increasing'
        elif volume_trend < -np.std(volumes) * 0.1:
            return 'decreasing'
        elif volume_cv > 1.0:
            return 'erratic'
        else:
            return 'stable'
    
    def _classify_overall_regime(self, volatility_level: str, trend_direction: str, 
                               momentum: float, mean_reversion: float) -> str:
        """Classify overall market regime"""
        
        # Combine indicators to determine regime
        regime_parts = []
        
        # Add volatility component
        regime_parts.append(volatility_level)
        
        # Add trend component
        if 'bull' in trend_direction:
            regime_parts.append('bull')
        elif 'bear' in trend_direction:
            regime_parts.append('bear')
        else:
            regime_parts.append('sideways')
        
        # Add momentum/mean reversion component
        if abs(momentum) > 0.05:  # Strong momentum
            regime_parts.append('trending')
        elif mean_reversion > 0.1:  # Mean reverting
            regime_parts.append('ranging')
        else:
            regime_parts.append('mixed')
        
        return '_'.join(regime_parts)
    
    def _calculate_confidence_score(self, volatility: float, trend: float, 
                                  momentum: float, mean_reversion: float) -> float:
        """Calculate confidence score for regime classification"""
        
        # Base confidence on strength of signals
        vol_confidence = min(volatility / 0.5, 1.0)  # Normalize to [0,1]
        trend_confidence = min(abs(trend) / 0.2, 1.0)
        momentum_confidence = min(abs(momentum) / 0.1, 1.0)
        mr_confidence = min(abs(mean_reversion) / 0.2, 1.0)
        
        # Weighted average
        confidence = (0.3 * vol_confidence + 0.3 * trend_confidence + 
                     0.2 * momentum_confidence + 0.2 * mr_confidence)
        
        return min(confidence, 1.0)

class MarketConditionBacktester:
    """Backtest performance across different market conditions"""
    
    def __init__(self, regime_detector: AdvancedMarketRegimeDetector = None):
        self.regime_detector = regime_detector or AdvancedMarketRegimeDetector()
        
    def analyze_regime_performance(self, backtest_results: List, 
                                 market_conditions: List[MarketCondition]) -> Dict[str, RegimePerformance]:
        """Analyze performance by market regime"""
        
        # Group results by regime
        regime_results = {}
        
        for result in backtest_results:
            # Find matching market condition
            matching_condition = self._find_matching_condition(result, market_conditions)
            
            if matching_condition:
                regime = matching_condition.regime
                if regime not in regime_results:
                    regime_results[regime] = []
                regime_results[regime].append((result, matching_condition))
        
        # Calculate performance metrics for each regime
        regime_performance = {}
        
        for regime, results_conditions in regime_results.items():
            if len(results_conditions) < 2:  # Skip if insufficient data
                continue
            
            results = [rc[0] for rc in results_conditions]
            conditions = [rc[1] for rc in results_conditions]
            
            # Calculate aggregate metrics
            returns = [r.total_return for r in results]
            sharpes = [r.sharpe_ratio for r in results if not np.isnan(r.sharpe_ratio)]
            drawdowns = [r.max_drawdown for r in results]
            win_rates = [r.win_rate for r in results]
            
            # Market characteristics
            volatilities = [c.raw_volatility for c in conditions if hasattr(c, 'raw_volatility')]
            trends = [c.raw_trend for c in conditions if hasattr(c, 'raw_trend')]
            
            # Create regime performance summary
            regime_performance[regime] = RegimePerformance(
                regime_name=regime,
                num_periods=len(results),
                total_duration=sum(r.duration_days for r in results),
                
                # Performance metrics
                avg_return=np.mean(returns),
                avg_sharpe=np.mean(sharpes) if sharpes else 0,
                avg_max_drawdown=np.mean(drawdowns),
                win_rate=np.mean([1 if r > 0 else 0 for r in returns]),
                
                # Consistency metrics
                return_std=np.std(returns),
                sharpe_std=np.std(sharpes) if sharpes else 0,
                best_period_return=np.max(returns),
                worst_period_return=np.min(returns),
                
                # Market characteristics
                avg_volatility=np.mean(volatilities) if volatilities else 0,
                avg_trend_strength=np.mean([abs(t) for t in trends]) if trends else 0,
                dominant_conditions=self._get_dominant_conditions(conditions)
            )
        
        return regime_performance
    
    def _find_matching_condition(self, result, conditions: List[MarketCondition]) -> Optional[MarketCondition]:
        """Find the market condition that best matches a backtest result"""
        
        # Simple matching based on index overlap
        result_start = getattr(result, 'start_idx', 0)
        result_end = getattr(result, 'end_idx', result_start + 1000)
        
        best_match = None
        best_overlap = 0
        
        for condition in conditions:
            # Calculate overlap
            overlap_start = max(result_start, condition.start_idx)
            overlap_end = min(result_end, condition.end_idx)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = condition
        
        return best_match
    
    def _get_dominant_conditions(self, conditions: List[MarketCondition]) -> List[str]:
        """Get dominant market conditions"""
        
        # Count frequency of different characteristics
        volatility_levels = [c.volatility_level for c in conditions]
        trend_directions = [c.trend_direction for c in conditions]
        
        # Find most common
        from collections import Counter
        vol_counter = Counter(volatility_levels)
        trend_counter = Counter(trend_directions)
        
        dominant = []
        if vol_counter:
            dominant.append(vol_counter.most_common(1)[0][0])
        if trend_counter:
            dominant.append(trend_counter.most_common(1)[0][0])
        
        return dominant
    
    def generate_regime_comparison_report(self, regime_performance: Dict[str, RegimePerformance]) -> str:
        """Generate regime comparison report"""
        
        report = f"""
        
{'='*80}
MARKET REGIME PERFORMANCE ANALYSIS
{'='*80}

REGIME PERFORMANCE SUMMARY
{'─'*40}
"""
        
        for regime_name, perf in regime_performance.items():
            report += f"""
{regime_name.upper().replace('_', ' ')}:
  Periods Tested:           {perf.num_periods}
  Average Return:           {perf.avg_return:.2%}
  Average Sharpe:           {perf.avg_sharpe:.3f}
  Win Rate:                 {perf.win_rate:.1%}
  Average Max Drawdown:     {perf.avg_max_drawdown:.2%}
  Return Consistency:       {perf.return_std:.3f}
  Best Period:              {perf.best_period_return:.2%}
  Worst Period:             {perf.worst_period_return:.2%}
  Average Volatility:       {perf.avg_volatility:.1%}
  Dominant Conditions:      {', '.join(perf.dominant_conditions)}
"""
        
        # Ranking analysis
        regimes_by_sharpe = sorted(regime_performance.items(), 
                                 key=lambda x: x[1].avg_sharpe, reverse=True)
        
        report += f"""

REGIME RANKINGS BY SHARPE RATIO
{'─'*40}
"""
        
        for i, (regime_name, perf) in enumerate(regimes_by_sharpe, 1):
            report += f"{i:2d}. {regime_name.replace('_', ' ').title():<25} {perf.avg_sharpe:>8.3f}\n"
        
        # Best/worst analysis
        if regimes_by_sharpe:
            best_regime = regimes_by_sharpe[0]
            worst_regime = regimes_by_sharpe[-1]
            
            report += f"""

BEST VS WORST PERFORMING REGIMES
{'─'*40}
Best:  {best_regime[0].replace('_', ' ').title()}
       Sharpe: {best_regime[1].avg_sharpe:.3f}, Return: {best_regime[1].avg_return:.2%}

Worst: {worst_regime[0].replace('_', ' ').title()}
       Sharpe: {worst_regime[1].avg_sharpe:.3f}, Return: {worst_regime[1].avg_return:.2%}

Performance Gap: {best_regime[1].avg_sharpe - worst_regime[1].avg_sharpe:.3f} Sharpe points
"""
        
        report += f"""

{'='*80}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        return report

class EventStudyAnalyzer:
    """Analyze performance around specific market events"""
    
    def __init__(self):
        self.event_windows = {
            'crash': (-5, 10),      # 5 days before, 10 days after
            'rally': (-5, 10),
            'volatility_spike': (-3, 7),
            'volume_surge': (-2, 5)
        }
    
    def detect_market_events(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict[str, List[int]]:
        """Detect significant market events"""
        
        returns = np.diff(np.log(prices))
        
        events = {
            'crash': [],
            'rally': [],
            'volatility_spike': [],
            'volume_surge': []
        }
        
        # Crash detection (large negative returns)
        crash_threshold = np.percentile(returns, 5)  # Bottom 5%
        for i, ret in enumerate(returns):
            if ret <= crash_threshold:
                events['crash'].append(i)
        
        # Rally detection (large positive returns)
        rally_threshold = np.percentile(returns, 95)  # Top 5%
        for i, ret in enumerate(returns):
            if ret >= rally_threshold:
                events['rally'].append(i)
        
        # Volatility spike detection
        rolling_vol = pd.Series(returns).rolling(20).std()
        vol_threshold = np.percentile(rolling_vol.dropna(), 90)
        for i, vol in enumerate(rolling_vol):
            if not np.isnan(vol) and vol >= vol_threshold:
                events['volatility_spike'].append(i)
        
        # Volume surge detection
        if volumes is not None:
            rolling_vol_avg = pd.Series(volumes).rolling(20).mean()
            vol_surge_threshold = rolling_vol_avg * 2  # 2x average volume
            for i, (vol, threshold) in enumerate(zip(volumes, vol_surge_threshold)):
                if not np.isnan(threshold) and vol >= threshold:
                    events['volume_surge'].append(i)
        
        return events
    
    def analyze_event_performance(self, events: Dict[str, List[int]], 
                                backtest_results: List) -> Dict[str, Dict]:
        """Analyze performance around events"""
        
        event_analysis = {}
        
        for event_type, event_indices in events.items():
            if not event_indices:
                continue
            
            window = self.event_windows.get(event_type, (-5, 5))
            
            # Find results that overlap with event windows
            event_performances = []
            
            for event_idx in event_indices:
                window_start = event_idx + window[0]
                window_end = event_idx + window[1]
                
                # Find overlapping backtest results
                for result in backtest_results:
                    result_start = getattr(result, 'start_idx', 0)
                    result_end = getattr(result, 'end_idx', result_start + 1000)
                    
                    # Check for overlap
                    if (result_start <= window_end and result_end >= window_start):
                        event_performances.append(result)
            
            # Calculate event-specific metrics
            if event_performances:
                returns = [r.total_return for r in event_performances]
                sharpes = [r.sharpe_ratio for r in event_performances if not np.isnan(r.sharpe_ratio)]
                
                event_analysis[event_type] = {
                    'num_events': len(event_indices),
                    'num_overlapping_results': len(event_performances),
                    'avg_return': np.mean(returns),
                    'avg_sharpe': np.mean(sharpes) if sharpes else 0,
                    'win_rate': np.mean([1 if r > 0 else 0 for r in returns]),
                    'best_performance': np.max(returns),
                    'worst_performance': np.min(returns)
                }
        
        return event_analysis

def main():
    """Example usage of market condition backtester"""
    
    # Generate sample data for testing
    np.random.seed(42)
    n_points = 10000
    
    # Create synthetic price data with different regimes
    prices = []
    current_price = 100
    
    for i in range(n_points):
        # Create different market regimes
        if i < 2000:  # Low vol bull market
            drift = 0.0005
            vol = 0.01
        elif i < 4000:  # High vol bear market
            drift = -0.001
            vol = 0.03
        elif i < 6000:  # Sideways market
            drift = 0.0001
            vol = 0.015
        elif i < 8000:  # Strong bull
            drift = 0.002
            vol = 0.02
        else:  # Mean reverting
            drift = -0.001 if current_price > 100 else 0.001
            vol = 0.012
        
        change = np.random.normal(drift, vol)
        current_price *= (1 + change)
        prices.append(current_price)
    
    prices = np.array(prices)
    volumes = np.random.exponential(1000, n_points)  # Synthetic volume
    
    print("🚀 Market Condition Analysis Demo")
    print("=" * 50)
    
    # 1. Detect market regimes
    print("\n📊 Detecting market regimes...")
    detector = AdvancedMarketRegimeDetector(lookback_period=200)
    market_conditions = detector.detect_regimes(prices, volumes)
    
    print(f"Detected {len(market_conditions)} market regimes")
    
    # Show regime distribution
    regime_counts = {}
    for condition in market_conditions:
        regime = condition.regime
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    print("\nRegime Distribution:")
    for regime, count in sorted(regime_counts.items()):
        print(f"  {regime.replace('_', ' ').title()}: {count} periods")
    
    # 2. Event analysis
    print("\n📈 Detecting market events...")
    event_analyzer = EventStudyAnalyzer()
    events = event_analyzer.detect_market_events(prices, volumes)
    
    for event_type, event_list in events.items():
        print(f"  {event_type.replace('_', ' ').title()}: {len(event_list)} events")
    
    print("\n✅ Market condition analysis complete!")
    
    return market_conditions, events

if __name__ == "__main__":
    conditions, events = main()