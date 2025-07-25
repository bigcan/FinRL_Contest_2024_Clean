"""
Regime Detection Threshold Tuner
Advanced tuning of regime detection thresholds for optimal market classification
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import components
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

@dataclass
class RegimeThresholds:
    """Regime detection thresholds configuration"""
    
    # Volatility thresholds
    low_volatility_threshold: float = 0.015
    high_volatility_threshold: float = 0.035
    
    # Trend thresholds  
    strong_trend_threshold: float = 0.02
    weak_trend_threshold: float = 0.005
    
    # Momentum thresholds
    momentum_threshold: float = 0.01
    momentum_acceleration_threshold: float = 0.005
    
    # Volume thresholds
    high_volume_threshold: float = 1.5  # multiplier of average volume
    low_volume_threshold: float = 0.7
    
    # Market stress thresholds
    stress_threshold: float = 0.05
    crisis_threshold: float = 0.08
    
    # Regime stability thresholds
    regime_stability_window: int = 20
    min_regime_duration: int = 10
    regime_confidence_threshold: float = 0.7
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'low_volatility_threshold': self.low_volatility_threshold,
            'high_volatility_threshold': self.high_volatility_threshold,
            'strong_trend_threshold': self.strong_trend_threshold,
            'weak_trend_threshold': self.weak_trend_threshold,
            'momentum_threshold': self.momentum_threshold,
            'momentum_acceleration_threshold': self.momentum_acceleration_threshold,
            'high_volume_threshold': self.high_volume_threshold,
            'low_volume_threshold': self.low_volume_threshold,
            'stress_threshold': self.stress_threshold,
            'crisis_threshold': self.crisis_threshold,
            'regime_stability_window': self.regime_stability_window,
            'min_regime_duration': self.min_regime_duration,
            'regime_confidence_threshold': self.regime_confidence_threshold
        }

class RegimeDetector:
    """Advanced regime detection with tunable thresholds"""
    
    def __init__(self, thresholds: RegimeThresholds):
        self.thresholds = thresholds
        self.regime_history = []
        self.feature_history = []
    
    def detect_regime(self, price_data: np.ndarray, volume_data: Optional[np.ndarray] = None, 
                     window: int = 50) -> Dict:
        """Detect market regime based on price and volume data"""
        
        if len(price_data) < window:
            return {'regime': 'insufficient_data', 'confidence': 0.0}
        
        # Calculate features
        features = self._calculate_regime_features(price_data, volume_data, window)
        
        # Classify regime
        regime, confidence = self._classify_regime(features)
        
        # Apply stability filtering
        stable_regime = self._apply_stability_filter(regime, confidence)
        
        # Store history 
        self.regime_history.append(stable_regime)
        self.feature_history.append(features)
        
        return {
            'regime': stable_regime['regime'],
            'confidence': stable_regime['confidence'],
            'features': features,
            'raw_regime': regime
        }
    
    def _calculate_regime_features(self, prices: np.ndarray, volumes: Optional[np.ndarray], 
                                 window: int) -> Dict:
        """Calculate comprehensive regime features"""
        
        # Price-based features
        returns = np.diff(np.log(prices[-window:]))
        
        # Volatility features
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        volatility_of_volatility = np.std([np.std(returns[i:i+10]) for i in range(len(returns)-10)])
        
        # Trend features
        trend = np.mean(returns) * 252  # Annualized
        trend_strength = abs(trend)
        
        # Momentum features
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-30:])
        momentum = (short_ma - long_ma) / long_ma
        
        # Momentum acceleration
        recent_momentum = (np.mean(prices[-5:]) - np.mean(prices[-15:-5])) / np.mean(prices[-15:-5])
        momentum_acceleration = recent_momentum - momentum
        
        # Range features
        price_range = (np.max(prices[-window:]) - np.min(prices[-window:])) / np.mean(prices[-window:])
        
        # Volume features (if available)
        volume_features = {}
        if volumes is not None and len(volumes) >= window:
            avg_volume = np.mean(volumes[-window:])
            recent_volume = np.mean(volumes[-10:])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            volume_volatility = np.std(volumes[-window:]) / avg_volume if avg_volume > 0 else 0.0
            
            volume_features = {
                'volume_ratio': volume_ratio,
                'volume_volatility': volume_volatility
            }
        
        # Market stress indicators
        drawdown = self._calculate_max_drawdown(prices[-window:])
        extreme_moves = np.sum(np.abs(returns) > 2 * np.std(returns)) / len(returns)
        
        return {
            'volatility': volatility,
            'volatility_of_volatility': volatility_of_volatility,
            'trend': trend,
            'trend_strength': trend_strength,
            'momentum': momentum,
            'momentum_acceleration': momentum_acceleration,
            'price_range': price_range,
            'drawdown': drawdown,
            'extreme_moves': extreme_moves,
            **volume_features
        }
    
    def _classify_regime(self, features: Dict) -> Tuple[str, float]:
        """Classify market regime based on features"""
        
        volatility = features['volatility']
        trend = features['trend']
        trend_strength = features['trend_strength']
        momentum = features['momentum']
        drawdown = features['drawdown']
        extreme_moves = features['extreme_moves']
        
        # Initialize confidence
        confidence = 0.5
        
        # Crisis detection (highest priority)
        if (volatility > self.thresholds.crisis_threshold or 
            drawdown > self.thresholds.crisis_threshold or
            extreme_moves > 0.1):
            return 'crisis', 0.9
        
        # Market stress detection
        if (volatility > self.thresholds.stress_threshold or
            drawdown > self.thresholds.stress_threshold):
            confidence = 0.8
            
            if trend > self.thresholds.weak_trend_threshold:
                return 'stressed_bull', confidence
            elif trend < -self.thresholds.weak_trend_threshold:
                return 'stressed_bear', confidence
            else:
                return 'stressed_sideways', confidence
        
        # Normal regime classification
        is_high_vol = volatility > self.thresholds.high_volatility_threshold
        is_low_vol = volatility < self.thresholds.low_volatility_threshold
        is_strong_trend = trend_strength > self.thresholds.strong_trend_threshold
        is_weak_trend = trend_strength < self.thresholds.weak_trend_threshold
        
        # Calculate confidence based on feature clarity
        vol_confidence = self._calculate_threshold_confidence(volatility, 
                                                            self.thresholds.low_volatility_threshold,
                                                            self.thresholds.high_volatility_threshold)
        trend_confidence = self._calculate_threshold_confidence(trend_strength,
                                                              self.thresholds.weak_trend_threshold,
                                                              self.thresholds.strong_trend_threshold)
        confidence = (vol_confidence + trend_confidence) / 2
        
        # Regime classification logic
        if is_strong_trend:
            if trend > 0:
                if is_high_vol:
                    return 'high_vol_bull_trending', confidence
                elif is_low_vol:
                    return 'low_vol_bull_trending', confidence
                else:
                    return 'bull_trending', confidence
            else:
                if is_high_vol:
                    return 'high_vol_bear_trending', confidence
                elif is_low_vol:
                    return 'low_vol_bear_trending', confidence
                else:
                    return 'bear_trending', confidence
        
        elif is_weak_trend:
            if is_high_vol:
                return 'high_vol_sideways', confidence
            elif is_low_vol:
                return 'low_vol_sideways', confidence
            else:
                return 'sideways', confidence
        
        else:  # Medium trend
            if trend > self.thresholds.momentum_threshold:
                return 'weak_bull', confidence
            elif trend < -self.thresholds.momentum_threshold:
                return 'weak_bear', confidence
            else:
                if is_high_vol:
                    return 'choppy_high_vol', confidence
                else:
                    return 'choppy', confidence
    
    def _calculate_threshold_confidence(self, value: float, low_threshold: float, 
                                      high_threshold: float) -> float:
        """Calculate confidence based on distance from thresholds"""
        
        mid_point = (low_threshold + high_threshold) / 2
        range_size = high_threshold - low_threshold
        
        if value <= low_threshold:
            distance = low_threshold - value
            confidence = min(1.0, 0.5 + distance / range_size)
        elif value >= high_threshold:
            distance = value - high_threshold
            confidence = min(1.0, 0.5 + distance / range_size)
        else:
            # In the middle range - lower confidence
            distance_from_mid = abs(value - mid_point)
            confidence = max(0.2, 0.5 - distance_from_mid / range_size)
        
        return confidence
    
    def _apply_stability_filter(self, regime: str, confidence: float) -> Dict:
        """Apply stability filtering to regime detection"""
        
        if len(self.regime_history) < self.thresholds.min_regime_duration:
            return {'regime': regime, 'confidence': confidence}
        
        # Check recent regime stability
        recent_regimes = [r['regime'] for r in self.regime_history[-self.thresholds.regime_stability_window:]]
        
        # Calculate regime consistency
        if recent_regimes:
            most_common = max(set(recent_regimes), key=recent_regimes.count)
            consistency = recent_regimes.count(most_common) / len(recent_regimes)
            
            # If current regime is consistent with recent history
            if regime == most_common and consistency > self.thresholds.regime_confidence_threshold:
                return {'regime': regime, 'confidence': min(1.0, confidence + 0.1)}
            
            # If switching regimes, require higher confidence
            elif regime != most_common:
                if confidence > self.thresholds.regime_confidence_threshold + 0.1:
                    return {'regime': regime, 'confidence': confidence}
                else:
                    # Stay with previous regime
                    return {'regime': most_common, 'confidence': confidence * 0.9}
        
        return {'regime': regime, 'confidence': confidence}
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        return np.max(drawdown)

class RegimeDetectionTuner:
    """Tune regime detection thresholds for optimal performance"""
    
    def __init__(self, price_data: np.ndarray, volume_data: Optional[np.ndarray] = None):
        self.price_data = price_data
        self.volume_data = volume_data
        self.optimization_history = []
        self.best_thresholds = None
        self.best_score = -np.inf
        
        # Create ground truth labels (simplified approach)
        self.ground_truth = self._create_ground_truth_labels()
    
    def _create_ground_truth_labels(self) -> List[str]:
        """Create ground truth regime labels using statistical analysis"""
        
        window = 100
        labels = []
        
        for i in range(window, len(self.price_data), 20):  # Every 20 steps
            segment = self.price_data[i-window:i]
            returns = np.diff(np.log(segment))
            
            volatility = np.std(returns) * np.sqrt(252)
            trend = np.mean(returns) * 252
            
            # Simple classification rules for ground truth
            if volatility > 0.04:
                vol_class = 'high_vol'
            elif volatility < 0.02:
                vol_class = 'low_vol'  
            else:
                vol_class = 'med_vol'
            
            if trend > 0.02:
                trend_class = 'bull'
            elif trend < -0.02:
                trend_class = 'bear'
            else:
                trend_class = 'sideways'
            
            labels.append(f"{vol_class}_{trend_class}")
        
        return labels
    
    def tune_thresholds(self, method='grid_search', n_trials=100) -> RegimeThresholds:
        """Tune regime detection thresholds"""
        
        print(f"🎯 Tuning regime detection thresholds")
        print(f"   Method: {method}")
        print(f"   Trials: {n_trials}")
        print(f"   Ground truth samples: {len(self.ground_truth)}")
        
        if method == 'grid_search':
            best_thresholds = self._grid_search_tuning()
        elif method == 'random_search':
            best_thresholds = self._random_search_tuning(n_trials)
        elif method == 'bayesian_optimization':
            best_thresholds = self._bayesian_optimization_tuning(n_trials)
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        
        self.best_thresholds = best_thresholds
        print(f"🎉 Tuning completed. Best score: {self.best_score:.4f}")
        
        return best_thresholds
    
    def _grid_search_tuning(self) -> RegimeThresholds:
        """Grid search threshold tuning"""
        
        # Define search grid (limited for computational efficiency)
        search_grid = {
            'low_volatility_threshold': [0.01, 0.015, 0.02],
            'high_volatility_threshold': [0.03, 0.035, 0.04],
            'strong_trend_threshold': [0.015, 0.02, 0.025],
            'weak_trend_threshold': [0.003, 0.005, 0.007],
            'regime_confidence_threshold': [0.6, 0.7, 0.8]
        }
        
        # Generate all combinations
        import itertools
        keys = list(search_grid.keys())
        values = list(search_grid.values())
        combinations = list(itertools.product(*values))
        
        print(f"   Evaluating {len(combinations)} combinations")
        
        best_score = -np.inf
        best_combination = None
        
        for i, combination in enumerate(combinations):
            # Create threshold configuration
            thresholds = RegimeThresholds()
            for key, value in zip(keys, combination):
                setattr(thresholds, key, value)
            
            # Evaluate configuration
            score = self._evaluate_threshold_configuration(thresholds)
            
            if score > best_score:
                best_score = score
                best_combination = thresholds
                print(f"     New best at {i+1}/{len(combinations)}: {score:.4f}")
            
            # Store in history
            self.optimization_history.append({
                'thresholds': thresholds,
                'score': score,
                'method': 'grid_search'
            })
        
        self.best_score = best_score
        return best_combination
    
    def _random_search_tuning(self, n_trials: int) -> RegimeThresholds:
        """Random search threshold tuning"""
        
        print(f"   Random search with {n_trials} trials")
        
        best_score = -np.inf
        best_thresholds = None
        
        for trial in range(n_trials):
            # Sample random thresholds
            thresholds = self._sample_random_thresholds()
            
            # Evaluate configuration
            score = self._evaluate_threshold_configuration(thresholds)
            
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
                print(f"     New best at trial {trial+1}: {score:.4f}")
            
            # Store in history
            self.optimization_history.append({
                'thresholds': thresholds,
                'score': score,
                'method': 'random_search'
            })
            
            if (trial + 1) % 20 == 0:
                print(f"   Progress: {trial + 1}/{n_trials}")
        
        self.best_score = best_score
        return best_thresholds
    
    def _bayesian_optimization_tuning(self, n_trials: int) -> RegimeThresholds:
        """Bayesian optimization threshold tuning (simplified)"""
        
        print(f"   Bayesian optimization with {n_trials} trials")
        
        # Initialize with random samples
        init_samples = min(20, n_trials // 3)
        
        for trial in range(init_samples):
            thresholds = self._sample_random_thresholds()
            score = self._evaluate_threshold_configuration(thresholds)
            
            if score > self.best_score:
                self.best_score = score
                self.best_thresholds = thresholds
            
            self.optimization_history.append({
                'thresholds': thresholds,
                'score': score,
                'method': 'bayesian_init'
            })
        
        # Continue with informed sampling
        for trial in range(init_samples, n_trials):
            thresholds = self._sample_informed_thresholds()
            score = self._evaluate_threshold_configuration(thresholds)
            
            if score > self.best_score:
                self.best_score = score
                self.best_thresholds = thresholds
                print(f"     New best at trial {trial+1}: {score:.4f}")
            
            self.optimization_history.append({
                'thresholds': thresholds,
                'score': score,
                'method': 'bayesian_informed'
            })
        
        return self.best_thresholds
    
    def _sample_random_thresholds(self) -> RegimeThresholds:
        """Sample random threshold configuration"""
        
        return RegimeThresholds(
            low_volatility_threshold=np.random.uniform(0.005, 0.025),
            high_volatility_threshold=np.random.uniform(0.025, 0.05),
            strong_trend_threshold=np.random.uniform(0.01, 0.03),
            weak_trend_threshold=np.random.uniform(0.002, 0.01),
            momentum_threshold=np.random.uniform(0.005, 0.02),
            momentum_acceleration_threshold=np.random.uniform(0.002, 0.01),
            high_volume_threshold=np.random.uniform(1.2, 2.0),
            low_volume_threshold=np.random.uniform(0.5, 0.8),
            stress_threshold=np.random.uniform(0.03, 0.07),
            crisis_threshold=np.random.uniform(0.06, 0.12),
            regime_stability_window=np.random.randint(10, 30),
            min_regime_duration=np.random.randint(5, 15),
            regime_confidence_threshold=np.random.uniform(0.5, 0.9)
        )
    
    def _sample_informed_thresholds(self) -> RegimeThresholds:
        """Sample thresholds based on optimization history"""
        
        if not self.optimization_history:
            return self._sample_random_thresholds()
        
        # Get top performing configurations
        sorted_history = sorted(self.optimization_history, key=lambda x: x['score'], reverse=True)
        top_configs = sorted_history[:max(1, len(sorted_history) // 4)]
        
        # Sample around best configuration
        base_config = np.random.choice(top_configs)['thresholds']
        
        # Add noise to base configuration
        noise_factor = 0.1
        
        return RegimeThresholds(
            low_volatility_threshold=max(0.005, base_config.low_volatility_threshold * 
                                       (1 + np.random.normal(0, noise_factor))),
            high_volatility_threshold=max(0.025, base_config.high_volatility_threshold * 
                                        (1 + np.random.normal(0, noise_factor))),
            strong_trend_threshold=max(0.01, base_config.strong_trend_threshold * 
                                     (1 + np.random.normal(0, noise_factor))),
            weak_trend_threshold=max(0.002, base_config.weak_trend_threshold * 
                                   (1 + np.random.normal(0, noise_factor))),
            momentum_threshold=max(0.005, base_config.momentum_threshold * 
                                 (1 + np.random.normal(0, noise_factor))),
            momentum_acceleration_threshold=max(0.002, base_config.momentum_acceleration_threshold * 
                                              (1 + np.random.normal(0, noise_factor))),
            high_volume_threshold=max(1.2, base_config.high_volume_threshold * 
                                    (1 + np.random.normal(0, noise_factor))),
            low_volume_threshold=max(0.5, base_config.low_volume_threshold * 
                                   (1 + np.random.normal(0, noise_factor))),
            stress_threshold=max(0.03, base_config.stress_threshold * 
                               (1 + np.random.normal(0, noise_factor))),
            crisis_threshold=max(0.06, base_config.crisis_threshold * 
                               (1 + np.random.normal(0, noise_factor))),
            regime_stability_window=max(10, int(base_config.regime_stability_window * 
                                              (1 + np.random.normal(0, noise_factor)))),
            min_regime_duration=max(5, int(base_config.min_regime_duration * 
                                         (1 + np.random.normal(0, noise_factor)))),
            regime_confidence_threshold=np.clip(base_config.regime_confidence_threshold * 
                                               (1 + np.random.normal(0, noise_factor)), 0.5, 0.9)
        )
    
    def _evaluate_threshold_configuration(self, thresholds: RegimeThresholds) -> float:
        """Evaluate threshold configuration against ground truth"""
        
        try:
            # Create detector with these thresholds
            detector = RegimeDetector(thresholds)
            
            # Get predictions for evaluation points
            predictions = []
            window = 100
            
            for i, true_label in enumerate(self.ground_truth):
                start_idx = i * 20  # Matches ground truth creation
                if start_idx + window < len(self.price_data):
                    segment = self.price_data[start_idx:start_idx + window]
                    volume_segment = None
                    if self.volume_data is not None:
                        volume_segment = self.volume_data[start_idx:start_idx + window]
                    
                    result = detector.detect_regime(segment, volume_segment, window)
                    predictions.append(result['regime'])
                else:
                    predictions.append('insufficient_data')
            
            # Calculate evaluation metrics
            # Simplified accuracy (regime family matching)
            family_accuracy = self._calculate_regime_family_accuracy(self.ground_truth, predictions)
            
            # Regime transition smoothness (penalize too frequent changes)
            transition_penalty = self._calculate_transition_penalty(predictions)
            
            # Confidence consistency (reward high-confidence predictions)
            confidence_bonus = self._calculate_confidence_bonus(detector)
            
            # Combined score
            score = 0.6 * family_accuracy + 0.2 * (1 - transition_penalty) + 0.2 * confidence_bonus
            
            return score
            
        except Exception as e:
            print(f"       Evaluation error: {e}")
            return 0.0
    
    def _calculate_regime_family_accuracy(self, true_labels: List[str], 
                                        predicted_labels: List[str]) -> float:
        """Calculate accuracy based on regime families"""
        
        def get_regime_family(label):
            if 'bull' in label:
                return 'bullish'
            elif 'bear' in label:
                return 'bearish'
            elif 'sideways' in label or 'choppy' in label:
                return 'sideways'
            elif 'crisis' in label or 'stressed' in label:
                return 'stressed'
            else:
                return 'other'
        
        true_families = [get_regime_family(label) for label in true_labels]
        pred_families = [get_regime_family(label) for label in predicted_labels]
        
        # Calculate accuracy
        correct = sum(1 for t, p in zip(true_families, pred_families) if t == p)
        total = len(true_families)
        
        return correct / max(total, 1)
    
    def _calculate_transition_penalty(self, predictions: List[str]) -> float:
        """Calculate penalty for excessive regime transitions"""
        
        if len(predictions) < 2:
            return 0.0
        
        transitions = sum(1 for i in range(1, len(predictions)) 
                         if predictions[i] != predictions[i-1])
        
        # Normalize by length and apply penalty
        transition_rate = transitions / (len(predictions) - 1)
        
        # Penalize transition rates above 0.3 (30% of periods)
        if transition_rate > 0.3:
            return (transition_rate - 0.3) / 0.7  # Scale to [0, 1]
        else:
            return 0.0
    
    def _calculate_confidence_bonus(self, detector: RegimeDetector) -> float:
        """Calculate bonus for high-quality confidence scores"""
        
        if not detector.regime_history:
            return 0.0
        
        confidences = [r['confidence'] for r in detector.regime_history]
        
        # Reward high mean confidence and low confidence volatility
        mean_confidence = np.mean(confidences)
        confidence_stability = 1.0 / (1.0 + np.std(confidences))
        
        return 0.7 * mean_confidence + 0.3 * confidence_stability
    
    def save_tuning_results(self, output_path: str):
        """Save threshold tuning results"""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save best thresholds
        if self.best_thresholds:
            thresholds_path = os.path.join(output_path, 'best_thresholds.json')
            with open(thresholds_path, 'w') as f:
                json.dump(self.best_thresholds.to_dict(), f, indent=2)
        
        # Save optimization history
        history_path = os.path.join(output_path, 'tuning_history.json')
        history_for_json = []
        for entry in self.optimization_history:
            history_for_json.append({
                'thresholds': entry['thresholds'].to_dict(),
                'score': entry['score'],
                'method': entry['method']
            })
        
        with open(history_path, 'w') as f:
            json.dump(history_for_json, f, indent=2)
        
        # Create visualization
        self._create_tuning_plots(output_path)
        
        # Create report
        self._create_tuning_report(output_path)
        
        print(f"📊 Tuning results saved to: {output_path}")
    
    def _create_tuning_plots(self, output_path: str):
        """Create tuning visualization plots"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Regime Detection Threshold Tuning Results', fontsize=16)
            
            # Plot 1: Score progression
            ax1 = axes[0, 0]
            scores = [entry['score'] for entry in self.optimization_history]
            ax1.plot(scores, 'b-', alpha=0.7)
            ax1.axhline(self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.3f}')
            ax1.set_title('Optimization Score Progression')
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Score')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Score distribution
            ax2 = axes[0, 1]
            ax2.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.3f}')
            ax2.set_title('Score Distribution')
            ax2.set_xlabel('Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            
            # Plot 3: Parameter sensitivity (example with volatility threshold)
            ax3 = axes[1, 0]
            vol_thresholds = [entry['thresholds'].low_volatility_threshold for entry in self.optimization_history]
            ax3.scatter(vol_thresholds, scores, alpha=0.6)
            ax3.set_title('Score vs Low Volatility Threshold')
            ax3.set_xlabel('Low Volatility Threshold')
            ax3.set_ylabel('Score')
            ax3.grid(True)
            
            # Plot 4: Method comparison (if multiple methods used)
            ax4 = axes[1, 1]
            methods = [entry['method'] for entry in self.optimization_history]
            method_scores = {}
            for method, score in zip(methods, scores):
                if method not in method_scores:
                    method_scores[method] = []
                method_scores[method].append(score)
            
            if len(method_scores) > 1:
                method_names = list(method_scores.keys())
                method_means = [np.mean(method_scores[method]) for method in method_names]
                ax4.bar(method_names, method_means)
                ax4.set_title('Mean Score by Method')
                ax4.set_ylabel('Mean Score')
            else:
                ax4.text(0.5, 0.5, 'Single Method Used', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Method Analysis')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'tuning_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Could not create tuning plots: {e}")
    
    def _create_tuning_report(self, output_path: str):
        """Create comprehensive tuning report"""
        
        report_path = os.path.join(output_path, 'tuning_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Regime Detection Threshold Tuning Report\n\n")
            
            # Summary
            f.write("## Tuning Summary\n\n")
            f.write(f"- **Trials**: {len(self.optimization_history)}\n")
            f.write(f"- **Best Score**: {self.best_score:.4f}\n")
            if self.optimization_history:
                scores = [entry['score'] for entry in self.optimization_history]
                f.write(f"- **Mean Score**: {np.mean(scores):.4f}\n")
                f.write(f"- **Score Std**: {np.std(scores):.4f}\n")
            f.write("\n")
            
            # Best thresholds
            if self.best_thresholds:
                f.write("## Best Threshold Configuration\n\n")
                thresholds_dict = self.best_thresholds.to_dict()
                for param, value in thresholds_dict.items():
                    f.write(f"- **{param}**: {value}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if self.best_score > 0.8:
                f.write("✅ **Excellent Performance**: Thresholds are well-tuned for regime detection.\n\n")
            elif self.best_score > 0.6:
                f.write("⚠️ **Good Performance**: Thresholds show good regime detection with room for improvement.\n\n")
            else:
                f.write("❌ **Needs Improvement**: Threshold tuning requires further optimization.\n\n")

def run_regime_tuning(price_data_path: str, output_path: str = "regime_tuning_results", 
                     method: str = 'random_search', n_trials: int = 100):
    """Run regime detection threshold tuning"""
    
    print(f"🎯 Starting Regime Detection Threshold Tuning")
    print(f"📂 Data: {price_data_path}")
    print(f"📂 Output: {output_path}")
    
    try:
        # Load price data
        if price_data_path.endswith('.csv'):
            data = pd.read_csv(price_data_path)
            price_data = data['midprice'].values if 'midprice' in data.columns else data.iloc[:, 0].values
            volume_data = data['volume'].values if 'volume' in data.columns else None
        else:
            price_data = np.load(price_data_path)
            volume_data = None
        
        print(f"📊 Loaded {len(price_data)} price points")
        
        # Initialize tuner
        tuner = RegimeDetectionTuner(price_data, volume_data)
        
        # Run tuning
        best_thresholds = tuner.tune_thresholds(method, n_trials)
        
        # Save results
        tuner.save_tuning_results(output_path)
        
        print(f"🎉 Tuning completed successfully!")
        print(f"📊 Best score: {tuner.best_score:.4f}")
        
        return best_thresholds
        
    except Exception as e:
        print(f"❌ Tuning failed: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    # Default parameters
    price_data_path = "../../../data/raw/task1/BTC_1sec.csv"
    output_path = "regime_tuning_results"
    method = 'random_search'
    n_trials = 100
    
    if len(sys.argv) > 1:
        price_data_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        method = sys.argv[3]
    if len(sys.argv) > 4:
        n_trials = int(sys.argv[4])
    
    best_thresholds = run_regime_tuning(price_data_path, output_path, method, n_trials)