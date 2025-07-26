"""
Enhanced Features V3 - Next Generation Feature Engineering
Advanced microstructure indicators and sophisticated transformations for crypto trading
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from scipy.signal import find_peaks
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AdvancedMicrostructureFeatures:
    """Advanced microstructure features for cryptocurrency LOB data"""
    
    def __init__(self, lookback_window: int = 100):
        """
        Initialize advanced microstructure calculator
        
        Args:
            lookback_window: Window size for rolling calculations
        """
        self.lookback_window = lookback_window
        self.logger = logging.getLogger(__name__)
        
    def compute_order_arrival_rates(self, lob_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute order arrival rates using exponential smoothing
        
        Args:
            lob_data: LOB data with bid/ask information
            
        Returns:
            Dictionary of arrival rate features
        """
        features = {}
        
        # Extract bid/ask volumes for multiple levels
        bid_volumes = []
        ask_volumes = []
        
        for level in range(3):  # 3 levels of book depth
            bid_col = f'bids_quantity_{level}'
            ask_col = f'asks_quantity_{level}'
            
            if bid_col in lob_data.columns and ask_col in lob_data.columns:
                bid_volumes.append(lob_data[bid_col].values)
                ask_volumes.append(lob_data[ask_col].values)
        
        if not bid_volumes:
            # Fallback if specific columns not available
            self.logger.warning("Specific volume columns not found, using approximations")
            features['bid_arrival_rate'] = np.zeros(len(lob_data))
            features['ask_arrival_rate'] = np.zeros(len(lob_data))
            features['order_arrival_imbalance'] = np.zeros(len(lob_data))
            return features
        
        bid_total = np.sum(bid_volumes, axis=0)
        ask_total = np.sum(ask_volumes, axis=0)
        
        # Calculate arrival rates using exponential smoothing
        alpha = 2.0 / (self.lookback_window + 1)  # EMA smoothing factor
        
        # Bid/ask volume changes as proxy for order arrivals
        bid_changes = np.diff(bid_total, prepend=bid_total[0])
        ask_changes = np.diff(ask_total, prepend=ask_total[0])
        
        # Apply exponential smoothing
        bid_arrival_rate = np.zeros_like(bid_changes)
        ask_arrival_rate = np.zeros_like(ask_changes)
        
        for i in range(1, len(bid_changes)):
            bid_arrival_rate[i] = alpha * abs(bid_changes[i]) + (1 - alpha) * bid_arrival_rate[i-1]
            ask_arrival_rate[i] = alpha * abs(ask_changes[i]) + (1 - alpha) * ask_arrival_rate[i-1]
        
        # Normalize by recent volatility
        volatility = pd.Series(lob_data['midpoint']).rolling(window=20).std().fillna(method='bfill').values
        volatility = np.maximum(volatility, 1e-8)  # Avoid division by zero
        
        features['bid_arrival_rate'] = bid_arrival_rate / volatility
        features['ask_arrival_rate'] = ask_arrival_rate / volatility
        features['order_arrival_imbalance'] = (bid_arrival_rate - ask_arrival_rate) / (bid_arrival_rate + ask_arrival_rate + 1e-8)
        
        return features
    
    def compute_cancellation_rates(self, lob_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute order cancellation rates across price levels
        
        Args:
            lob_data: LOB data
            
        Returns:
            Dictionary of cancellation rate features
        """
        features = {}
        
        # Get spread as proxy for market stress
        spread = lob_data['spread'].values if 'spread' in lob_data.columns else np.ones(len(lob_data))
        spread_norm = spread / np.median(spread[spread > 0]) if np.any(spread > 0) else np.ones_like(spread)
        
        # Calculate cancellation rates using spread dynamics
        spread_changes = np.diff(spread_norm, prepend=spread_norm[0])
        spread_volatility = pd.Series(spread_changes).rolling(window=20).std().fillna(method='bfill').values
        
        # Higher spread volatility indicates more cancellations
        features['cancellation_rate'] = np.abs(spread_changes) / (spread_volatility + 1e-8)
        
        # Bid/ask specific cancellation proxies
        features['bid_cancellation_rate'] = np.maximum(0, spread_changes) / (spread_volatility + 1e-8)
        features['ask_cancellation_rate'] = np.maximum(0, -spread_changes) / (spread_volatility + 1e-8)
        
        return features
    
    def compute_price_impact_measures(self, lob_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute Kyle's lambda and Amihud illiquidity measures
        
        Args:
            lob_data: LOB data
            
        Returns:
            Dictionary of price impact features
        """
        features = {}
        
        midpoint = lob_data['midpoint'].values
        volume_proxy = lob_data['spread'].values if 'spread' in lob_data.columns else np.ones(len(lob_data))
        
        # Calculate returns
        returns = np.diff(np.log(midpoint), prepend=0)
        returns[0] = 0  # Set first return to 0
        
        # Kyle's Lambda (price impact per unit volume)
        # Approximated using rolling regression of returns on volume
        window = min(50, len(returns) // 10)
        kyles_lambda = np.zeros_like(returns)
        
        for i in range(window, len(returns)):
            start_idx = i - window
            y = returns[start_idx:i]
            x = volume_proxy[start_idx:i]
            
            if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                correlation = np.corrcoef(x, y)[0, 1]
                if not np.isnan(correlation):
                    kyles_lambda[i] = correlation * np.std(y) / np.std(x)
        
        # Amihud Illiquidity Measure
        # |return| / volume_proxy
        amihud_illiquidity = np.abs(returns) / (volume_proxy + 1e-8)
        
        # Rolling average for stability
        features['kyles_lambda'] = pd.Series(kyles_lambda).rolling(window=10).mean().fillna(0).values
        features['amihud_illiquidity'] = pd.Series(amihud_illiquidity).rolling(window=10).mean().fillna(0).values
        
        return features
    
    def compute_market_regime_detection(self, lob_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute ADX-based market regime detection
        
        Args:
            lob_data: LOB data
            
        Returns:
            Dictionary of regime detection features
        """
        features = {}
        
        # Use midpoint for price data
        prices = lob_data['midpoint'].values
        
        # Calculate high, low, close approximations
        window = 5
        high = pd.Series(prices).rolling(window=window).max().fillna(method='bfill').values
        low = pd.Series(prices).rolling(window=window).min().fillna(method='bfill').values
        close = prices
        
        # Calculate True Range (TR)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate +DM and -DM
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with exponential moving average
        period = 14
        alpha = 2.0 / (period + 1)
        
        atr = self._ema(tr, alpha)
        plus_di = 100 * self._ema(plus_dm, alpha) / (atr + 1e-8)
        minus_di = 100 * self._ema(minus_dm, alpha) / (atr + 1e-8)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = self._ema(dx, alpha)
        
        features['adx'] = adx / 100.0  # Normalize to [0,1]
        features['plus_di'] = plus_di / 100.0
        features['minus_di'] = minus_di / 100.0
        features['trend_strength'] = np.where(adx > 25, 1, 0)  # Strong trend indicator
        
        return features
    
    def _ema(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """Calculate exponential moving average"""
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema


class AdvancedDataTransformations:
    """Advanced data transformation techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def fractional_differentiation(
        self, 
        series: np.ndarray, 
        d: float = 0.5, 
        threshold: float = 1e-5
    ) -> np.ndarray:
        """
        Apply fractional differentiation to achieve stationarity while preserving memory
        
        Args:
            series: Input time series
            d: Degree of differentiation (0 < d < 1)
            threshold: Threshold for weight truncation
            
        Returns:
            Fractionally differentiated series
        """
        # Calculate weights
        weights = [1.0]
        k = 1
        
        while True:
            weight = -weights[-1] * (d - k + 1) / k
            if abs(weight) < threshold:
                break
            weights.append(weight)
            k += 1
        
        weights = np.array(weights)
        
        # Apply fractional differentiation
        n = len(series)
        result = np.zeros(n)
        
        for i in range(len(weights), n):
            result[i] = np.dot(series[i-len(weights)+1:i+1][::-1], weights)
        
        # Fill initial values with original differences
        if len(weights) > 1:
            initial_diffs = np.diff(series[:len(weights)], prepend=series[0])
            result[:len(initial_diffs)] = initial_diffs
        
        return result
    
    def time_based_features(self, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create sinusoidal time-based features
        
        Args:
            timestamps: Array of timestamps
            
        Returns:
            Dictionary of time-based features
        """
        if len(timestamps) == 0:
            return {}
        
        # Convert to pandas datetime if needed
        if not isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
            # Assume sequential minute data
            timestamps = pd.date_range(start='2020-01-01', periods=len(timestamps), freq='1min')
        else:
            timestamps = pd.to_datetime(timestamps)
        
        features = {}
        
        # Hour of day (0-23)
        hour = timestamps.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6)
        day_of_week = timestamps.dayofweek
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Minute of hour (0-59)
        minute = timestamps.minute
        features['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        features['minute_cos'] = np.cos(2 * np.pi * minute / 60)
        
        return features
    
    def rolling_z_scores(
        self, 
        data: np.ndarray, 
        windows: List[int] = [5, 20, 60]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate rolling z-scores for multiple windows
        
        Args:
            data: Input data array
            windows: List of window sizes
            
        Returns:
            Dictionary of rolling z-score features
        """
        features = {}
        series = pd.Series(data)
        
        for window in windows:
            if window < len(data):
                rolling_mean = series.rolling(window=window).mean()
                rolling_std = series.rolling(window=window).std()
                
                z_scores = (series - rolling_mean) / (rolling_std + 1e-8)
                features[f'z_score_{window}'] = z_scores.fillna(0).values
        
        return features
    
    def regime_adaptive_normalization(
        self, 
        data: np.ndarray, 
        volatility: np.ndarray
    ) -> np.ndarray:
        """
        Apply regime-adaptive normalization based on market volatility
        
        Args:
            data: Input data
            volatility: Volatility measure for regime detection
            
        Returns:
            Normalized data
        """
        # Define volatility regimes
        vol_percentiles = np.percentile(volatility[volatility > 0], [33, 67])
        
        low_vol_mask = volatility <= vol_percentiles[0]
        med_vol_mask = (volatility > vol_percentiles[0]) & (volatility <= vol_percentiles[1])
        high_vol_mask = volatility > vol_percentiles[1]
        
        normalized = np.zeros_like(data)
        
        # Normalize by regime
        for mask in [low_vol_mask, med_vol_mask, high_vol_mask]:
            if np.any(mask):
                regime_data = data[mask]
                if len(regime_data) > 1:
                    mean_val = np.mean(regime_data)
                    std_val = np.std(regime_data)
                    if std_val > 1e-8:
                        normalized[mask] = (regime_data - mean_val) / std_val
                    else:
                        normalized[mask] = regime_data - mean_val
        
        return normalized


class VolatilityClustering:
    """GARCH-based volatility clustering features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def simple_garch_volatility(
        self, 
        returns: np.ndarray, 
        alpha: float = 0.1, 
        beta: float = 0.85
    ) -> np.ndarray:
        """
        Simple GARCH(1,1) volatility estimation
        
        Args:
            returns: Return series
            alpha: GARCH alpha parameter
            beta: GARCH beta parameter
            
        Returns:
            Volatility estimates
        """
        if len(returns) < 2:
            return np.ones_like(returns)
        
        # Initialize
        volatility = np.zeros_like(returns)
        volatility[0] = np.std(returns[:min(50, len(returns))])  # Initial volatility
        
        omega = (1 - alpha - beta) * volatility[0]**2
        
        # GARCH recursion
        for i in range(1, len(returns)):
            volatility[i] = np.sqrt(
                omega + alpha * returns[i-1]**2 + beta * volatility[i-1]**2
            )
        
        return volatility
    
    def volatility_regimes(self, volatility: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Classify volatility regimes
        
        Args:
            volatility: Volatility series
            
        Returns:
            Dictionary of regime features
        """
        features = {}
        
        # Define regimes based on percentiles
        percentiles = np.percentile(volatility[volatility > 0], [25, 50, 75])
        
        features['vol_regime_low'] = (volatility <= percentiles[0]).astype(float)
        features['vol_regime_medium'] = ((volatility > percentiles[0]) & 
                                       (volatility <= percentiles[2])).astype(float)
        features['vol_regime_high'] = (volatility > percentiles[2]).astype(float)
        
        # Volatility persistence
        vol_change = np.diff(volatility, prepend=volatility[0])
        features['vol_persistence'] = pd.Series(vol_change).rolling(window=10).mean().fillna(0).values
        
        return features


class EnhancedFeatureEngineering:
    """Main class for enhanced feature engineering v3"""
    
    def __init__(
        self, 
        use_cache: bool = True, 
        cache_dir: str = "data",
        validation_split: bool = True
    ):
        """
        Initialize enhanced feature engineering
        
        Args:
            use_cache: Whether to use caching
            cache_dir: Directory for caching
            validation_split: Whether to implement validation-aware processing
        """
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.validation_split = validation_split
        
        # Initialize component modules
        self.microstructure = AdvancedMicrostructureFeatures()
        self.transformations = AdvancedDataTransformations()
        self.volatility = VolatilityClustering()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Feature storage
        self.feature_names = []
        self.train_stats = {}  # Statistics from training data only
        
    def generate_enhanced_features_v3(
        self, 
        csv_path: str, 
        predict_path: str = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        save_path: str = None
    ) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Generate next-generation enhanced features (v3)
        
        Args:
            csv_path: Path to raw CSV data
            predict_path: Path to existing predict array
            split_ratios: (train, val, test) ratios
            save_path: Path to save enhanced features
            
        Returns:
            Tuple of (feature_array, feature_names, metadata)
        """
        self.logger.info("Starting enhanced feature generation v3...")
        
        # Load raw data
        self.logger.info(f"Loading data from {csv_path}")
        lob_data = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(lob_data)} rows with {len(lob_data.columns)} columns")
        
        # Create temporal splits for validation-aware processing
        if self.validation_split:
            n_samples = len(lob_data)
            train_end = int(n_samples * split_ratios[0])
            val_end = train_end + int(n_samples * split_ratios[1])
            
            train_data = lob_data.iloc[:train_end]
            val_data = lob_data.iloc[train_end:val_end]
            test_data = lob_data.iloc[val_end:]
            
            self.logger.info(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        else:
            train_data = lob_data
        
        # Generate all feature categories
        all_features = {}
        feature_names = ['position_norm', 'holding_norm']  # Position features
        
        # 1. Advanced Microstructure Features
        self.logger.info("Computing advanced microstructure features...")
        
        # Order arrival rates
        arrival_features = self.microstructure.compute_order_arrival_rates(lob_data)
        for name, values in arrival_features.items():
            feature_names.append(f'micro_{name}')
            all_features[f'micro_{name}'] = values
        
        # Cancellation rates
        cancel_features = self.microstructure.compute_cancellation_rates(lob_data)
        for name, values in cancel_features.items():
            feature_names.append(f'micro_{name}')
            all_features[f'micro_{name}'] = values
        
        # Price impact measures
        impact_features = self.microstructure.compute_price_impact_measures(lob_data)
        for name, values in impact_features.items():
            feature_names.append(f'micro_{name}')
            all_features[f'micro_{name}'] = values
        
        # Market regime detection
        regime_features = self.microstructure.compute_market_regime_detection(lob_data)
        for name, values in regime_features.items():
            feature_names.append(f'regime_{name}')
            all_features[f'regime_{name}'] = values
        
        # 2. Advanced Data Transformations
        self.logger.info("Computing advanced data transformations...")
        
        # Fractional differentiation on midpoint
        midpoint = lob_data['midpoint'].values
        frac_diff = self.transformations.fractional_differentiation(midpoint)
        feature_names.append('transform_frac_diff')
        all_features['transform_frac_diff'] = frac_diff
        
        # Time-based features
        timestamps = None
        if 'timestamp' in lob_data.columns:
            timestamps = lob_data['timestamp'].values
        else:
            # Create synthetic timestamps
            timestamps = pd.date_range('2020-01-01', periods=len(lob_data), freq='1min')
        
        time_features = self.transformations.time_based_features(timestamps)
        for name, values in time_features.items():
            feature_names.append(f'time_{name}')
            all_features[f'time_{name}'] = values
        
        # Rolling z-scores on key price features
        for col in ['midpoint', 'spread']:
            if col in lob_data.columns:
                z_features = self.transformations.rolling_z_scores(lob_data[col].values)
                for name, values in z_features.items():
                    feature_names.append(f'zscore_{col}_{name}')
                    all_features[f'zscore_{col}_{name}'] = values
        
        # 3. Volatility Clustering Features
        self.logger.info("Computing volatility clustering features...")
        
        # Calculate returns
        returns = np.diff(np.log(midpoint), prepend=0)
        returns[0] = 0
        
        # GARCH volatility
        garch_vol = self.volatility.simple_garch_volatility(returns)
        feature_names.append('volatility_garch')
        all_features['volatility_garch'] = garch_vol
        
        # Volatility regimes
        vol_regime_features = self.volatility.volatility_regimes(garch_vol)
        for name, values in vol_regime_features.items():
            feature_names.append(f'vol_{name}')
            all_features[f'vol_{name}'] = values
        
        # Regime-adaptive normalization for spread
        if 'spread' in lob_data.columns:
            spread_norm = self.transformations.regime_adaptive_normalization(
                lob_data['spread'].values, garch_vol
            )
            feature_names.append('transform_spread_regime_norm')
            all_features['transform_spread_regime_norm'] = spread_norm
        
        # 4. Load and include existing features
        if predict_path and os.path.exists(predict_path):
            self.logger.info(f"Loading existing features from {predict_path}")
            existing_features = np.load(predict_path)
            
            # Add select existing features
            n_existing = min(10, existing_features.shape[1])  # Limit to prevent feature explosion
            for i in range(n_existing):
                feature_names.append(f'existing_{i}')
                all_features[f'existing_{i}'] = existing_features[:, i]
        
        # 5. Combine and align features
        self.logger.info("Combining and aligning features...")
        
        # Find minimum length
        feature_lengths = [len(values) for values in all_features.values()]
        min_length = min(feature_lengths) if feature_lengths else len(lob_data)
        
        # Create feature matrix (excluding position features)
        feature_matrix = []
        aligned_names = []
        
        for name in feature_names[2:]:  # Skip position features
            if name in all_features:
                values = all_features[name]
                if len(values) >= min_length:
                    aligned_values = values[-min_length:]  # Take last min_length values
                    feature_matrix.append(aligned_values)
                    aligned_names.append(name)
        
        if not feature_matrix:
            raise ValueError("No valid features generated")
        
        feature_array = np.column_stack(feature_matrix)
        self.feature_names = ['position_norm', 'holding_norm'] + aligned_names
        
        # 6. Train-only normalization (validation-aware)
        if self.validation_split:
            train_feature_end = int(min_length * split_ratios[0])
            train_features = feature_array[:train_feature_end]
            
            # Calculate normalization parameters from training data only
            for i in range(feature_array.shape[1]):
                col_data = train_features[:, i]
                if len(col_data) > 0:
                    mean_val = np.mean(col_data)
                    std_val = np.std(col_data)
                    
                    if std_val > 1e-8:
                        feature_array[:, i] = (feature_array[:, i] - mean_val) / std_val
                        self.train_stats[aligned_names[i]] = {'mean': mean_val, 'std': std_val}
                    else:
                        feature_array[:, i] = feature_array[:, i] - mean_val
                        self.train_stats[aligned_names[i]] = {'mean': mean_val, 'std': 1.0}
        else:
            # Standard normalization
            for i in range(feature_array.shape[1]):
                col_data = feature_array[:, i]
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                
                if std_val > 1e-8:
                    feature_array[:, i] = (col_data - mean_val) / std_val
                    self.train_stats[aligned_names[i]] = {'mean': mean_val, 'std': std_val}
                else:
                    feature_array[:, i] = col_data - mean_val
                    self.train_stats[aligned_names[i]] = {'mean': mean_val, 'std': 1.0}
        
        # 7. Add position features (zeros for now, filled at runtime)  
        position_features = np.zeros((len(feature_array), 2))
        final_array = np.column_stack([position_features, feature_array])
        
        self.logger.info(f"Enhanced features v3 generated: {final_array.shape}")
        self.logger.info(f"Feature count: {len(self.feature_names)}")
        
        # 8. Save features and metadata
        if save_path:
            self._save_enhanced_features(final_array, save_path, split_ratios)
        
        # Create metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'data_shape': final_array.shape,
            'normalization_params': self.train_stats,
            'split_ratios': split_ratios,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'feature_categories': self._categorize_features()
        }
        
        return final_array, self.feature_names, metadata
    
    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features by type"""
        categories = {
            'position': [],
            'microstructure': [],
            'regime': [],
            'transform': [],
            'time': [],
            'volatility': [],
            'zscore': [],
            'existing': []
        }
        
        for name in self.feature_names:
            if name.startswith('position') or name.startswith('holding'):
                categories['position'].append(name)
            elif name.startswith('micro_'):
                categories['microstructure'].append(name)
            elif name.startswith('regime_'):
                categories['regime'].append(name)
            elif name.startswith('transform_'):
                categories['transform'].append(name)
            elif name.startswith('time_'):
                categories['time'].append(name)
            elif name.startswith('vol_') or name.startswith('volatility_'):
                categories['volatility'].append(name)
            elif name.startswith('zscore_'):
                categories['zscore'].append(name)
            elif name.startswith('existing_'):
                categories['existing'].append(name)
        
        return categories
    
    def _save_enhanced_features(
        self, 
        feature_array: np.ndarray, 
        save_path: str, 
        split_ratios: Tuple[float, float, float]
    ):
        """Save enhanced features and metadata"""
        # Determine save paths
        if save_path.endswith('.npy'):
            base_path = save_path.replace('.npy', '')
        else:
            base_path = save_path
        
        enhanced_path = f"{base_path}_enhanced_v3.npy"
        metadata_path = f"{base_path}_enhanced_v3_metadata.npy"
        
        # Save feature array
        np.save(enhanced_path, feature_array)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'data_shape': feature_array.shape,
            'normalization_params': self.train_stats,
            'split_ratios': split_ratios,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'feature_categories': self._categorize_features()
        }
        
        np.save(metadata_path, metadata)
        
        self.logger.info(f"Enhanced features v3 saved to {enhanced_path}")
        self.logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    # Example usage
    from data_config import ConfigData
    
    # Initialize
    feature_engine = EnhancedFeatureEngineering(validation_split=True)
    
    # Load data configuration
    args = ConfigData()
    
    # Generate enhanced features
    feature_array, feature_names, metadata = feature_engine.generate_enhanced_features_v3(
        csv_path=args.csv_path,
        predict_path=args.predict_ary_path,
        save_path=args.predict_ary_path
    )
    
    print(f"Enhanced features v3 generated successfully!")
    print(f"Shape: {feature_array.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Categories: {list(metadata['feature_categories'].keys())}")