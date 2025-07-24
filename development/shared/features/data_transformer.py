"""
Data Transformer Module

Handles data transformation for stationarity and model stability:
- Log returns transformation for price data
- Rolling statistics calculation (volatility, skewness, kurtosis)
- Time-based cyclical feature encoding
- Feature normalization with StandardScaler
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataTransformer:
    """Main data transformation pipeline"""
    
    def __init__(self, lookback_window=50):
        """
        Initialize data transformer
        
        Args:
            lookback_window: Window size for rolling calculations
        """
        self.lookback_window = lookback_window
        self.log_transformer = LogReturnsTransformer()
        self.rolling_calc = RollingStatsCalculator(lookback_window)
        self.time_features = TimeBasedFeatures()
        self.normalizer = FeatureNormalizer()
        
    def transform_data(self, lob_data, apply_normalization=True):
        """
        Apply all transformations to the data
        
        Args:
            lob_data: DataFrame with LOB data
            apply_normalization: Whether to apply StandardScaler normalization
            
        Returns:
            Dictionary of transformed features
        """
        features = {}
        
        # Log returns transformation for price data
        price_features = self.log_transformer.compute_log_returns(lob_data)
        features.update(price_features)
        
        # Rolling statistics
        rolling_features = self.rolling_calc.compute_rolling_stats(lob_data)
        features.update(rolling_features)
        
        # Time-based features
        if 'system_time' in lob_data.columns:
            time_features = self.time_features.compute_time_features(lob_data['system_time'])
            features.update(time_features)
        
        # Feature normalization
        if apply_normalization:
            normalized_features = self.normalizer.normalize_features(features)
            return normalized_features
        
        return features

class LogReturnsTransformer:
    """Transforms price data to stationary log returns"""
    
    def compute_log_returns(self, lob_data):
        """
        Compute log returns for price stationarity
        
        Args:
            lob_data: DataFrame with price data
            
        Returns:
            Dictionary of log return features
        """
        features = {}
        
        # Midpoint log returns
        midpoint = lob_data['midpoint'].values
        log_returns = np.diff(np.log(midpoint + 1e-8))  # Add small epsilon for numerical stability
        log_returns = np.concatenate([[0], log_returns])
        features['midpoint_log_returns'] = log_returns
        
        # Spread log returns (if available)
        if 'spread' in lob_data.columns:
            spread = lob_data['spread'].values
            spread_log_returns = np.diff(np.log(spread + 1e-8))
            spread_log_returns = np.concatenate([[0], spread_log_returns])
            features['spread_log_returns'] = spread_log_returns
        
        # Bid-Ask log return difference (measure of asymmetric price movements)
        if all(col in lob_data.columns for col in ['bids_distance_0', 'asks_distance_0']):
            bid_prices = midpoint * (1 + lob_data['bids_distance_0'].values)
            ask_prices = midpoint * (1 + lob_data['asks_distance_0'].values)
            
            bid_log_returns = np.diff(np.log(bid_prices + 1e-8))
            bid_log_returns = np.concatenate([[0], bid_log_returns])
            
            ask_log_returns = np.diff(np.log(ask_prices + 1e-8))
            ask_log_returns = np.concatenate([[0], ask_log_returns])
            
            features['bid_log_returns'] = bid_log_returns
            features['ask_log_returns'] = ask_log_returns
            features['bid_ask_return_diff'] = bid_log_returns - ask_log_returns
        
        return features

class RollingStatsCalculator:
    """Computes rolling statistical measures"""
    
    def __init__(self, lookback_window=50):
        """
        Initialize rolling statistics calculator
        
        Args:
            lookback_window: Window size for rolling calculations
        """
        self.lookback_window = lookback_window
        
    def compute_rolling_stats(self, lob_data):
        """
        Compute rolling statistical features
        
        Args:
            lob_data: DataFrame with LOB data
            
        Returns:
            Dictionary of rolling statistical features
        """
        features = {}
        
        # Get midpoint data
        midpoint = lob_data['midpoint'].values
        
        # Compute log returns first
        log_returns = np.diff(np.log(midpoint + 1e-8))
        log_returns = np.concatenate([[0], log_returns])
        
        # Rolling volatility (standard deviation of log returns)
        rolling_vol = self._rolling_calculation(np.std, pd.Series(log_returns), self.lookback_window)
        features['rolling_volatility'] = rolling_vol
        
        # Rolling skewness (asymmetry of return distribution)
        rolling_skew = self._rolling_calculation(self._safe_skewness, pd.Series(log_returns), self.lookback_window)
        features['rolling_skewness'] = rolling_skew
        
        # Rolling kurtosis (tail heaviness of return distribution)
        rolling_kurt = self._rolling_calculation(self._safe_kurtosis, pd.Series(log_returns), self.lookback_window)
        features['rolling_kurtosis'] = rolling_kurt
        
        # Rolling mean absolute deviation
        rolling_mad = self._rolling_calculation(self._mean_absolute_deviation, pd.Series(log_returns), self.lookback_window)
        features['rolling_mad'] = rolling_mad
        
        # Rolling range (max - min)
        rolling_range = self._rolling_calculation(self._price_range, pd.Series(midpoint), self.lookback_window)
        features['rolling_range'] = rolling_range / (midpoint + 1e-8)  # Normalize by price level
        
        return features
    
    def _rolling_calculation(self, func, data, window):
        """Apply rolling calculation function"""
        results = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data.iloc[start_idx:i+1]
            try:
                result = func(window_data)
                if np.isnan(result) or np.isinf(result):
                    result = 0.0
            except:
                result = 0.0
            results.append(result)
        return np.array(results)
    
    def _safe_skewness(self, data):
        """Compute skewness with numerical stability"""
        if len(data) < 3:
            return 0.0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val < 1e-8:
            return 0.0
        skew = np.mean(((data - mean_val) / std_val) ** 3)
        return skew
    
    def _safe_kurtosis(self, data):
        """Compute kurtosis with numerical stability"""
        if len(data) < 4:
            return 0.0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val < 1e-8:
            return 0.0
        kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3  # Excess kurtosis
        return kurt
    
    def _mean_absolute_deviation(self, data):
        """Compute mean absolute deviation"""
        if len(data) < 2:
            return 0.0
        return np.mean(np.abs(data - np.mean(data)))
    
    def _price_range(self, data):
        """Compute price range (max - min)"""
        if len(data) < 2:
            return 0.0
        return np.max(data) - np.min(data)

class TimeBasedFeatures:
    """Computes cyclical time-based features"""
    
    def compute_time_features(self, timestamps):
        """
        Compute cyclical time features
        
        Args:
            timestamps: Array of timestamp strings or datetime objects
            
        Returns:
            Dictionary of time-based features
        """
        features = {}
        
        # Convert to datetime if needed
        if isinstance(timestamps.iloc[0], str):
            timestamps = pd.to_datetime(timestamps)
        
        # Hour of day (0-23) -> cyclical encoding
        hours = timestamps.dt.hour.values
        features['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        
        # Day of week (0-6) -> cyclical encoding  
        days = timestamps.dt.dayofweek.values
        features['day_sin'] = np.sin(2 * np.pi * days / 7)
        features['day_cos'] = np.cos(2 * np.pi * days / 7)
        
        # Minute of hour (for intraday patterns)
        minutes = timestamps.dt.minute.values
        features['minute_sin'] = np.sin(2 * np.pi * minutes / 60)
        features['minute_cos'] = np.cos(2 * np.pi * minutes / 60)
        
        return features

class FeatureNormalizer:
    """Handles feature normalization with StandardScaler"""
    
    def __init__(self):
        """Initialize feature normalizer"""
        self.scaler = None
        self.is_fitted = False
        
    def normalize_features(self, features_dict):
        """
        Normalize features using StandardScaler
        
        Args:
            features_dict: Dictionary of feature arrays
            
        Returns:
            Dictionary of normalized features
        """
        # Convert to DataFrame for easier handling
        feature_names = list(features_dict.keys())
        feature_matrix = np.column_stack([features_dict[name] for name in feature_names])
        
        # Initialize and fit scaler if not already done
        if not self.is_fitted:
            self.scaler = StandardScaler()
            normalized_matrix = self.scaler.fit_transform(feature_matrix)
            self.is_fitted = True
        else:
            normalized_matrix = self.scaler.transform(feature_matrix)
        
        # Convert back to dictionary
        normalized_features = {}
        for i, name in enumerate(feature_names):
            normalized_features[name] = normalized_matrix[:, i]
            
        return normalized_features
    
    def get_normalization_params(self):
        """Get normalization parameters for debugging"""
        if self.scaler is None:
            return None
        return {
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_,
            'var': self.scaler.var_
        }