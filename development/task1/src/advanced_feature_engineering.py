#!/usr/bin/env python3
"""
Advanced Feature Engineering for FinRL Contest 2024
High-impact features to improve Sharpe ratio > 1.0
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering specifically designed for cryptocurrency trading
    Focus on market microstructure and regime-aware features
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_microstructure_features(self, price_data: np.ndarray) -> np.ndarray:
        """
        Create market microstructure features from LOB data
        
        Args:
            price_data: Array with columns [bid, ask, mid]
            
        Returns:
            Microstructure features array
        """
        bid, ask, mid = price_data[:, 0], price_data[:, 1], price_data[:, 2]
        features = []
        feature_names = []
        
        # 1. Bid-Ask Spread Features
        spread = ask - bid
        spread_bps = (spread / mid) * 10000  # Basis points
        features.extend([spread, spread_bps])
        feature_names.extend(['spread_absolute', 'spread_bps'])
        
        # 2. Mid-price Returns (multiple horizons)
        for window in [1, 5, 10, 30]:
            returns = np.log(mid / np.roll(mid, window))
            returns[:window] = 0  # Handle initial values
            features.append(returns)
            feature_names.append(f'return_{window}min')
        
        # 3. Price Momentum and Mean Reversion
        # Short-term momentum (1-5 min)
        mom_1min = np.gradient(mid)
        mom_5min = mid - np.roll(mid, 5)
        mom_5min[:5] = 0
        
        # Mean reversion signals
        sma_20 = pd.Series(mid).rolling(20, min_periods=1).mean().values
        mean_reversion = (mid - sma_20) / sma_20
        
        features.extend([mom_1min, mom_5min, mean_reversion])
        feature_names.extend(['momentum_1min', 'momentum_5min', 'mean_reversion_20'])
        
        # 4. Volatility Features
        # Rolling volatility (multiple windows)
        for window in [10, 30, 60]:
            returns_1min = np.log(mid / np.roll(mid, 1))
            returns_1min[0] = 0
            vol = pd.Series(returns_1min).rolling(window, min_periods=1).std().values
            features.append(vol)
            feature_names.append(f'volatility_{window}min')
        
        # 5. Technical Indicators
        # RSI (Relative Strength Index)
        rsi = self._calculate_rsi(mid, window=14)
        features.append(rsi)
        feature_names.append('rsi_14')
        
        # MACD
        macd, macd_signal, macd_hist = self._calculate_macd(mid)
        features.extend([macd, macd_signal, macd_hist])
        feature_names.extend(['macd', 'macd_signal', 'macd_histogram'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(mid)
        bb_position = (mid - bb_lower) / (bb_upper - bb_lower)  # Position within bands
        bb_squeeze = (bb_upper - bb_lower) / bb_middle  # Band width
        features.extend([bb_position, bb_squeeze])
        feature_names.extend(['bollinger_position', 'bollinger_squeeze'])
        
        # 6. Order Flow Proxies (estimated from price/spread data)
        # Price impact proxy
        price_impact = np.abs(np.gradient(mid)) / spread_bps
        price_impact = np.nan_to_num(price_impact, 0)
        
        # Tick direction
        tick_direction = np.sign(np.gradient(mid))
        
        features.extend([price_impact, tick_direction])
        feature_names.extend(['price_impact_proxy', 'tick_direction'])
        
        # 7. Regime Detection Features
        # Volatility regime (high/low vol periods)
        vol_30min = pd.Series(np.log(mid / np.roll(mid, 1))).rolling(30, min_periods=1).std().values
        vol_regime = (vol_30min > np.percentile(vol_30min[30:], 75)).astype(float)
        
        # Trend regime
        trend_strength = np.abs(mean_reversion)
        trend_regime = (trend_strength > np.percentile(trend_strength[20:], 75)).astype(float)
        
        features.extend([vol_regime, trend_regime])
        feature_names.extend(['high_vol_regime', 'trend_regime'])
        
        # Stack all features
        feature_array = np.column_stack(features)
        self.feature_names = feature_names
        
        return feature_array
    
    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        try:
            rsi = talib.RSI(prices.astype(float), timeperiod=window)
            return np.nan_to_num(rsi, 50)  # Fill NaN with neutral value
        except:
            # Fallback manual calculation
            deltas = np.gradient(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = pd.Series(gains).rolling(window, min_periods=1).mean().values
            avg_losses = pd.Series(losses).rolling(window, min_periods=1).mean().values
            
            rs = avg_gains / (avg_losses + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        try:
            macd, macd_signal, macd_hist = talib.MACD(prices.astype(float))
            return (np.nan_to_num(macd, 0), 
                   np.nan_to_num(macd_signal, 0), 
                   np.nan_to_num(macd_hist, 0))
        except:
            # Fallback manual calculation
            ema_12 = pd.Series(prices).ewm(span=12).mean().values
            ema_26 = pd.Series(prices).ewm(span=26).mean().values
            macd = ema_12 - ema_26
            macd_signal = pd.Series(macd).ewm(span=9).mean().values
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, window: int = 20, num_std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        try:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices.astype(float), 
                                                        timeperiod=window, 
                                                        nbdevup=num_std, 
                                                        nbdevdn=num_std)
            return (np.nan_to_num(bb_upper, prices), 
                   np.nan_to_num(bb_middle, prices), 
                   np.nan_to_num(bb_lower, prices))
        except:
            # Fallback manual calculation
            sma = pd.Series(prices).rolling(window, min_periods=1).mean().values
            std = pd.Series(prices).rolling(window, min_periods=1).std().values
            bb_upper = sma + (std * num_std)
            bb_lower = sma - (std * num_std)
            return bb_upper, sma, bb_lower
    
    def create_enhanced_features(self, price_data: np.ndarray, factor_data: np.ndarray = None) -> Dict:
        """
        Create comprehensive enhanced feature set
        
        Args:
            price_data: Price data [bid, ask, mid]
            factor_data: Optional existing factor data
            
        Returns:
            Dictionary with enhanced features and metadata
        """
        print("🔧 Creating advanced microstructure features...")
        
        # Create microstructure features
        micro_features = self.create_microstructure_features(price_data)
        
        # Combine with existing factor data if provided
        if factor_data is not None:
            print(f"📊 Combining with existing {factor_data.shape[1]} features")
            all_features = np.column_stack([micro_features, factor_data])
            feature_names = self.feature_names + [f'factor_{i}' for i in range(factor_data.shape[1])]
        else:
            all_features = micro_features
            feature_names = self.feature_names
        
        # Normalize features
        print("📏 Normalizing features...")
        normalized_features = self.scaler.fit_transform(all_features)
        
        # Add position features (will be updated dynamically during trading)
        position_features = np.zeros((len(normalized_features), 2))  # [position, holding]
        final_features = np.column_stack([position_features, normalized_features])
        
        result = {
            'features': final_features,
            'feature_names': ['position', 'holding'] + feature_names,
            'n_features': final_features.shape[1],
            'n_microstructure': len(self.feature_names),
            'scaler': self.scaler
        }
        
        print(f"✅ Created {result['n_features']} total features:")
        print(f"   - 2 position features")
        print(f"   - {result['n_microstructure']} microstructure features")
        if factor_data is not None:
            print(f"   - {factor_data.shape[1]} original factor features")
        
        return result


def create_enhanced_dataset(csv_path: str, predict_path: str, output_path: str):
    """
    Create enhanced dataset with advanced features
    """
    print("🚀 Creating Enhanced Feature Dataset")
    print("="*50)
    
    # Load data
    print(f"📂 Loading data from {csv_path}")
    price_df = pd.read_csv(csv_path)
    price_data = price_df[["bids_distance_3", "asks_distance_3", "midpoint"]].values
    
    # Convert relative distances to absolute prices
    price_data[:, 0] = price_data[:, 2] * (1 + price_data[:, 0])  # Bid
    price_data[:, 1] = price_data[:, 2] * (1 + price_data[:, 1])  # Ask
    
    print(f"💰 Price data shape: {price_data.shape}")
    print(f"   Price range: ${price_data[:, 2].min():.2f} - ${price_data[:, 2].max():.2f}")
    
    # Load existing factors if available
    factor_data = None
    if predict_path and os.path.exists(predict_path):
        print(f"📊 Loading existing factors from {predict_path}")
        factor_data = np.load(predict_path)
        print(f"   Factor data shape: {factor_data.shape}")
    
    # Create enhanced features
    engineer = AdvancedFeatureEngineer()
    result = engineer.create_enhanced_features(price_data, factor_data)
    
    # Save enhanced dataset
    enhanced_features = result['features']
    
    print(f"💾 Saving enhanced features to {output_path}")
    np.save(output_path, enhanced_features)
    
    # Save metadata
    metadata = {
        'feature_names': result['feature_names'],
        'n_features': result['n_features'],
        'n_microstructure': result['n_microstructure'],
        'creation_date': pd.Timestamp.now().isoformat(),
        'source_files': {'csv': csv_path, 'factors': predict_path}
    }
    
    metadata_path = output_path.replace('.npy', '_metadata.npy')
    np.save(metadata_path, metadata, allow_pickle=True)
    
    print(f"📋 Metadata saved to {metadata_path}")
    print(f"✅ Enhanced dataset created successfully!")
    print(f"   Final shape: {enhanced_features.shape}")
    print(f"   Feature names: {result['feature_names'][:5]}... (showing first 5)")
    
    return enhanced_features, result


if __name__ == "__main__":
    import os
    
    # Paths
    csv_path = "../../../data/raw/task1/BTC_1sec.csv"
    predict_path = "../../../data/raw/task1/BTC_1sec_predict.npy"
    output_path = "../../../data/raw/task1/BTC_1sec_predict_enhanced_v2.npy"
    
    # Create enhanced dataset
    if os.path.exists(csv_path):
        enhanced_features, metadata = create_enhanced_dataset(csv_path, predict_path, output_path)
        
        print(f"\n🎯 Ready to use enhanced features in TradeSimulator!")
        print(f"   Use: {output_path}")
        print(f"   Features: {metadata['n_features']}")
    else:
        print(f"❌ Error: {csv_path} not found!")
        print("   Please ensure BTC_1sec.csv exists in the data directory")