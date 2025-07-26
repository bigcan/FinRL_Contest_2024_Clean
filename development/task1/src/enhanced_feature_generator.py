#!/usr/bin/env python3
"""
Enhanced Feature Generator - Build upon existing 8-feature optimization
Adds complementary features to the existing optimized 8-feature set
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_existing_optimized_features():
    """Load existing optimized 8-feature dataset"""
    
    print("📊 Loading existing optimized feature set...")
    
    # Load optimized features
    feature_path = "../../../data/raw/task1/BTC_1sec_predict_optimized.npy" 
    metadata_path = "../../../data/raw/task1/BTC_1sec_predict_optimized_metadata.npy"
    
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Optimized features not found: {feature_path}")
    
    # Load data
    features = np.load(feature_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    
    print(f"✅ Loaded optimized features: {features.shape}")
    print(f"   Selected features: {metadata.get('feature_names', [])}")
    
    return features, metadata

def load_original_price_data():
    """Load original BTC price data for additional feature calculation"""
    
    print("💰 Loading original price data...")
    
    price_path = "../../../data/raw/task1/BTC_1sec.csv"
    
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Price data not found: {price_path}")
    
    # Load price data
    df = pd.read_csv(price_path)
    print(f"✅ Loaded price data: {df.shape}")
    
    # Extract key price columns
    price_columns = ['close', 'high', 'low', 'volume']
    available_columns = [col for col in price_columns if col in df.columns]
    
    if not available_columns:
        print("⚠️  Standard price columns not found, using available columns")
        available_columns = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
    
    price_data = df[available_columns].values
    print(f"   Using columns: {available_columns}")
    
    return price_data, available_columns

def calculate_volatility_features(price_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate volatility-based features"""
    
    print("📊 Calculating volatility features...")
    
    close_prices = price_data[:, 0]  # Assume first column is close
    
    features = {}
    
    # Historical volatility (different windows)
    for window in [10, 20, 50]:
        returns = np.diff(np.log(close_prices))
        volatility = pd.Series(returns).rolling(window=window).std() * np.sqrt(86400)  # Annualized
        volatility = volatility.fillna(volatility.mean())
        features[f'volatility_{window}'] = volatility.values
    
    # GARCH-style volatility proxy
    returns = np.diff(np.log(close_prices))
    returns = np.concatenate([[0], returns])  # Pad for length
    squared_returns = returns ** 2
    garch_vol = pd.Series(squared_returns).ewm(alpha=0.1).mean()
    features['garch_volatility'] = np.sqrt(garch_vol.values)
    
    # Volatility regime indicator
    vol_20 = features['volatility_20']
    vol_regime = np.where(vol_20 > np.percentile(vol_20, 75), 1,  # High vol
                 np.where(vol_20 < np.percentile(vol_20, 25), -1, 0))  # Low vol
    features['volatility_regime'] = vol_regime
    
    print(f"   ✅ Created {len(features)} volatility features")
    return features

def calculate_bollinger_bands(price_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate Bollinger Bands features"""
    
    print("📈 Calculating Bollinger Bands features...")
    
    close_prices = price_data[:, 0]
    features = {}
    
    # Standard Bollinger Bands (20, 2)
    window = 20
    sma = pd.Series(close_prices).rolling(window=window).mean()
    std = pd.Series(close_prices).rolling(window=window).std()
    
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    
    # Bollinger Band Position (0 = lower band, 1 = upper band)
    bb_position = (close_prices - lower_band) / (upper_band - lower_band)
    bb_position = np.clip(bb_position, 0, 1)
    features['bb_position'] = bb_position.fillna(0.5).values
    
    # Bollinger Band Width (normalized volatility)
    bb_width = (upper_band - lower_band) / sma
    features['bb_width'] = bb_width.fillna(bb_width.mean()).values
    
    # Bollinger Band Squeeze (low width periods)
    bb_squeeze = (bb_width < bb_width.rolling(50).quantile(0.2)).astype(int)
    features['bb_squeeze'] = bb_squeeze.fillna(0).values
    
    print(f"   ✅ Created {len(features)} Bollinger Bands features")
    return features

def calculate_macd_features(price_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate MACD features"""
    
    print("🔄 Calculating MACD features...")
    
    close_prices = price_data[:, 0]
    features = {}
    
    # MACD parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    # Calculate EMAs
    ema_fast = pd.Series(close_prices).ewm(span=fast_period).mean()
    ema_slow = pd.Series(close_prices).ewm(span=slow_period).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    features['macd_line'] = macd_line.fillna(0).values
    
    # Signal line
    signal_line = macd_line.ewm(span=signal_period).mean()
    features['macd_signal'] = signal_line.fillna(0).values
    
    # MACD histogram
    macd_histogram = macd_line - signal_line
    features['macd_histogram'] = macd_histogram.fillna(0).values
    
    # MACD cross signals
    macd_cross = np.where(macd_line > signal_line, 1, -1)
    features['macd_cross'] = macd_cross
    
    print(f"   ✅ Created {len(features)} MACD features")
    return features

def calculate_regime_detection_features(price_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate market regime detection features"""
    
    print("🌍 Calculating regime detection features...")
    
    close_prices = price_data[:, 0]
    if price_data.shape[1] > 3:
        volume = price_data[:, 3]
    else:
        volume = np.ones_like(close_prices)  # Fallback
    
    features = {}
    
    # Trend strength (ADX proxy)
    high = close_prices  # Simplified - use close as high
    low = close_prices   # Simplified - use close as low
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close_prices, 1))
    tr3 = np.abs(low - np.roll(close_prices, 1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Average True Range
    atr = pd.Series(true_range).rolling(window=14).mean()
    features['atr'] = atr.fillna(atr.mean()).values
    
    # Trend direction
    returns = np.diff(np.log(close_prices))
    returns = np.concatenate([[0], returns])
    
    # Trend strength indicator
    trend_strength = pd.Series(np.abs(returns)).rolling(window=20).mean()
    features['trend_strength'] = trend_strength.fillna(trend_strength.mean()).values
    
    # Range vs Trend classification
    range_indicator = pd.Series(returns).rolling(window=50).std()
    trend_indicator = pd.Series(returns).rolling(window=50).mean()
    
    regime = np.where(np.abs(trend_indicator) > range_indicator, 1, 0)  # 1=trend, 0=range
    features['market_regime'] = regime
    
    # Volume regime
    volume_ma = pd.Series(volume).rolling(window=20).mean()
    volume_regime = np.where(volume > volume_ma.fillna(volume_ma.mean()).values, 1, 0)  # 1=high volume, 0=normal
    features['volume_regime'] = volume_regime
    
    print(f"   ✅ Created {len(features)} regime detection features")
    return features

def calculate_multi_timeframe_features(price_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate multi-timeframe features"""
    
    print("⏰ Calculating multi-timeframe features...")
    
    close_prices = price_data[:, 0]
    features = {}
    
    # Multi-timeframe moving averages
    timeframes = [5, 10, 30, 60]  # Different lookback periods
    
    for tf in timeframes:
        # Simple moving average
        sma = pd.Series(close_prices).rolling(window=tf).mean()
        
        # Price relative to SMA
        price_vs_sma = (close_prices / sma - 1) * 100  # Percentage deviation
        features[f'price_vs_sma_{tf}'] = price_vs_sma.fillna(0).values
        
        # SMA slope (trend direction)
        sma_slope = sma.diff() / sma * 100
        features[f'sma_slope_{tf}'] = sma_slope.fillna(0).values
    
    # Multi-timeframe RSI
    for tf in [7, 14, 28]:
        delta = pd.Series(close_prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=tf).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=tf).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features[f'rsi_{tf}'] = rsi.fillna(50).values
    
    print(f"   ✅ Created {len(features)} multi-timeframe features")
    return features

def select_best_features(all_features: Dict[str, np.ndarray], 
                        existing_features: np.ndarray,
                        target_count: int = 12) -> Tuple[np.ndarray, List[str]]:
    """Select best additional features using correlation and importance analysis"""
    
    print(f"🎯 Selecting best {target_count} additional features...")
    
    # Ensure all features have the same length
    target_length = existing_features.shape[0]
    aligned_features = {}
    
    for name, feature in all_features.items():
        if len(feature) > target_length:
            aligned_features[name] = feature[:target_length]
        elif len(feature) < target_length:
            # Pad with the last value
            padded = np.concatenate([feature, np.full(target_length - len(feature), feature[-1])])
            aligned_features[name] = padded
        else:
            aligned_features[name] = feature
    
    # Convert to DataFrame for analysis
    feature_df = pd.DataFrame(aligned_features)
    existing_df = pd.DataFrame(existing_features)
    
    # Calculate correlations with existing features
    correlations = {}
    for new_feature in feature_df.columns:
        max_corr = 0
        for i in range(existing_df.shape[1]):
            corr = abs(np.corrcoef(feature_df[new_feature].fillna(0), 
                                 existing_df.iloc[:, i])[0, 1])
            max_corr = max(max_corr, corr)
        correlations[new_feature] = max_corr
    
    # Calculate feature variance (importance proxy)
    variances = feature_df.var()
    
    # Score features (low correlation with existing, high variance)
    scores = {}
    for feature in feature_df.columns:
        correlation_penalty = correlations[feature]
        variance_score = min(variances[feature] / variances.max(), 1.0)
        
        # Combined score (lower correlation + higher variance = better)
        scores[feature] = variance_score * (1 - correlation_penalty)
    
    # Select top features
    selected_features = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:target_count]
    
    # Create feature matrix
    selected_matrix = feature_df[selected_features].fillna(0).values
    
    print(f"   ✅ Selected features: {selected_features}")
    print(f"   📊 Average correlation with existing: {np.mean([correlations[f] for f in selected_features]):.3f}")
    
    return selected_matrix, selected_features

def combine_feature_sets(existing_features: np.ndarray,
                        new_features: np.ndarray,
                        existing_names: List[str],
                        new_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Combine existing and new features"""
    
    print("🔄 Combining feature sets...")
    
    # Ensure same number of samples
    min_samples = min(existing_features.shape[0], new_features.shape[0])
    existing_features = existing_features[:min_samples]
    new_features = new_features[:min_samples]
    
    # Combine features
    combined_features = np.concatenate([existing_features, new_features], axis=1)
    combined_names = existing_names + new_names
    
    print(f"   ✅ Combined shape: {combined_features.shape}")
    print(f"   📊 Features: {len(combined_names)}")
    
    return combined_features, combined_names

def save_enhanced_dataset(features: np.ndarray, 
                         feature_names: List[str],
                         metadata: Dict) -> str:
    """Save enhanced dataset with metadata"""
    
    print("💾 Saving enhanced dataset...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save enhanced features
    enhanced_path = f"../../../data/raw/task1/BTC_1sec_predict_enhanced_v2.npy"
    np.save(enhanced_path, features)
    
    # Save metadata
    enhanced_metadata = {
        'creation_date': datetime.now().strftime("%Y-%m-%d"),
        'phase': 'Phase 3 - Enhanced Feature Engineering',
        'base_system': 'Phase 2 optimized 8-feature system',
        'enhancement_method': 'Complementary feature addition',
        'feature_names': feature_names,
        'original_shape': metadata.get('optimized_shape', 'Unknown'),
        'enhanced_shape': features.shape,
        'base_features': metadata.get('feature_names', []),
        'enhancement_count': len(feature_names) - len(metadata.get('feature_names', [])),
        'selection_criteria': 'Low correlation + high variance + domain expertise',
        'expected_improvement': 'Better market regime detection and volatility handling'
    }
    
    metadata_path = f"../../../data/raw/task1/BTC_1sec_predict_enhanced_v2_metadata.npy"
    np.save(metadata_path, enhanced_metadata)
    
    print(f"   ✅ Saved to: {enhanced_path}")
    print(f"   📊 Final shape: {features.shape}")
    print(f"   🎯 Total features: {len(feature_names)}")
    
    return enhanced_path

def main():
    """Main feature enhancement process"""
    
    print("🚀 ENHANCED FEATURE GENERATION SYSTEM")
    print("Building upon existing 8-feature optimization")
    print("=" * 70)
    
    try:
        # Load existing optimized features
        existing_features, metadata = load_existing_optimized_features()
        existing_names = metadata.get('feature_names', [])
        
        # Load original price data for new feature calculation
        price_data, price_columns = load_original_price_data()
        
        # Calculate new feature categories
        all_new_features = {}
        
        # 1. Volatility features (missing from current set)
        volatility_features = calculate_volatility_features(price_data)
        all_new_features.update(volatility_features)
        
        # 2. Bollinger Bands features (complement RSI)
        bollinger_features = calculate_bollinger_bands(price_data)
        all_new_features.update(bollinger_features)
        
        # 3. MACD features (momentum indicator)
        macd_features = calculate_macd_features(price_data)
        all_new_features.update(macd_features)
        
        # 4. Regime detection features (market state awareness)
        regime_features = calculate_regime_detection_features(price_data)
        all_new_features.update(regime_features)
        
        # 5. Multi-timeframe features (different horizons)
        mtf_features = calculate_multi_timeframe_features(price_data)
        all_new_features.update(mtf_features)
        
        print(f"\n📊 FEATURE GENERATION SUMMARY:")
        print(f"   Total new features calculated: {len(all_new_features)}")
        print(f"   Categories: Volatility, Bollinger, MACD, Regime, Multi-timeframe")
        
        # Select best additional features
        selected_new_features, selected_names = select_best_features(
            all_new_features, existing_features, target_count=12
        )
        
        # Combine with existing features
        final_features, final_names = combine_feature_sets(
            existing_features, selected_new_features,
            existing_names, selected_names
        )
        
        # Save enhanced dataset
        enhanced_path = save_enhanced_dataset(final_features, final_names, metadata)
        
        print(f"\n🎉 ENHANCEMENT COMPLETE!")
        print(f"   Base features: {len(existing_names)} (Phase 2 optimized)")
        print(f"   Added features: {len(selected_names)} (Phase 3 enhanced)")
        print(f"   Total features: {len(final_names)}")
        print(f"   Dataset saved: {enhanced_path}")
        
        print(f"\n📋 FEATURE BREAKDOWN:")
        print(f"   Existing (8): {existing_names}")
        print(f"   Added (12): {selected_names}")
        
        return enhanced_path, final_names
        
    except Exception as e:
        print(f"❌ Error in feature enhancement: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    enhanced_path, feature_names = main()
    
    if enhanced_path:
        print(f"\n✅ Enhanced dataset ready for training!")
        print(f"   Use: {enhanced_path}")
        print(f"   Features: {len(feature_names)} total")
    else:
        print(f"\n❌ Feature enhancement failed!")