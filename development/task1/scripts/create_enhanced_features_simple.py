"""
Simple Enhanced Features Creation

Create enhanced features by combining the most important ones from the subset analysis.
"""

import os
import numpy as np
import pandas as pd
from data_config import ConfigData

def main():
    """Create simplified enhanced features"""
    print("=" * 60)
    print("CREATING SIMPLIFIED ENHANCED FEATURES")
    print("=" * 60)
    
    config = ConfigData()
    
    # Load original data
    predict_original = np.load(config.predict_ary_path)
    print(f"Original features shape: {predict_original.shape}")
    
    # Load CSV for additional features
    print("Loading CSV for technical indicators...")
    df = pd.read_csv(config.csv_path)
    df_aligned = df.tail(len(predict_original)).copy()
    
    # Extract key features from CSV
    midpoint = df_aligned['midpoint'].values
    spread = df_aligned['spread'].values
    buys = df_aligned['buys'].values
    sells = df_aligned['sells'].values
    
    print("Computing simplified technical indicators...")
    
    # Technical indicators (simplified)
    def ema(data, period):
        alpha = 2.0 / (period + 1)
        ema_vals = np.zeros_like(data)
        ema_vals[0] = data[0]
        for i in range(1, len(data)):
            ema_vals[i] = alpha * data[i] + (1 - alpha) * ema_vals[i-1]
        return ema_vals
    
    def rsi(data, period=14):
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(data)
        avg_losses = np.zeros_like(data)
        
        for i in range(period, len(data)):
            avg_gains[i] = np.mean(gains[max(0, i-period):i])
            avg_losses[i] = np.mean(losses[max(0, i-period):i])
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals / 100.0  # Normalize
    
    # Compute technical features
    ema_20 = ema(midpoint, 20)
    ema_50 = ema(midpoint, 50)
    rsi_14 = rsi(midpoint, 14)
    
    # Price momentum
    returns = np.diff(midpoint) / midpoint[:-1]
    returns = np.concatenate([[0], returns])
    
    # Moving averages of returns
    momentum_5 = np.zeros_like(returns)
    momentum_20 = np.zeros_like(returns)
    
    for i in range(len(returns)):
        start_5 = max(0, i-4)
        start_20 = max(0, i-19)
        momentum_5[i] = np.mean(returns[start_5:i+1])
        momentum_20[i] = np.mean(returns[start_20:i+1])
    
    # LOB features (simplified)
    spread_norm = (spread - np.mean(spread)) / (np.std(spread) + 1e-8)
    
    # Trade imbalance
    trade_imbalance = (buys - sells) / (buys + sells + 1e-8)
    
    # Order flow (approximation)
    order_flow_5 = np.zeros_like(trade_imbalance)
    for i in range(len(trade_imbalance)):
        start_idx = max(0, i-4)
        order_flow_5[i] = np.mean(trade_imbalance[start_idx:i+1])
    
    # Combine all enhanced features
    enhanced_features = []
    
    # Position features (filled at runtime)
    enhanced_features.append(np.zeros_like(midpoint))  # position_norm
    enhanced_features.append(np.zeros_like(midpoint))  # holding_norm
    
    # Technical indicators (normalized)
    enhanced_features.append((ema_20 - np.mean(ema_20)) / (np.std(ema_20) + 1e-8))
    enhanced_features.append((ema_50 - np.mean(ema_50)) / (np.std(ema_50) + 1e-8))
    enhanced_features.append(rsi_14)
    enhanced_features.append((momentum_5 - np.mean(momentum_5)) / (np.std(momentum_5) + 1e-8))
    enhanced_features.append((momentum_20 - np.mean(momentum_20)) / (np.std(momentum_20) + 1e-8))
    
    # LOB features (normalized)
    enhanced_features.append(spread_norm)
    enhanced_features.append((trade_imbalance - np.mean(trade_imbalance)) / (np.std(trade_imbalance) + 1e-8))
    enhanced_features.append((order_flow_5 - np.mean(order_flow_5)) / (np.std(order_flow_5) + 1e-8))
    
    # EMA crossover signal
    ema_cross = (ema_20 > ema_50).astype(float)
    enhanced_features.append(ema_cross)
    
    # Original features (most important ones)
    for i in [0, 1, 2, 4, 5]:  # Based on subset analysis
        if i < predict_original.shape[1]:
            original_norm = (predict_original[:, i] - np.mean(predict_original[:, i])) / (np.std(predict_original[:, i]) + 1e-8)
            enhanced_features.append(original_norm)
    
    # Combine into final array
    enhanced_array = np.column_stack(enhanced_features)
    
    # Feature names
    feature_names = [
        'position_norm', 'holding_norm',
        'ema_20', 'ema_50', 'rsi_14', 'momentum_5', 'momentum_20',
        'spread_norm', 'trade_imbalance', 'order_flow_5', 'ema_crossover',
        'original_0', 'original_1', 'original_2', 'original_4', 'original_5'
    ]
    
    print(f"Enhanced features shape: {enhanced_array.shape}")
    print(f"Feature names ({len(feature_names)}): {feature_names}")
    
    # Validation
    nan_count = np.isnan(enhanced_array).sum()
    if nan_count > 0:
        print(f"⚠️  Warning: {nan_count} NaN values found, replacing with 0")
        enhanced_array[np.isnan(enhanced_array)] = 0
    
    print(f"✓ Value range: [{enhanced_array.min():.3f}, {enhanced_array.max():.3f}]")
    print(f"✓ Mean: {enhanced_array.mean():.3f}")
    print(f"✓ Std: {enhanced_array.std():.3f}")
    
    # Save enhanced features
    enhanced_path = config.predict_ary_path.replace('.npy', '_enhanced.npy')
    np.save(enhanced_path, enhanced_array)
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'state_dim': len(feature_names),
        'original_shape': predict_original.shape,
        'enhanced_shape': enhanced_array.shape
    }
    metadata_path = enhanced_path.replace('.npy', '_metadata.npy')
    np.save(metadata_path, metadata)
    
    print("\n" + "=" * 60)
    print("SIMPLIFIED ENHANCED FEATURES COMPLETE")
    print("=" * 60)
    print(f"✓ Enhanced features saved: {enhanced_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    print(f"✓ Original state_dim: 10")
    print(f"✓ Enhanced state_dim: {len(feature_names)}")
    print(f"✓ Ready for TradeSimulator integration")

if __name__ == "__main__":
    main()