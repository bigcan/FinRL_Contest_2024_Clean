#!/usr/bin/env python3
"""
Microstructure Feature Generator v3
Combines existing optimized features with advanced microstructure features
and forces inclusion of proven technical indicators for maximum alpha generation.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from advanced_microstructure_features import LOBMicrostructureFeatures

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

def calculate_forced_technical_indicators(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Calculate the technical indicators that were filtered out but should be included
    These are proven indicators that provide unique value to neural networks
    """
    
    print("📈 Calculating forced technical indicators...")
    
    # Load price data
    df = pd.read_csv(csv_path)
    
    if 'midpoint' in df.columns:
        close_prices = df['midpoint'].values
    else:
        # Fallback to first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        close_prices = df[numeric_cols[0]].values
    
    features = {}
    
    # 1. Complete Bollinger Bands Suite (Force Include)
    print("   🎯 Bollinger Bands (20, 2) - Force Include")
    window = 20
    sma = pd.Series(close_prices).rolling(window=window).mean()
    std = pd.Series(close_prices).rolling(window=window).std()
    
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    
    # %B (Bollinger Band Position)
    bb_position = (close_prices - lower_band) / (upper_band - lower_band)
    bb_position = np.clip(bb_position, 0, 1)
    features['bb_position'] = bb_position.fillna(0.5).values
    
    # Bollinger Band Width (Bandwidth)
    bb_width = (upper_band - lower_band) / sma
    features['bb_width'] = bb_width.fillna(bb_width.mean()).values
    
    # Bollinger Band Squeeze
    bb_squeeze = (bb_width < bb_width.rolling(50).quantile(0.2)).astype(int)
    features['bb_squeeze'] = bb_squeeze.fillna(0).values
    
    # 2. Complete MACD Suite (Force Include)
    print("   🎯 MACD (12, 26, 9) - Force Include")
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
    
    # MACD cross signals (already exists, but ensure consistency)
    macd_cross = np.where(macd_line > signal_line, 1, -1)
    features['macd_cross'] = macd_cross
    
    # 3. ATR (Average True Range) - Enhanced Implementation
    print("   🎯 ATR (14) - Enhanced")
    
    # For LOB data, we'll use price volatility as proxy for True Range
    returns = np.diff(np.log(close_prices))
    returns = np.concatenate([[0], returns])
    
    # Calculate ATR using absolute returns (proxy for True Range)
    abs_returns = np.abs(returns)
    atr = pd.Series(abs_returns).rolling(window=14).mean()
    features['atr'] = atr.fillna(atr.mean()).values
    
    # ATR-based volatility regime
    atr_percentiles = np.percentile(features['atr'], [25, 75])
    atr_regime = np.where(features['atr'] > atr_percentiles[1], 1,  # High volatility
                         np.where(features['atr'] < atr_percentiles[0], -1, 0))  # Low volatility
    features['atr_regime'] = atr_regime
    
    print(f"   ✅ Created {len(features)} forced technical indicators")
    return features

def smart_feature_selection(all_features: Dict[str, np.ndarray], 
                          existing_features: np.ndarray,
                          existing_names: List[str],
                          forced_features: List[str],
                          target_additional: int = 15) -> Tuple[np.ndarray, List[str]]:
    """
    Smart feature selection with microstructure bias and forced inclusion
    
    Args:
        all_features: All available new features
        existing_features: Existing 8 optimized features
        existing_names: Names of existing features
        forced_features: Features that must be included
        target_additional: Target number of additional features
    """
    
    print(f"🎯 Smart feature selection (microstructure-aware)...")
    print(f"   Target additional features: {target_additional}")
    print(f"   Forced includes: {len(forced_features)}")
    
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
            if not np.isnan(corr):
                max_corr = max(max_corr, corr)
        correlations[new_feature] = max_corr
    
    # Calculate feature variance (importance proxy)
    variances = feature_df.var()
    
    # Smart scoring with microstructure bias
    scores = {}
    for feature in feature_df.columns:
        correlation_penalty = correlations[feature]
        variance_score = min(variances[feature] / variances.max(), 1.0) if variances.max() > 0 else 0
        
        # Base score (lower correlation + higher variance = better)
        base_score = variance_score * (1 - correlation_penalty)
        
        # Apply bonuses
        final_score = base_score
        
        # 2x bonus for microstructure features (OBI, microprice, spread, flow)
        microstructure_keywords = ['obi', 'microprice', 'spread', 'flow', 'liquidity']
        if any(keyword in feature.lower() for keyword in microstructure_keywords):
            final_score *= 2.0
            print(f"   🎯 2x Microstructure bonus: {feature}")
        
        # 1.5x bonus for proven technical indicators
        technical_keywords = ['atr', 'bb_', 'macd']
        if any(keyword in feature.lower() for keyword in technical_keywords):
            final_score *= 1.5
            print(f"   📈 1.5x Technical bonus: {feature}")
        
        scores[feature] = final_score
    
    # Force include specified features
    selected_features = forced_features.copy()
    print(f"   🔒 Force included: {forced_features}")
    
    # Select remaining features by score
    remaining_features = [f for f in scores.keys() if f not in forced_features]
    remaining_sorted = sorted(remaining_features, key=lambda x: scores[x], reverse=True)
    
    remaining_needed = target_additional - len(forced_features)
    selected_features.extend(remaining_sorted[:remaining_needed])
    
    # Create feature matrix
    selected_matrix = feature_df[selected_features].fillna(0).values
    
    print(f"   ✅ Selected {len(selected_features)} additional features")
    print(f"   📊 Average correlation with existing: {np.mean([correlations[f] for f in selected_features]):.3f}")
    
    return selected_matrix, selected_features

def create_microstructure_dataset():
    """Create the complete microstructure-enhanced dataset (v3)"""
    
    print("🚀 CREATING MICROSTRUCTURE-ENHANCED DATASET V3")
    print("=" * 70)
    
    try:
        # 1. Load existing optimized features (base 8)
        existing_features, existing_metadata = load_existing_optimized_features()
        existing_names = existing_metadata.get('feature_names', [])
        
        # 2. Generate advanced microstructure features
        print("\n🔬 PHASE 1: Advanced Microstructure Features")
        lob_features = LOBMicrostructureFeatures(max_levels=5)
        csv_path = "../../../data/raw/task1/BTC_1sec.csv"
        microstructure_dict, microstructure_names = lob_features.generate_all_microstructure_features(csv_path)
        
        # 3. Generate forced technical indicators
        print("\n⚡ PHASE 2: Forced Technical Indicators")
        technical_dict = calculate_forced_technical_indicators(csv_path)
        
        # 4. Combine all new features
        all_new_features = {**microstructure_dict, **technical_dict}
        print(f"\n📊 Total new features available: {len(all_new_features)}")
        
        # 5. Smart feature selection with forced inclusion
        forced_features = [
            # Complete Bollinger Bands
            'bb_position', 'bb_width', 'bb_squeeze',
            # Complete MACD  
            'macd_line', 'macd_histogram', 'macd_cross',
            # Enhanced ATR
            'atr', 'atr_regime',
            # Top microstructure features (highest alpha potential)
            'obi_3_level', 'obi_weighted', 'microprice', 
            'spread_volatility', 'order_flow_ratio'
        ]
        
        selected_new_features, selected_names = smart_feature_selection(
            all_new_features, existing_features, existing_names,
            forced_features, target_additional=15
        )
        
        # 6. Combine existing + new features
        print(f"\n🔄 PHASE 3: Feature Integration")
        
        # Ensure same length
        min_length = min(existing_features.shape[0], selected_new_features.shape[0])
        existing_features = existing_features[:min_length]
        selected_new_features = selected_new_features[:min_length]
        
        # Combine
        final_features = np.concatenate([existing_features, selected_new_features], axis=1)
        final_names = existing_names + selected_names
        
        print(f"   ✅ Final feature set: {final_features.shape}")
        print(f"   📊 Base features: {len(existing_names)}")
        print(f"   📊 Added features: {len(selected_names)}")
        print(f"   📊 Total features: {len(final_names)}")
        
        # 7. Save enhanced dataset
        print(f"\n💾 PHASE 4: Dataset Generation")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save enhanced features
        enhanced_path = f"../../../data/raw/task1/BTC_1sec_predict_microstructure_v3.npy"
        np.save(enhanced_path, final_features)
        
        # Save metadata
        enhanced_metadata = {
            'creation_date': datetime.now().strftime("%Y-%m-%d"),
            'phase': 'Phase 3 - Advanced Microstructure Features',
            'base_system': 'Phase 2 optimized 8-feature system',
            'enhancement_method': 'Microstructure + Forced Technical Indicators',
            'feature_names': final_names,
            'base_features': existing_names,
            'microstructure_features': [f for f in selected_names if any(k in f.lower() for k in ['obi', 'microprice', 'spread', 'flow'])],
            'technical_features': [f for f in selected_names if any(k in f.lower() for k in ['bb_', 'macd', 'atr'])],
            'forced_features': forced_features,
            'original_shape': existing_metadata.get('optimized_shape', 'Unknown'),
            'enhanced_shape': final_features.shape,
            'enhancement_count': len(selected_names),
            'selection_criteria': 'Microstructure bias + Technical indicators + Low correlation + High variance',
            'expected_improvement': 'Superior alpha from LOB signals + Proven technical indicators',
            'microstructure_bonus': '2x scoring weight for LOB features',
            'technical_bonus': '1.5x scoring weight for proven indicators'
        }
        
        metadata_path = f"../../../data/raw/task1/BTC_1sec_predict_microstructure_v3_metadata.npy"
        np.save(metadata_path, enhanced_metadata)
        
        print(f"   ✅ Saved to: {enhanced_path}")
        print(f"   📊 Final shape: {final_features.shape}")
        
        # 8. Feature breakdown summary
        print(f"\n🎉 MICROSTRUCTURE DATASET V3 COMPLETE!")
        print(f"   📋 Base Features (8): {existing_names}")
        print(f"   🔬 Microstructure Features: {enhanced_metadata['microstructure_features']}")
        print(f"   📈 Technical Features: {enhanced_metadata['technical_features']}")
        print(f"   🎯 Total Features: {len(final_names)}")
        
        return enhanced_path, final_names, enhanced_metadata
        
    except Exception as e:
        print(f"❌ Error in microstructure dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main execution"""
    
    enhanced_path, feature_names, metadata = create_microstructure_dataset()
    
    if enhanced_path:
        print(f"\n✅ Microstructure-enhanced dataset V3 created successfully!")
        print(f"   📁 Dataset: {enhanced_path}")
        print(f"   🎯 Features: {len(feature_names)}")
        print(f"   🚀 Ready for training with target Sharpe ratio >1.0")
    else:
        print(f"\n❌ Dataset creation failed!")

if __name__ == "__main__":
    main()