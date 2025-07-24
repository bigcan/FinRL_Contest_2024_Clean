"""
Generate Enhanced Features for FinRL Bitcoin Trading

This script processes the raw Bitcoin LOB data to create an enhanced feature set
including technical indicators and LOB-specific features.
"""

import os
import sys
import numpy as np
import pandas as pd
from development.shared.features import FeatureProcessor
from data_config import ConfigData

def main():
    """Generate enhanced features"""
    print("=" * 60)
    print("ENHANCED FEATURE GENERATION")
    print("=" * 60)
    
    # Configuration
    config = ConfigData()
    
    # Check if data files exist
    if not os.path.exists(config.csv_path):
        print(f"❌ CSV data not found: {config.csv_path}")
        print("Please download the data first.")
        return
    
    if not os.path.exists(config.predict_ary_path):
        print(f"❌ Predict array not found: {config.predict_ary_path}")
        print("Please download the data first.")
        return
    
    print(f"✓ CSV data found: {config.csv_path}")
    print(f"✓ Predict array found: {config.predict_ary_path}")
    
    # Initialize feature processor
    processor = FeatureProcessor(cache_dir="data", use_cache=True)
    
    # Step 1: Compute all features
    print("\n" + "=" * 40)
    print("STEP 1: COMPUTING ALL FEATURES")
    print("=" * 40)
    
    try:
        feature_array, feature_names = processor.compute_all_features(
            csv_path=config.csv_path,
            predict_path=config.predict_ary_path,
            force_recompute=False
        )
        
        print(f"✓ Computed {feature_array.shape[1]} raw features")
        print(f"✓ Data length: {feature_array.shape[0]} timesteps")
        
    except Exception as e:
        print(f"❌ Failed to compute features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Feature selection and processing
    print("\n" + "=" * 40)
    print("STEP 2: FEATURE SELECTION")
    print("=" * 40)
    
    try:
        # Create proxy target using price changes
        price_col = None
        for i, name in enumerate(feature_names):
            if 'midpoint' in name or 'original_' in name:
                price_col = i - 2  # Adjust for position features
                break
        
        if price_col is not None and price_col < feature_array.shape[1]:
            # Use price returns as target
            price_data = feature_array[:, price_col + 2]  # +2 for position features
            returns = np.diff(price_data) / price_data[:-1]
            returns = np.concatenate([[0], returns])  # Align lengths
        else:
            # Use random target (less optimal but works)
            print("Warning: Using random target for feature selection")
            returns = np.random.randn(len(feature_array)) * 0.01
        
        processed_array, selected_names = processor.select_and_process_features(
            feature_array=feature_array,
            target_returns=returns,
            max_features=27,  # 2 position + 25 selected
            save_path=config.predict_ary_path
        )
        
        print(f"✓ Selected {len(selected_names)} features")
        print(f"✓ Final array shape: {processed_array.shape}")
        
    except Exception as e:
        print(f"❌ Failed to select features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Generate feature importance report
    print("\n" + "=" * 40)
    print("STEP 3: FEATURE ANALYSIS")
    print("=" * 40)
    
    try:
        report = processor.get_feature_importance_report()
        print(report)
        
        # Save report
        with open("enhanced_features_report.txt", "w") as f:
            f.write("Enhanced Features Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original features: {len(feature_names)}\n")
            f.write(f"Selected features: {len(selected_names)}\n")
            f.write(f"Data length: {processed_array.shape[0]}\n\n")
            f.write(report)
            f.write("\n\nSelected Features:\n")
            f.write("-" * 30 + "\n")
            for i, name in enumerate(selected_names):
                f.write(f"{i+1:2d}. {name}\n")
        
        print("✓ Feature report saved to enhanced_features_report.txt")
        
    except Exception as e:
        print(f"Warning: Could not generate feature report: {e}")
    
    # Step 4: Validation
    print("\n" + "=" * 40)
    print("STEP 4: VALIDATION")
    print("=" * 40)
    
    try:
        # Check for NaN values
        nan_count = np.isnan(processed_array).sum()
        if nan_count > 0:
            print(f"⚠️  Warning: {nan_count} NaN values found")
        else:
            print("✓ No NaN values detected")
        
        # Check feature statistics
        print(f"✓ Feature value range: [{processed_array.min():.3f}, {processed_array.max():.3f}]")
        print(f"✓ Feature mean: {processed_array.mean():.3f}")
        print(f"✓ Feature std: {processed_array.std():.3f}")
        
        # Test compatibility with existing simulator
        expected_shape = (processed_array.shape[0], 27)  # 2 + 25
        if processed_array.shape[1] != expected_shape[1]:
            print(f"⚠️  Warning: Expected {expected_shape[1]} features, got {processed_array.shape[1]}")
        else:
            print(f"✓ Feature dimensions compatible with simulator")
        
    except Exception as e:
        print(f"⚠️  Validation warning: {e}")
    
    print("\n" + "=" * 60)
    print("ENHANCED FEATURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"✓ Enhanced features saved")
    print(f"✓ Original state_dim: 10 (2 + 8)")
    print(f"✓ Enhanced state_dim: {processed_array.shape[1]} (2 + {processed_array.shape[1]-2})")
    print(f"✓ Feature improvement: {processed_array.shape[1]/10:.1f}x more features")
    print("\nNext steps:")
    print("1. Update TradeSimulator to use enhanced features")
    print("2. Update state_dim in all configurations")
    print("3. Retrain ensemble models")

if __name__ == "__main__":
    main()