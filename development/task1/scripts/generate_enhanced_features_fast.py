"""
Fast Enhanced Feature Generation for FinRL Bitcoin Trading

This script processes a subset of the data to quickly test feature generation.
"""

import os
import sys
import numpy as np
import pandas as pd
from development.shared.features import FeatureProcessor
from data_config import ConfigData

def main():
    """Generate enhanced features on subset"""
    print("=" * 60)
    print("FAST ENHANCED FEATURE GENERATION (SUBSET)")
    print("=" * 60)
    
    # Configuration
    config = ConfigData()
    
    # Load subset of data (last 10000 rows to align with predict array)
    print("Loading subset of data...")
    df = pd.read_csv(config.csv_path)
    print(f"Full data shape: {df.shape}")
    
    # Take last 10000 rows
    subset_size = 10000
    df_subset = df.tail(subset_size).copy()
    print(f"Subset shape: {df_subset.shape}")
    
    # Load corresponding predict array
    predict_full = np.load(config.predict_ary_path)
    predict_subset = predict_full[-subset_size:].copy()
    print(f"Predict subset shape: {predict_subset.shape}")
    
    # Save temporary files
    temp_csv = "temp_subset.csv"
    temp_predict = "temp_subset.npy"
    
    df_subset.to_csv(temp_csv, index=False)
    np.save(temp_predict, predict_subset)
    
    try:
        # Initialize feature processor
        processor = FeatureProcessor(cache_dir="data", use_cache=False)
        
        # Step 1: Compute all features
        print("\n" + "=" * 40)
        print("STEP 1: COMPUTING ALL FEATURES")
        print("=" * 40)
        
        feature_array, feature_names = processor.compute_all_features(
            csv_path=temp_csv,
            predict_path=temp_predict,
            force_recompute=True
        )
        
        print(f"✓ Computed {feature_array.shape[1]} raw features")
        print(f"✓ Data length: {feature_array.shape[0]} timesteps")
        
        # Step 2: Feature selection and processing
        print("\n" + "=" * 40)
        print("STEP 2: FEATURE SELECTION")
        print("=" * 40)
        
        # Create target using simple price returns
        if 'midpoint' in df_subset.columns:
            price_data = df_subset['midpoint'].values[-len(feature_array):]
            returns = np.diff(price_data) / price_data[:-1]
            returns = np.concatenate([[0], returns])
        else:
            returns = np.random.randn(len(feature_array)) * 0.01
        
        processed_array, selected_names = processor.select_and_process_features(
            feature_array=feature_array,
            target_returns=returns,
            max_features=27,
            save_path=None
        )
        
        print(f"✓ Selected {len(selected_names)} features")
        print(f"✓ Final array shape: {processed_array.shape}")
        
        # Step 3: Generate report
        print("\n" + "=" * 40)
        print("STEP 3: FEATURE ANALYSIS")
        print("=" * 40)
        
        report = processor.get_feature_importance_report()
        print(report)
        
        # Step 4: Validation
        print("\n" + "=" * 40)
        print("STEP 4: VALIDATION")
        print("=" * 40)
        
        nan_count = np.isnan(processed_array).sum()
        if nan_count > 0:
            print(f"⚠️  Warning: {nan_count} NaN values found")
        else:
            print("✓ No NaN values detected")
        
        print(f"✓ Feature value range: [{processed_array.min():.3f}, {processed_array.max():.3f}]")
        print(f"✓ Feature dimensions: {processed_array.shape}")
        
        # Save subset results for full processing
        np.save("enhanced_features_subset.npy", processed_array)
        np.save("enhanced_feature_names.npy", np.array(selected_names))
        
        print("\n" + "=" * 60)
        print("FAST FEATURE GENERATION COMPLETE")
        print("=" * 60)
        print(f"✓ Enhanced features saved to enhanced_features_subset.npy")
        print(f"✓ Selected features: {len(selected_names)}")
        print(f"✓ Ready for full dataset processing")
        
    finally:
        # Cleanup
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        if os.path.exists(temp_predict):
            os.remove(temp_predict)

if __name__ == "__main__":
    main()