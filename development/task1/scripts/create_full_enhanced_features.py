"""
Create Full Enhanced Features Dataset

Process the complete dataset efficiently by using the feature selection
results from the subset to create the full enhanced feature array.
"""

import os
import numpy as np
import pandas as pd
from development.shared.features import FeatureProcessor
from data_config import ConfigData

def main():
    """Create full enhanced features using subset results"""
    print("=" * 60)
    print("CREATING FULL ENHANCED FEATURES")
    print("=" * 60)
    
    # Check if subset results exist
    if not os.path.exists("enhanced_feature_names.npy"):
        print("❌ Run generate_enhanced_features_fast.py first to get feature selection")
        return
    
    # Load subset results
    selected_names = np.load("enhanced_feature_names.npy", allow_pickle=True).tolist()
    print(f"✓ Loaded {len(selected_names)} selected feature names")
    
    # Configuration
    config = ConfigData()
    
    print("Loading full dataset...")
    df = pd.read_csv(config.csv_path)
    predict_full = np.load(config.predict_ary_path)
    
    print(f"Full CSV shape: {df.shape}")
    print(f"Predict array shape: {predict_full.shape}")
    
    # Take the last part of CSV to align with predict array
    df_aligned = df.tail(len(predict_full)).copy()
    print(f"Aligned CSV shape: {df_aligned.shape}")
    
    # Initialize processor
    processor = FeatureProcessor(cache_dir="data", use_cache=False)
    
    # Process in chunks to avoid memory issues
    chunk_size = 50000
    total_rows = len(df_aligned)
    
    all_features = []
    
    print(f"\nProcessing {total_rows} rows in chunks of {chunk_size}...")
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_size_actual = end_idx - start_idx
        
        print(f"Processing chunk {start_idx}-{end_idx} ({chunk_size_actual} rows)...")
        
        # Extract chunk
        df_chunk = df_aligned.iloc[start_idx:end_idx].copy()
        predict_chunk = predict_full[start_idx:end_idx]
        
        # Compute features for chunk
        try:
            # Technical indicators
            price_data = df_chunk[['bids_distance_3', 'asks_distance_3', 'midpoint']].values
            tech_features = processor.tech_indicators.compute_indicators(price_data, None)
            
            # LOB features (simplified for speed)
            processor.lob_features.lookback_window = 10  # Reduce window for speed
            lob_features = processor.lob_features.compute_lob_features(df_chunk)
            
            # Combine features based on selected names
            chunk_features = np.zeros((chunk_size_actual, len(selected_names)))
            
            # Position features (will be filled at runtime)
            chunk_features[:, 0] = 0  # position_norm
            chunk_features[:, 1] = 0  # holding_norm
            
            # Fill selected features
            for i, name in enumerate(selected_names[2:], 2):  # Skip position features
                if name.startswith('tech_'):
                    tech_name = name[5:]  # Remove 'tech_' prefix
                    if tech_name in tech_features:
                        values = tech_features[tech_name]
                        chunk_features[:, i] = values[-chunk_size_actual:]  # Take last values
                elif name.startswith('lob_'):
                    lob_name = name[4:]  # Remove 'lob_' prefix
                    if lob_name in lob_features:
                        values = lob_features[lob_name]
                        chunk_features[:, i] = values[-chunk_size_actual:]  # Take last values
                elif name.startswith('original_'):
                    orig_idx = int(name.split('_')[1])
                    if orig_idx < predict_chunk.shape[1]:
                        chunk_features[:, i] = predict_chunk[:, orig_idx]
            
            all_features.append(chunk_features)
            print(f"✓ Processed chunk {start_idx}-{end_idx}")
            
        except Exception as e:
            print(f"❌ Error processing chunk {start_idx}-{end_idx}: {e}")
            # Create zero features for this chunk
            chunk_features = np.zeros((chunk_size_actual, len(selected_names)))
            all_features.append(chunk_features)
    
    # Combine all chunks
    print("\nCombining all chunks...")
    full_features = np.vstack(all_features)
    print(f"Combined features shape: {full_features.shape}")
    
    # Apply normalization (using statistics from subset)
    print("Applying normalization...")
    subset_features = np.load("enhanced_features_subset.npy")
    
    # Calculate normalization parameters from subset (excluding position features)
    for i in range(2, full_features.shape[1]):
        if i < subset_features.shape[1]:
            col_mean = np.mean(subset_features[:, i])
            col_std = np.std(subset_features[:, i])
            
            if col_std > 1e-8:
                full_features[:, i] = (full_features[:, i] - col_mean) / col_std
    
    # Save enhanced features
    enhanced_path = config.predict_ary_path.replace('.npy', '_enhanced.npy')
    np.save(enhanced_path, full_features)
    
    # Save metadata
    metadata = {
        'feature_names': selected_names,
        'original_shape': predict_full.shape,
        'enhanced_shape': full_features.shape,
        'state_dim': len(selected_names)
    }
    metadata_path = enhanced_path.replace('.npy', '_metadata.npy')
    np.save(metadata_path, metadata)
    
    print("\n" + "=" * 60)
    print("FULL ENHANCED FEATURES COMPLETE")
    print("=" * 60)
    print(f"✓ Enhanced features saved: {enhanced_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    print(f"✓ Original state_dim: 10")
    print(f"✓ Enhanced state_dim: {len(selected_names)}")
    print(f"✓ Shape: {full_features.shape}")
    print(f"✓ Selected features: {selected_names}")

if __name__ == "__main__":
    main()