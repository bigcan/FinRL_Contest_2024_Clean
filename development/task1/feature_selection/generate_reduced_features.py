"""
Generate Reduced Feature Set Based on Feature Selection Analysis
Creates new feature files with only the selected 15 features
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import sys
sys.path.append('../src')
from data_config import ConfigData

class ReducedFeatureGenerator:
    """
    Generate reduced feature set based on feature selection results
    """
    
    def __init__(self, selection_results_path: str = './feature_selection_results.json'):
        """
        Initialize with feature selection results
        
        Args:
            selection_results_path: Path to feature selection results JSON
        """
        # Load selection results
        with open(selection_results_path, 'r') as f:
            self.selection_results = json.load(f)
        
        self.selected_features = self.selection_results['selected_features']
        self.all_features = self.selection_results['feature_names']
        
        print(f"Loaded feature selection results:")
        print(f"  Original features: {len(self.all_features)}")
        print(f"  Selected features: {len(self.selected_features)}")
        
        # Get feature indices
        self.selected_indices = [self.all_features.index(feat) for feat in self.selected_features]
        
    def generate_reduced_features(self, 
                                input_path: str,
                                output_path: str,
                                metadata_path: str = None):
        """
        Generate reduced feature file from full feature file
        
        Args:
            input_path: Path to full feature file
            output_path: Path for reduced feature file
            metadata_path: Optional path for metadata
        """
        print(f"\nProcessing: {input_path}")
        
        # Load full features
        full_features = np.load(input_path)
        print(f"  Full shape: {full_features.shape}")
        
        # Extract selected features
        reduced_features = full_features[:, self.selected_indices]
        print(f"  Reduced shape: {reduced_features.shape}")
        
        # Save reduced features
        np.save(output_path, reduced_features)
        print(f"  Saved to: {output_path}")
        
        # Save metadata if requested
        if metadata_path:
            metadata = {
                'feature_names': self.selected_features,
                'n_features': len(self.selected_features),
                'original_features': len(self.all_features),
                'generation_timestamp': datetime.now().isoformat(),
                'selection_method': 'XGBoost + RF + MI with redundancy removal',
                'feature_indices': self.selected_indices
            }
            np.save(metadata_path, metadata)
            print(f"  Metadata saved to: {metadata_path}")
        
        return reduced_features
    
    def validate_reduced_features(self, reduced_features: np.ndarray):
        """
        Validate the reduced feature array
        
        Args:
            reduced_features: The reduced feature array
        """
        print("\nValidating reduced features:")
        
        # Check shape
        assert reduced_features.shape[1] == len(self.selected_features), \
            f"Expected {len(self.selected_features)} features, got {reduced_features.shape[1]}"
        
        # Check for NaN/Inf
        n_nan = np.sum(np.isnan(reduced_features))
        n_inf = np.sum(np.isinf(reduced_features))
        
        print(f"  NaN values: {n_nan}")
        print(f"  Inf values: {n_inf}")
        
        # Basic statistics
        print(f"  Mean: {np.mean(reduced_features):.6f}")
        print(f"  Std: {np.std(reduced_features):.6f}")
        print(f"  Min: {np.min(reduced_features):.6f}")
        print(f"  Max: {np.max(reduced_features):.6f}")
        
        # Feature-wise statistics
        print("\n  Feature statistics:")
        for i, feat_name in enumerate(self.selected_features):
            feat_data = reduced_features[:, i]
            print(f"    {feat_name}: mean={np.mean(feat_data):.4f}, std={np.std(feat_data):.4f}")
    
    def generate_all_splits(self):
        """
        Generate reduced features for all data splits (train, validation, test)
        """
        print("="*60)
        print("Generating Reduced Feature Sets")
        print("="*60)
        
        # Base paths
        base_path = '../../../data/raw/task1/'
        
        # Process enhanced_v3 features
        files_to_process = [
            {
                'input': os.path.join(base_path, 'BTC_1sec_predict_enhanced_v3.npy'),
                'output': os.path.join(base_path, 'BTC_1sec_predict_reduced.npy'),
                'metadata': os.path.join(base_path, 'BTC_1sec_predict_reduced_metadata.npy')
            }
        ]
        
        # Check for validation splits
        val_split_path = os.path.join(base_path, 'validation_splits')
        if os.path.exists(val_split_path):
            # Training split
            if os.path.exists(os.path.join(val_split_path, 'train_features_enhanced_v3.npy')):
                files_to_process.append({
                    'input': os.path.join(val_split_path, 'train_features_enhanced_v3.npy'),
                    'output': os.path.join(val_split_path, 'train_features_reduced.npy'),
                    'metadata': None
                })
            
            # Validation split
            if os.path.exists(os.path.join(val_split_path, 'val_features_enhanced_v3.npy')):
                files_to_process.append({
                    'input': os.path.join(val_split_path, 'val_features_enhanced_v3.npy'),
                    'output': os.path.join(val_split_path, 'val_features_reduced.npy'),
                    'metadata': None
                })
        
        # Process each file
        all_reduced = []
        for file_info in files_to_process:
            if os.path.exists(file_info['input']):
                reduced = self.generate_reduced_features(
                    file_info['input'],
                    file_info['output'],
                    file_info['metadata']
                )
                all_reduced.append(reduced)
            else:
                print(f"Warning: File not found: {file_info['input']}")
        
        # Validate the main file
        if all_reduced:
            self.validate_reduced_features(all_reduced[0])
        
        print("\n" + "="*60)
        print("Reduced feature generation complete!")
        print("="*60)
        
        # Print summary
        print("\nSummary of selected features:")
        for i, feat in enumerate(self.selected_features):
            print(f"  {i+1:2d}. {feat}")
        
        print(f"\nReduction: {len(self.all_features)} → {len(self.selected_features)} features")
        print(f"Compression: {(1 - len(self.selected_features)/len(self.all_features))*100:.1f}%")
        
        # Update configs to use reduced features
        self.update_configs()
    
    def update_configs(self):
        """
        Create configuration updates for using reduced features
        """
        config_update = f"""
# Configuration Update for Reduced Features

To use the reduced feature set, update the following in your training scripts:

## In trade_simulator.py:

Replace the feature loading priority with:
```python
# Priority loading: reduced > enhanced_v3 > optimized > enhanced > original
reduced_path = args.predict_ary_path.replace('.npy', '_reduced.npy')

if os.path.exists(reduced_path):
    print(f"Loading reduced features from {{reduced_path}}")
    self.factor_ary = np.load(reduced_path)
    
    # Load metadata for reduced features
    metadata_path = reduced_path.replace('.npy', '_metadata.npy')
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.feature_names = metadata.get('feature_names', [])
        print(f"Reduced features loaded: {{len(self.feature_names)}} features")
```

## Selected Features ({len(self.selected_features)}):
{chr(10).join([f'- {feat}' for feat in self.selected_features])}

## Performance Benefits:
- Reduced training time (fewer features)
- Less overfitting (removed redundant features)
- Better generalization (focused on most predictive features)
- Lower memory usage
"""
        
        config_path = './reduced_features_config.md'
        with open(config_path, 'w') as f:
            f.write(config_update)
        
        print(f"\nConfiguration guide saved to: {config_path}")


if __name__ == "__main__":
    # Generate reduced features
    generator = ReducedFeatureGenerator()
    generator.generate_all_splits()