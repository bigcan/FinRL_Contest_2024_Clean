"""
Feature Processor Module

Main interface for feature engineering pipeline:
- Orchestrates technical indicators and LOB features
- Handles data preprocessing and normalization
- Integrates with feature selection
- Provides caching and persistence
"""

import numpy as np
import pandas as pd
import os
from .technical_indicators import TechnicalIndicators
from .lob_features import LOBFeatures
from .feature_selector import FeatureSelector

class FeatureProcessor:
    """Main feature processing pipeline"""
    
    def __init__(self, cache_dir="data", use_cache=True):
        """
        Initialize feature processor
        
        Args:
            cache_dir: Directory to cache computed features
            use_cache: Whether to use cached features if available
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Initialize components
        self.tech_indicators = TechnicalIndicators()
        self.lob_features = LOBFeatures()
        self.feature_selector = FeatureSelector(max_features=25)
        
        # State
        self.feature_names = None
        self.selected_features = None
        self.normalization_params = None
        
    def compute_all_features(self, csv_path, predict_path=None, force_recompute=False):
        """
        Compute all features from raw data
        
        Args:
            csv_path: Path to raw CSV data
            predict_path: Path to existing predict array (optional)
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            feature_array: Combined feature array (n_steps, n_features)
            feature_names: List of feature names
        """
        cache_path = os.path.join(self.cache_dir, "BTC_1sec_features_raw.npy")
        names_path = os.path.join(self.cache_dir, "BTC_1sec_feature_names.npy")
        
        # Check cache
        if self.use_cache and not force_recompute and os.path.exists(cache_path):
            print("Loading features from cache...")
            feature_array = np.load(cache_path)
            self.feature_names = np.load(names_path, allow_pickle=True).tolist()
            return feature_array, self.feature_names
        
        print("Computing features from raw data...")
        
        # Load raw data
        print("Loading raw data...")
        lob_data = pd.read_csv(csv_path)
        print(f"Loaded {len(lob_data)} rows with {len(lob_data.columns)} columns")
        
        # Load existing features if available
        original_features = None
        if predict_path and os.path.exists(predict_path):
            original_features = np.load(predict_path)
            print(f"Loaded original features: {original_features.shape}")
        
        # Compute technical indicators
        print("Computing technical indicators...")
        price_data = lob_data[['bids_distance_3', 'asks_distance_3', 'midpoint']].values
        volume_data = None  # Will use synthetic volume from LOB
        tech_features = self.tech_indicators.compute_indicators(price_data, volume_data)
        
        # Compute LOB features
        print("Computing LOB features...")
        lob_feat_dict = self.lob_features.compute_lob_features(lob_data)
        
        # Combine all features
        all_features = {}
        
        # Position features (will be added during runtime)
        feature_names = ['position_norm', 'holding_norm']
        
        # Technical indicators
        for name, values in tech_features.items():
            feature_names.append(f'tech_{name}')
            all_features[f'tech_{name}'] = values
            
        # LOB features
        for name, values in lob_feat_dict.items():
            feature_names.append(f'lob_{name}')
            all_features[f'lob_{name}'] = values
            
        # Original features
        if original_features is not None:
            for i in range(original_features.shape[1]):
                feature_names.append(f'original_{i}')
                all_features[f'original_{i}'] = original_features[:, i]
        
        # Align all features to same length
        min_length = min(len(values) for values in all_features.values())
        print(f"Aligning features to length: {min_length}")
        
        # Create feature matrix (excluding position features for now)
        feature_matrix = []
        aligned_names = []
        
        for name in feature_names[2:]:  # Skip position features
            if name in all_features:
                values = all_features[name][-min_length:]  # Take last min_length values
                feature_matrix.append(values)
                aligned_names.append(name)
        
        feature_array = np.column_stack(feature_matrix)
        self.feature_names = ['position_norm', 'holding_norm'] + aligned_names
        
        print(f"Computed feature array shape: {feature_array.shape}")
        print(f"Total features: {len(self.feature_names)}")
        
        # Cache results
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_path, feature_array)
            np.save(names_path, np.array(self.feature_names))
            print(f"Cached features to {cache_path}")
        
        return feature_array, self.feature_names
    
    def select_and_process_features(self, feature_array, target_returns=None, 
                                  max_features=25, save_path=None):
        """
        Select best features and create final processed array
        
        Args:
            feature_array: Raw feature array (n_steps, n_features)
            target_returns: Target returns for feature selection
            max_features: Maximum number of features to select
            save_path: Path to save processed features
            
        Returns:
            processed_array: Selected and normalized features
            selected_names: Names of selected features
        """
        print("Starting feature selection and processing...")
        
        if target_returns is None:
            # Use midpoint returns as proxy target
            print("Using midpoint returns as target for feature selection")
            target_returns = np.diff(feature_array[:, -1]) / feature_array[:-1, -1]  # Assuming last feature is price-related
            feature_array = feature_array[1:]  # Align lengths
        
        # Feature selection (excluding position features)
        X_for_selection = feature_array[:, 2:]  # Skip position features
        names_for_selection = self.feature_names[2:]
        
        self.feature_selector.max_features = max_features - 2  # Reserve 2 for position features
        selected_indices, selected_names = self.feature_selector.select_features(
            X_for_selection, target_returns, names_for_selection)
        
        # Combine position features with selected features
        position_features = np.zeros((len(feature_array), 2))  # Will be filled at runtime
        selected_feature_array = feature_array[:, 2:][:, selected_indices]
        
        # Final feature array
        final_array = np.column_stack([position_features, selected_feature_array])
        final_names = ['position_norm', 'holding_norm'] + selected_names
        
        # Normalize features (except position features which are normalized at runtime)
        self.normalization_params = {}
        for i in range(2, final_array.shape[1]):
            col_data = final_array[:, i]
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            
            if std_val > 1e-8:
                final_array[:, i] = (col_data - mean_val) / std_val
                self.normalization_params[final_names[i]] = {'mean': mean_val, 'std': std_val}
            else:
                self.normalization_params[final_names[i]] = {'mean': mean_val, 'std': 1.0}
        
        self.selected_features = list(range(len(final_names)))
        
        print(f"Final processed features: {final_array.shape}")
        print(f"Selected features: {final_names}")
        
        # Save processed features
        if save_path:
            enhanced_path = save_path.replace('.npy', '_enhanced.npy')
            np.save(enhanced_path, final_array)
            
            # Save metadata
            metadata = {
                'feature_names': final_names,
                'normalization_params': self.normalization_params,
                'selected_indices': self.selected_features,
                'feature_importance': self.feature_selector.get_feature_rankings()
            }
            metadata_path = save_path.replace('.npy', '_metadata.npy')
            np.save(metadata_path, metadata)
            
            print(f"Saved enhanced features to {enhanced_path}")
            print(f"Saved metadata to {metadata_path}")
        
        return final_array, final_names
    
    def get_feature_importance_report(self):
        """Generate feature importance report"""
        if self.feature_selector.feature_importance is None:
            return "Feature selection has not been run yet."
        
        rankings = self.feature_selector.get_feature_rankings()
        
        report = "Feature Importance Rankings:\n"
        report += "=" * 50 + "\n"
        
        for i, (name, importance) in enumerate(rankings[:20], 1):
            report += f"{i:2d}. {name:<30} {importance:.4f}\n"
        
        return report
    
    def load_processed_features(self, enhanced_path):
        """Load previously processed features"""
        if not os.path.exists(enhanced_path):
            raise FileNotFoundError(f"Enhanced features not found: {enhanced_path}")
        
        # Load features
        feature_array = np.load(enhanced_path)
        
        # Load metadata
        metadata_path = enhanced_path.replace('_enhanced.npy', '_metadata.npy')
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()
            self.feature_names = metadata['feature_names']
            self.normalization_params = metadata['normalization_params']
            self.selected_features = metadata['selected_indices']
        
        print(f"Loaded processed features: {feature_array.shape}")
        return feature_array, self.feature_names
    
    def normalize_features_runtime(self, features, feature_names):
        """Normalize features at runtime using stored parameters"""
        if self.normalization_params is None:
            return features
        
        normalized = features.copy()
        for i, name in enumerate(feature_names):
            if name in self.normalization_params:
                params = self.normalization_params[name]
                normalized[:, i] = (features[:, i] - params['mean']) / params['std']
        
        return normalized