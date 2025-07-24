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
from .data_transformer import DataTransformer

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
        self.data_transformer = DataTransformer()
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
        
        # Compute data transformation features
        print("Computing data transformation features...")
        transform_feat_dict = self.data_transformer.transform_data(lob_data, apply_normalization=False)
        
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
        
        # Data transformation features
        for name, values in transform_feat_dict.items():
            feature_names.append(f'transform_{name}')
            all_features[f'transform_{name}'] = values
            
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
    
    def validate_features(self, feature_array, feature_names, verbose=True):
        """
        Comprehensive feature validation
        
        Args:
            feature_array: Feature array to validate
            feature_names: List of feature names
            verbose: Whether to print detailed validation results
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {
            'total_features': len(feature_names),
            'array_shape': feature_array.shape,
            'nan_count': 0,
            'inf_count': 0,
            'constant_features': [],
            'high_correlation_pairs': [],
            'feature_stats': {}
        }
        
        if verbose:
            print("=" * 60)
            print("FEATURE VALIDATION REPORT")
            print("=" * 60)
            print(f"Total features: {validation_results['total_features']}")
            print(f"Array shape: {validation_results['array_shape']}")
        
        # Check for NaN and Inf values
        nan_mask = np.isnan(feature_array)
        inf_mask = np.isinf(feature_array)
        
        validation_results['nan_count'] = np.sum(nan_mask)
        validation_results['inf_count'] = np.sum(inf_mask)
        
        if verbose:
            print(f"NaN values: {validation_results['nan_count']}")
            print(f"Inf values: {validation_results['inf_count']}")
        
        # Feature statistics
        for i, name in enumerate(feature_names):
            if i < feature_array.shape[1]:
                col_data = feature_array[:, i]
                stats = {
                    'mean': np.mean(col_data),
                    'std': np.std(col_data),
                    'min': np.min(col_data),
                    'max': np.max(col_data),
                    'nan_count': np.sum(np.isnan(col_data)),
                    'zero_count': np.sum(col_data == 0)
                }
                validation_results['feature_stats'][name] = stats
                
                # Check for constant features
                if stats['std'] < 1e-8:
                    validation_results['constant_features'].append(name)
        
        # Check for high correlations (excluding position features)
        if feature_array.shape[1] > 2:
            corr_matrix = np.corrcoef(feature_array[:, 2:].T)
            high_corr_threshold = 0.95
            
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > high_corr_threshold:
                        name1 = feature_names[i+2] if i+2 < len(feature_names) else f"feature_{i+2}"
                        name2 = feature_names[j+2] if j+2 < len(feature_names) else f"feature_{j+2}"
                        validation_results['high_correlation_pairs'].append(
                            (name1, name2, corr_matrix[i, j])
                        )
        
        if verbose:
            print(f"Constant features: {len(validation_results['constant_features'])}")
            if validation_results['constant_features']:
                for name in validation_results['constant_features'][:5]:
                    print(f"  - {name}")
                if len(validation_results['constant_features']) > 5:
                    print(f"  ... and {len(validation_results['constant_features']) - 5} more")
            
            print(f"High correlation pairs: {len(validation_results['high_correlation_pairs'])}")
            for name1, name2, corr in validation_results['high_correlation_pairs'][:3]:
                print(f"  - {name1} <-> {name2}: {corr:.3f}")
            if len(validation_results['high_correlation_pairs']) > 3:
                print(f"  ... and {len(validation_results['high_correlation_pairs']) - 3} more")
            
            print("\nTop feature statistics:")
            sorted_features = sorted(validation_results['feature_stats'].items(), 
                                   key=lambda x: abs(x[1]['std']), reverse=True)
            for name, stats in sorted_features[:5]:
                print(f"  {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            
            print("=" * 60)
        
        return validation_results
    
    def get_feature_metadata(self):
        """Get comprehensive feature metadata"""
        if self.feature_names is None:
            return None
        
        metadata = {
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_categories': self._categorize_features(),
            'normalization_params': self.normalization_params,
            'selected_features': self.selected_features
        }
        
        if hasattr(self.feature_selector, 'feature_importance') and self.feature_selector.feature_importance is not None:
            metadata['feature_importance'] = self.feature_selector.get_feature_rankings()
        
        return metadata
    
    def _categorize_features(self):
        """Categorize features by type"""
        categories = {
            'position': [],
            'technical': [],
            'lob': [],
            'transform': [],
            'original': []
        }
        
        for name in self.feature_names:
            if name.startswith('position') or name.startswith('holding'):
                categories['position'].append(name)
            elif name.startswith('tech_'):
                categories['technical'].append(name)
            elif name.startswith('lob_'):
                categories['lob'].append(name)
            elif name.startswith('transform_'):
                categories['transform'].append(name)
            elif name.startswith('original_'):
                categories['original'].append(name)
        
        return categories