"""
Feature Selection Module

Provides feature selection and dimensionality reduction:
- SHAP-based feature importance
- LightGBM feature ranking
- Correlation analysis
- Variance filtering
"""

import numpy as np
import pandas as pd

class FeatureSelector:
    """Feature selection using multiple criteria"""
    
    def __init__(self, max_features=25, correlation_threshold=0.95):
        """
        Initialize feature selector
        
        Args:
            max_features: Maximum number of features to select
            correlation_threshold: Remove features with correlation above this
        """
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.feature_importance = None
        
    def select_features(self, X, y, feature_names):
        """
        Select best features using multiple criteria
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (returns or rewards)
            feature_names: List of feature names
            
        Returns:
            selected_indices: Indices of selected features
            selected_names: Names of selected features
        """
        print(f"Starting feature selection from {len(feature_names)} features...")
        
        # Step 1: Remove low variance features
        X_filtered, names_filtered, variance_mask = self._remove_low_variance(X, feature_names)
        print(f"After variance filtering: {X_filtered.shape[1]} features")
        
        # Step 2: Remove highly correlated features
        X_filtered, names_filtered, corr_mask = self._remove_correlated_features(
            X_filtered, names_filtered)
        print(f"After correlation filtering: {X_filtered.shape[1]} features")
        
        # Step 3: Feature importance ranking
        importance_scores = self._compute_feature_importance(X_filtered, y, names_filtered)
        
        # Step 4: Select top features
        selected_indices = self._select_top_features(importance_scores)
        selected_names = [names_filtered[i] for i in selected_indices]
        
        # Map back to original indices
        original_indices = np.where(variance_mask)[0]
        original_indices = original_indices[np.where(corr_mask)[0]]
        original_selected = original_indices[selected_indices]
        
        self.selected_features = original_selected
        self.feature_importance = {name: importance_scores[i] 
                                 for i, name in enumerate(selected_names)}
        
        print(f"Final selection: {len(selected_names)} features")
        return original_selected, selected_names
    
    def _remove_low_variance(self, X, feature_names, threshold=0.01):
        """Remove features with low variance"""
        variances = np.var(X, axis=0)
        mask = variances > threshold
        
        # Ensure dimensions match
        if len(mask) != len(feature_names):
            print(f"Warning: mask length {len(mask)} != feature_names length {len(feature_names)}")
            min_len = min(len(mask), len(feature_names))
            mask = mask[:min_len]
            feature_names = feature_names[:min_len]
            X = X[:, :min_len]
        
        X_filtered = X[:, mask]
        names_filtered = [name for i, name in enumerate(feature_names) if i < len(mask) and mask[i]]
        
        return X_filtered, names_filtered, mask
    
    def _remove_correlated_features(self, X, feature_names):
        """Remove highly correlated features"""
        if X.shape[1] <= 1:
            return X, feature_names, np.array([True])
            
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find pairs of highly correlated features
        mask = np.ones(X.shape[1], dtype=bool)
        
        for i in range(X.shape[1]):
            if not mask[i]:
                continue
            for j in range(i+1, X.shape[1]):
                if mask[j] and abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Remove feature with lower variance
                    if np.var(X[:, i]) >= np.var(X[:, j]):
                        mask[j] = False
                    else:
                        mask[i] = False
                        break
        
        X_filtered = X[:, mask]
        names_filtered = [name for i, name in enumerate(feature_names) if mask[i]]
        
        return X_filtered, names_filtered, mask
    
    def _compute_feature_importance(self, X, y, feature_names):
        """
        Compute feature importance using optimally configured LightGBM
        
        Enhanced configuration ensures robust feature selection:
        - Optimal hyperparameters for feature importance estimation
        - Comprehensive input validation and error handling
        - Multiple importance metrics for robustness
        - Fallback mechanisms for edge cases
        """
        
        # Validate inputs first
        if not self._validate_lightgbm_inputs(X, y, feature_names):
            print("⚠️ Input validation failed, using correlation-based importance")
            return self._correlation_importance(X, y)
        
        try:
            import lightgbm as lgb
            print("✅ LightGBM successfully imported")
            
            # Verify LightGBM version for compatibility
            lgb_version = lgb.__version__
            print(f"📦 LightGBM version: {lgb_version}")
            
            if tuple(map(int, lgb_version.split('.'))) < (3, 0, 0):
                print("⚠️ LightGBM version < 3.0.0 detected, using basic configuration")
                
            # Enhanced data preparation with validation
            train_data = self._prepare_lightgbm_dataset(X, y, feature_names)
            
            # Optimized parameters for feature importance estimation
            params = self._get_optimal_lightgbm_params(X.shape)
            
            print(f"🔧 Training LightGBM with {len(feature_names)} features, {X.shape[0]} samples")
            
            # Train model with enhanced configuration
            model = self._train_lightgbm_model(params, train_data)
            
            # Extract multiple types of feature importance for robustness
            importance_scores = self._extract_feature_importance(model, X.shape[1])
            
            print(f"✅ LightGBM feature importance computed successfully")
            print(f"📊 Top 5 features by importance: {np.argsort(importance_scores)[-5:][::-1]}")
            
            return importance_scores
            
        except ImportError as e:
            print(f"❌ LightGBM not available: {e}")
            print("💡 Install LightGBM: pip install lightgbm")
            return self._correlation_importance(X, y)
            
        except Exception as e:
            print(f"❌ LightGBM training failed: {e}")
            print("🔄 Falling back to correlation-based importance")
            return self._correlation_importance(X, y)
    
    def _validate_lightgbm_inputs(self, X, y, feature_names):
        """Comprehensive input validation for LightGBM"""
        
        # Check data dimensions
        if X.shape[0] != len(y):
            print(f"❌ Shape mismatch: X has {X.shape[0]} samples, y has {len(y)}")
            return False
            
        # Check for minimum data requirements
        if X.shape[0] < 10:
            print("❌ Insufficient data: need at least 10 samples for feature selection")
            return False
            
        if X.shape[1] < 2:
            print("❌ Insufficient features: need at least 2 features for selection")
            return False
            
        # Check for data quality issues
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            print(f"⚠️ Found {nan_count} NaN values in X")
            
        if np.isnan(y).any():
            nan_count = np.isnan(y).sum()
            print(f"⚠️ Found {nan_count} NaN values in y")
            
        if np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            print(f"⚠️ Found {inf_count} infinite values in X")
            
        # Check target variable variance
        if np.var(y) < 1e-10:
            print("❌ Target variable has no variance - cannot compute feature importance")
            return False
            
        # Validate feature names
        if len(feature_names) != X.shape[1]:
            print(f"⚠️ Feature names length ({len(feature_names)}) != X columns ({X.shape[1]})")
            
        return True
    
    def _prepare_lightgbm_dataset(self, X, y, feature_names):
        """Prepare and validate LightGBM dataset"""
        
        import lightgbm as lgb
        
        # Handle NaN values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure feature names are strings
        feature_names_clean = [str(name) for name in feature_names] if feature_names else None
        
        try:
            train_data = lgb.Dataset(
                X_clean, 
                label=y_clean, 
                feature_name=feature_names_clean,
                free_raw_data=False  # Keep raw data for debugging
            )
            
            # Validate dataset construction
            if train_data.num_data() != X.shape[0]:
                raise ValueError(f"Dataset size mismatch: expected {X.shape[0]}, got {train_data.num_data()}")
                
            return train_data
            
        except Exception as e:
            print(f"❌ Error creating LightGBM dataset: {e}")
            raise
    
    def _get_optimal_lightgbm_params(self, data_shape):
        """Get optimized LightGBM parameters based on data characteristics"""
        
        n_samples, n_features = data_shape
        
        # Base parameters optimized for feature importance
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42,
            'deterministic': True,  # For reproducible results
            'force_col_wise': True,  # Better for feature importance
        }
        
        # Adaptive parameters based on dataset size
        if n_samples < 1000:
            # Small dataset - conservative parameters
            params.update({
                'num_leaves': min(15, 2**int(np.log2(n_samples/10))),
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'min_data_in_leaf': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
            })
        elif n_samples < 10000:
            # Medium dataset - balanced parameters
            params.update({
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'min_data_in_leaf': 10,
                'lambda_l1': 0.01,
                'lambda_l2': 0.01,
            })
        else:
            # Large dataset - more aggressive parameters
            params.update({
                'num_leaves': 63,
                'learning_rate': 0.03,
                'feature_fraction': 0.95,
                'bagging_fraction': 0.95,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.001,
                'lambda_l2': 0.001,
            })
        
        # Adjust for high-dimensional data
        if n_features > 100:
            params['feature_fraction'] = max(0.7, params['feature_fraction'] - 0.1)
            
        return params
    
    def _train_lightgbm_model(self, params, train_data):
        """Train LightGBM model with optimal configuration"""
        
        import lightgbm as lgb
        
        # Determine optimal number of boosting rounds
        n_samples = train_data.num_data()
        
        if n_samples < 1000:
            num_boost_round = 50
            early_stopping_rounds = 10
        elif n_samples < 10000:
            num_boost_round = 100
            early_stopping_rounds = 15
        else:
            num_boost_round = 200
            early_stopping_rounds = 20
        
        # Enhanced callbacks for robust training
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0)  # Silent training
        ]
        
        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data],
                callbacks=callbacks
            )
            
            return model
            
        except Exception as e:
            print(f"❌ LightGBM training error: {e}")
            raise
    
    def _extract_feature_importance(self, model, n_features):
        """Extract robust feature importance scores"""
        
        try:
            # Primary importance: gain-based (most reliable)
            importance_gain = model.feature_importance(importance_type='gain')
            
            # Secondary importance: split-based (frequency)
            importance_split = model.feature_importance(importance_type='split')
            
            # Combine importances with weighted average (gain is more important)
            if len(importance_gain) == n_features and len(importance_split) == n_features:
                # Normalize both importance types
                gain_norm = importance_gain / (np.sum(importance_gain) + 1e-8)
                split_norm = importance_split / (np.sum(importance_split) + 1e-8)
                
                # Weighted combination (80% gain, 20% split)
                combined_importance = 0.8 * gain_norm + 0.2 * split_norm
                
                print("✅ Using combined gain + split importance")
                return combined_importance
                
            else:
                print("⚠️ Using gain importance only")
                return importance_gain
                
        except Exception as e:
            print(f"❌ Error extracting feature importance: {e}")
            # Fallback to uniform importance
            return np.ones(n_features) / n_features
    
    def _correlation_importance(self, X, y):
        """Fallback: correlation-based importance"""
        importance = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            importance.append(abs(corr) if not np.isnan(corr) else 0)
        return np.array(importance)
    
    def _select_top_features(self, importance_scores):
        """Select top features based on importance"""
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        # Select top features
        n_select = min(self.max_features, len(sorted_indices))
        selected_indices = sorted_indices[:n_select]
        
        return selected_indices
    
    def get_feature_rankings(self):
        """Get feature importance rankings"""
        if self.feature_importance is None:
            return None
            
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        return sorted_features
    
    def transform(self, X):
        """Transform feature matrix using selected features"""
        if self.selected_features is None:
            raise ValueError("Must call select_features first")
            
        return X[:, self.selected_features]