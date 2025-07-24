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
        """Compute feature importance using LightGBM"""
        try:
            import lightgbm as lgb
            
            # Prepare data
            train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
            
            # Parameters for feature selection
            params = {
                'objective': 'regression',
                'metric': 'mse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1
            }
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Get feature importance
            importance = model.feature_importance(importance_type='gain')
            
        except ImportError:
            print("LightGBM not available, using correlation-based importance")
            importance = self._correlation_importance(X, y)
        except Exception as e:
            print(f"LightGBM failed ({e}), using correlation-based importance")
            importance = self._correlation_importance(X, y)
            
        return importance
    
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