"""
Feature Selection Analysis for FinRL Contest 2024
Reduce from 41 features to 10-15 most predictive, non-redundant features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
import os
from datetime import datetime
from typing import Dict, List, Tuple
import json

warnings.filterwarnings('ignore')

class FeatureSelector:
    """
    Comprehensive feature selection for high-frequency trading
    """
    
    def __init__(self, data_path: str = '../../../data/raw/task1/', 
                 output_path: str = './', 
                 n_target_features: int = 15):
        """
        Initialize feature selector
        
        Args:
            data_path: Path to raw data files
            output_path: Path for output files and visualizations
            n_target_features: Target number of features to select
        """
        self.data_path = data_path
        self.output_path = output_path
        self.n_target_features = n_target_features
        self.feature_names = []
        self.selected_features = []
        self.analysis_results = {}
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load feature data and create target variable
        
        Returns:
            features: Feature array
            target: Target variable (1-minute ahead return direction)
            feature_names: List of feature names
        """
        print("Loading enhanced features v3...")
        
        # Load enhanced features
        feature_path = os.path.join(self.data_path, 'BTC_1sec_predict_enhanced_v3.npy')
        features = np.load(feature_path)
        
        # Load metadata
        metadata_path = os.path.join(self.data_path, 'BTC_1sec_predict_enhanced_v3_metadata.npy')
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.feature_names = metadata.get('feature_names', [])
        
        # Load price data for target calculation
        csv_path = os.path.join(self.data_path, 'BTC_1sec.csv')
        price_df = pd.read_csv(csv_path)
        
        # Calculate 1-minute ahead returns
        midpoint = price_df['midpoint'].values
        returns_1min = (midpoint[60:] - midpoint[:-60]) / midpoint[:-60]
        
        # Create binary target (1 for positive returns, 0 for negative)
        target = (returns_1min > 0).astype(int)
        
        # Align features with target (remove last 60 observations from features)
        features_aligned = features[:-60]
        
        # Ensure alignment
        min_len = min(len(features_aligned), len(target))
        features_aligned = features_aligned[:min_len]
        target = target[:min_len]
        
        print(f"Loaded {len(self.feature_names)} features")
        print(f"Feature shape: {features_aligned.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target distribution: {np.mean(target):.3f} positive")
        
        return features_aligned, target, self.feature_names
    
    def calculate_feature_importance_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using XGBoost
        
        Returns:
            Dictionary of feature importances
        """
        print("\nCalculating XGBoost feature importance...")
        
        # Use time series split for proper validation
        tscv = TimeSeriesSplit(n_splits=3)
        importances = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     verbose=False)
            
            importances.append(model.feature_importances_)
            
            # Calculate validation AUC
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            print(f"  Fold {fold + 1} AUC: {auc:.4f}")
        
        # Average importances across folds
        avg_importances = np.mean(importances, axis=0)
        importance_dict = {name: imp for name, imp in zip(self.feature_names, avg_importances)}
        
        return importance_dict
    
    def calculate_feature_importance_rf(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using Random Forest
        
        Returns:
            Dictionary of feature importances
        """
        print("\nCalculating Random Forest feature importance...")
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Use only a subset of data for faster computation
        sample_size = min(100000, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        
        rf.fit(X[indices], y[indices])
        
        importance_dict = {name: imp for name, imp in zip(self.feature_names, rf.feature_importances_)}
        
        return importance_dict
    
    def calculate_mutual_information(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate mutual information between features and target
        
        Returns:
            Dictionary of mutual information scores
        """
        print("\nCalculating Mutual Information scores...")
        
        # Use subset for faster computation
        sample_size = min(50000, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        
        mi_scores = mutual_info_classif(X[indices], y[indices], random_state=42)
        mi_dict = {name: score for name, score in zip(self.feature_names, mi_scores)}
        
        return mi_dict
    
    def calculate_correlation_matrix(self, X: np.ndarray) -> pd.DataFrame:
        """
        Calculate correlation matrix for all features
        
        Returns:
            Correlation matrix as DataFrame
        """
        print("\nCalculating correlation matrix...")
        
        # Use subset for faster computation
        sample_size = min(50000, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        
        # Create DataFrame
        df = pd.DataFrame(X[indices], columns=self.feature_names)
        corr_matrix = df.corr()
        
        return corr_matrix
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, save_path: str = None):
        """
        Create correlation heatmap visualization
        """
        plt.figure(figsize=(16, 14))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                    mask=mask, 
                    cmap='coolwarm', 
                    center=0,
                    square=True, 
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1)
        
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def find_redundant_features(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> Dict[str, List[str]]:
        """
        Find groups of highly correlated features
        
        Returns:
            Dictionary mapping feature to its correlated features
        """
        print(f"\nFinding redundant features (correlation > {threshold})...")
        
        redundant_groups = {}
        processed = set()
        
        for i, feature1 in enumerate(self.feature_names):
            if feature1 in processed:
                continue
                
            correlated = []
            for j, feature2 in enumerate(self.feature_names):
                if i != j and abs(corr_matrix.iloc[i, j]) > threshold:
                    correlated.append(feature2)
                    processed.add(feature2)
            
            if correlated:
                redundant_groups[feature1] = correlated
                
        return redundant_groups
    
    def select_features(self, importance_scores: Dict[str, Dict[str, float]], 
                       corr_matrix: pd.DataFrame,
                       redundancy_threshold: float = 0.7) -> List[str]:
        """
        Select top features based on importance and redundancy
        
        Returns:
            List of selected feature names
        """
        print("\nSelecting optimal features...")
        
        # Combine importance scores (average across methods)
        combined_scores = {}
        for feature in self.feature_names:
            scores = []
            for method, method_scores in importance_scores.items():
                # Normalize scores within each method
                max_score = max(method_scores.values())
                if max_score > 0:
                    normalized_score = method_scores[feature] / max_score
                    scores.append(normalized_score)
            combined_scores[feature] = np.mean(scores)
        
        # Sort features by combined importance
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select features using greedy approach
        selected = []
        for feature, score in sorted_features:
            # Check correlation with already selected features
            if selected:
                max_corr = max([abs(corr_matrix.loc[feature, sel]) for sel in selected])
                if max_corr > redundancy_threshold:
                    continue
            
            selected.append(feature)
            
            if len(selected) >= self.n_target_features:
                break
        
        # Ensure we have key feature categories
        categories = {
            'microstructure': ['micro_', 'bid_', 'ask_', 'order_', 'spread'],
            'price_action': ['return', 'volatility', 'momentum'],
            'volume': ['volume', 'imbalance', 'flow'],
            'position': ['position', 'holding']
        }
        
        for category, keywords in categories.items():
            has_category = any(any(kw in feat for kw in keywords) for feat in selected)
            if not has_category and len(selected) < self.n_target_features:
                # Find best feature from this category
                for feature, score in sorted_features:
                    if feature not in selected and any(kw in feature for kw in keywords):
                        selected.append(feature)
                        break
        
        return selected[:self.n_target_features]
    
    def generate_report(self, importance_scores: Dict[str, Dict[str, float]], 
                       selected_features: List[str],
                       corr_matrix: pd.DataFrame):
        """
        Generate comprehensive feature selection report
        """
        report = f"""# Feature Selection Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Original Features**: {len(self.feature_names)}
- **Selected Features**: {len(selected_features)}
- **Reduction**: {(1 - len(selected_features)/len(self.feature_names))*100:.1f}%

## Selected Features

The following {len(selected_features)} features were selected based on:
1. High predictive power (XGBoost, Random Forest, Mutual Information)
2. Low redundancy (correlation < 0.7 with other selected features)
3. Coverage across key categories (microstructure, price action, volume, position)

### Final Feature Set:
"""
        
        # Add selected features with their scores
        combined_scores = {}
        for feature in self.feature_names:
            scores = []
            for method, method_scores in importance_scores.items():
                max_score = max(method_scores.values())
                if max_score > 0:
                    normalized_score = method_scores[feature] / max_score
                    scores.append(normalized_score)
            combined_scores[feature] = np.mean(scores)
        
        for i, feature in enumerate(selected_features):
            score = combined_scores[feature]
            report += f"\n{i+1}. **{feature}** (score: {score:.4f})"
        
        # Add correlation analysis
        report += f"\n\n## Correlation Analysis\n\n"
        report += f"Maximum pairwise correlation among selected features: "
        
        selected_corr = corr_matrix.loc[selected_features, selected_features]
        np.fill_diagonal(selected_corr.values, 0)
        max_corr = np.max(np.abs(selected_corr.values))
        report += f"{max_corr:.3f}\n"
        
        # Add feature importance rankings
        report += f"\n## Feature Importance Rankings\n\n"
        
        for method, scores in importance_scores.items():
            report += f"### {method}\n\n"
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            report += "| Rank | Feature | Score |\n"
            report += "|------|---------|-------|\n"
            
            for i, (feature, score) in enumerate(sorted_scores[:20]):
                selected_mark = "✓" if feature in selected_features else ""
                report += f"| {i+1} | {feature} {selected_mark} | {score:.4f} |\n"
            
            report += "\n"
        
        # Save report
        report_path = os.path.join(self.output_path, 'FEATURE_SELECTION_REPORT.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_path}")
    
    def save_results(self, selected_features: List[str], importance_scores: Dict[str, Dict[str, float]]):
        """
        Save selection results to JSON
        """
        # Convert numpy float32 to Python float for JSON serialization
        json_scores = {}
        for method, scores in importance_scores.items():
            json_scores[method] = {k: float(v) for k, v in scores.items()}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'original_features': len(self.feature_names),
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'importance_scores': json_scores,
            'feature_names': self.feature_names
        }
        
        results_path = os.path.join(self.output_path, 'feature_selection_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    
    def run_analysis(self):
        """
        Run complete feature selection analysis
        """
        print("="*60)
        print("Feature Selection Analysis - FinRL Contest 2024")
        print("="*60)
        
        # Load data
        X, y, feature_names = self.load_data()
        
        # Calculate importance scores
        importance_scores = {
            'XGBoost': self.calculate_feature_importance_xgboost(X, y),
            'RandomForest': self.calculate_feature_importance_rf(X, y),
            'MutualInformation': self.calculate_mutual_information(X, y)
        }
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(X)
        
        # Plot correlation heatmap
        heatmap_path = os.path.join(self.output_path, 'visualizations', 'correlation_heatmap.png')
        self.plot_correlation_heatmap(corr_matrix, heatmap_path)
        print(f"Correlation heatmap saved to: {heatmap_path}")
        
        # Find redundant features
        redundant_groups = self.find_redundant_features(corr_matrix)
        print(f"Found {len(redundant_groups)} groups of redundant features")
        
        # Select features
        selected_features = self.select_features(importance_scores, corr_matrix)
        self.selected_features = selected_features
        
        print(f"\nSelected {len(selected_features)} features:")
        for i, feature in enumerate(selected_features):
            print(f"  {i+1}. {feature}")
        
        # Generate report
        self.generate_report(importance_scores, selected_features, corr_matrix)
        
        # Save results
        self.save_results(selected_features, importance_scores)
        
        print("\n" + "="*60)
        print("Feature selection analysis complete!")
        print("="*60)
        
        return selected_features


if __name__ == "__main__":
    # Run feature selection
    selector = FeatureSelector(
        data_path='../../../data/raw/task1/',
        output_path='./',
        n_target_features=15
    )
    
    selected_features = selector.run_analysis()