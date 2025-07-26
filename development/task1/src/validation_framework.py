"""
Validation Framework for FinRL Contest 2024
Provides comprehensive validation strategies for time series financial data
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict, Any
from datetime import datetime
import logging

class TemporalDataSplitter:
    """
    Temporal data splitter that maintains chronological order
    and prevents look-ahead bias in financial time series
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_gap: int = 60,  # Gap between splits to prevent contamination
        min_samples_per_split: int = 1000
    ):
        """
        Initialize temporal data splitter
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            purge_gap: Number of samples to skip between splits
            min_samples_per_split: Minimum samples required per split
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
            
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_gap = purge_gap
        self.min_samples_per_split = min_samples_per_split
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def split_data(
        self, 
        data: np.ndarray, 
        timestamps: np.ndarray = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, slice]]:
        """
        Split data temporally into train/validation/test sets
        
        Args:
            data: Input data array (n_samples, n_features)
            timestamps: Optional timestamp array for logging
            
        Returns:
            splits: Dictionary containing train/val/test data splits
            indices: Dictionary containing slice objects for each split
        """
        n_samples = len(data)
        
        if n_samples < self.min_samples_per_split * 3:
            raise ValueError(f"Insufficient data: {n_samples} samples, need at least {self.min_samples_per_split * 3}")
        
        # Calculate split points with purge gaps
        train_end = int(n_samples * self.train_ratio)
        val_start = train_end + self.purge_gap
        val_end = val_start + int(n_samples * self.val_ratio)
        test_start = val_end + self.purge_gap
        
        # Ensure we don't exceed data bounds
        if test_start >= n_samples:
            # Adjust splits if purge gaps are too large
            self.logger.warning("Purge gaps too large, adjusting splits")
            train_end = int(n_samples * 0.7)
            val_start = train_end + min(self.purge_gap, 10)
            val_end = val_start + int(n_samples * 0.15)
            test_start = val_end + min(self.purge_gap, 10)
        
        # Create splits
        train_data = data[:train_end]
        val_data = data[val_start:val_end]
        test_data = data[test_start:]
        
        # Create index slices
        train_slice = slice(0, train_end)
        val_slice = slice(val_start, val_end)
        test_slice = slice(test_start, None)
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        indices = {
            'train': train_slice,
            'val': val_slice,
            'test': test_slice
        }
        
        # Log split information
        self.logger.info("Data split completed:")
        self.logger.info(f"  Total samples: {n_samples}")
        self.logger.info(f"  Train: {len(train_data)} samples ({len(train_data)/n_samples:.1%})")
        self.logger.info(f"  Validation: {len(val_data)} samples ({len(val_data)/n_samples:.1%})")
        self.logger.info(f"  Test: {len(test_data)} samples ({len(test_data)/n_samples:.1%})")
        self.logger.info(f"  Purge gap: {self.purge_gap} samples")
        
        if timestamps is not None:
            self.logger.info(f"  Train period: {timestamps[0]} to {timestamps[train_end-1]}")
            self.logger.info(f"  Val period: {timestamps[val_start]} to {timestamps[val_end-1]}")
            self.logger.info(f"  Test period: {timestamps[test_start]} to {timestamps[-1]}")
        
        return splits, indices
    
    def create_cv_folds(
        self, 
        train_data: np.ndarray, 
        n_folds: int = 5,
        expanding_window: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds for temporal data
        
        Args:
            train_data: Training data array
            n_folds: Number of CV folds
            expanding_window: If True, use expanding window; if False, use sliding window
            
        Returns:
            List of (train_fold, val_fold) tuples
        """
        n_samples = len(train_data)
        fold_size = n_samples // (n_folds + 1)  # Reserve space for validation
        
        folds = []
        
        for i in range(n_folds):
            if expanding_window:
                # Expanding window: train on all data up to validation period
                train_end = (i + 1) * fold_size
                val_start = train_end + self.purge_gap
                val_end = val_start + fold_size
                
                if val_end > n_samples:
                    break
                    
                train_fold = train_data[:train_end]
                val_fold = train_data[val_start:val_end]
            else:
                # Sliding window: fixed-size training window
                val_start = (i + 1) * fold_size + self.purge_gap
                val_end = val_start + fold_size
                train_start = max(0, val_start - 2 * fold_size - self.purge_gap)
                
                if val_end > n_samples:
                    break
                    
                train_fold = train_data[train_start:val_start - self.purge_gap]
                val_fold = train_data[val_start:val_end]
            
            if len(train_fold) >= self.min_samples_per_split and len(val_fold) >= self.min_samples_per_split // 2:
                folds.append((train_fold, val_fold))
        
        self.logger.info(f"Created {len(folds)} CV folds ({'expanding' if expanding_window else 'sliding'} window)")
        
        return folds


class ValidationMetricsCalculator:
    """Calculate comprehensive validation metrics for trading models"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_financial_metrics(
        self, 
        returns: np.ndarray, 
        positions: np.ndarray = None,
        prices: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive financial performance metrics
        
        Args:
            returns: Array of portfolio returns
            positions: Array of trading positions (optional)
            prices: Array of asset prices (optional)
            
        Returns:
            Dictionary of financial metrics
        """
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Remove any NaN or infinite values
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic return metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Risk-adjusted returns
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # RoMaD (Return over Maximum Drawdown)
        romad = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading-specific metrics
        if positions is not None:
            # Win rate
            profitable_trades = np.sum(returns > 0)
            total_trades = len(returns)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # Average trade metrics
            avg_win = np.mean(returns[returns > 0]) if profitable_trades > 0 else 0
            avg_loss = np.mean(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            metrics.update({
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            })
        
        # VaR and CVaR (5% level)
        var_5 = np.percentile(returns, 5)
        cvar_5 = np.mean(returns[returns <= var_5]) if np.any(returns <= var_5) else var_5
        
        metrics.update({
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'romad': romad,
            'downside_deviation': downside_deviation,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        })
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'romad': 0.0,
            'downside_deviation': 0.0,
            'var_5': 0.0,
            'cvar_5': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    def compare_metrics(
        self, 
        metrics_dict: Dict[str, Dict[str, float]],
        primary_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Compare metrics across different models/configurations
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            primary_metric: Primary metric for ranking
            
        Returns:
            Comparison results with rankings
        """
        if not metrics_dict:
            return {}
        
        # Create comparison table
        comparison = pd.DataFrame(metrics_dict).T
        
        # Rank by primary metric
        comparison['rank'] = comparison[primary_metric].rank(ascending=False)
        comparison = comparison.sort_values('rank')
        
        # Calculate relative performance
        best_value = comparison[primary_metric].max()
        comparison[f'{primary_metric}_relative'] = comparison[primary_metric] / best_value if best_value != 0 else 0
        
        results = {
            'comparison_table': comparison,
            'best_model': comparison.index[0],
            'best_value': best_value,
            'ranking': comparison[['rank', primary_metric]].to_dict()
        }
        
        self.logger.info(f"Model comparison completed. Best model: {results['best_model']} "
                        f"({primary_metric}: {best_value:.4f})")
        
        return results
    
    def validate_features(
        self, 
        feature_array: np.ndarray, 
        feature_names: List[str], 
        verbose: bool = True
    ) -> Dict[str, Any]:
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
            try:
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
            except:
                if verbose:
                    print("Warning: Could not compute correlation matrix")
        
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


class ValidationFramework:
    """
    Comprehensive validation framework for FinRL models
    Combines data splitting, cross-validation, and metrics calculation
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        cv_folds: int = 5,
        purge_gap: int = 60,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize validation framework
        
        Args:
            train_ratio: Training data ratio
            val_ratio: Validation data ratio  
            test_ratio: Test data ratio
            cv_folds: Number of cross-validation folds
            purge_gap: Gap between data splits
            risk_free_rate: Risk-free rate for metrics
        """
        self.data_splitter = TemporalDataSplitter(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_gap=purge_gap
        )
        
        self.metrics_calculator = ValidationMetricsCalculator(
            risk_free_rate=risk_free_rate
        )
        
        self.cv_folds = cv_folds
        self.logger = logging.getLogger(__name__)
        
        # Storage for validation results
        self.splits = None
        self.indices = None
        self.cv_results = None
        
    def setup_validation(
        self, 
        data: np.ndarray, 
        timestamps: np.ndarray = None,
        save_splits: bool = True,
        save_path: str = None
    ) -> Dict[str, Any]:
        """
        Setup validation framework with data splits
        
        Args:
            data: Input data array
            timestamps: Optional timestamps
            save_splits: Whether to save split information
            save_path: Path to save splits
            
        Returns:
            Dictionary containing setup information
        """
        # Create data splits
        self.splits, self.indices = self.data_splitter.split_data(data, timestamps)
        
        # Create CV folds from training data
        cv_folds = self.data_splitter.create_cv_folds(
            self.splits['train'], 
            n_folds=self.cv_folds
        )
        
        setup_info = {
            'n_samples': len(data),
            'train_samples': len(self.splits['train']),
            'val_samples': len(self.splits['val']),
            'test_samples': len(self.splits['test']),
            'cv_folds': len(cv_folds),
            'splits': self.splits,
            'indices': self.indices,
            'cv_folds': cv_folds
        }
        
        # Save splits if requested
        if save_splits and save_path:
            self._save_splits(setup_info, save_path)
        
        self.logger.info("Validation framework setup completed")
        
        return setup_info
    
    def validate_model_performance(
        self, 
        model_results: Dict[str, np.ndarray],
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Validate model performance across all splits
        
        Args:
            model_results: Dictionary with 'train', 'val', 'test' results
            model_name: Name of the model being validated
            
        Returns:
            Comprehensive validation results
        """
        validation_results = {
            'model_name': model_name,
            'metrics': {},
            'overfitting_analysis': {},
            'stability_analysis': {}
        }
        
        # Calculate metrics for each split
        for split_name, returns in model_results.items():
            if split_name in ['train', 'val', 'test']:
                metrics = self.metrics_calculator.calculate_financial_metrics(returns)
                validation_results['metrics'][split_name] = metrics
        
        # Overfitting analysis
        if 'train' in validation_results['metrics'] and 'val' in validation_results['metrics']:
            train_sharpe = validation_results['metrics']['train']['sharpe_ratio']
            val_sharpe = validation_results['metrics']['val']['sharpe_ratio']
            
            overfitting_score = (train_sharpe - val_sharpe) / train_sharpe if train_sharpe != 0 else 0
            validation_results['overfitting_analysis'] = {
                'overfitting_score': overfitting_score,
                'train_val_gap': train_sharpe - val_sharpe,
                'is_overfitting': overfitting_score > 0.1  # 10% threshold
            }
        
        # Stability analysis across CV folds (if available)
        if hasattr(self, 'cv_results') and self.cv_results:
            cv_sharpes = [result.get('sharpe_ratio', 0) for result in self.cv_results]
            validation_results['stability_analysis'] = {
                'cv_mean_sharpe': np.mean(cv_sharpes),
                'cv_std_sharpe': np.std(cv_sharpes),
                'cv_stability': 1 - (np.std(cv_sharpes) / abs(np.mean(cv_sharpes))) if np.mean(cv_sharpes) != 0 else 0
            }
        
        self.logger.info(f"Model validation completed for {model_name}")
        if 'val' in validation_results['metrics']:
            val_sharpe = validation_results['metrics']['val']['sharpe_ratio']
            self.logger.info(f"  Validation Sharpe ratio: {val_sharpe:.4f}")
        
        return validation_results
    
    def _save_splits(self, setup_info: Dict[str, Any], save_path: str):
        """Save validation splits and metadata"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save data splits
        for split_name, split_data in setup_info['splits'].items():
            split_path = os.path.join(save_path, f"{split_name}_split.npy")
            np.save(split_path, split_data)
        
        # Save indices
        indices_info = {k: {'start': v.start, 'stop': v.stop, 'step': v.step} 
                       for k, v in setup_info['indices'].items()}
        
        # Save metadata
        metadata = {
            'n_samples': setup_info['n_samples'],
            'split_sizes': {
                'train': setup_info['train_samples'],
                'val': setup_info['val_samples'], 
                'test': setup_info['test_samples']
            },
            'cv_folds': setup_info['cv_folds'],
            'indices': indices_info,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(save_path, "validation_metadata.npy")
        np.save(metadata_path, metadata)
        
        self.logger.info(f"Validation splits saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Validation Framework Test")
    
    # Create synthetic data
    n_samples = 10000
    n_features = 20
    data = np.random.randn(n_samples, n_features).cumsum(axis=0)
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    
    # Initialize framework
    framework = ValidationFramework()
    
    # Setup validation
    setup_info = framework.setup_validation(data, timestamps.values)
    
    # Create synthetic model results
    model_results = {
        'train': np.random.randn(setup_info['train_samples']) * 0.01,
        'val': np.random.randn(setup_info['val_samples']) * 0.01,
        'test': np.random.randn(setup_info['test_samples']) * 0.01
    }
    
    # Validate model
    validation_results = framework.validate_model_performance(model_results, "test_model")
    
    print("Validation completed successfully!")
    print(f"Validation Sharpe ratio: {validation_results['metrics']['val']['sharpe_ratio']:.4f}")