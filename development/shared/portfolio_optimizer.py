"""
Multi-Asset Portfolio Optimizer
Advanced portfolio optimization for both Task 1 (crypto) and Task 2 (equities)
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
from scipy import linalg
from sklearn.covariance import LedoitWolf
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_RETURN = "max_return"
    BLACK_LITTERMAN = "black_litterman"
    KELLY_CRITERION = "kelly_criterion"
    CVaR = "cvar"

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    
    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_individual_weight: float = 0.3
    
    # Portfolio constraints
    max_leverage: float = 1.0
    min_positions: int = 1
    max_positions: int = 10
    
    # Risk constraints
    max_volatility: float = 0.25
    max_drawdown: float = 0.2
    max_var: float = 0.05  # Value at Risk
    max_cvar: float = 0.07  # Conditional Value at Risk
    
    # Turnover constraints
    max_turnover: float = 0.5
    transaction_cost: float = 0.001
    
    # Sector/Asset constraints (for diversification)
    sector_limits: Optional[Dict[str, float]] = None
    asset_class_limits: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.sector_limits is None:
            self.sector_limits = {}
        if self.asset_class_limits is None:
            self.asset_class_limits = {}

class PortfolioOptimizer:
    """Advanced multi-asset portfolio optimizer"""
    
    def __init__(self, constraints: OptimizationConstraints = None):
        self.constraints = constraints or OptimizationConstraints()
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
        # Optimization history
        self.optimization_history = []
        
        # Covariance estimation method
        self.covariance_method = "ledoit_wolf"  # ledoit_wolf, sample, shrinkage
    
    def optimize_portfolio(self, 
                          returns: Union[np.ndarray, pd.DataFrame],
                          expected_returns: Optional[np.ndarray] = None,
                          objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                          current_weights: Optional[np.ndarray] = None,
                          asset_names: Optional[List[str]] = None) -> Dict:
        """
        Optimize portfolio weights based on specified objective
        
        Args:
            returns: Historical returns matrix (n_periods x n_assets)
            expected_returns: Expected returns vector (n_assets,)
            objective: Optimization objective
            current_weights: Current portfolio weights for turnover constraints
            asset_names: Asset names for reporting
            
        Returns:
            Dictionary containing optimized weights and metrics
        """
        
        # Validate and prepare data
        returns_matrix = self._prepare_returns_data(returns)
        n_assets = returns_matrix.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        # Estimate expected returns if not provided
        if expected_returns is None:
            expected_returns = self._estimate_expected_returns(returns_matrix)
        
        # Estimate covariance matrix
        cov_matrix = self._estimate_covariance_matrix(returns_matrix)
        
        # Optimize based on objective
        optimization_result = self._optimize_by_objective(
            expected_returns, cov_matrix, objective, current_weights, returns_matrix
        )
        
        # Validate and adjust weights
        optimal_weights = self._validate_and_adjust_weights(
            optimization_result['weights'], expected_returns, cov_matrix
        )
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            optimal_weights, expected_returns, cov_matrix, returns_matrix
        )
        
        # Create result dictionary
        result = {
            'weights': optimal_weights,
            'asset_names': asset_names,
            'objective': objective.value,
            'metrics': portfolio_metrics,
            'optimization_status': optimization_result.get('status', 'unknown'),
            'covariance_matrix': cov_matrix,
            'expected_returns': expected_returns
        }
        
        # Store in history
        self.optimization_history.append(result)
        
        return result
    
    def _prepare_returns_data(self, returns: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Prepare and validate returns data"""
        
        if isinstance(returns, pd.DataFrame):
            returns_matrix = returns.values
        else:
            returns_matrix = np.array(returns)
        
        # Ensure 2D array
        if returns_matrix.ndim == 1:
            returns_matrix = returns_matrix.reshape(-1, 1)
        
        # Remove any NaN values
        if np.any(np.isnan(returns_matrix)):
            # Forward fill and backward fill
            returns_df = pd.DataFrame(returns_matrix)
            returns_df = returns_df.fillna(method='ffill').fillna(method='bfill')
            returns_matrix = returns_df.values
        
        return returns_matrix
    
    def _estimate_expected_returns(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Estimate expected returns using various methods"""
        
        # Simple historical mean (can be enhanced with more sophisticated methods)
        historical_mean = np.mean(returns_matrix, axis=0)
        
        # Exponentially weighted average (gives more weight to recent observations)
        ewm_span = min(30, returns_matrix.shape[0] // 2)
        if ewm_span > 1:
            ewm_returns = pd.DataFrame(returns_matrix).ewm(span=ewm_span).mean().iloc[-1].values
        else:
            ewm_returns = historical_mean
        
        # Combine methods (70% EWM, 30% historical)
        expected_returns = 0.7 * ewm_returns + 0.3 * historical_mean
        
        # Annualize returns (assuming daily data)
        expected_returns = expected_returns * 252
        
        return expected_returns
    
    def _estimate_covariance_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix using advanced methods"""
        
        if self.covariance_method == "ledoit_wolf":
            # Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_matrix).covariance_
        
        elif self.covariance_method == "sample":
            # Sample covariance matrix
            cov_matrix = np.cov(returns_matrix.T)
        
        elif self.covariance_method == "shrinkage":
            # Custom shrinkage towards diagonal
            sample_cov = np.cov(returns_matrix.T)
            diagonal_cov = np.diag(np.diag(sample_cov))
            shrinkage_intensity = 0.2
            cov_matrix = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * diagonal_cov
        
        else:
            # Default to sample covariance
            cov_matrix = np.cov(returns_matrix.T)
        
        # Annualize covariance matrix
        cov_matrix = cov_matrix * 252
        
        # Ensure positive semi-definite
        cov_matrix = self._make_positive_semidefinite(cov_matrix)
        
        return cov_matrix
    
    def _make_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive semi-definite"""
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = linalg.eigh(matrix)
        
        # Set negative eigenvalues to small positive value
        eigenvals = np.maximum(eigenvals, 1e-8)
        
        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    def _optimize_by_objective(self, 
                              expected_returns: np.ndarray, 
                              cov_matrix: np.ndarray,
                              objective: OptimizationObjective,
                              current_weights: Optional[np.ndarray],
                              returns_matrix: np.ndarray) -> Dict:
        """Optimize portfolio based on specified objective"""
        
        n_assets = len(expected_returns)
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds (individual weight constraints)
        bounds = [(self.constraints.min_weight, 
                  min(self.constraints.max_weight, self.constraints.max_individual_weight)) 
                 for _ in range(n_assets)]
        
        # Constraints
        constraints = self._build_optimization_constraints(n_assets, current_weights)
        
        # Objective function
        if objective == OptimizationObjective.MAX_SHARPE:
            objective_func = lambda w: -self._sharpe_ratio(w, expected_returns, cov_matrix)
            
        elif objective == OptimizationObjective.MIN_VARIANCE:
            objective_func = lambda w: self._portfolio_variance(w, cov_matrix)
            
        elif objective == OptimizationObjective.MAX_RETURN:
            objective_func = lambda w: -np.dot(w, expected_returns)
            
        elif objective == OptimizationObjective.RISK_PARITY:
            objective_func = lambda w: self._risk_parity_objective(w, cov_matrix)
            
        elif objective == OptimizationObjective.KELLY_CRITERION:
            objective_func = lambda w: -self._kelly_criterion(w, expected_returns, cov_matrix)
            
        elif objective == OptimizationObjective.CVaR:
            objective_func = lambda w: self._cvar_objective(w, returns_matrix)
            
        else:  # Default to mean-variance
            objective_func = lambda w: self._mean_variance_objective(w, expected_returns, cov_matrix)
        
        # Optimize
        try:
            result = sco.minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            return {
                'weights': result.x,
                'status': 'success' if result.success else 'failed',
                'message': result.message
            }
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return {
                'weights': x0,  # Return equal weights as fallback
                'status': 'failed',
                'message': str(e)
            }
    
    def _build_optimization_constraints(self, n_assets: int, current_weights: Optional[np.ndarray]) -> List:
        """Build optimization constraints"""
        
        constraints = []
        
        # Sum of weights constraint (fully invested or leverage constraint)
        if self.constraints.max_leverage <= 1.0:
            # Fully invested
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
        else:
            # Leverage constraint
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.constraints.max_leverage - np.sum(np.abs(w))
            })
        
        # Maximum volatility constraint
        if self.constraints.max_volatility < np.inf:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.constraints.max_volatility - np.sqrt(self._portfolio_variance(w, self.cov_matrix))
            })
        
        # Turnover constraint
        if current_weights is not None and self.constraints.max_turnover < np.inf:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.constraints.max_turnover - np.sum(np.abs(w - current_weights))
            })
        
        # Position constraints (min/max number of positions)
        if self.constraints.min_positions > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.sum(np.abs(w) > 1e-6) - self.constraints.min_positions
            })
        
        if self.constraints.max_positions < n_assets:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.constraints.max_positions - np.sum(np.abs(w) > 1e-6)
            })
        
        return constraints
    
    def _sharpe_ratio(self, weights: np.ndarray, expected_returns: np.ndarray, 
                     cov_matrix: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(self._portfolio_variance(weights, cov_matrix))
        
        if portfolio_volatility == 0:
            return 0.0
        
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility
    
    def _portfolio_variance(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio variance"""
        return np.dot(weights, np.dot(cov_matrix, weights))
    
    def _mean_variance_objective(self, weights: np.ndarray, expected_returns: np.ndarray, 
                                cov_matrix: np.ndarray, risk_aversion: float = 1.0) -> float:
        """Mean-variance optimization objective"""
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = self._portfolio_variance(weights, cov_matrix)
        
        # Utility = Return - (risk_aversion/2) * Variance
        return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
    
    def _risk_parity_objective(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Risk parity optimization objective"""
        
        # Calculate risk contributions
        portfolio_volatility = np.sqrt(self._portfolio_variance(weights, cov_matrix))
        
        if portfolio_volatility == 0:
            return 1e10  # Large penalty for zero volatility
        
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
        contrib = weights * marginal_contrib
        
        # Target equal risk contribution
        target_contrib = np.full(len(weights), np.sum(contrib) / len(weights))
        
        # Minimize sum of squared deviations from target
        return np.sum((contrib - target_contrib) ** 2)
    
    def _kelly_criterion(self, weights: np.ndarray, expected_returns: np.ndarray, 
                        cov_matrix: np.ndarray) -> float:
        """Kelly criterion objective"""
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = self._portfolio_variance(weights, cov_matrix)
        
        if portfolio_variance == 0:
            return 0.0
        
        # Kelly fraction = (expected_return - risk_free_rate) / variance
        kelly_fraction = (portfolio_return - self.risk_free_rate) / portfolio_variance
        
        return kelly_fraction
    
    def _cvar_objective(self, weights: np.ndarray, returns_matrix: np.ndarray, 
                       alpha: float = 0.05) -> float:
        """Conditional Value at Risk (CVaR) objective"""
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_matrix, weights)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, alpha * 100)
        
        # Calculate CVaR (expected shortfall)
        cvar = np.mean(portfolio_returns[portfolio_returns <= var])
        
        return -cvar  # Minimize negative CVaR (maximize CVaR)
    
    def _validate_and_adjust_weights(self, weights: np.ndarray, expected_returns: np.ndarray, 
                                   cov_matrix: np.ndarray) -> np.ndarray:
        """Validate and adjust weights to meet constraints"""
        
        # Clip weights to bounds
        weights = np.clip(weights, self.constraints.min_weight, 
                         min(self.constraints.max_weight, self.constraints.max_individual_weight))
        
        # Renormalize to sum to 1 (or max leverage)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            target_sum = min(1.0, self.constraints.max_leverage)
            weights = weights * (target_sum / weight_sum)
        
        # Apply position constraints
        if np.sum(weights > 1e-6) > self.constraints.max_positions:
            # Keep only top positions
            top_indices = np.argsort(np.abs(weights))[-self.constraints.max_positions:]
            new_weights = np.zeros_like(weights)
            new_weights[top_indices] = weights[top_indices]
            
            # Renormalize
            weight_sum = np.sum(new_weights)
            if weight_sum > 0:
                new_weights = new_weights * (1.0 / weight_sum)
            weights = new_weights
        
        return weights
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray, returns_matrix: np.ndarray) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        
        # Basic metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = self._portfolio_variance(weights, cov_matrix)
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = self._sharpe_ratio(weights, expected_returns, cov_matrix)
        
        # Risk metrics
        portfolio_returns = np.dot(returns_matrix, weights)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        # Diversification metrics
        effective_positions = np.sum(weights > 1e-6)
        concentration = np.sum(weights ** 2)  # Herfindahl index
        diversification_ratio = self._calculate_diversification_ratio(weights, cov_matrix)
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'effective_positions': effective_positions,
            'concentration': concentration,
            'diversification_ratio': diversification_ratio,
            'total_weight': np.sum(weights),
            'long_weight': np.sum(weights[weights > 0]),
            'short_weight': np.sum(weights[weights < 0])
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        
        return np.max(drawdown)
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        
        # Individual volatilities
        individual_vols = np.sqrt(np.diag(cov_matrix))
        
        # Weighted average of individual volatilities
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(self._portfolio_variance(weights, cov_matrix))
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_avg_vol / portfolio_vol
    
    def optimize_multi_period(self, 
                             returns_history: pd.DataFrame,
                             rebalance_frequency: int = 21,  # Monthly rebalancing
                             objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                             lookback_window: int = 252) -> Dict:
        """Optimize portfolio with periodic rebalancing"""
        
        print(f"🔄 Multi-period optimization with {rebalance_frequency}-day rebalancing")
        
        dates = returns_history.index
        assets = returns_history.columns
        
        # Results storage
        weights_history = []
        performance_history = []
        rebalance_dates = []
        
        # Initial portfolio value
        portfolio_value = 100000.0
        current_weights = np.ones(len(assets)) / len(assets)  # Equal weights initially
        
        for i in range(lookback_window, len(dates), rebalance_frequency):
            rebalance_date = dates[i]
            
            # Get training data
            training_data = returns_history.iloc[i-lookback_window:i]
            
            # Optimize portfolio
            optimization_result = self.optimize_portfolio(
                returns=training_data,
                objective=objective,
                current_weights=current_weights,
                asset_names=list(assets)
            )
            
            new_weights = optimization_result['weights']
            
            # Calculate performance for the next period
            next_period_end = min(i + rebalance_frequency, len(dates))
            period_returns = returns_history.iloc[i:next_period_end]
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(period_returns.values, new_weights)
            period_performance = np.prod(1 + portfolio_returns) - 1
            
            # Update portfolio value
            portfolio_value *= (1 + period_performance)
            
            # Store results
            weights_history.append({
                'date': rebalance_date,
                'weights': dict(zip(assets, new_weights)),
                'metrics': optimization_result['metrics']
            })
            
            performance_history.append({
                'date': rebalance_date,
                'period_return': period_performance,
                'portfolio_value': portfolio_value,
                'sharpe_ratio': optimization_result['metrics']['sharpe_ratio']
            })
            
            rebalance_dates.append(rebalance_date)
            current_weights = new_weights
            
            if len(rebalance_dates) % 4 == 0:  # Progress update every 4 rebalances
                print(f"   Completed {len(rebalance_dates)} rebalances, "
                      f"Portfolio value: ${portfolio_value:,.2f}")
        
        # Calculate overall performance
        total_return = (portfolio_value / 100000.0) - 1
        period_returns = [p['period_return'] for p in performance_history]
        overall_sharpe = (np.mean(period_returns) / np.std(period_returns)) * np.sqrt(252/rebalance_frequency) if np.std(period_returns) > 0 else 0
        
        return {
            'weights_history': weights_history,
            'performance_history': performance_history,
            'rebalance_dates': rebalance_dates,
            'final_portfolio_value': portfolio_value,
            'total_return': total_return,
            'overall_sharpe': overall_sharpe,
            'num_rebalances': len(rebalance_dates)
        }
    
    def create_efficient_frontier(self, 
                                 returns: Union[np.ndarray, pd.DataFrame],
                                 num_portfolios: int = 100,
                                 asset_names: Optional[List[str]] = None) -> Dict:
        """Create efficient frontier"""
        
        print(f"📈 Creating efficient frontier with {num_portfolios} portfolios")
        
        returns_matrix = self._prepare_returns_data(returns)
        expected_returns = self._estimate_expected_returns(returns_matrix)
        cov_matrix = self._estimate_covariance_matrix(returns_matrix)
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(len(expected_returns))]
        
        # Calculate min and max returns
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)
        
        # Target returns for efficient frontier
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Optimize for minimum variance given target return
            n_assets = len(expected_returns)
            x0 = np.ones(n_assets) / n_assets
            bounds = [(0, 1) for _ in range(n_assets)]
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}  # Target return
            ]
            
            try:
                result = sco.minimize(
                    lambda w: self._portfolio_variance(w, cov_matrix),
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    weights = result.x
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_vol = np.sqrt(self._portfolio_variance(weights, cov_matrix))
                    sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                    
                    efficient_portfolios.append({
                        'weights': weights,
                        'return': portfolio_return,
                        'volatility': portfolio_vol,
                        'sharpe_ratio': sharpe
                    })
                    
            except:
                continue
        
        return {
            'efficient_portfolios': efficient_portfolios,
            'asset_names': asset_names,
            'expected_returns': expected_returns,
            'covariance_matrix': cov_matrix
        }

def create_sample_data(n_assets: int = 5, n_periods: int = 1000) -> pd.DataFrame:
    """Create sample returns data for testing"""
    
    np.random.seed(42)
    
    # Generate correlated returns
    correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Generate returns
    returns = np.random.multivariate_normal(
        mean=np.random.uniform(0.0005, 0.002, n_assets),
        cov=correlation_matrix * 0.001,
        size=n_periods
    )
    
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    
    return pd.DataFrame(returns, index=dates, columns=asset_names)

def run_portfolio_optimization_demo():
    """Run comprehensive portfolio optimization demo"""
    
    print("🚀 Portfolio Optimization Demo")
    
    # Create sample data
    returns_data = create_sample_data(n_assets=7, n_periods=1000)
    print(f"📊 Generated sample data: {returns_data.shape}")
    
    # Initialize optimizer
    constraints = OptimizationConstraints(
        max_individual_weight=0.4,
        max_positions=5,
        max_volatility=0.20,
        max_turnover=0.3
    )
    
    optimizer = PortfolioOptimizer(constraints)
    
    # Test different optimization objectives
    objectives = [
        OptimizationObjective.MAX_SHARPE,
        OptimizationObjective.MIN_VARIANCE,
        OptimizationObjective.RISK_PARITY,
        OptimizationObjective.KELLY_CRITERION
    ]
    
    results = {}
    
    for objective in objectives:
        print(f"\n🎯 Optimizing for {objective.value}")
        
        result = optimizer.optimize_portfolio(
            returns=returns_data,
            objective=objective,
            asset_names=list(returns_data.columns)
        )
        
        results[objective.value] = result
        
        print(f"   Status: {result['optimization_status']}")
        print(f"   Expected Return: {result['metrics']['expected_return']:.4f}")
        print(f"   Volatility: {result['metrics']['volatility']:.4f}")
        print(f"   Sharpe Ratio: {result['metrics']['sharpe_ratio']:.4f}")
        print(f"   Effective Positions: {result['metrics']['effective_positions']}")
    
    # Multi-period optimization
    print(f"\n🔄 Multi-period optimization")
    multi_period_result = optimizer.optimize_multi_period(
        returns_data,
        rebalance_frequency=21,
        objective=OptimizationObjective.MAX_SHARPE,
        lookback_window=252
    )
    
    print(f"   Final Portfolio Value: ${multi_period_result['final_portfolio_value']:,.2f}")
    print(f"   Total Return: {multi_period_result['total_return']:.2%}")
    print(f"   Overall Sharpe: {multi_period_result['overall_sharpe']:.4f}")
    print(f"   Number of Rebalances: {multi_period_result['num_rebalances']}")
    
    print(f"\n🎉 Portfolio optimization demo completed!")
    return results, multi_period_result

if __name__ == "__main__":
    results, multi_period_result = run_portfolio_optimization_demo()