"""
Advanced Performance Metrics Suite for Backtesting
Comprehensive financial metrics calculation and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedMetrics:
    """Advanced financial metrics calculator"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252
        
    def calculate_all_metrics(self, returns: np.ndarray, prices: np.ndarray = None, 
                            trades: List[Dict] = None) -> Dict[str, float]:
        """Calculate comprehensive set of performance metrics"""
        
        # Handle edge cases
        if len(returns) == 0:
            return self._empty_metrics()
            
        # Clean returns
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Calculate all metrics
        metrics = {}
        
        # Basic Performance Metrics
        metrics.update(self._basic_performance_metrics(returns))
        
        # Risk-Adjusted Return Metrics
        metrics.update(self._risk_adjusted_metrics(returns))
        
        # Drawdown Metrics
        metrics.update(self._drawdown_metrics(returns))
        
        # Risk Metrics
        metrics.update(self._risk_metrics(returns))
        
        # Distribution Metrics
        metrics.update(self._distribution_metrics(returns))
        
        # Trade-Based Metrics (if trades provided)
        if trades:
            metrics.update(self._trade_based_metrics(trades))
        
        # Time Series Metrics
        if prices is not None and len(prices) > 1:
            metrics.update(self._time_series_metrics(prices))
            
        return metrics
    
    def _basic_performance_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        
        # Cumulative returns
        cum_returns = np.cumprod(1 + returns) - 1
        total_return = cum_returns[-1] if len(cum_returns) > 0 else 0
        
        # Annualized return
        if len(returns) > 0:
            annualized_return = (1 + total_return) ** (self.trading_days / len(returns)) - 1
        else:
            annualized_return = 0
            
        # Average return
        avg_return = np.mean(returns)
        avg_positive_return = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_negative_return = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_return': avg_return,
            'avg_positive_return': avg_positive_return,
            'avg_negative_return': avg_negative_return,
            'volatility': np.std(returns) * np.sqrt(self.trading_days),
            'downside_volatility': self._downside_deviation(returns) * np.sqrt(self.trading_days)
        }
    
    def _risk_adjusted_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics"""
        
        if len(returns) <= 1:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'information_ratio': 0
            }
        
        # Excess returns
        excess_returns = returns - (self.risk_free_rate / self.trading_days)
        
        # Sharpe Ratio
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(self.trading_days) if np.std(returns) > 0 else 0
        
        # Sortino Ratio
        downside_dev = self._downside_deviation(returns)
        sortino_ratio = np.mean(excess_returns) / downside_dev * np.sqrt(self.trading_days) if downside_dev > 0 else 0
        
        # Calmar Ratio
        max_dd = self._calculate_max_drawdown(returns)
        calmar_ratio = (np.mean(returns) * self.trading_days) / abs(max_dd) if max_dd != 0 else 0
        
        # Information Ratio (assuming benchmark return of 0)
        information_ratio = np.mean(returns) / np.std(returns) * np.sqrt(self.trading_days) if np.std(returns) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def _drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown-related metrics"""
        
        if len(returns) == 0:
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'max_drawdown_duration': 0,
                'avg_drawdown_duration': 0,
                'recovery_factor': 0,
                'ulcer_index': 0
            }
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdowns
        drawdowns = (cum_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
        
        # Drawdown durations
        drawdown_durations = self._calculate_drawdown_durations(drawdowns)
        max_drawdown_duration = np.max(drawdown_durations) if len(drawdown_durations) > 0 else 0
        avg_drawdown_duration = np.mean(drawdown_durations) if len(drawdown_durations) > 0 else 0
        
        # Recovery factor
        total_return = cum_returns[-1] - 1
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'recovery_factor': recovery_factor,
            'ulcer_index': ulcer_index
        }
    
    def _risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        if len(returns) <= 1:
            return {
                'var_95': 0,
                'var_99': 0,
                'cvar_95': 0,
                'cvar_99': 0,
                'expected_shortfall': 0,
                'tail_ratio': 0,
                'gain_to_pain_ratio': 0
            }
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else 0
        cvar_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else 0
        
        # Expected Shortfall (same as CVaR 95%)
        expected_shortfall = cvar_95
        
        # Tail Ratio
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(gains) > 0 and len(losses) > 0:
            tail_ratio = np.percentile(gains, 95) / abs(np.percentile(losses, 5))
        else:
            tail_ratio = 0
        
        # Gain to Pain Ratio
        positive_returns = np.sum(returns[returns > 0])
        negative_returns = abs(np.sum(returns[returns < 0]))
        gain_to_pain_ratio = positive_returns / negative_returns if negative_returns > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'expected_shortfall': expected_shortfall,
            'tail_ratio': tail_ratio,
            'gain_to_pain_ratio': gain_to_pain_ratio
        }
    
    def _distribution_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate distribution-related metrics"""
        
        if len(returns) <= 3:
            return {
                'skewness': 0,
                'kurtosis': 0,
                'normality_test_pvalue': 0,
                'win_rate': 0,
                'loss_rate': 0,
                'expectancy': 0,
                'kelly_criterion': 0
            }
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Normality test
        try:
            _, normality_pvalue = stats.jarque_bera(returns)
        except:
            normality_pvalue = 0
        
        # Win/Loss rates
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns)
        loss_rate = len(negative_returns) / len(returns)
        
        # Expectancy
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        # Kelly Criterion
        if avg_loss > 0:
            kelly_criterion = win_rate - (loss_rate / (avg_win / avg_loss))
        else:
            kelly_criterion = 0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_test_pvalue': normality_pvalue,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'expectancy': expectancy,
            'kelly_criterion': kelly_criterion
        }
    
    def _trade_based_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate trade-based metrics"""
        
        if not trades:
            return {
                'num_trades': 0,
                'avg_trade_duration': 0,
                'profit_factor': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'avg_bars_in_trade': 0
            }
        
        # Extract trade information
        trade_returns = []
        trade_durations = []
        
        # Simple trade analysis (assuming trades have profit/loss info)
        for trade in trades:
            if 'pnl' in trade:
                trade_returns.append(trade['pnl'])
            if 'duration' in trade:
                trade_durations.append(trade['duration'])
        
        if not trade_returns:
            return {
                'num_trades': len(trades),
                'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0,
                'profit_factor': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'avg_bars_in_trade': np.mean(trade_durations) if trade_durations else 0
            }
        
        # Separate wins and losses
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1e-10
        profit_factor = total_wins / total_losses
        
        # Largest win/loss
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Consecutive wins/losses
        consecutive_wins = self._max_consecutive(trade_returns, lambda x: x > 0)
        consecutive_losses = self._max_consecutive(trade_returns, lambda x: x < 0)
        
        return {
            'num_trades': len(trades),
            'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'avg_bars_in_trade': np.mean(trade_durations) if trade_durations else 0
        }
    
    def _time_series_metrics(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate time series specific metrics"""
        
        if len(prices) <= 1:
            return {
                'beta': 0,
                'alpha': 0,
                'r_squared': 0,
                'tracking_error': 0
            }
        
        # Simple market proxy (assuming benchmark is flat)
        market_returns = np.zeros(len(prices) - 1)
        asset_returns = np.diff(prices) / prices[:-1]
        
        # Beta calculation
        try:
            beta = np.cov(asset_returns, market_returns)[0, 1] / np.var(market_returns) if np.var(market_returns) > 0 else 0
        except:
            beta = 0
        
        # Alpha calculation
        alpha = np.mean(asset_returns) - beta * np.mean(market_returns)
        
        # R-squared
        try:
            correlation = np.corrcoef(asset_returns, market_returns)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
        except:
            r_squared = 0
        
        # Tracking error
        tracking_error = np.std(asset_returns - market_returns) * np.sqrt(self.trading_days)
        
        return {
            'beta': beta,
            'alpha': alpha * self.trading_days,  # Annualized
            'r_squared': r_squared,
            'tracking_error': tracking_error
        }
    
    def _downside_deviation(self, returns: np.ndarray, target: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target]
        return np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
            
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        return np.min(drawdowns)
    
    def _calculate_drawdown_durations(self, drawdowns: np.ndarray) -> List[int]:
        """Calculate drawdown durations"""
        durations = []
        current_duration = 0
        
        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add final duration if still in drawdown
        if current_duration > 0:
            durations.append(current_duration)
            
        return durations
    
    def _max_consecutive(self, values: List, condition) -> int:
        """Calculate maximum consecutive occurrences of condition"""
        max_consecutive = 0
        current_consecutive = 0
        
        for value in values:
            if condition(value):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        return {
            'total_return': 0,
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'win_rate': 0,
            'var_95': 0,
            'num_trades': 0
        }

class BenchmarkComparator:
    """Compare strategy performance against benchmarks"""
    
    def __init__(self):
        self.metrics_calculator = AdvancedMetrics()
    
    def compare_to_benchmark(self, strategy_returns: np.ndarray, 
                           benchmark_returns: np.ndarray) -> Dict[str, Any]:
        """Compare strategy to benchmark"""
        
        # Calculate metrics for both
        strategy_metrics = self.metrics_calculator.calculate_all_metrics(strategy_returns)
        benchmark_metrics = self.metrics_calculator.calculate_all_metrics(benchmark_returns)
        
        # Calculate relative metrics
        relative_metrics = {}
        for key in strategy_metrics:
            if key in benchmark_metrics:
                if benchmark_metrics[key] != 0:
                    relative_metrics[f"relative_{key}"] = strategy_metrics[key] / benchmark_metrics[key]
                else:
                    relative_metrics[f"relative_{key}"] = 0
                relative_metrics[f"excess_{key}"] = strategy_metrics[key] - benchmark_metrics[key]
        
        return {
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics,
            'relative_metrics': relative_metrics
        }
    
    def calculate_information_ratio(self, strategy_returns: np.ndarray, 
                                  benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio"""
        if len(strategy_returns) != len(benchmark_returns):
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        excess_returns = strategy_returns - benchmark_returns
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

class MetricsReportGenerator:
    """Generate comprehensive metrics reports"""
    
    def __init__(self):
        self.metrics_calculator = AdvancedMetrics()
    
    def generate_metrics_report(self, returns: np.ndarray, prices: np.ndarray = None,
                              trades: List[Dict] = None, strategy_name: str = "Strategy") -> str:
        """Generate comprehensive metrics report"""
        
        metrics = self.metrics_calculator.calculate_all_metrics(returns, prices, trades)
        
        report = f"""
        
{'='*60}
{strategy_name.upper()} PERFORMANCE REPORT
{'='*60}

BASIC PERFORMANCE METRICS
{'─'*30}
Total Return:                {metrics.get('total_return', 0):.2%}
Annualized Return:           {metrics.get('annualized_return', 0):.2%}
Volatility (Annualized):     {metrics.get('volatility', 0):.2%}
Average Return:              {metrics.get('avg_return', 0):.4f}

RISK-ADJUSTED METRICS
{'─'*30}
Sharpe Ratio:                {metrics.get('sharpe_ratio', 0):.3f}
Sortino Ratio:               {metrics.get('sortino_ratio', 0):.3f}
Calmar Ratio:                {metrics.get('calmar_ratio', 0):.3f}
Information Ratio:           {metrics.get('information_ratio', 0):.3f}

DRAWDOWN ANALYSIS
{'─'*30}
Maximum Drawdown:            {metrics.get('max_drawdown', 0):.2%}
Average Drawdown:            {metrics.get('avg_drawdown', 0):.2%}
Max Drawdown Duration:       {metrics.get('max_drawdown_duration', 0):.0f} periods
Recovery Factor:             {metrics.get('recovery_factor', 0):.3f}
Ulcer Index:                 {metrics.get('ulcer_index', 0):.4f}

RISK METRICS
{'─'*30}
Value at Risk (95%):         {metrics.get('var_95', 0):.4f}
Conditional VaR (95%):       {metrics.get('cvar_95', 0):.4f}
Expected Shortfall:          {metrics.get('expected_shortfall', 0):.4f}
Tail Ratio:                  {metrics.get('tail_ratio', 0):.3f}
Gain to Pain Ratio:          {metrics.get('gain_to_pain_ratio', 0):.3f}

DISTRIBUTION ANALYSIS
{'─'*30}
Skewness:                    {metrics.get('skewness', 0):.3f}
Kurtosis:                    {metrics.get('kurtosis', 0):.3f}
Win Rate:                    {metrics.get('win_rate', 0):.2%}
Loss Rate:                   {metrics.get('loss_rate', 0):.2%}
Expectancy:                  {metrics.get('expectancy', 0):.4f}
Kelly Criterion:             {metrics.get('kelly_criterion', 0):.3f}

TRADING STATISTICS
{'─'*30}
Number of Trades:            {metrics.get('num_trades', 0):.0f}
Profit Factor:               {metrics.get('profit_factor', 0):.3f}
Largest Win:                 {metrics.get('largest_win', 0):.4f}
Largest Loss:                {metrics.get('largest_loss', 0):.4f}
Max Consecutive Wins:        {metrics.get('consecutive_wins', 0):.0f}
Max Consecutive Losses:      {metrics.get('consecutive_losses', 0):.0f}

{'='*60}
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
        
        """
        
        return report
    
    def save_metrics_to_csv(self, metrics: Dict[str, float], filepath: str):
        """Save metrics to CSV file"""
        df = pd.DataFrame([metrics])
        df.to_csv(filepath, index=False)

def main():
    """Example usage of advanced metrics"""
    
    # Generate sample returns for testing
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
    prices = 100 * np.cumprod(1 + returns)
    
    # Calculate metrics
    calculator = AdvancedMetrics()
    metrics = calculator.calculate_all_metrics(returns, prices)
    
    # Generate report
    report_generator = MetricsReportGenerator()
    report = report_generator.generate_metrics_report(returns, prices, 
                                                    strategy_name="Sample Strategy")
    
    print(report)
    
    # Compare to benchmark
    benchmark_returns = np.random.normal(0, 0.01, 1000)
    comparator = BenchmarkComparator()
    comparison = comparator.compare_to_benchmark(returns, benchmark_returns)
    
    print(f"\nInformation Ratio vs Benchmark: {comparator.calculate_information_ratio(returns, benchmark_returns):.3f}")

if __name__ == "__main__":
    main()