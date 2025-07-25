"""
Comprehensive Backtesting Framework for FinRL Contest 2024
Advanced backtesting with walk-forward validation, regime analysis, and statistical testing
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass, asdict
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Import trading components
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_config import Config
from data_config import ConfigData
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    # Data parameters
    data_split_ratio: float = 0.7  # Train/test split
    walk_forward_window: int = 5000  # Number of steps per window
    overlap_ratio: float = 0.2  # Overlap between consecutive windows
    
    # Ensemble parameters
    ensemble_path: str = "ensemble_optimized_phase2"
    agent_classes: List = None
    
    # Trading parameters
    starting_cash: float = 1e6
    max_position: int = 1
    slippage: float = 7e-7
    step_gap: int = 2
    
    # Analysis parameters
    confidence_level: float = 0.95
    monte_carlo_runs: int = 1000
    regime_window: int = 100
    
    # Performance thresholds
    min_sharpe: float = 0.5
    max_drawdown_threshold: float = 0.05
    min_win_rate: float = 0.5
    
    def __post_init__(self):
        if self.agent_classes is None:
            self.agent_classes = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]

@dataclass
class BacktestResult:
    """Single backtest result container"""
    period_name: str
    start_idx: int
    end_idx: int
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    romad: float
    win_rate: float
    
    # Trade statistics
    num_trades: int
    avg_trade_return: float
    profit_factor: float
    
    # Risk metrics
    volatility: float
    var_95: float
    cvar_95: float
    
    # Market conditions
    market_regime: str
    volatility_regime: str
    
    # Execution details
    equity_curve: np.ndarray
    positions: np.ndarray
    trade_log: List[Dict]
    
    # Metadata
    duration_days: float
    timestamp: datetime

class MarketRegimeClassifier:
    """Classifies market regimes based on price action"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    def classify_regime(self, prices: np.ndarray) -> Tuple[str, Dict]:
        """Classify market regime and return regime info"""
        if len(prices) < self.window_size:
            return 'insufficient_data', {}
            
        # Calculate returns and volatility
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        trend = np.mean(returns) * 252 * 100  # Annualized percentage
        
        # Define regime thresholds
        high_vol_threshold = 0.6  # 60% annual volatility
        medium_vol_threshold = 0.3  # 30% annual volatility
        trend_threshold = 5  # 5% annual trend
        
        # Classify volatility regime
        if volatility > high_vol_threshold:
            vol_regime = 'high_volatility'
        elif volatility > medium_vol_threshold:
            vol_regime = 'medium_volatility'
        else:
            vol_regime = 'low_volatility'
            
        # Classify trend regime
        if abs(trend) < trend_threshold:
            trend_regime = 'sideways'
        elif trend > trend_threshold:
            trend_regime = 'bull'
        else:
            trend_regime = 'bear'
            
        # Combined regime
        regime = f"{vol_regime}_{trend_regime}"
        
        regime_info = {
            'volatility': volatility,
            'trend': trend,
            'vol_regime': vol_regime,
            'trend_regime': trend_regime,
            'regime_strength': abs(trend) / volatility if volatility > 0 else 0
        }
        
        return regime, regime_info

class ComprehensiveBacktester:
    """Main backtesting engine with advanced features"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = []
        self.regime_classifier = MarketRegimeClassifier(config.regime_window)
        
        # Load data
        self._load_data()
        
        # Initialize agents
        self._initialize_agents()
        
    def _load_data(self):
        """Load and prepare market data"""
        data_config = ConfigData()
        
        # Load price data
        self.price_df = pd.read_csv(data_config.csv_path)
        
        # Load features (try optimized first)
        optimized_path = data_config.predict_ary_path.replace('.npy', '_optimized.npy')
        if os.path.exists(optimized_path):
            print(f"Loading optimized features from {optimized_path}")
            self.features = np.load(optimized_path)
        else:
            print(f"Loading original features from {data_config.predict_ary_path}")
            self.features = np.load(data_config.predict_ary_path)
            
        # Align data
        min_length = min(len(self.price_df), len(self.features))
        self.price_df = self.price_df.iloc[-min_length:]
        self.features = self.features[-min_length:]
        
        # Extract price columns
        self.prices = self.price_df["midpoint"].values
        
        print(f"Loaded {len(self.prices)} data points")
        
    def _initialize_agents(self):
        """Initialize ensemble agents"""
        try:
            # Initialize config
            temp_sim = TradeSimulator(num_sims=1)
            state_dim = temp_sim.state_dim
            
            config = Config()
            config.state_dim = state_dim
            config.net_dims = (128, 64, 32) if state_dim == 8 else (128, 128, 128)
            
            # Load agents
            self.agents = []
            for agent_class in self.config.agent_classes:
                agent = agent_class(
                    config.net_dims,
                    config.state_dim,
                    config.action_dim,
                    gpu_id=-1,  # Use CPU for backtesting stability
                    args=config,
                )
                
                # Load model weights
                agent_name = agent_class.__name__
                model_path = os.path.join(self.config.ensemble_path, f"{agent_name}_0.pth")
                if os.path.exists(model_path):
                    agent.save_or_load_agent(model_path, if_save=False)
                    self.agents.append(agent)
                    print(f"Loaded {agent_name}")
                else:
                    print(f"Warning: Model not found at {model_path}")
                    
        except Exception as e:
            print(f"Error initializing agents: {e}")
            self.agents = []
    
    def run_standard_backtest(self, start_idx: int = None, end_idx: int = None) -> BacktestResult:
        """Run standard backtest on specified period"""
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.prices)
            
        # Classify market regime for this period
        period_prices = self.prices[start_idx:end_idx]
        regime, regime_info = self.regime_classifier.classify_regime(period_prices)
        
        # Run simulation
        result = self._simulate_trading(start_idx, end_idx, f"standard_{start_idx}_{end_idx}")
        result.market_regime = regime
        result.volatility_regime = regime_info.get('vol_regime', 'unknown')
        
        return result
    
    def run_walk_forward_analysis(self) -> List[BacktestResult]:
        """Run walk-forward analysis with overlapping windows"""
        print("Starting walk-forward analysis...")
        
        results = []
        window_size = self.config.walk_forward_window
        overlap_size = int(window_size * self.config.overlap_ratio)
        step_size = window_size - overlap_size
        
        # Calculate windows
        total_length = len(self.prices)
        start_indices = range(0, total_length - window_size, step_size)
        
        print(f"Testing {len(start_indices)} windows of size {window_size} with {overlap_size} overlap")
        
        for i, start_idx in enumerate(start_indices):
            end_idx = min(start_idx + window_size, total_length)
            
            if end_idx - start_idx < window_size // 2:  # Skip if window too small
                continue
                
            print(f"Processing window {i+1}/{len(start_indices)}: {start_idx}-{end_idx}")
            
            try:
                result = self._simulate_trading(start_idx, end_idx, f"walk_forward_window_{i}")
                
                # Add regime classification
                period_prices = self.prices[start_idx:end_idx]
                regime, regime_info = self.regime_classifier.classify_regime(period_prices)
                result.market_regime = regime
                result.volatility_regime = regime_info.get('vol_regime', 'unknown')
                
                results.append(result)
                
            except Exception as e:
                print(f"Error in window {i}: {e}")
                continue
                
        print(f"Completed walk-forward analysis with {len(results)} valid windows")
        return results
    
    def run_regime_specific_backtests(self) -> Dict[str, List[BacktestResult]]:
        """Run backtests grouped by market regime"""
        print("Running regime-specific backtests...")
        
        # First, classify all periods
        window_size = self.config.regime_window
        regime_periods = {}
        
        for start_idx in range(0, len(self.prices) - window_size, window_size // 2):
            end_idx = min(start_idx + window_size, len(self.prices))
            
            period_prices = self.prices[start_idx:end_idx]
            regime, regime_info = self.regime_classifier.classify_regime(period_prices)
            
            if regime not in regime_periods:
                regime_periods[regime] = []
                
            regime_periods[regime].append((start_idx, end_idx))
        
        # Run backtests for each regime
        regime_results = {}
        for regime, periods in regime_periods.items():
            if len(periods) < 2:  # Skip regimes with insufficient data
                continue
                
            print(f"Testing {len(periods)} periods for regime: {regime}")
            regime_results[regime] = []
            
            for i, (start_idx, end_idx) in enumerate(periods):
                try:
                    result = self._simulate_trading(start_idx, end_idx, f"{regime}_period_{i}")
                    result.market_regime = regime
                    regime_results[regime].append(result)
                except Exception as e:
                    print(f"Error in {regime} period {i}: {e}")
                    continue
        
        return regime_results
    
    def run_monte_carlo_backtest(self, num_runs: int = None) -> List[BacktestResult]:
        """Run Monte Carlo backtests with randomized parameters"""
        if num_runs is None:
            num_runs = self.config.monte_carlo_runs
            
        print(f"Running {num_runs} Monte Carlo simulations...")
        
        results = []
        total_length = len(self.prices)
        
        for run in range(num_runs):
            # Randomize window selection
            window_size = np.random.randint(1000, min(5000, total_length // 2))
            start_idx = np.random.randint(0, total_length - window_size)
            end_idx = start_idx + window_size
            
            try:
                result = self._simulate_trading(start_idx, end_idx, f"monte_carlo_run_{run}")
                
                # Add regime classification
                period_prices = self.prices[start_idx:end_idx]
                regime, regime_info = self.regime_classifier.classify_regime(period_prices)
                result.market_regime = regime
                result.volatility_regime = regime_info.get('vol_regime', 'unknown')
                
                results.append(result)
                
                if (run + 1) % 100 == 0:
                    print(f"Completed {run + 1}/{num_runs} Monte Carlo runs")
                    
            except Exception as e:
                print(f"Error in Monte Carlo run {run}: {e}")
                continue
        
        print(f"Completed {len(results)} successful Monte Carlo runs")
        return results
    
    def _simulate_trading(self, start_idx: int, end_idx: int, period_name: str) -> BacktestResult:
        """Simulate trading for specified period"""
        
        # Extract period data
        period_length = end_idx - start_idx
        if period_length < 100:  # Minimum period length
            raise ValueError(f"Period too short: {period_length}")
        
        # Initialize trading simulation
        num_sims = 1
        max_step = (period_length - 60) // self.config.step_gap
        
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": num_sims,
            "max_step": max_step,
            "state_dim": self.features.shape[1],
            "action_dim": 3,
            "if_discrete": True,
            "max_position": self.config.max_position,
            "slippage": self.config.slippage,
            "num_sims": num_sims,
            "step_gap": self.config.step_gap,
            "dataset_path": None,  # We'll use custom data
        }
        
        # Create custom environment with period data
        config = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
        config.starting_cash = self.config.starting_cash
        
        # Simulate trading
        equity_curve = [self.config.starting_cash]
        positions = []
        trade_log = []
        cash = self.config.starting_cash
        position = 0
        
        # Get period prices
        period_prices = self.prices[start_idx:end_idx]
        
        # Simple simulation loop (placeholder - would integrate with actual trading sim)
        for i in range(1, len(period_prices)):
            if len(self.agents) == 0:
                action = 0  # Hold if no agents
            else:
                # Simplified action selection (would use actual agent prediction)
                action = np.random.choice([-1, 0, 1])  # Random for now
            
            current_price = period_prices[i]
            prev_price = period_prices[i-1]
            
            # Execute trade
            if action == 1 and cash >= current_price:  # Buy
                cash -= current_price
                position += 1
                trade_log.append({
                    'timestamp': i,
                    'action': 'buy',
                    'price': current_price,
                    'position': position,
                    'cash': cash
                })
            elif action == -1 and position > 0:  # Sell
                cash += current_price
                position -= 1
                trade_log.append({
                    'timestamp': i,
                    'action': 'sell',
                    'price': current_price,
                    'position': position,
                    'cash': cash
                })
            
            # Update equity
            equity = cash + position * current_price
            equity_curve.append(equity)
            positions.append(position)
        
        # Calculate performance metrics
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Handle edge cases
        if len(returns) == 0 or np.all(returns == 0):
            returns = np.array([0.0])
        
        # Calculate metrics
        total_return = (equity_curve[-1] / equity_curve[0] - 1) if equity_curve[0] != 0 else 0
        sharpe = sharpe_ratio(returns) if len(returns) > 1 else 0
        max_dd = max_drawdown(returns) if len(returns) > 1 else 0
        romad = return_over_max_drawdown(returns) if len(returns) > 1 else 0
        
        # Trade statistics
        buy_trades = [t for t in trade_log if t['action'] == 'buy']
        sell_trades = [t for t in trade_log if t['action'] == 'sell']
        win_rate = 0.5  # Placeholder
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        var_95 = np.percentile(returns, 5) if len(returns) > 1 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 1 and np.any(returns <= var_95) else 0
        
        # Create result
        result = BacktestResult(
            period_name=period_name,
            start_idx=start_idx,
            end_idx=end_idx,
            
            # Performance metrics
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            romad=romad,
            win_rate=win_rate,
            
            # Trade statistics
            num_trades=len(trade_log),
            avg_trade_return=np.mean([abs(t['price']) for t in trade_log]) if trade_log else 0,
            profit_factor=1.0,  # Placeholder
            
            # Risk metrics
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            
            # Market conditions (to be filled by caller)
            market_regime='unknown',
            volatility_regime='unknown',
            
            # Execution details
            equity_curve=equity_curve,
            positions=np.array(positions),
            trade_log=trade_log,
            
            # Metadata
            duration_days=period_length / (60 * 60 * 24),  # Assuming 1-second data
            timestamp=datetime.now()
        )
        
        return result
    
    def generate_summary_statistics(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Generate summary statistics across all results"""
        if not results:
            return {}
        
        # Aggregate metrics
        metrics = {
            'total_return': [r.total_return for r in results],
            'sharpe_ratio': [r.sharpe_ratio for r in results],
            'max_drawdown': [r.max_drawdown for r in results],
            'romad': [r.romad for r in results],
            'win_rate': [r.win_rate for r in results],
            'volatility': [r.volatility for r in results],
            'num_trades': [r.num_trades for r in results]
        }
        
        # Calculate summary statistics
        summary = {}
        for metric, values in metrics.items():
            values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        # Performance by regime
        regime_performance = {}
        for result in results:
            regime = result.market_regime
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(result.sharpe_ratio)
        
        summary['regime_performance'] = {
            regime: {
                'count': len(sharpes),
                'mean_sharpe': np.mean(sharpes),
                'std_sharpe': np.std(sharpes)
            }
            for regime, sharpes in regime_performance.items()
        }
        
        # Overall assessment
        valid_sharpes = [r.sharpe_ratio for r in results if not np.isnan(r.sharpe_ratio)]
        summary['overall_assessment'] = {
            'total_periods': len(results),
            'profitable_periods': len([r for r in results if r.total_return > 0]),
            'profitable_percentage': len([r for r in results if r.total_return > 0]) / len(results) * 100,
            'positive_sharpe_periods': len([s for s in valid_sharpes if s > 0]),
            'mean_sharpe': np.mean(valid_sharpes) if valid_sharpes else 0,
            'sharpe_consistency': np.std(valid_sharpes) if valid_sharpes else 0
        }
        
        return summary

def main():
    """Example usage of comprehensive backtester"""
    
    # Configuration
    config = BacktestConfig(
        ensemble_path="ensemble_optimized_phase2",
        walk_forward_window=3000,
        monte_carlo_runs=100,
        confidence_level=0.95
    )
    
    # Initialize backtester
    backtester = ComprehensiveBacktester(config)
    
    print("🚀 Starting Comprehensive Backtesting")
    print("=" * 60)
    
    # Run different types of backtests
    
    # 1. Standard backtest
    print("\n📊 Running standard backtest...")
    standard_result = backtester.run_standard_backtest()
    print(f"Standard backtest - Sharpe: {standard_result.sharpe_ratio:.3f}, "
          f"Return: {standard_result.total_return:.1%}")
    
    # 2. Walk-forward analysis
    print("\n🚶 Running walk-forward analysis...")
    walkforward_results = backtester.run_walk_forward_analysis()
    if walkforward_results:
        wf_summary = backtester.generate_summary_statistics(walkforward_results)
        print(f"Walk-forward results: {len(walkforward_results)} periods, "
              f"Mean Sharpe: {wf_summary['overall_assessment']['mean_sharpe']:.3f}")
    
    # 3. Regime-specific backtests
    print("\n📈 Running regime-specific backtests...")
    regime_results = backtester.run_regime_specific_backtests()
    for regime, results in regime_results.items():
        if results:
            avg_sharpe = np.mean([r.sharpe_ratio for r in results])
            print(f"Regime {regime}: {len(results)} periods, Avg Sharpe: {avg_sharpe:.3f}")
    
    # 4. Monte Carlo backtests
    print("\n🎲 Running Monte Carlo backtests...")
    mc_results = backtester.run_monte_carlo_backtest(num_runs=50)  # Reduced for demo
    if mc_results:
        mc_summary = backtester.generate_summary_statistics(mc_results)
        print(f"Monte Carlo results: {len(mc_results)} runs, "
              f"Mean Sharpe: {mc_summary['overall_assessment']['mean_sharpe']:.3f}")
    
    print("\n✅ Comprehensive backtesting completed!")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine all results
    all_results = [standard_result] + walkforward_results + mc_results
    for regime_result_list in regime_results.values():
        all_results.extend(regime_result_list)
    
    # Generate final summary
    final_summary = backtester.generate_summary_statistics(all_results)
    
    print(f"\n📋 Final Summary:")
    print(f"Total periods tested: {len(all_results)}")
    print(f"Overall mean Sharpe: {final_summary['overall_assessment']['mean_sharpe']:.3f}")
    print(f"Profitable periods: {final_summary['overall_assessment']['profitable_percentage']:.1f}%")
    
    return all_results, final_summary

if __name__ == "__main__":
    results, summary = main()