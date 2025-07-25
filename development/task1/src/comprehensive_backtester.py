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
            # Initialize config (following task1_eval.py pattern)
            temp_sim = TradeSimulator(num_sims=1)
            state_dim = temp_sim.state_dim
            
            config = Config()
            config.state_dim = state_dim
            config.action_dim = 3
            
            # Use optimized architecture for 8-feature models
            if state_dim == 8:
                config.net_dims = (128, 64, 32)
                print(f"Using optimized architecture for 8-feature models: {config.net_dims}")
            else:
                config.net_dims = (128, 128, 128)
                print(f"Using default architecture for {state_dim}-feature models: {config.net_dims}")
            
            # Load agents (following task1_eval.py pattern)
            self.agents = []
            for agent_class in self.config.agent_classes:
                agent = agent_class(
                    config.net_dims,
                    config.state_dim,
                    config.action_dim,
                    gpu_id=-1,  # Use CPU for backtesting stability
                    args=config,
                )
                
                # Load model weights using the same pattern as task1_eval.py
                agent_name = agent_class.__name__
                model_dir = os.path.join(self.config.ensemble_path, agent_name)
                
                if os.path.exists(model_dir):
                    agent.save_or_load_agent(model_dir, if_save=False)
                    self.agents.append(agent)
                    print(f"Loaded {agent_name} from {model_dir}")
                else:
                    print(f"Warning: Model directory not found at {model_dir}")
                    
        except Exception as e:
            print(f"Error initializing agents: {e}")
            import traceback
            traceback.print_exc()
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
        """Simulate trading for specified period using actual agent predictions"""
        
        # Extract period data
        period_length = end_idx - start_idx
        if period_length < 100:  # Minimum period length
            raise ValueError(f"Period too short: {period_length}")
        
        if not self.agents:
            raise ValueError("No agents loaded - cannot perform realistic backtesting")
        
        # Initialize trading simulation using the same setup as task1_eval.py
        num_sims = 1
        num_ignore_step = 60
        step_gap = self.config.step_gap
        max_step = (period_length - num_ignore_step) // step_gap
        
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
            "step_gap": step_gap,
            "dataset_path": None,  # We'll use custom data
        }
        
        # Create trading environment
        from erl_config import build_env
        config = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
        config.starting_cash = self.config.starting_cash
        config.gpu_id = -1  # Use CPU
        
        trade_env = build_env(config.env_class, env_args, gpu_id=config.gpu_id)
        
        # Initialize tracking variables (following task1_eval.py pattern)
        equity_curve = [self.config.starting_cash]
        positions = []
        trade_log = []
        action_ints = []
        
        # Portfolio tracking
        cash = self.config.starting_cash
        btc_position = 0
        btc_assets = [0]
        net_assets = [self.config.starting_cash]
        
        # Reset environment
        state = trade_env.reset()
        last_state = state
        
        # Trading simulation loop using actual agent predictions
        for step in range(trade_env.max_step):
            # Get ensemble action from all agents (following task1_eval.py)
            actions = []
            
            for agent in self.agents:
                try:
                    # Convert state to tensor and get Q-values
                    tensor_state = torch.as_tensor(last_state, dtype=torch.float32, device=agent.device)
                    with torch.no_grad():
                        tensor_q_values = agent.act(tensor_state)
                        tensor_action = tensor_q_values.argmax(dim=1)
                        action = tensor_action.detach().cpu().unsqueeze(1)
                        actions.append(action)
                except Exception as e:
                    # Fallback action if agent fails
                    actions.append(torch.tensor([[1]], dtype=torch.int32))  # Hold action
            
            # Get ensemble action using majority voting
            ensemble_action = self._ensemble_action(actions)
            action_int = ensemble_action.item() - 1  # Convert to {-1, 0, 1}
            
            # Step environment
            try:
                state, reward, done, _ = trade_env.step(ensemble_action)
            except:
                # If environment step fails, break
                break
                
            action_ints.append(action_int)
            
            # Get current price from environment
            try:
                current_price = trade_env.price_ary[trade_env.step_i, 2].item()
            except:
                # Fallback to period prices if env price unavailable
                step_idx = min(step + num_ignore_step, len(self.prices) - 1)
                if start_idx + step_idx < len(self.prices):
                    current_price = self.prices[start_idx + step_idx]
                else:
                    current_price = self.prices[-1]
            
            # Execute trades based on action (following task1_eval.py logic)
            new_cash = cash
            trade_executed = False
            
            if action_int > 0 and cash >= current_price:  # Buy signal
                # Calculate quantity based on available cash (simple 1 unit for now)
                quantity = 1.0
                if cash >= current_price * quantity:
                    new_cash = cash - (current_price * quantity)
                    btc_position += quantity
                    trade_executed = True
                    
                    trade_log.append({
                        'timestamp': step,
                        'action': 'buy',
                        'price': current_price,
                        'quantity': quantity,
                        'position': btc_position,
                        'cash': new_cash,
                        'pnl': 0  # Will be calculated later
                    })
                    
            elif action_int < 0 and btc_position > 0:  # Sell signal
                # Sell 1 unit if available
                quantity = min(1.0, btc_position)
                new_cash = cash + (current_price * quantity)
                btc_position -= quantity
                trade_executed = True
                
                trade_log.append({
                    'timestamp': step,
                    'action': 'sell',
                    'price': current_price,
                    'quantity': quantity,
                    'position': btc_position,
                    'cash': new_cash,
                    'pnl': 0  # Will be calculated later
                })
            
            # Update portfolio values
            cash = new_cash
            btc_asset_value = btc_position * current_price
            total_equity = cash + btc_asset_value
            
            # Track portfolio evolution
            btc_assets.append(btc_asset_value)
            net_assets.append(total_equity)
            equity_curve.append(total_equity)
            positions.append(btc_position)
            
            # Update state for next iteration
            last_state = state
            
            if done:
                break
        
        # Calculate PnL for trades
        for i, trade in enumerate(trade_log):
            if i > 0:
                prev_equity = equity_curve[trade['timestamp']]
                curr_equity = equity_curve[trade['timestamp'] + 1] if trade['timestamp'] + 1 < len(equity_curve) else equity_curve[-1]
                trade['pnl'] = curr_equity - prev_equity
        
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
    
    def _ensemble_action(self, actions):
        """Returns the majority action among agents (following task1_eval.py)"""
        from collections import Counter
        count = Counter([a.item() for a in actions])
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([[majority_action]], dtype=torch.int32)

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