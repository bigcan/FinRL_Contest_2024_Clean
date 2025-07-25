"""
Task 2 Comprehensive Evaluation Enhancement
Advanced evaluation framework for LLM-based signal generation with detailed analytics
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import task2 components
from task2_news import get_news
from task2_signal import generate_eval_signal
from task2_config import Task2Config

@dataclass
class ComprehensiveEvaluationConfig:
    """Configuration for comprehensive evaluation"""
    
    # Model parameters
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    use_quantization: bool = True
    quantization_bits: int = 8
    
    # Evaluation parameters
    signal_strength: float = 5.0
    threshold: float = 0.5
    num_long: int = 3
    num_short: int = 3
    
    # Portfolio parameters
    equal_weight: bool = True
    portfolio_value: float = 100000.0
    transaction_cost: float = 0.001
    max_position_size: float = 0.2
    
    # Risk management
    max_daily_loss: float = 0.05
    position_sizing_method: str = "equal"  # equal, kelly, risk_parity
    
    # Evaluation periods
    warmup_days: int = 10
    lookback_window: int = 20
    
    # Output configuration
    save_detailed_results: bool = True
    create_visualizations: bool = True
    generate_report: bool = True

class ComprehensiveTask2Evaluator:
    """Comprehensive evaluation framework for Task 2"""
    
    def __init__(self, config: ComprehensiveEvaluationConfig, output_dir: str = "task2_comprehensive_results"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.daily_results = []
        self.signal_history = []
        self.portfolio_history = []
        self.performance_metrics = {}
        
        # Load model and tokenizer
        self._initialize_model()
        
        # Load data
        self._load_data()
        
        # Initialize evaluation tracking
        self._initialize_tracking()
    
    def _initialize_model(self):
        """Initialize LLM model and tokenizer"""
        print("🤖 Initializing LLM model...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if self.config.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=self.config.quantization_bits == 8,
                load_in_4bit=self.config.quantization_bits == 4
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto"
            )
        
        print("✅ Model initialized successfully")
    
    def _load_data(self):
        """Load stock and news data"""
        print("📊 Loading evaluation data...")
        
        # Load stock data
        self.stock_data = pd.read_csv("task2_dsets/test/task2_stocks_test.csv")
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        
        # Load news data
        self.news_data = pd.read_csv("task2_dsets/test/task2_news_test.csv")
        self.news_data['Date'] = pd.to_datetime(self.news_data['Date'])
        
        # Get available tickers and dates
        self.tickers = self.stock_data['Ticker'].unique().tolist()
        self.eval_dates = sorted(self.stock_data['Date'].unique())
        
        print(f"   Loaded {len(self.stock_data)} stock observations")
        print(f"   Loaded {len(self.news_data)} news items")
        print(f"   Tickers: {self.tickers}")
        print(f"   Date range: {self.eval_dates[0]} to {self.eval_dates[-1]}")
    
    def _initialize_tracking(self):
        """Initialize performance tracking variables"""
        self.portfolio_value = self.config.portfolio_value
        self.positions = {ticker: 0.0 for ticker in self.tickers}
        self.cash = self.config.portfolio_value
        
        # Performance tracking
        self.returns_history = []
        self.sharpe_history = []
        self.drawdown_history = []
        self.volatility_history = []
        
        # Signal quality tracking
        self.signal_accuracy = []
        self.signal_strength_distribution = []
        self.regime_performance = {}
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation with detailed analytics"""
        
        print("🚀 Starting comprehensive evaluation...")
        print(f"   Evaluation period: {len(self.eval_dates)} trading days")
        print(f"   Portfolio value: ${self.config.portfolio_value:,.2f}")
        
        # Main evaluation loop
        for date_idx, current_date in enumerate(tqdm(self.eval_dates, desc="Evaluating")):
            daily_result = self._evaluate_single_day(current_date, date_idx)
            self.daily_results.append(daily_result)
            
            # Update portfolio tracking
            self._update_portfolio_tracking(daily_result)
            
            # Calculate rolling metrics
            if date_idx >= self.config.lookback_window:
                self._calculate_rolling_metrics(date_idx)
        
        # Calculate final performance metrics
        self._calculate_final_metrics()
        
        # Generate outputs
        if self.config.save_detailed_results:
            self._save_detailed_results()
        
        if self.config.create_visualizations:
            self._create_comprehensive_visualizations()
        
        if self.config.generate_report:
            self._generate_comprehensive_report()
        
        print("🎉 Comprehensive evaluation completed!")
        return self.performance_metrics
    
    def _evaluate_single_day(self, current_date: datetime, date_idx: int) -> Dict:
        """Evaluate a single trading day"""
        
        # Get stock prices for current date
        daily_prices = self.stock_data[self.stock_data['Date'] == current_date]
        
        if daily_prices.empty:
            return {'date': current_date, 'no_data': True}
        
        # Generate signals for all tickers
        ticker_signals = {}
        signal_details = {}
        
        for ticker in self.tickers:
            ticker_prices = daily_prices[daily_prices['Ticker'] == ticker]
            
            if ticker_prices.empty:
                continue
            
            # Get news for the ticker
            news = get_news(
                ticker,
                (current_date - timedelta(days=1)).strftime('%Y-%m-%d'),
                (current_date - timedelta(days=11)).strftime('%Y-%m-%d'),
                "task2_dsets/test/task2_news_test.csv"
            )
            
            # Generate signal
            signal_score = generate_eval_signal(
                self.tokenizer,
                self.model,
                self.device,
                news,
                ticker_prices.copy().drop("Future_Close", axis=1),
                self.config.signal_strength,
                self.config.threshold
            )
            
            ticker_signals[ticker] = signal_score
            signal_details[ticker] = {
                'signal': signal_score,
                'news_count': len(news) if news else 0,
                'current_price': ticker_prices['Close'].iloc[0],
                'future_price': ticker_prices['Future_Close'].iloc[0] if 'Future_Close' in ticker_prices.columns else None
            }
        
        # Portfolio construction
        portfolio_result = self._construct_portfolio(ticker_signals, signal_details, current_date)
        
        # Calculate actual returns
        actual_returns = self._calculate_actual_returns(portfolio_result, signal_details)
        
        return {
            'date': current_date,
            'ticker_signals': ticker_signals,
            'signal_details': signal_details,
            'portfolio': portfolio_result,
            'returns': actual_returns,
            'portfolio_value': self.portfolio_value
        }
    
    def _construct_portfolio(self, signals: Dict, signal_details: Dict, current_date: datetime) -> Dict:
        """Construct portfolio based on signals"""
        
        # Sort tickers by signal strength
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        
        # Select long and short positions
        long_candidates = [(ticker, signal) for ticker, signal in sorted_signals 
                          if signal > self.config.threshold]
        short_candidates = [(ticker, signal) for ticker, signal in sorted_signals 
                           if signal < -self.config.threshold]
        
        # Limit positions
        selected_long = long_candidates[:self.config.num_long]
        selected_short = short_candidates[:self.config.num_short]
        
        # Calculate position sizes
        position_sizes = self._calculate_position_sizes(selected_long + selected_short, signals)
        
        return {
            'long_positions': selected_long,
            'short_positions': selected_short,
            'position_sizes': position_sizes,
            'total_positions': len(selected_long) + len(selected_short)
        }
    
    def _calculate_position_sizes(self, selected_positions: List, signals: Dict) -> Dict:
        """Calculate position sizes based on configuration"""
        
        if not selected_positions:
            return {}
        
        position_sizes = {}
        total_positions = len(selected_positions)
        
        if self.config.position_sizing_method == "equal":
            # Equal weight allocation
            weight_per_position = 1.0 / total_positions
            for ticker, signal in selected_positions:
                position_sizes[ticker] = min(weight_per_position, self.config.max_position_size)
        
        elif self.config.position_sizing_method == "signal_weighted":
            # Weight by signal strength
            total_signal_strength = sum(abs(signals[ticker]) for ticker, _ in selected_positions)
            for ticker, signal in selected_positions:
                if total_signal_strength > 0:
                    weight = abs(signal) / total_signal_strength
                    position_sizes[ticker] = min(weight, self.config.max_position_size)
                else:
                    position_sizes[ticker] = 1.0 / total_positions
        
        elif self.config.position_sizing_method == "risk_parity":
            # Risk parity (simplified - equal risk contribution)
            risk_per_position = 1.0 / total_positions
            for ticker, signal in selected_positions:
                position_sizes[ticker] = min(risk_per_position, self.config.max_position_size)
        
        # Normalize to ensure sum <= 1.0
        total_weight = sum(position_sizes.values())
        if total_weight > 1.0:
            for ticker in position_sizes:
                position_sizes[ticker] /= total_weight
        
        return position_sizes
    
    def _calculate_actual_returns(self, portfolio: Dict, signal_details: Dict) -> Dict:
        """Calculate actual portfolio returns"""
        
        daily_return = 0.0
        position_returns = {}
        
        # Long positions
        for ticker, signal in portfolio['long_positions']:
            if ticker in signal_details and signal_details[ticker]['future_price'] is not None:
                current_price = signal_details[ticker]['current_price']
                future_price = signal_details[ticker]['future_price']
                
                # Calculate return (including transaction costs)
                gross_return = (future_price - current_price) / current_price
                net_return = gross_return - self.config.transaction_cost
                
                position_size = portfolio['position_sizes'].get(ticker, 0)
                weighted_return = net_return * position_size
                
                daily_return += weighted_return
                position_returns[ticker] = {'return': net_return, 'weight': position_size, 'direction': 'long'}
        
        # Short positions
        for ticker, signal in portfolio['short_positions']:
            if ticker in signal_details and signal_details[ticker]['future_price'] is not None:
                current_price = signal_details[ticker]['current_price']
                future_price = signal_details[ticker]['future_price']
                
                # Calculate short return (including transaction costs)
                gross_return = -(future_price - current_price) / current_price
                net_return = gross_return - self.config.transaction_cost
                
                position_size = portfolio['position_sizes'].get(ticker, 0)
                weighted_return = net_return * position_size
                
                daily_return += weighted_return
                position_returns[ticker] = {'return': net_return, 'weight': position_size, 'direction': 'short'}
        
        return {
            'daily_return': daily_return,
            'position_returns': position_returns,
            'gross_exposure': sum(portfolio['position_sizes'].values()),
            'num_positions': len(position_returns)
        }
    
    def _update_portfolio_tracking(self, daily_result: Dict):
        """Update portfolio tracking with daily results"""
        
        if 'returns' not in daily_result:
            return
        
        daily_return = daily_result['returns']['daily_return']
        
        # Update portfolio value
        self.portfolio_value *= (1 + daily_return)
        
        # Track returns
        self.returns_history.append(daily_return)
        
        # Track signal quality
        if 'ticker_signals' in daily_result:
            signals = list(daily_result['ticker_signals'].values())
            self.signal_strength_distribution.extend(signals)
            
            # Calculate signal accuracy (simplified)
            if 'returns' in daily_result and 'position_returns' in daily_result['returns']:
                accurate_signals = sum(1 for ticker, pos_data in daily_result['returns']['position_returns'].items()
                                     if pos_data['return'] > 0)
                total_signals = len(daily_result['returns']['position_returns'])
                if total_signals > 0:
                    self.signal_accuracy.append(accurate_signals / total_signals)
    
    def _calculate_rolling_metrics(self, current_idx: int):
        """Calculate rolling performance metrics"""
        
        if len(self.returns_history) < self.config.lookback_window:
            return
        
        recent_returns = self.returns_history[-self.config.lookback_window:]
        
        # Rolling Sharpe ratio
        if len(recent_returns) > 1:
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            self.sharpe_history.append(sharpe)
        
        # Rolling volatility
        volatility = np.std(recent_returns) * np.sqrt(252) if len(recent_returns) > 1 else 0
        self.volatility_history.append(volatility)
        
        # Rolling drawdown
        cumulative_returns = np.cumprod([1 + r for r in recent_returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdown)
        self.drawdown_history.append(max_drawdown)
    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics"""
        
        if not self.returns_history:
            self.performance_metrics = {'error': 'No returns data available'}
            return
        
        returns = np.array(self.returns_history)
        
        # Basic performance metrics
        total_return = (self.portfolio_value / self.config.portfolio_value) - 1
        mean_daily_return = np.mean(returns)
        annual_return = mean_daily_return * 252
        
        # Risk metrics
        daily_volatility = np.std(returns)
        annual_volatility = daily_volatility * np.sqrt(252)
        sharpe_ratio = (annual_return / annual_volatility) if annual_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Win/Loss analysis
        winning_days = np.sum(returns > 0)
        losing_days = np.sum(returns < 0)
        win_rate = winning_days / len(returns) if len(returns) > 0 else 0
        
        # Average win/loss
        avg_win = np.mean(returns[returns > 0]) if winning_days > 0 else 0
        avg_loss = np.mean(returns[returns < 0]) if losing_days > 0 else 0
        
        # Signal quality metrics
        avg_signal_accuracy = np.mean(self.signal_accuracy) if self.signal_accuracy else 0
        signal_strength_std = np.std(self.signal_strength_distribution) if self.signal_strength_distribution else 0
        
        # Risk-adjusted metrics
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        self.performance_metrics = {
            # Return metrics
            'total_return': total_return,
            'annual_return': annual_return,
            'mean_daily_return': mean_daily_return,
            
            # Risk metrics
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            
            # Win/Loss metrics
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_days / (avg_loss * losing_days)) if avg_loss != 0 and losing_days > 0 else 0,
            
            # Portfolio metrics
            'final_portfolio_value': self.portfolio_value,
            'trading_days': len(returns),
            
            # Signal quality metrics
            'avg_signal_accuracy': avg_signal_accuracy,
            'signal_strength_std': signal_strength_std,
            'total_signals_generated': len(self.signal_strength_distribution)
        }
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        annual_return = np.mean(returns) * 252
        
        return annual_return / downside_deviation if downside_deviation > 0 else 0
    
    def _save_detailed_results(self):
        """Save detailed results to files"""
        
        print("💾 Saving detailed results...")
        
        # Save daily results
        daily_results_path = os.path.join(self.output_dir, 'daily_results.json')
        with open(daily_results_path, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_results = []
            for result in self.daily_results:
                serializable_result = result.copy()
                if 'date' in serializable_result:
                    serializable_result['date'] = serializable_result['date'].isoformat()
                serializable_results.append(serializable_result)
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save performance metrics
        metrics_path = os.path.join(self.output_dir, 'performance_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        # Save time series data
        time_series_data = {
            'dates': [d.isoformat() for d in self.eval_dates[:len(self.returns_history)]],
            'daily_returns': self.returns_history,
            'portfolio_values': [self.config.portfolio_value * np.prod([1 + r for r in self.returns_history[:i+1]]) 
                               for i in range(len(self.returns_history))],
            'sharpe_history': self.sharpe_history,
            'volatility_history': self.volatility_history,
            'drawdown_history': self.drawdown_history
        }
        
        timeseries_path = os.path.join(self.output_dir, 'time_series_data.json')
        with open(timeseries_path, 'w') as f:
            json.dump(time_series_data, f, indent=2)
        
        print(f"   Results saved to {self.output_dir}")
    
    def _create_comprehensive_visualizations(self):
        """Create comprehensive visualization suite"""
        
        print("📊 Creating comprehensive visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
        # Create main performance dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio Value Over Time
        ax1 = fig.add_subplot(gs[0, :])
        portfolio_values = [self.config.portfolio_value * np.prod([1 + r for r in self.returns_history[:i+1]]) 
                           for i in range(len(self.returns_history))]
        dates = self.eval_dates[:len(self.returns_history)]
        ax1.plot(dates, portfolio_values, linewidth=2, color=colors[0])
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Daily Returns Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.returns_history, bins=50, alpha=0.7, color=colors[1], edgecolor='black')
        ax2.axvline(np.mean(self.returns_history), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.returns_history):.4f}')
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio
        ax3 = fig.add_subplot(gs[1, 1])
        if self.sharpe_history:
            rolling_dates = dates[self.config.lookback_window:]
            ax3.plot(rolling_dates, self.sharpe_history, color=colors[2], linewidth=2)
            ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('Rolling Sharpe Ratio')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Drawdown Chart
        ax4 = fig.add_subplot(gs[1, 2])
        if self.drawdown_history:
            rolling_dates = dates[self.config.lookback_window:]
            ax4.fill_between(rolling_dates, 0, [-d for d in self.drawdown_history], 
                           color=colors[3], alpha=0.7)
            ax4.set_title('Rolling Drawdown')
            ax4.set_ylabel('Drawdown')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Signal Strength Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        if self.signal_strength_distribution:
            ax5.hist(self.signal_strength_distribution, bins=50, alpha=0.7, 
                    color=colors[4], edgecolor='black')
            ax5.axvline(self.config.threshold, color='red', linestyle='--', label='Long Threshold')
            ax5.axvline(-self.config.threshold, color='red', linestyle='--', label='Short Threshold')
            ax5.set_title('Signal Strength Distribution')
            ax5.set_xlabel('Signal Strength')
            ax5.set_ylabel('Frequency')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Win Rate Over Time
        ax6 = fig.add_subplot(gs[2, 1])
        if self.signal_accuracy:
            accuracy_dates = dates[:len(self.signal_accuracy)]
            ax6.plot(accuracy_dates, self.signal_accuracy, color=colors[5], linewidth=2)
            ax6.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Random')
            ax6.set_title('Signal Accuracy Over Time')
            ax6.set_ylabel('Accuracy')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. Risk-Return Scatter (Monthly)
        ax7 = fig.add_subplot(gs[2, 2])
        if len(self.returns_history) >= 21:  # At least 21 days for monthly analysis
            monthly_returns = []
            monthly_volatilities = []
            
            for i in range(21, len(self.returns_history), 21):
                month_returns = self.returns_history[i-21:i]
                monthly_returns.append(np.mean(month_returns))
                monthly_volatilities.append(np.std(month_returns))
            
            if monthly_returns and monthly_volatilities:
                ax7.scatter(monthly_volatilities, monthly_returns, 
                           color=colors[6], alpha=0.7, s=50)
                ax7.set_title('Risk-Return Profile (Monthly)')
                ax7.set_xlabel('Volatility')
                ax7.set_ylabel('Return')
                ax7.grid(True, alpha=0.3)
        
        # 8. Performance Metrics Summary
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        metrics_text = f"""
        PERFORMANCE SUMMARY
        
        Total Return: {self.performance_metrics.get('total_return', 0):.2%}
        Annual Return: {self.performance_metrics.get('annual_return', 0):.2%}
        Sharpe Ratio: {self.performance_metrics.get('sharpe_ratio', 0):.3f}
        Max Drawdown: {self.performance_metrics.get('max_drawdown', 0):.2%}
        
        Win Rate: {self.performance_metrics.get('win_rate', 0):.2%}
        Avg Win: {self.performance_metrics.get('avg_win', 0):.4f}
        Avg Loss: {self.performance_metrics.get('avg_loss', 0):.4f}
        Profit Factor: {self.performance_metrics.get('profit_factor', 0):.2f}
        
        Final Portfolio Value: ${self.performance_metrics.get('final_portfolio_value', 0):,.2f}
        Trading Days: {self.performance_metrics.get('trading_days', 0)}
        Signal Accuracy: {self.performance_metrics.get('avg_signal_accuracy', 0):.2%}
        """
        
        ax8.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('Task 2 Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, 'comprehensive_performance_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Visualizations saved to {plot_path}")
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive markdown report"""
        
        print("📝 Generating comprehensive report...")
        
        report_path = os.path.join(self.output_dir, 'comprehensive_evaluation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Task 2 Comprehensive Evaluation Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents a comprehensive evaluation of the LLM-based signal generation system for Task 2 of the FinRL Contest 2024.\n\n")
            
            # Key Performance Metrics
            f.write("## Key Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in self.performance_metrics.items():
                if isinstance(value, float):
                    if 'rate' in key or 'return' in key or 'drawdown' in key:
                        f.write(f"| {key.replace('_', ' ').title()} | {value:.2%} |\n")
                    else:
                        f.write(f"| {key.replace('_', ' ').title()} | {value:.4f} |\n")
                else:
                    f.write(f"| {key.replace('_', ' ').title()} | {value} |\n")
            f.write("\n")
            
            # Configuration
            f.write("## Evaluation Configuration\n\n")
            f.write(f"- **Model**: {self.config.model_name}\n")
            f.write(f"- **Signal Threshold**: {self.config.threshold}\n")
            f.write(f"- **Long Positions**: {self.config.num_long}\n")
            f.write(f"- **Short Positions**: {self.config.num_short}\n")
            f.write(f"- **Portfolio Value**: ${self.config.portfolio_value:,.2f}\n")
            f.write(f"- **Transaction Cost**: {self.config.transaction_cost:.3f}\n")
            f.write(f"- **Position Sizing**: {self.config.position_sizing_method}\n\n")
            
            # Analysis
            f.write("## Performance Analysis\n\n")
            
            # Risk-Return Analysis
            f.write("### Risk-Return Profile\n")
            total_return = self.performance_metrics.get('total_return', 0)
            sharpe_ratio = self.performance_metrics.get('sharpe_ratio', 0)
            max_drawdown = self.performance_metrics.get('max_drawdown', 0)
            
            if total_return > 0.1 and sharpe_ratio > 1.0:
                f.write("✅ **Strong Performance**: High returns with good risk-adjusted performance.\n\n")
            elif total_return > 0 and sharpe_ratio > 0.5:
                f.write("⚠️ **Moderate Performance**: Positive returns but room for improvement in risk management.\n\n")
            else:
                f.write("❌ **Underperformance**: Returns and risk metrics need improvement.\n\n")
            
            # Signal Quality Analysis
            f.write("### Signal Quality Assessment\n")
            signal_accuracy = self.performance_metrics.get('avg_signal_accuracy', 0)
            
            if signal_accuracy > 0.6:
                f.write("✅ **High Signal Quality**: Signals consistently predict market direction.\n\n")
            elif signal_accuracy > 0.5:
                f.write("⚠️ **Moderate Signal Quality**: Signals show some predictive power.\n\n")
            else:
                f.write("❌ **Poor Signal Quality**: Signals need significant improvement.\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### Short-term Improvements\n")
            f.write("1. **Signal Enhancement**: Focus on improving signal accuracy through better prompt engineering\n")
            f.write("2. **Risk Management**: Implement dynamic position sizing based on signal confidence\n")
            f.write("3. **Transaction Costs**: Optimize trading frequency to minimize costs\n\n")
            
            f.write("### Long-term Strategy\n")
            f.write("1. **Model Fine-tuning**: Consider fine-tuning the LLM on financial data\n")
            f.write("2. **Multi-timeframe Analysis**: Incorporate multiple time horizons\n")
            f.write("3. **Alternative Data**: Integrate additional data sources beyond news\n\n")
            
            # Data Summary
            f.write("## Data Summary\n\n")
            f.write(f"- **Evaluation Period**: {self.eval_dates[0].strftime('%Y-%m-%d')} to {self.eval_dates[-1].strftime('%Y-%m-%d')}\n")
            f.write(f"- **Trading Days**: {len(self.returns_history)}\n")
            f.write(f"- **Tickers Analyzed**: {len(self.tickers)}\n")
            f.write(f"- **Total Signals Generated**: {self.performance_metrics.get('total_signals_generated', 0)}\n\n")
        
        print(f"   Report saved to {report_path}")

def run_comprehensive_task2_evaluation(output_dir: str = "task2_comprehensive_results"):
    """Run comprehensive Task 2 evaluation"""
    
    print("🚀 Starting Comprehensive Task 2 Evaluation")
    print(f"📂 Output Directory: {output_dir}")
    
    try:
        # Initialize configuration
        config = ComprehensiveEvaluationConfig()
        
        # Create evaluator
        evaluator = ComprehensiveTask2Evaluator(config, output_dir)
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        print("\n📊 Final Results Summary:")
        print(f"   Total Return: {results.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
        print(f"   Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"   Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"   Signal Accuracy: {results.get('avg_signal_accuracy', 0):.2%}")
        
        return results
        
    except Exception as e:
        print(f"❌ Comprehensive evaluation failed: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "task2_comprehensive_results"
    
    results = run_comprehensive_task2_evaluation(output_dir)