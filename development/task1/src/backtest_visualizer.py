"""
Backtest Visualization Dashboard
Interactive and static visualizations for comprehensive backtest analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BacktestVisualizer:
    """Comprehensive backtest visualization framework"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'neutral': '#6c757d'
        }
        
    def create_comprehensive_dashboard(self, backtest_results: List, 
                                     market_conditions: List = None,
                                     save_path: str = None) -> plt.Figure:
        """Create comprehensive backtest dashboard"""
        
        if not backtest_results:
            print("No backtest results provided")
            return None
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curves (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curves(ax1, backtest_results)
        
        # 2. Performance Metrics Heatmap (second row, left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_performance_heatmap(ax2, backtest_results)
        
        # 3. Return Distribution (second row, middle)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_return_distribution(ax3, backtest_results)
        
        # 4. Drawdown Analysis (second row, right)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_drawdown_analysis(ax4, backtest_results)
        
        # 5. Rolling Performance (third row, left two)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_rolling_performance(ax5, backtest_results)
        
        # 6. Risk Metrics (third row, right)
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_risk_metrics(ax6, backtest_results)
        
        # 7. Regime Performance (fourth row, full width)
        if market_conditions:
            ax7 = fig.add_subplot(gs[3, :])
            self._plot_regime_performance(ax7, backtest_results, market_conditions)
        else:
            ax7 = fig.add_subplot(gs[3, :])
            self._plot_performance_by_period(ax7, backtest_results)
        
        # 8. Trade Analysis (fifth row)
        ax8 = fig.add_subplot(gs[4, 0])
        self._plot_trade_analysis(ax8, backtest_results)
        
        # 9. Volatility Analysis (fifth row, middle)
        ax9 = fig.add_subplot(gs[4, 1])
        self._plot_volatility_analysis(ax9, backtest_results)
        
        # 10. Performance Correlation (fifth row, right)
        ax10 = fig.add_subplot(gs[4, 2])
        self._plot_performance_correlation(ax10, backtest_results)
        
        # 11. Summary Statistics Table (bottom row)
        ax11 = fig.add_subplot(gs[5, :])
        self._plot_summary_table(ax11, backtest_results)
        
        # Add overall title
        fig.suptitle('Comprehensive Backtest Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        return fig
    
    def _plot_equity_curves(self, ax, backtest_results):
        """Plot equity curves for all backtest periods"""
        ax.set_title('Equity Curves Over Time', fontsize=14, fontweight='bold')
        
        # Combine all equity curves
        all_curves = []
        for i, result in enumerate(backtest_results[:10]):  # Limit to first 10 for clarity
            if hasattr(result, 'equity_curve'):
                equity = result.equity_curve
                if len(equity) > 1:
                    normalized_equity = equity / equity[0]  # Normalize to starting value
                    ax.plot(normalized_equity, alpha=0.6, linewidth=1, 
                           label=f'Period {i+1}' if i < 5 else None)
                    all_curves.append(normalized_equity)
        
        # Plot average curve
        if all_curves:
            # Align lengths and calculate mean
            min_length = min(len(curve) for curve in all_curves)
            aligned_curves = [curve[:min_length] for curve in all_curves]
            mean_curve = np.mean(aligned_curves, axis=0)
            
            ax.plot(mean_curve, color=self.colors['danger'], linewidth=3, 
                   label='Average', alpha=0.9)
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Normalized Portfolio Value')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
    def _plot_performance_heatmap(self, ax, backtest_results):
        """Plot performance metrics heatmap"""
        ax.set_title('Performance Metrics Heatmap', fontsize=12, fontweight='bold')
        
        # Collect metrics
        metrics_data = []
        for i, result in enumerate(backtest_results[:20]):  # Limit for readability
            metrics_data.append([
                result.total_return * 100,
                result.sharpe_ratio,
                result.max_drawdown * 100,
                result.win_rate * 100,
                result.volatility * 100 if hasattr(result, 'volatility') else 0
            ])
        
        if metrics_data:
            df = pd.DataFrame(metrics_data, 
                            columns=['Return (%)', 'Sharpe', 'Max DD (%)', 'Win Rate (%)', 'Volatility (%)'])
            
            # Create heatmap
            im = ax.imshow(df.T, cmap='RdYlGn', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels([f'P{i+1}' for i in range(len(df))], rotation=45)
            ax.set_yticks(range(len(df.columns)))
            ax.set_yticklabels(df.columns)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
    def _plot_return_distribution(self, ax, backtest_results):
        """Plot return distribution analysis"""
        ax.set_title('Return Distribution', fontsize=12, fontweight='bold')
        
        # Collect all returns
        all_returns = []
        for result in backtest_results:
            if hasattr(result, 'equity_curve') and len(result.equity_curve) > 1:
                returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
                all_returns.extend(returns)
        
        if all_returns:
            # Plot histogram
            ax.hist(np.array(all_returns) * 100, bins=50, alpha=0.7, 
                   color=self.colors['primary'], density=True)
            
            # Overlay normal distribution
            mean_ret = np.mean(all_returns) * 100
            std_ret = np.std(all_returns) * 100
            x = np.linspace(mean_ret - 4*std_ret, mean_ret + 4*std_ret, 100)
            normal_dist = (1/(std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_ret) / std_ret) ** 2)
            ax.plot(x, normal_dist, 'r--', linewidth=2, label='Normal Distribution')
            
            ax.set_xlabel('Returns (%)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
    def _plot_drawdown_analysis(self, ax, backtest_results):
        """Plot drawdown analysis"""
        ax.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
        
        # Collect drawdown data
        drawdowns = []
        for result in backtest_results:
            if hasattr(result, 'equity_curve') and len(result.equity_curve) > 1:
                equity = result.equity_curve
                running_max = np.maximum.accumulate(equity)
                drawdown = (equity - running_max) / running_max
                drawdowns.extend(drawdown * 100)
        
        if drawdowns:
            # Plot underwater curve (average)
            ax.fill_between(range(len(drawdowns)), drawdowns, 0, 
                          color=self.colors['danger'], alpha=0.3)
            ax.plot(drawdowns, color=self.colors['danger'], linewidth=1)
            
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            max_dd = min(drawdowns)
            avg_dd = np.mean([dd for dd in drawdowns if dd < 0])
            ax.text(0.02, 0.98, f'Max DD: {max_dd:.1f}%\nAvg DD: {avg_dd:.1f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    def _plot_rolling_performance(self, ax, backtest_results):
        """Plot rolling performance metrics"""
        ax.set_title('Rolling Performance Metrics', fontsize=12, fontweight='bold')
        
        # Calculate rolling metrics
        window = min(20, len(backtest_results) // 4)
        if window < 5:
            window = len(backtest_results)
        
        rolling_sharpe = []
        rolling_returns = []
        
        for i in range(window, len(backtest_results) + 1):
            window_results = backtest_results[i-window:i]
            
            # Average Sharpe ratio
            sharpes = [r.sharpe_ratio for r in window_results if not np.isnan(r.sharpe_ratio)]
            if sharpes:
                rolling_sharpe.append(np.mean(sharpes))
            
            # Average returns
            returns = [r.total_return for r in window_results]
            rolling_returns.append(np.mean(returns))
        
        if rolling_sharpe and rolling_returns:
            ax2 = ax.twinx()
            
            # Plot rolling Sharpe
            line1 = ax.plot(range(window, len(backtest_results) + 1), rolling_sharpe, 
                           color=self.colors['primary'], linewidth=2, label='Rolling Sharpe')
            ax.set_ylabel('Rolling Sharpe Ratio', color=self.colors['primary'])
            
            # Plot rolling returns
            line2 = ax2.plot(range(window, len(backtest_results) + 1), 
                           np.array(rolling_returns) * 100,
                           color=self.colors['secondary'], linewidth=2, label='Rolling Return (%)')
            ax2.set_ylabel('Rolling Return (%)', color=self.colors['secondary'])
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.set_xlabel('Period')
            ax.grid(True, alpha=0.3)
        
    def _plot_risk_metrics(self, ax, backtest_results):
        """Plot risk metrics radar chart"""
        ax.set_title('Risk Metrics', fontsize=12, fontweight='bold')
        
        # Calculate average risk metrics
        var_95_values = []
        volatility_values = []
        max_dd_values = []
        
        for result in backtest_results:
            if hasattr(result, 'var_95'):
                var_95_values.append(abs(result.var_95))
            if hasattr(result, 'volatility'):
                volatility_values.append(result.volatility)
            max_dd_values.append(abs(result.max_drawdown))
        
        # Create bar chart of risk metrics
        metrics = ['VaR 95%', 'Volatility', 'Max Drawdown']
        values = [
            np.mean(var_95_values) * 100 if var_95_values else 0,
            np.mean(volatility_values) * 100 if volatility_values else 0,
            np.mean(max_dd_values) * 100
        ]
        
        colors = [self.colors['warning'], self.colors['info'], self.colors['danger']]
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%', ha='center', va='bottom')
        
        ax.set_ylabel('Value (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_regime_performance(self, ax, backtest_results, market_conditions):
        """Plot performance by market regime"""
        ax.set_title('Performance by Market Regime', fontsize=12, fontweight='bold')
        
        # Group results by regime
        regime_performance = {}
        
        for result in backtest_results:
            regime = getattr(result, 'market_regime', 'unknown')
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(result.total_return * 100)
        
        if regime_performance:
            # Create box plot
            regimes = list(regime_performance.keys())
            data = [regime_performance[regime] for regime in regimes]
            
            bp = ax.boxplot(data, labels=regimes, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(regimes)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Return (%)')
            ax.set_xlabel('Market Regime')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_performance_by_period(self, ax, backtest_results):
        """Plot performance by time period (fallback)"""
        ax.set_title('Performance by Period', fontsize=12, fontweight='bold')
        
        returns = [r.total_return * 100 for r in backtest_results]
        sharpes = [r.sharpe_ratio for r in backtest_results if not np.isnan(r.sharpe_ratio)]
        
        if returns:
            periods = range(1, len(returns) + 1)
            
            # Create dual-axis plot
            ax2 = ax.twinx()
            
            # Plot returns as bars
            bars = ax.bar(periods, returns, alpha=0.6, color=self.colors['primary'], 
                         label='Returns (%)')
            
            # Plot Sharpe as line
            if sharpes and len(sharpes) == len(returns):
                line = ax2.plot(periods, sharpes, color=self.colors['danger'], 
                               linewidth=2, marker='o', label='Sharpe Ratio')
                ax2.set_ylabel('Sharpe Ratio', color=self.colors['danger'])
            
            ax.set_xlabel('Period')
            ax.set_ylabel('Return (%)', color=self.colors['primary'])
            ax.grid(True, alpha=0.3)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
    def _plot_trade_analysis(self, ax, backtest_results):
        """Plot trade analysis"""
        ax.set_title('Trade Analysis', fontsize=12, fontweight='bold')
        
        # Collect trade data
        num_trades = [r.num_trades for r in backtest_results if hasattr(r, 'num_trades')]
        win_rates = [r.win_rate * 100 for r in backtest_results]
        
        if num_trades and win_rates:
            # Scatter plot of trades vs win rate
            ax.scatter(num_trades, win_rates, alpha=0.6, s=50, 
                      color=self.colors['success'])
            
            # Add trend line
            if len(num_trades) > 1:
                z = np.polyfit(num_trades, win_rates, 1)
                p = np.poly1d(z)
                ax.plot(sorted(num_trades), p(sorted(num_trades)), 
                       "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Number of Trades')
            ax.set_ylabel('Win Rate (%)')
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            if len(num_trades) > 1:
                corr = np.corrcoef(num_trades, win_rates)[0, 1]
                ax.text(0.02, 0.98, f'Correlation: {corr:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    def _plot_volatility_analysis(self, ax, backtest_results):
        """Plot volatility analysis"""
        ax.set_title('Volatility Analysis', fontsize=12, fontweight='bold')
        
        # Extract volatility data
        volatilities = []
        for result in backtest_results:
            if hasattr(result, 'volatility') and result.volatility > 0:
                volatilities.append(result.volatility * 100)
        
        if volatilities:
            # Plot histogram
            ax.hist(volatilities, bins=20, alpha=0.7, color=self.colors['info'], 
                   density=True, edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_vol = np.mean(volatilities)
            std_vol = np.std(volatilities)
            
            ax.axvline(mean_vol, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_vol:.1f}%')
            ax.axvline(mean_vol + std_vol, color='orange', linestyle=':', 
                      linewidth=2, label=f'+1σ: {mean_vol + std_vol:.1f}%')
            ax.axvline(mean_vol - std_vol, color='orange', linestyle=':', 
                      linewidth=2, label=f'-1σ: {mean_vol - std_vol:.1f}%')
            
            ax.set_xlabel('Volatility (%)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
    def _plot_performance_correlation(self, ax, backtest_results):
        """Plot performance correlation matrix"""
        ax.set_title('Performance Correlation', fontsize=12, fontweight='bold')
        
        # Create correlation matrix of different metrics
        metrics_data = []
        for result in backtest_results:
            metrics_data.append([
                result.total_return,
                result.sharpe_ratio if not np.isnan(result.sharpe_ratio) else 0,
                result.max_drawdown,
                result.win_rate,
                getattr(result, 'volatility', 0)
            ])
        
        if len(metrics_data) > 1:
            df = pd.DataFrame(metrics_data, 
                            columns=['Return', 'Sharpe', 'Max DD', 'Win Rate', 'Volatility'])
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            
            # Add labels
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45)
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation')
        
    def _plot_summary_table(self, ax, backtest_results):
        """Plot summary statistics table"""
        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Calculate summary statistics
        returns = [r.total_return * 100 for r in backtest_results]
        sharpes = [r.sharpe_ratio for r in backtest_results if not np.isnan(r.sharpe_ratio)]
        max_dds = [r.max_drawdown * 100 for r in backtest_results]
        win_rates = [r.win_rate * 100 for r in backtest_results]
        
        # Create summary data
        summary_data = [
            ['Metric', 'Mean', 'Std', 'Min', 'Max', 'Count'],
            ['Return (%)', f'{np.mean(returns):.2f}', f'{np.std(returns):.2f}', 
             f'{np.min(returns):.2f}', f'{np.max(returns):.2f}', len(returns)],
            ['Sharpe Ratio', f'{np.mean(sharpes):.3f}', f'{np.std(sharpes):.3f}', 
             f'{np.min(sharpes):.3f}', f'{np.max(sharpes):.3f}', len(sharpes)] if sharpes else 
            ['Sharpe Ratio', 'N/A', 'N/A', 'N/A', 'N/A', 0],
            ['Max DD (%)', f'{np.mean(max_dds):.2f}', f'{np.std(max_dds):.2f}', 
             f'{np.min(max_dds):.2f}', f'{np.max(max_dds):.2f}', len(max_dds)],
            ['Win Rate (%)', f'{np.mean(win_rates):.1f}', f'{np.std(win_rates):.1f}', 
             f'{np.min(win_rates):.1f}', f'{np.max(win_rates):.1f}', len(win_rates)]
        ]
        
        # Create table
        table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                        cellLoc='center', loc='center', 
                        colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows with alternating colors
        for i in range(1, len(summary_data)):
            color = '#f8f9fa' if i % 2 == 0 else 'white'
            for j in range(len(summary_data[0])):
                table[(i, j)].set_facecolor(color)

def create_interactive_dashboard(backtest_results: List) -> go.Figure:
    """Create interactive Plotly dashboard"""
    
    if not backtest_results:
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Equity Curves', 'Performance Distribution', 
                       'Risk-Return Scatter', 'Drawdown Analysis',
                       'Performance by Period', 'Summary Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"type": "table"}]]
    )
    
    # 1. Equity curves
    for i, result in enumerate(backtest_results[:5]):  # Limit for performance
        if hasattr(result, 'equity_curve'):
            equity = result.equity_curve
            if len(equity) > 1:
                normalized_equity = equity / equity[0]
                fig.add_trace(
                    go.Scatter(y=normalized_equity, mode='lines', 
                             name=f'Period {i+1}', opacity=0.7),
                    row=1, col=1
                )
    
    # 2. Performance distribution
    returns = [r.total_return * 100 for r in backtest_results]
    fig.add_trace(
        go.Histogram(x=returns, nbinsx=30, name='Return Distribution'),
        row=1, col=2
    )
    
    # 3. Risk-return scatter
    sharpes = [r.sharpe_ratio for r in backtest_results if not np.isnan(r.sharpe_ratio)]
    volatilities = [getattr(r, 'volatility', 0) * 100 for r in backtest_results]
    
    fig.add_trace(
        go.Scatter(x=volatilities, y=returns, mode='markers',
                  name='Risk-Return', marker=dict(size=8, opacity=0.6)),
        row=2, col=1
    )
    
    # 4. Drawdown analysis
    max_dds = [r.max_drawdown * 100 for r in backtest_results]
    periods = list(range(1, len(max_dds) + 1))
    
    fig.add_trace(
        go.Scatter(x=periods, y=max_dds, mode='lines+markers',
                  name='Max Drawdown', line=dict(color='red')),
        row=2, col=2
    )
    
    # 5. Performance by period (bar chart)
    fig.add_trace(
        go.Bar(x=periods, y=returns, name='Returns by Period'),
        row=3, col=1
    )
    
    # Add trend line for Sharpe ratio
    if sharpes and len(sharpes) == len(periods):
        fig.add_trace(
            go.Scatter(x=periods, y=sharpes, mode='lines+markers',
                      name='Sharpe Ratio', yaxis='y2', 
                      line=dict(color='orange')),
            row=3, col=1, secondary_y=True
        )
    
    # 6. Summary table
    summary_data = [
        ['Total Periods', len(backtest_results)],
        ['Avg Return (%)', f'{np.mean(returns):.2f}'],
        ['Avg Sharpe', f'{np.mean(sharpes):.3f}' if sharpes else 'N/A'],
        ['Avg Max DD (%)', f'{np.mean(max_dds):.2f}'],
        ['Win Rate (%)', f'{np.mean([r.win_rate * 100 for r in backtest_results]):.1f}']
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'], 
                       fill_color='lightblue', align='left'),
            cells=dict(values=[[row[0] for row in summary_data], 
                              [row[1] for row in summary_data]],
                      fill_color='white', align='left')
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        title_text="Interactive Backtest Dashboard",
        title_x=0.5,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
    
    fig.update_xaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="Volatility (%)", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    
    fig.update_xaxes(title_text="Period", row=2, col=2)
    fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=2)
    
    fig.update_xaxes(title_text="Period", row=3, col=1)
    fig.update_yaxes(title_text="Return (%)", row=3, col=1)
    
    return fig

def main():
    """Example usage of backtest visualizer"""
    
    print("🚀 Backtest Visualizer Demo")
    print("=" * 50)
    
    from comprehensive_backtester import BacktestResult
    
    # Generate sample backtest results
    np.random.seed(42)
    sample_results = []
    
    for i in range(20):
        # Generate sample equity curve
        returns = np.random.normal(0.001, 0.02, 100)
        equity_curve = 100000 * np.cumprod(1 + returns)
        
        # Create sample result
        result = BacktestResult(
            period_name=f"Period_{i}",
            start_idx=i * 100,
            end_idx=(i + 1) * 100,
            total_return=np.random.normal(0.05, 0.15),
            sharpe_ratio=np.random.normal(0.8, 0.5),
            max_drawdown=np.random.uniform(-0.1, -0.01),
            romad=np.random.normal(2.0, 1.0),
            win_rate=np.random.uniform(0.4, 0.7),
            num_trades=np.random.randint(10, 100),
            avg_trade_return=np.random.normal(0.001, 0.005),
            profit_factor=np.random.uniform(0.8, 2.0),
            volatility=np.random.uniform(0.1, 0.4),
            var_95=np.random.uniform(-0.05, -0.01),
            cvar_95=np.random.uniform(-0.08, -0.02),
            market_regime=np.random.choice(['bull', 'bear', 'sideways', 'high_vol']),
            volatility_regime=np.random.choice(['low', 'medium', 'high']),
            equity_curve=equity_curve,
            positions=np.random.randint(-1, 2, 100),
            trade_log=[],
            duration_days=100,
            timestamp=datetime.now()
        )
        
        sample_results.append(result)
    
    # Create visualizer
    visualizer = BacktestVisualizer()
    
    # Create comprehensive dashboard
    print("📊 Creating comprehensive dashboard...")
    fig = visualizer.create_comprehensive_dashboard(
        sample_results, 
        save_path="backtest_dashboard.png"
    )
    
    # Create interactive dashboard
    print("🌐 Creating interactive dashboard...")
    interactive_fig = create_interactive_dashboard(sample_results)
    
    # Save interactive dashboard
    interactive_fig.write_html("interactive_backtest_dashboard.html")
    print("Interactive dashboard saved to 'interactive_backtest_dashboard.html'")
    
    # Show static dashboard
    plt.show()
    
    print("✅ Visualization demo completed!")

if __name__ == "__main__":
    main()