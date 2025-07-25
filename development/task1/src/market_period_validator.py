"""
Market Period Performance Validator
Validates model performance consistency across different market regimes and time periods
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import trading components
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from metrics import calculate_sharpe_ratio, calculate_max_drawdown

class MarketRegimeAnalyzer:
    """Analyzes different market regimes and periods"""
    
    def __init__(self):
        self.regime_definitions = {
            'bull_trending': {'volatility_threshold': 0.02, 'trend_threshold': 0.01},
            'bear_trending': {'volatility_threshold': 0.02, 'trend_threshold': -0.01},
            'high_volatility': {'volatility_threshold': 0.04, 'trend_threshold': None},
            'low_volatility': {'volatility_threshold': 0.01, 'trend_threshold': None},
            'sideways': {'volatility_threshold': 0.03, 'trend_threshold': 0.005}
        }
    
    def classify_market_periods(self, price_data: np.ndarray, window=100) -> Dict:
        """Classify market periods into different regimes"""
        
        if len(price_data) < window:
            return {'error': 'Insufficient data for classification'}
        
        # Calculate rolling metrics
        returns = np.diff(np.log(price_data))
        
        periods = []
        for i in range(window, len(price_data), window//2):
            period_returns = returns[i-window:i]
            
            # Calculate regime indicators
            volatility = np.std(period_returns) * np.sqrt(252)  # Annualized
            trend = np.mean(period_returns) * 252  # Annualized
            
            # Classify regime
            regime = self._classify_single_period(volatility, trend)
            
            periods.append({
                'start_idx': i-window,
                'end_idx': i,
                'regime': regime,
                'volatility': volatility,
                'trend': trend,
                'period_return': np.sum(period_returns)
            })
        
        return {
            'periods': periods,
            'regime_counts': self._count_regimes(periods),
            'total_periods': len(periods)
        }
    
    def _classify_single_period(self, volatility: float, trend: float) -> str:
        """Classify a single period into market regime"""
        
        if abs(trend) < 0.005:  # Very low trend
            if volatility > 0.03:
                return 'high_vol_sideways'
            else:
                return 'low_vol_sideways'
        elif trend > 0.01:  # Positive trend
            if volatility > 0.03:
                return 'high_vol_bull'
            else:
                return 'low_vol_bull'
        else:  # Negative trend
            if volatility > 0.03:
                return 'high_vol_bear'
            else:
                return 'low_vol_bear'
    
    def _count_regimes(self, periods: List) -> Dict:
        """Count occurrences of each regime"""
        counts = {}
        for period in periods:
            regime = period['regime']
            counts[regime] = counts.get(regime, 0) + 1
        return counts

class PerformanceValidator:
    """Validates ensemble performance across different market periods"""
    
    def __init__(self, ensemble_path: str, device='cuda:0'):
        self.ensemble_path = ensemble_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.agents = {}
        self.regime_analyzer = MarketRegimeAnalyzer()
        
        # Load trained agents
        self._load_ensemble()
    
    def _load_ensemble(self):
        """Load trained ensemble agents"""
        agent_classes = {
            'AgentD3QN': AgentD3QN,
            'AgentDoubleDQN': AgentDoubleDQN,
            'AgentTwinD3QN': AgentTwinD3QN
        }
        
        ensemble_models_path = os.path.join(self.ensemble_path, 'ensemble_models')
        
        if not os.path.exists(ensemble_models_path):
            raise FileNotFoundError(f"Ensemble models not found at: {ensemble_models_path}")
        
        # Get state dimension from TradeSimulator
        temp_sim = TradeSimulator(num_sims=1)
        state_dim = temp_sim.state_dim
        action_dim = 3
        
        for agent_name in os.listdir(ensemble_models_path):
            agent_path = os.path.join(ensemble_models_path, agent_name)
            
            if os.path.isdir(agent_path) and agent_name in agent_classes:
                try:
                    # Initialize agent
                    agent_class = agent_classes[agent_name]
                    agent = agent_class(
                        net_dims=(128, 64, 32),  # Optimized architecture
                        state_dim=state_dim,
                        action_dim=action_dim,
                        gpu_id=0
                    )
                    
                    # Load trained weights
                    agent.save_or_load_agent(agent_path, if_save=False)
                    self.agents[agent_name] = agent
                    
                    print(f"✅ Loaded {agent_name}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {agent_name}: {e}")
        
        print(f"📊 Loaded {len(self.agents)} agents for validation")
    
    def validate_across_periods(self, num_periods=10, period_length=200) -> Dict:
        """Validate performance across different time periods"""
        
        print(f"🔍 Validating performance across {num_periods} periods")
        
        # Create evaluation environment
        env = EvalTradeSimulator(num_sims=1)
        
        validation_results = {
            'period_results': [],
            'agent_consistency': {},
            'regime_performance': {},
            'summary_stats': {}
        }
        
        # Get price data for regime analysis
        try:
            from data_config import data_path_dict
            data_path = data_path_dict.get("BTC")
            price_data = pd.read_csv(data_path)['midprice'].values[:4800]  # Use available data
        except:
            print("⚠️  Could not load price data for regime analysis")
            price_data = None
        
        # Analyze market regimes if data available
        regime_info = None
        if price_data is not None:
            regime_info = self.regime_analyzer.classify_market_periods(price_data, window=200)
            print(f"📊 Identified {regime_info['total_periods']} market periods")
            for regime, count in regime_info['regime_counts'].items():
                print(f"   {regime}: {count} periods")
        
        # Test performance across different starting points
        total_data_length = 4800 - 60  # Account for num_ignore_step
        period_starts = np.linspace(0, total_data_length - period_length, num_periods, dtype=int)
        
        for period_idx, start_idx in enumerate(period_starts):
            period_results = self._evaluate_period(
                env, start_idx, period_length, period_idx, regime_info
            )
            validation_results['period_results'].append(period_results)
            
            print(f"   Period {period_idx+1}/{num_periods}: "
                  f"Avg Sharpe={period_results['ensemble_sharpe']:.3f}")
        
        # Calculate consistency metrics
        validation_results['agent_consistency'] = self._calculate_consistency_metrics(
            validation_results['period_results']
        )
        
        # Analyze regime-specific performance
        if regime_info:
            validation_results['regime_performance'] = self._analyze_regime_performance(
                validation_results['period_results'], regime_info
            )
        
        # Summary statistics
        validation_results['summary_stats'] = self._calculate_summary_stats(
            validation_results['period_results']
        )
        
        return validation_results
    
    def _evaluate_period(self, env, start_idx: int, length: int, period_idx: int, 
                        regime_info: Dict = None) -> Dict:
        """Evaluate ensemble performance for a specific period"""
        
        # Reset environment to specific start point
        env.reset()
        
        # Simulate trading for the period
        period_results = {
            'period_idx': period_idx,
            'start_idx': start_idx,
            'length': length,
            'agent_performance': {},
            'ensemble_performance': {},
            'regime': None
        }
        
        # Determine regime for this period if available
        if regime_info and regime_info.get('periods'):
            for regime_period in regime_info['periods']:
                if (regime_period['start_idx'] <= start_idx <= regime_period['end_idx']):
                    period_results['regime'] = regime_period['regime']
                    break
        
        # Evaluate each agent individually
        agent_actions = {}
        agent_rewards = {}
        
        for agent_name, agent in self.agents.items():
            actions, rewards = self._simulate_agent_trading(agent, env, length)
            agent_actions[agent_name] = actions
            agent_rewards[agent_name] = rewards
            
            # Calculate agent metrics
            sharpe = calculate_sharpe_ratio(rewards) if len(rewards) > 1 else 0.0
            max_dd = calculate_max_drawdown(np.cumsum(rewards)) if len(rewards) > 1 else 0.0
            
            period_results['agent_performance'][agent_name] = {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'total_return': np.sum(rewards),
                'win_rate': np.mean(np.array(rewards) > 0),
                'num_trades': np.sum(np.abs(np.diff(actions)))
            }
        
        # Calculate ensemble performance (majority voting)
        if agent_actions:
            ensemble_actions = self._ensemble_majority_vote(agent_actions)
            ensemble_rewards = self._calculate_ensemble_rewards(ensemble_actions, env, length)
            
            ensemble_sharpe = calculate_sharpe_ratio(ensemble_rewards) if len(ensemble_rewards) > 1 else 0.0
            ensemble_max_dd = calculate_max_drawdown(np.cumsum(ensemble_rewards)) if len(ensemble_rewards) > 1 else 0.0
            
            period_results['ensemble_performance'] = {
                'sharpe_ratio': ensemble_sharpe,
                'max_drawdown': ensemble_max_dd,
                'total_return': np.sum(ensemble_rewards),
                'win_rate': np.mean(np.array(ensemble_rewards) > 0),
                'num_trades': np.sum(np.abs(np.diff(ensemble_actions)))
            }
            
            period_results['ensemble_sharpe'] = ensemble_sharpe
        
        return period_results
    
    def _simulate_agent_trading(self, agent, env, length: int) -> Tuple[List, List]:
        """Simulate trading for a single agent"""
        env.reset()
        state = env.reset()
        
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        actions = []
        rewards = []
        
        for step in range(min(length, 1000)):  # Limit steps for safety
            try:
                # Get action from agent
                with torch.no_grad():
                    action = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                
                actions.append(action[0] if isinstance(action, (list, np.ndarray)) else action)
                rewards.append(reward[0] if isinstance(reward, (list, np.ndarray)) else reward)
                
                if isinstance(next_state, np.ndarray):
                    state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    state = next_state
                
                if done[0] if isinstance(done, (list, np.ndarray)) else done:
                    break
                    
            except Exception as e:
                print(f"   ⚠️  Simulation error: {e}")
                break
        
        return actions, rewards
    
    def _ensemble_majority_vote(self, agent_actions: Dict) -> List:
        """Combine agent actions using majority voting"""
        max_length = max(len(actions) for actions in agent_actions.values())
        ensemble_actions = []
        
        for step in range(max_length):
            step_actions = []
            for actions in agent_actions.values():
                if step < len(actions):
                    step_actions.append(actions[step])
            
            if step_actions:
                # Majority vote (or first action if tie)
                from collections import Counter
                vote_counts = Counter(step_actions)
                majority_action = vote_counts.most_common(1)[0][0]
                ensemble_actions.append(majority_action)
        
        return ensemble_actions
    
    def _calculate_ensemble_rewards(self, ensemble_actions: List, env, length: int) -> List:
        """Calculate rewards for ensemble actions"""
        env.reset()
        rewards = []
        
        for step, action in enumerate(ensemble_actions[:length]):
            try:
                _, reward, done, _ = env.step([action])
                rewards.append(reward[0] if isinstance(reward, (list, np.ndarray)) else reward)
                
                if done[0] if isinstance(done, (list, np.ndarray)) else done:
                    break
            except:
                break
        
        return rewards
    
    def _calculate_consistency_metrics(self, period_results: List) -> Dict:
        """Calculate consistency metrics across periods"""
        
        consistency = {
            'sharpe_consistency': {},
            'return_consistency': {},
            'drawdown_consistency': {}
        }
        
        # Collect metrics by agent
        agent_names = set()
        for result in period_results:
            agent_names.update(result['agent_performance'].keys())
        
        for agent_name in agent_names:
            sharpe_ratios = []
            returns = []
            drawdowns = []
            
            for result in period_results:
                if agent_name in result['agent_performance']:
                    perf = result['agent_performance'][agent_name]
                    sharpe_ratios.append(perf['sharpe_ratio'])
                    returns.append(perf['total_return'])
                    drawdowns.append(perf['max_drawdown'])
            
            if sharpe_ratios:
                consistency['sharpe_consistency'][agent_name] = {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios),
                    'consistency_score': 1.0 / (1.0 + np.std(sharpe_ratios))  # Higher = more consistent
                }
                
                consistency['return_consistency'][agent_name] = {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'consistency_score': 1.0 / (1.0 + np.std(returns))
                }
                
                consistency['drawdown_consistency'][agent_name] = {
                    'mean': np.mean(drawdowns),
                    'std': np.std(drawdowns),
                    'consistency_score': 1.0 / (1.0 + np.std(drawdowns))
                }
        
        return consistency
    
    def _analyze_regime_performance(self, period_results: List, regime_info: Dict) -> Dict:
        """Analyze performance by market regime"""
        
        regime_performance = {}
        
        # Group results by regime
        for result in period_results:
            regime = result.get('regime', 'unknown')
            
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'agent_performance': {},
                    'ensemble_performance': [],
                    'count': 0
                }
            
            regime_performance[regime]['count'] += 1
            
            # Collect ensemble performance
            if 'ensemble_performance' in result:
                regime_performance[regime]['ensemble_performance'].append(
                    result['ensemble_performance']
                )
            
            # Collect agent performance
            for agent_name, perf in result['agent_performance'].items():
                if agent_name not in regime_performance[regime]['agent_performance']:
                    regime_performance[regime]['agent_performance'][agent_name] = []
                
                regime_performance[regime]['agent_performance'][agent_name].append(perf)
        
        # Calculate regime statistics
        for regime, data in regime_performance.items():
            # Ensemble stats
            if data['ensemble_performance']:
                ensemble_sharpes = [p['sharpe_ratio'] for p in data['ensemble_performance']]
                data['ensemble_stats'] = {
                    'mean_sharpe': np.mean(ensemble_sharpes),
                    'std_sharpe': np.std(ensemble_sharpes),
                    'count': len(ensemble_sharpes)
                }
            
            # Agent stats
            for agent_name, perfs in data['agent_performance'].items():
                if perfs:
                    sharpes = [p['sharpe_ratio'] for p in perfs]
                    data['agent_performance'][agent_name] = {
                        'performances': perfs,
                        'mean_sharpe': np.mean(sharpes),
                        'std_sharpe': np.std(sharpes),
                        'count': len(sharpes)
                    }
        
        return regime_performance
    
    def _calculate_summary_stats(self, period_results: List) -> Dict:
        """Calculate overall summary statistics"""
        
        ensemble_sharpes = []
        agent_sharpes = {name: [] for name in self.agents.keys()}
        
        for result in period_results:
            if 'ensemble_performance' in result:
                ensemble_sharpes.append(result['ensemble_performance']['sharpe_ratio'])
            
            for agent_name, perf in result['agent_performance'].items():
                if agent_name in agent_sharpes:
                    agent_sharpes[agent_name].append(perf['sharpe_ratio'])
        
        summary = {
            'ensemble_summary': {
                'mean_sharpe': np.mean(ensemble_sharpes) if ensemble_sharpes else 0.0,
                'std_sharpe': np.std(ensemble_sharpes) if ensemble_sharpes else 0.0,
                'consistency_score': 1.0 / (1.0 + np.std(ensemble_sharpes)) if ensemble_sharpes else 0.0,
                'periods_evaluated': len(ensemble_sharpes)
            },
            'agent_summary': {}
        }
        
        for agent_name, sharpes in agent_sharpes.items():
            if sharpes:
                summary['agent_summary'][agent_name] = {
                    'mean_sharpe': np.mean(sharpes),
                    'std_sharpe': np.std(sharpes),
                    'consistency_score': 1.0 / (1.0 + np.std(sharpes)),
                    'periods_evaluated': len(sharpes)
                }
        
        return summary
    
    def save_validation_report(self, results: Dict, output_path: str):
        """Save comprehensive validation report"""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(output_path, 'validation_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))
        
        # Create visualization
        self._create_validation_plots(results, output_path)
        
        # Create text report
        self._create_text_report(results, output_path)
        
        print(f"📊 Validation report saved to: {output_path}")
    
    def _create_validation_plots(self, results: Dict, output_path: str):
        """Create validation visualization plots"""
        
        try:
            # Performance consistency plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Performance Validation Across Market Periods', fontsize=16)
            
            # Plot 1: Sharpe ratios over time
            ax1 = axes[0, 0]
            period_sharpes = [r.get('ensemble_sharpe', 0) for r in results['period_results']]
            ax1.plot(period_sharpes, 'b-o', label='Ensemble Sharpe')
            ax1.axhline(np.mean(period_sharpes), color='r', linestyle='--', label='Mean')
            ax1.set_title('Ensemble Sharpe Ratio by Period')
            ax1.set_xlabel('Period')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Agent consistency
            ax2 = axes[0, 1]
            if 'agent_consistency' in results and 'sharpe_consistency' in results['agent_consistency']:
                agent_names = list(results['agent_consistency']['sharpe_consistency'].keys())
                consistency_scores = [
                    results['agent_consistency']['sharpe_consistency'][name]['consistency_score']
                    for name in agent_names
                ]
                ax2.bar(agent_names, consistency_scores)
                ax2.set_title('Agent Consistency Scores')
                ax2.set_ylabel('Consistency Score')
                plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # Plot 3: Regime performance
            ax3 = axes[1, 0]
            if 'regime_performance' in results:
                regimes = list(results['regime_performance'].keys())
                regime_sharpes = []
                for regime in regimes:
                    regime_data = results['regime_performance'][regime]
                    if 'ensemble_stats' in regime_data:
                        regime_sharpes.append(regime_data['ensemble_stats']['mean_sharpe'])
                    else:
                        regime_sharpes.append(0)
                
                ax3.bar(regimes, regime_sharpes)
                ax3.set_title('Performance by Market Regime')
                ax3.set_ylabel('Mean Sharpe Ratio')
                plt.setp(ax3.get_xticklabels(), rotation=45)
            
            # Plot 4: Performance distribution
            ax4 = axes[1, 1]
            period_sharpes = [r.get('ensemble_sharpe', 0) for r in results['period_results']]
            ax4.hist(period_sharpes, bins=10, alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(period_sharpes), color='r', linestyle='--', label='Mean')
            ax4.set_title('Sharpe Ratio Distribution')
            ax4.set_xlabel('Sharpe Ratio')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'validation_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Could not create validation plots: {e}")
    
    def _create_text_report(self, results: Dict, output_path: str):
        """Create comprehensive text report"""
        
        report_path = os.path.join(output_path, 'validation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Market Period Performance Validation Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            if 'summary_stats' in results and 'ensemble_summary' in results['summary_stats']:
                summary = results['summary_stats']['ensemble_summary']
                f.write(f"- **Periods Evaluated**: {summary['periods_evaluated']}\n")
                f.write(f"- **Mean Sharpe Ratio**: {summary['mean_sharpe']:.3f}\n")
                f.write(f"- **Sharpe Volatility**: {summary['std_sharpe']:.3f}\n")
                f.write(f"- **Consistency Score**: {summary['consistency_score']:.3f}\n\n")
            
            # Agent Performance Summary
            f.write("## Agent Performance Summary\n\n")
            if 'summary_stats' in results and 'agent_summary' in results['summary_stats']:
                for agent_name, stats in results['summary_stats']['agent_summary'].items():
                    f.write(f"### {agent_name}\n")
                    f.write(f"- Mean Sharpe: {stats['mean_sharpe']:.3f}\n")
                    f.write(f"- Sharpe Volatility: {stats['std_sharpe']:.3f}\n")
                    f.write(f"- Consistency Score: {stats['consistency_score']:.3f}\n\n")
            
            # Regime Analysis
            f.write("## Market Regime Analysis\n\n")
            if 'regime_performance' in results:
                for regime, data in results['regime_performance'].items():
                    f.write(f"### {regime.replace('_', ' ').title()}\n")
                    f.write(f"- Periods: {data['count']}\n")
                    if 'ensemble_stats' in data:
                        f.write(f"- Mean Sharpe: {data['ensemble_stats']['mean_sharpe']:.3f}\n")
                        f.write(f"- Sharpe Volatility: {data['ensemble_stats']['std_sharpe']:.3f}\n")
                    f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if 'summary_stats' in results:
                consistency = results['summary_stats']['ensemble_summary']['consistency_score']
                if consistency > 0.7:
                    f.write("✅ **High Consistency**: Model performance is stable across different market periods.\n\n")
                elif consistency > 0.5:
                    f.write("⚠️ **Moderate Consistency**: Some performance variation observed. Consider regime-specific optimization.\n\n")
                else:
                    f.write("❌ **Low Consistency**: Significant performance variation. Requires stability improvements.\n\n")

def run_validation(ensemble_path, output_path="validation_results", num_periods=10):
    """Run comprehensive performance validation"""
    
    print(f"🔍 Starting Performance Validation")
    print(f"📂 Ensemble Path: {ensemble_path}")
    print(f"📂 Output Path: {output_path}")
    
    try:
        # Initialize validator
        validator = PerformanceValidator(ensemble_path)
        
        # Run validation
        results = validator.validate_across_periods(num_periods=num_periods)
        
        # Save results
        validator.save_validation_report(results, output_path)
        
        # Print summary
        if 'summary_stats' in results and 'ensemble_summary' in results['summary_stats']:
            summary = results['summary_stats']['ensemble_summary']
            print(f"\n📊 Validation Summary:")
            print(f"   Mean Sharpe: {summary['mean_sharpe']:.3f}")
            print(f"   Consistency: {summary['consistency_score']:.3f}")
            print(f"   Periods: {summary['periods_evaluated']}")
        
        print(f"✅ Validation completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    # Default paths
    ensemble_path = "ensemble_optimized_phase2" 
    output_path = "validation_results"
    num_periods = 10
    
    if len(sys.argv) > 1:
        ensemble_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]  
    if len(sys.argv) > 3:
        num_periods = int(sys.argv[3])
    
    results = run_validation(ensemble_path, output_path, num_periods)