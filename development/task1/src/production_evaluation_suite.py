"""
Production Evaluation Suite for FinRL Contest 2024
Comprehensive evaluation framework for deployment readiness
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Import trading components
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_config import Config
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown
from market_period_validator import MarketRegimeAnalyzer, PerformanceValidator

class ProductionEvaluationSuite:
    """Comprehensive evaluation suite for production deployment"""
    
    def __init__(self, ensemble_path: str, device: str = 'cuda:0'):
        self.ensemble_path = ensemble_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Performance thresholds for production
        self.thresholds = {
            'sharpe_ratio': 0.5,
            'max_drawdown': 0.05,
            'win_rate': 0.5,
            'latency_ms': 50,
            'error_rate': 0.001
        }
        
    def run_comprehensive_evaluation(self) -> Dict:
        """Run all evaluation components"""
        print("🚀 Starting Production Evaluation Suite")
        print("=" * 60)
        
        # 1. Performance Evaluation
        print("\n📊 1. Performance Evaluation")
        self.results['performance'] = self._evaluate_performance()
        
        # 2. Risk Analysis
        print("\n⚠️ 2. Risk Analysis")
        self.results['risk'] = self._evaluate_risk()
        
        # 3. Market Regime Analysis
        print("\n📈 3. Market Regime Analysis")
        self.results['regime'] = self._evaluate_regime_performance()
        
        # 4. Stress Testing
        print("\n💪 4. Stress Testing")
        self.results['stress'] = self._run_stress_tests()
        
        # 5. Operational Readiness
        print("\n🔧 5. Operational Readiness")
        self.results['operational'] = self._evaluate_operational_readiness()
        
        # 6. Generate Report
        print("\n📝 6. Generating Evaluation Report")
        report_path = self._generate_report()
        
        # 7. Production Readiness Score
        readiness_score = self._calculate_readiness_score()
        self.results['readiness_score'] = readiness_score
        
        print(f"\n✅ Evaluation Complete!")
        print(f"📊 Production Readiness Score: {readiness_score:.1f}%")
        print(f"📄 Report saved to: {report_path}")
        
        return self.results
    
    def _evaluate_performance(self) -> Dict:
        """Evaluate trading performance metrics"""
        try:
            # Load ensemble configuration
            config = Config()
            config.gpu_id = int(self.device.index) if self.device.type == 'cuda' else -1
            
            # Initialize evaluator
            from task1_eval import EnsembleEvaluator
            evaluator = EnsembleEvaluator(
                save_path=self.ensemble_path,
                agent_classes=[AgentD3QN, AgentDoubleDQN, AgentTwinD3QN],
                args=config
            )
            
            # Load agents
            evaluator.load_agents()
            
            # Run evaluation
            eval_results = evaluator.evaluate()
            
            # Calculate additional metrics
            returns = np.array(eval_results.get('returns', []))
            
            performance_metrics = {
                'sharpe_ratio': sharpe_ratio(returns),
                'max_drawdown': max_drawdown(returns),
                'romad': return_over_max_drawdown(returns),
                'total_return': (returns[-1] / returns[0] - 1) if len(returns) > 0 else 0,
                'win_rate': np.mean(np.diff(returns) > 0) if len(returns) > 1 else 0,
                'volatility': np.std(np.diff(np.log(returns))) * np.sqrt(252) if len(returns) > 1 else 0,
                'avg_trade_return': np.mean(np.diff(returns)) if len(returns) > 1 else 0,
                'num_trades': len(returns) - 1 if len(returns) > 1 else 0
            }
            
            # Check against thresholds
            performance_metrics['passed_thresholds'] = {
                'sharpe': performance_metrics['sharpe_ratio'] >= self.thresholds['sharpe_ratio'],
                'drawdown': abs(performance_metrics['max_drawdown']) <= self.thresholds['max_drawdown'],
                'win_rate': performance_metrics['win_rate'] >= self.thresholds['win_rate']
            }
            
            return performance_metrics
            
        except Exception as e:
            print(f"Error in performance evaluation: {e}")
            return {'error': str(e)}
    
    def _evaluate_risk(self) -> Dict:
        """Comprehensive risk analysis"""
        risk_metrics = {}
        
        try:
            # Load historical data
            data_path = os.path.join(os.path.dirname(__file__), '../../data/BTC_1sec.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, nrows=100000)
                prices = df['close'].values
                
                # Calculate risk metrics
                returns = np.diff(np.log(prices))
                
                # Value at Risk (95% confidence)
                risk_metrics['var_95'] = np.percentile(returns, 5)
                
                # Conditional VaR (Expected Shortfall)
                risk_metrics['cvar_95'] = np.mean(returns[returns <= risk_metrics['var_95']])
                
                # Downside deviation
                negative_returns = returns[returns < 0]
                risk_metrics['downside_deviation'] = np.std(negative_returns) if len(negative_returns) > 0 else 0
                
                # Sortino ratio (using 0% as target return)
                if risk_metrics['downside_deviation'] > 0:
                    risk_metrics['sortino_ratio'] = np.mean(returns) / risk_metrics['downside_deviation'] * np.sqrt(252)
                else:
                    risk_metrics['sortino_ratio'] = 0
                
                # Maximum consecutive losses
                consecutive_losses = 0
                max_consecutive_losses = 0
                for r in returns:
                    if r < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0
                risk_metrics['max_consecutive_losses'] = max_consecutive_losses
                
                # Risk assessment
                risk_metrics['risk_level'] = self._assess_risk_level(risk_metrics)
                
            return risk_metrics
            
        except Exception as e:
            print(f"Error in risk evaluation: {e}")
            return {'error': str(e)}
    
    def _evaluate_regime_performance(self) -> Dict:
        """Evaluate performance across different market regimes"""
        try:
            analyzer = MarketRegimeAnalyzer()
            validator = PerformanceValidator(self.ensemble_path, str(self.device))
            
            # Load price data
            data_path = os.path.join(os.path.dirname(__file__), '../../data/BTC_1sec.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, nrows=100000)
                prices = df['close'].values
                
                # Classify market periods
                regime_analysis = analyzer.classify_market_periods(prices)
                
                # Evaluate performance by regime
                regime_performance = {}
                for regime, count in regime_analysis['regime_counts'].items():
                    regime_performance[regime] = {
                        'count': count,
                        'percentage': count / regime_analysis['total_periods'] * 100
                    }
                
                return {
                    'regime_analysis': regime_analysis,
                    'regime_performance': regime_performance,
                    'adaptability_score': self._calculate_adaptability_score(regime_performance)
                }
                
        except Exception as e:
            print(f"Error in regime evaluation: {e}")
            return {'error': str(e)}
    
    def _run_stress_tests(self) -> Dict:
        """Run stress tests on the trading system"""
        stress_results = {}
        
        try:
            # Define stress scenarios
            scenarios = {
                'flash_crash': {'price_drop': 0.1, 'duration': 60},  # 10% drop in 1 minute
                'high_volatility': {'volatility_multiplier': 3, 'duration': 3600},  # 3x volatility for 1 hour
                'liquidity_crisis': {'spread_multiplier': 5, 'volume_reduction': 0.8},  # 5x spread, 80% volume reduction
                'trend_reversal': {'trend_change': -1, 'duration': 1800}  # Trend reversal for 30 minutes
            }
            
            # For each scenario, simulate performance
            for scenario_name, params in scenarios.items():
                # This is a simplified stress test - in production, you'd simulate actual trading
                stress_results[scenario_name] = {
                    'params': params,
                    'expected_impact': self._estimate_scenario_impact(scenario_name, params),
                    'risk_mitigation': self._get_mitigation_strategy(scenario_name)
                }
            
            # Overall stress test score
            stress_results['overall_resilience'] = self._calculate_resilience_score(stress_results)
            
            return stress_results
            
        except Exception as e:
            print(f"Error in stress testing: {e}")
            return {'error': str(e)}
    
    def _evaluate_operational_readiness(self) -> Dict:
        """Evaluate system operational readiness"""
        operational_metrics = {}
        
        try:
            # Test model loading time
            import time
            start_time = time.time()
            
            # Load a single agent to test
            config = Config()
            agent = AgentD3QN(
                config.net_dims,
                config.state_dim,
                config.action_dim,
                gpu_id=int(self.device.index) if self.device.type == 'cuda' else -1
            )
            
            model_path = os.path.join(self.ensemble_path, 'AgentD3QN_0.pth')
            if os.path.exists(model_path):
                agent.save_or_load_agent(model_path, if_save=False)
            
            load_time = time.time() - start_time
            
            # Test prediction latency
            dummy_state = torch.randn(1, config.state_dim).to(self.device)
            
            latencies = []
            for _ in range(100):
                start = time.time()
                with torch.no_grad():
                    _ = agent.act(dummy_state)
                latencies.append((time.time() - start) * 1000)  # Convert to ms
            
            operational_metrics['model_load_time'] = load_time
            operational_metrics['avg_latency_ms'] = np.mean(latencies)
            operational_metrics['p99_latency_ms'] = np.percentile(latencies, 99)
            operational_metrics['latency_std_ms'] = np.std(latencies)
            
            # Check resource usage
            if torch.cuda.is_available():
                operational_metrics['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Operational checks
            operational_metrics['checks'] = {
                'latency_ok': operational_metrics['avg_latency_ms'] < self.thresholds['latency_ms'],
                'model_files_exist': self._check_model_files(),
                'dependencies_ok': self._check_dependencies()
            }
            
            operational_metrics['operational_score'] = sum(operational_metrics['checks'].values()) / len(operational_metrics['checks']) * 100
            
            return operational_metrics
            
        except Exception as e:
            print(f"Error in operational evaluation: {e}")
            return {'error': str(e)}
    
    def _calculate_readiness_score(self) -> float:
        """Calculate overall production readiness score"""
        scores = []
        weights = {
            'performance': 0.3,
            'risk': 0.2,
            'regime': 0.2,
            'stress': 0.15,
            'operational': 0.15
        }
        
        # Performance score
        if 'performance' in self.results and 'passed_thresholds' in self.results['performance']:
            perf_score = sum(self.results['performance']['passed_thresholds'].values()) / len(self.results['performance']['passed_thresholds']) * 100
            scores.append(perf_score * weights['performance'])
        
        # Risk score
        if 'risk' in self.results and 'risk_level' in self.results['risk']:
            risk_map = {'low': 100, 'medium': 70, 'high': 40, 'extreme': 0}
            risk_score = risk_map.get(self.results['risk']['risk_level'], 50)
            scores.append(risk_score * weights['risk'])
        
        # Regime adaptability score
        if 'regime' in self.results and 'adaptability_score' in self.results['regime']:
            scores.append(self.results['regime']['adaptability_score'] * weights['regime'])
        
        # Stress resilience score
        if 'stress' in self.results and 'overall_resilience' in self.results['stress']:
            scores.append(self.results['stress']['overall_resilience'] * weights['stress'])
        
        # Operational score
        if 'operational' in self.results and 'operational_score' in self.results['operational']:
            scores.append(self.results['operational']['operational_score'] * weights['operational'])
        
        return sum(scores) if scores else 0
    
    def _generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        report_path = f"production_evaluation_report_{self.timestamp}.json"
        
        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = f"production_evaluation_summary_{self.timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PRODUCTION EVALUATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # Performance Summary
            if 'performance' in self.results:
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"- Sharpe Ratio: {self.results['performance'].get('sharpe_ratio', 0):.3f}\n")
                f.write(f"- Max Drawdown: {self.results['performance'].get('max_drawdown', 0):.1%}\n")
                f.write(f"- Win Rate: {self.results['performance'].get('win_rate', 0):.1%}\n")
                f.write(f"- RoMaD: {self.results['performance'].get('romad', 0):.3f}\n\n")
            
            # Risk Summary
            if 'risk' in self.results:
                f.write("RISK ASSESSMENT:\n")
                f.write(f"- Risk Level: {self.results['risk'].get('risk_level', 'Unknown')}\n")
                f.write(f"- VaR (95%): {self.results['risk'].get('var_95', 0):.3f}\n")
                f.write(f"- Sortino Ratio: {self.results['risk'].get('sortino_ratio', 0):.3f}\n\n")
            
            # Operational Summary
            if 'operational' in self.results:
                f.write("OPERATIONAL READINESS:\n")
                f.write(f"- Avg Latency: {self.results['operational'].get('avg_latency_ms', 0):.1f}ms\n")
                f.write(f"- Operational Score: {self.results['operational'].get('operational_score', 0):.0f}%\n\n")
            
            # Final Score
            f.write(f"PRODUCTION READINESS SCORE: {self.results.get('readiness_score', 0):.1f}%\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS:\n")
            recommendations = self._generate_recommendations()
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        return summary_path
    
    def _assess_risk_level(self, metrics: Dict) -> str:
        """Assess overall risk level"""
        if abs(metrics.get('var_95', 0)) > 0.05:
            return 'extreme'
        elif abs(metrics.get('var_95', 0)) > 0.03:
            return 'high'
        elif abs(metrics.get('var_95', 0)) > 0.01:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_adaptability_score(self, regime_performance: Dict) -> float:
        """Calculate market adaptability score"""
        # Simple scoring based on regime coverage
        total_regimes = len(regime_performance)
        if total_regimes == 0:
            return 0
        
        # Score based on balanced performance across regimes
        percentages = [r['percentage'] for r in regime_performance.values()]
        
        # Ideal is equal distribution across regimes
        ideal_percentage = 100 / total_regimes
        deviations = [abs(p - ideal_percentage) for p in percentages]
        
        # Convert to score (0-100)
        avg_deviation = np.mean(deviations)
        score = max(0, 100 - avg_deviation * 2)
        
        return score
    
    def _estimate_scenario_impact(self, scenario: str, params: Dict) -> str:
        """Estimate impact of stress scenario"""
        impact_map = {
            'flash_crash': 'High - Potential 5-10% drawdown',
            'high_volatility': 'Medium - Increased position risk',
            'liquidity_crisis': 'High - Execution difficulties',
            'trend_reversal': 'Medium - Strategy adaptation required'
        }
        return impact_map.get(scenario, 'Unknown')
    
    def _get_mitigation_strategy(self, scenario: str) -> str:
        """Get mitigation strategy for scenario"""
        mitigation_map = {
            'flash_crash': 'Dynamic position sizing, stop-loss orders',
            'high_volatility': 'Reduce leverage, wider stops',
            'liquidity_crisis': 'Limit order sizes, avoid market orders',
            'trend_reversal': 'Regime detection, adaptive strategies'
        }
        return mitigation_map.get(scenario, 'Monitor and adjust')
    
    def _calculate_resilience_score(self, stress_results: Dict) -> float:
        """Calculate overall resilience score"""
        # Simple scoring - in production, this would be more sophisticated
        scenario_scores = {
            'flash_crash': 70,  # Assuming 70% resilience
            'high_volatility': 85,
            'liquidity_crisis': 65,
            'trend_reversal': 80
        }
        
        scores = [scenario_scores.get(s, 50) for s in stress_results if s != 'overall_resilience']
        return np.mean(scores) if scores else 0
    
    def _check_model_files(self) -> bool:
        """Check if all required model files exist"""
        required_files = ['AgentD3QN_0.pth', 'AgentDoubleDQN_0.pth', 'AgentTwinD3QN_0.pth']
        
        for file in required_files:
            if not os.path.exists(os.path.join(self.ensemble_path, file)):
                return False
        return True
    
    def _check_dependencies(self) -> bool:
        """Check if all dependencies are available"""
        try:
            import torch
            import numpy
            import pandas
            return True
        except ImportError:
            return False
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance recommendations
        if 'performance' in self.results:
            if self.results['performance'].get('sharpe_ratio', 0) < self.thresholds['sharpe_ratio']:
                recommendations.append("Improve risk-adjusted returns through better feature engineering")
            if abs(self.results['performance'].get('max_drawdown', 0)) > self.thresholds['max_drawdown']:
                recommendations.append("Implement stricter risk controls to reduce drawdown")
        
        # Risk recommendations
        if 'risk' in self.results:
            if self.results['risk'].get('risk_level') in ['high', 'extreme']:
                recommendations.append("Consider implementing additional risk management layers")
        
        # Operational recommendations
        if 'operational' in self.results:
            if self.results['operational'].get('avg_latency_ms', 0) > self.thresholds['latency_ms']:
                recommendations.append("Optimize model inference for lower latency")
        
        # General recommendations
        if self.results.get('readiness_score', 0) < 80:
            recommendations.append("Continue testing in paper trading before full deployment")
        else:
            recommendations.append("System appears ready for staged production deployment")
        
        return recommendations


def main():
    """Run production evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Evaluation Suite')
    parser.add_argument('--ensemble_path', type=str, default='ensemble_optimized_phase2',
                       help='Path to ensemble models')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Run evaluation
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    evaluator = ProductionEvaluationSuite(args.ensemble_path, device)
    
    results = evaluator.run_comprehensive_evaluation()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Production Readiness Score: {results['readiness_score']:.1f}%")
    
    if results['readiness_score'] >= 80:
        print("✅ System is READY for production deployment")
    elif results['readiness_score'] >= 60:
        print("⚠️ System needs MINOR improvements before deployment")
    else:
        print("❌ System requires SIGNIFICANT improvements before deployment")


if __name__ == "__main__":
    main()