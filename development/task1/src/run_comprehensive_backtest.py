"""
Main Execution Script for Comprehensive Backtesting
Orchestrates all backtesting components for complete analysis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all backtesting components
from comprehensive_backtester import ComprehensiveBacktester, BacktestConfig
from backtest_metrics import AdvancedMetrics
from market_condition_backtester import AdvancedMarketRegimeDetector, MarketConditionBacktester
from transaction_cost_analyzer import TransactionCostAnalyzer, CostModel
from statistical_validator import StatisticalValidator, CrossValidationFramework
from backtest_visualizer import BacktestVisualizer, create_interactive_dashboard
from backtest_report_generator import BacktestReportGenerator

class ComprehensiveBacktestRunner:
    """Main orchestrator for comprehensive backtesting"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.results = {}
        self.components = {}
        
        # Initialize components
        self._initialize_components()
        
    def _load_config(self, config_file: str = None) -> dict:
        """Load configuration from file or use defaults"""
        
        default_config = {
            # Backtesting parameters
            'ensemble_path': 'ensemble_optimized_phase2/ensemble_models',
            'data_split_ratio': 0.7,
            'walk_forward_window': 3000,
            'overlap_ratio': 0.2,
            'monte_carlo_runs': 100,
            
            # Analysis parameters
            'confidence_level': 0.95,
            'regime_window': 100,
            'cost_analysis': True,
            'statistical_validation': True,
            
            # Output parameters
            'generate_visualizations': True,
            'generate_reports': True,
            'output_formats': ['html', 'markdown', 'json', 'pdf'],
            'save_results': True,
            
            # Performance thresholds
            'min_sharpe': 0.5,
            'max_drawdown_threshold': 0.05,
            'min_win_rate': 0.5,
            'min_validation_score': 70,
            
            # Benchmark data
            'benchmark_file': None,  # Path to benchmark data file (CSV with dates and returns)
            'benchmark_symbol': 'BTC',  # Symbol for benchmark
            
            # Technical parameters
            'gpu_id': 0,
            'random_seed': 42,
            'verbose': True
        }
        
        if config_file and os.path.exists(config_file):
            import json
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                print(f"✓ Configuration loaded from {config_file}")
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing JSON config file {config_file}: {e}")
                print("Using default configuration")
            except Exception as e:
                print(f"⚠️ Error reading config file {config_file}: {e}")
                print("Using default configuration")
            
        return default_config
    
    def _load_benchmark_data(self) -> np.ndarray:
        """Load benchmark data for comparison"""
        
        benchmark_file = self.config.get('benchmark_file')
        
        if not benchmark_file or not os.path.exists(benchmark_file):
            print("⚠️ No benchmark file specified or file not found")
            return None
        
        try:
            # Try to load benchmark data from CSV
            benchmark_df = pd.read_csv(benchmark_file)
            
            # Assume the file has columns: date, price or return
            if 'return' in benchmark_df.columns:
                benchmark_returns = benchmark_df['return'].values
            elif 'returns' in benchmark_df.columns:
                benchmark_returns = benchmark_df['returns'].values
            elif 'price' in benchmark_df.columns:
                # Calculate returns from prices
                prices = benchmark_df['price'].values
                benchmark_returns = np.diff(prices) / prices[:-1]
            else:
                # Try to use the second column as values
                price_col = benchmark_df.columns[1]
                prices = benchmark_df[price_col].values
                benchmark_returns = np.diff(prices) / prices[:-1]
            
            print(f"✓ Loaded benchmark data: {len(benchmark_returns)} return observations")
            return benchmark_returns
            
        except Exception as e:
            print(f"⚠️ Error loading benchmark data: {e}")
            return None
    
    def _initialize_components(self):
        """Initialize all backtesting components"""
        
        print("🔧 Initializing backtesting components...")
        
        try:
            # Backtesting engine
            backtest_config = BacktestConfig(
                ensemble_path=self.config['ensemble_path'],
                data_split_ratio=self.config['data_split_ratio'],
                walk_forward_window=self.config['walk_forward_window'],
                overlap_ratio=self.config['overlap_ratio'],
                confidence_level=self.config['confidence_level'],
                monte_carlo_runs=self.config['monte_carlo_runs'],
                regime_window=self.config['regime_window'],
                min_sharpe=self.config['min_sharpe'],
                max_drawdown_threshold=self.config['max_drawdown_threshold'],
                min_win_rate=self.config['min_win_rate']
            )
            
            self.components['backtester'] = ComprehensiveBacktester(backtest_config)
            print("✅ Backtesting engine initialized")
            
            # Metrics calculator
            self.components['metrics'] = AdvancedMetrics()
            print("✅ Metrics calculator initialized")
            
            # Market regime detector
            self.components['regime_detector'] = AdvancedMarketRegimeDetector(
                lookback_period=self.config['regime_window']
            )
            print("✅ Market regime detector initialized")
            
            # Market condition backtester
            self.components['market_backtester'] = MarketConditionBacktester(
                self.components['regime_detector']
            )
            print("✅ Market condition backtester initialized")
            
            # Transaction cost analyzer
            if self.config['cost_analysis']:
                cost_model = CostModel()
                self.components['cost_analyzer'] = TransactionCostAnalyzer(cost_model)
                print("✅ Transaction cost analyzer initialized")
            
            # Statistical validator
            if self.config['statistical_validation']:
                self.components['validator'] = StatisticalValidator(
                    confidence_level=self.config['confidence_level']
                )
                self.components['cv_framework'] = CrossValidationFramework(n_splits=5)
                print("✅ Statistical validator initialized")
            
            # Visualization components
            if self.config['generate_visualizations']:
                self.components['visualizer'] = BacktestVisualizer()
                print("✅ Visualizer initialized")
            
            # Report generator
            if self.config['generate_reports']:
                self.components['report_generator'] = BacktestReportGenerator()
                print("✅ Report generator initialized")
                
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            sys.exit(1)
    
    def run_comprehensive_analysis(self) -> dict:
        """Run complete comprehensive backtesting analysis"""
        
        print("\n🚀 Starting Comprehensive Backtesting Analysis")
        print("=" * 70)
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        
        try:
            # Phase 1: Core Backtesting
            print("\n📊 Phase 1: Core Backtesting")
            print("-" * 40)
            
            backtest_results = self._run_core_backtesting()
            self.results['backtest_results'] = backtest_results
            
            # Phase 2: Market Condition Analysis
            print("\n📈 Phase 2: Market Condition Analysis")
            print("-" * 40)
            
            market_analysis = self._run_market_analysis(backtest_results)
            self.results['market_analysis'] = market_analysis
            
            # Phase 3: Statistical Validation
            if self.config['statistical_validation']:
                print("\n🧪 Phase 3: Statistical Validation")
                print("-" * 40)
                
                validation_results = self._run_statistical_validation(backtest_results)
                self.results['validation_results'] = validation_results
            
            # Phase 4: Transaction Cost Analysis
            if self.config['cost_analysis']:
                print("\n💰 Phase 4: Transaction Cost Analysis")
                print("-" * 40)
                
                cost_analysis = self._run_cost_analysis(backtest_results)
                self.results['cost_analysis'] = cost_analysis
            
            # Phase 5: Advanced Metrics Calculation
            print("\n📈 Phase 5: Advanced Metrics")
            print("-" * 40)
            
            advanced_metrics = self._calculate_advanced_metrics(backtest_results)
            self.results['advanced_metrics'] = advanced_metrics
            
            # Phase 6: Visualization Generation
            if self.config['generate_visualizations']:
                print("\n🎨 Phase 6: Visualization Generation")
                print("-" * 40)
                
                visualizations = self._generate_visualizations(backtest_results)
                self.results['visualizations'] = visualizations
            
            # Phase 7: Report Generation
            if self.config['generate_reports']:
                print("\n📝 Phase 7: Report Generation")
                print("-" * 40)
                
                reports = self._generate_reports(backtest_results)
                self.results['reports'] = reports
            
            # Phase 8: Final Summary
            print("\n📋 Phase 8: Final Summary")
            print("-" * 40)
            
            summary = self._generate_final_summary()
            self.results['summary'] = summary
            
            # Save results if requested
            if self.config['save_results']:
                self._save_results()
            
            print("\n✅ Comprehensive backtesting analysis completed successfully!")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            if self.config['verbose']:
                import traceback
                traceback.print_exc()
            return {'error': str(e)}
    
    def _run_core_backtesting(self) -> dict:
        """Run core backtesting procedures"""
        
        backtester = self.components['backtester']
        results = {}
        
        # 1. Standard backtest
        print("Running standard backtest...")
        standard_result = backtester.run_standard_backtest()
        results['standard'] = standard_result
        print(f"✓ Standard backtest: Sharpe {standard_result.sharpe_ratio:.3f}, Return {standard_result.total_return:.2%}")
        
        # 2. Walk-forward analysis
        print("Running walk-forward analysis...")
        walkforward_results = backtester.run_walk_forward_analysis()
        results['walk_forward'] = walkforward_results
        
        if walkforward_results:
            avg_sharpe = np.mean([r.sharpe_ratio for r in walkforward_results if not np.isnan(r.sharpe_ratio)])
            print(f"✓ Walk-forward analysis: {len(walkforward_results)} periods, Avg Sharpe {avg_sharpe:.3f}")
        
        # 3. Regime-specific backtests
        print("Running regime-specific backtests...")
        regime_results = backtester.run_regime_specific_backtests()
        results['by_regime'] = regime_results
        
        total_regime_periods = sum(len(periods) for periods in regime_results.values())
        print(f"✓ Regime analysis: {len(regime_results)} regimes, {total_regime_periods} total periods")
        
        # 4. Monte Carlo backtests
        print("Running Monte Carlo backtests...")
        mc_results = backtester.run_monte_carlo_backtest(num_runs=self.config['monte_carlo_runs'])
        results['monte_carlo'] = mc_results
        
        if mc_results:
            mc_sharpe = np.mean([r.sharpe_ratio for r in mc_results if not np.isnan(r.sharpe_ratio)])
            print(f"✓ Monte Carlo: {len(mc_results)} runs, Avg Sharpe {mc_sharpe:.3f}")
        
        # Combine all results for summary
        all_results = [standard_result]
        if walkforward_results:
            all_results.extend(walkforward_results)
        if mc_results:
            all_results.extend(mc_results)
        for regime_list in regime_results.values():
            all_results.extend(regime_list)
        
        results['all_results'] = all_results
        results['summary'] = backtester.generate_summary_statistics(all_results)
        
        return results
    
    def _run_market_analysis(self, backtest_results: dict) -> dict:
        """Run market condition analysis"""
        
        market_backtester = self.components['market_backtester']
        all_results = backtest_results['all_results']
        
        # Detect market conditions from data
        print("Detecting market regimes...")
        
        # Get price data for regime detection
        try:
            backtester = self.components['backtester']
            prices = backtester.prices
            market_conditions = self.components['regime_detector'].detect_regimes(prices)
            
            print(f"✓ Detected {len(market_conditions)} market regime periods")
            
            # Analyze performance by regime
            print("Analyzing regime-specific performance...")
            regime_performance = market_backtester.analyze_regime_performance(
                all_results, market_conditions
            )
            
            print(f"✓ Analyzed performance across {len(regime_performance)} regimes")
            
            return {
                'market_conditions': market_conditions,
                'regime_performance': regime_performance,
                'regime_summary': market_backtester.generate_regime_comparison_report(regime_performance)
            }
            
        except Exception as e:
            print(f"⚠️ Market analysis error: {e}")
            return {'error': str(e)}
    
    def _run_statistical_validation(self, backtest_results: dict) -> dict:
        """Run statistical validation"""
        
        validator = self.components['validator']
        cv_framework = self.components['cv_framework']
        
        # Aggregate returns for validation
        all_results = backtest_results['all_results']
        all_returns = []
        
        for result in all_results:
            if hasattr(result, 'equity_curve') and len(result.equity_curve) > 1:
                denominator = result.equity_curve[:-1]
                # Avoid division by zero
                non_zero_mask = denominator != 0
                if np.any(non_zero_mask):
                    returns = np.zeros_like(denominator)
                    returns[non_zero_mask] = np.diff(result.equity_curve)[non_zero_mask] / denominator[non_zero_mask]
                    all_returns.extend(returns[non_zero_mask])  # Only add valid returns
        
        if not all_returns:
            print("⚠️ No returns available for statistical validation")
            return {'error': 'No returns data'}
        
        returns_array = np.array(all_returns)
        
        # Main statistical validation
        print("Running comprehensive statistical validation...")
        validation_result = validator.validate_strategy(returns_array, strategy_name="Trading Strategy")
        
        passed_tests = len([t for t in validation_result.test_results if t.passed])
        total_tests = len(validation_result.test_results)
        
        print(f"✓ Statistical validation: {passed_tests}/{total_tests} tests passed ({validation_result.overall_score:.1f}%)")
        print(f"✓ Reliability: {validation_result.reliability_assessment}")
        print(f"✓ Risk level: {validation_result.risk_level}")
        
        # Cross-validation
        print("Running cross-validation analysis...")
        cv_results = cv_framework.validate_strategy_performance(returns_array)
        
        print(f"✓ Cross-validation: Mean Sharpe {cv_results['mean_cv_sharpe']:.3f} ± {cv_results['std_cv_sharpe']:.3f}")
        
        return {
            'validation_result': validation_result,
            'cross_validation': cv_results
        }
    
    def _run_cost_analysis(self, backtest_results: dict) -> dict:
        """Run transaction cost analysis using actual trade logs"""
        
        cost_analyzer = self.components['cost_analyzer']
        
        # Analyze actual trades from backtest results
        print("Analyzing transaction costs from actual trades...")
        
        all_results = backtest_results['all_results']
        all_executions = []
        cost_bps_list = []
        
        for result in all_results:
            # Get actual trade log from backtest result
            if hasattr(result, 'trade_log') and result.trade_log:
                trades = result.trade_log
                
                for trade in trades:
                    try:
                        # Extract trade information
                        action = trade.get('action', 'buy')
                        price = trade.get('price', 50000)
                        quantity = trade.get('quantity', 1.0)
                        
                        # Create realistic market data based on trade price
                        spread_pct = np.random.uniform(0.0001, 0.0005)  # 1-5 bps spread
                        market_data = {
                            'bid': price * (1 - spread_pct/2),
                            'ask': price * (1 + spread_pct/2),
                            'volume': np.random.uniform(500, 2000),
                            'volatility': np.random.uniform(0.015, 0.035),
                            'mid': price
                        }
                        
                        # Calculate execution costs for actual trade
                        from transaction_cost_analyzer import OrderSide, OrderType
                        
                        order_side = OrderSide.BUY if action == 'buy' else OrderSide.SELL
                        order_type = OrderType.MARKET  # Assuming market orders
                        
                        execution = cost_analyzer.calculate_execution_costs(
                            order_side=order_side,
                            order_type=order_type,
                            quantity=quantity,
                            target_price=price,
                            market_data=market_data,
                            order_id=f"trade_{len(all_executions)}"
                        )
                        
                        all_executions.append(execution)
                        cost_bps = cost_analyzer.calculate_cost_basis_points(execution)
                        cost_bps_list.append(cost_bps)
                        
                    except Exception as e:
                        print(f"⚠️ Error analyzing trade: {e}")
                        continue
        
        # Fall back to simulation if no actual trades found
        if not all_executions:
            print("No actual trades found, falling back to simulation...")
            return self._simulate_cost_analysis(all_results, cost_analyzer)
        
        # Analyze costs from actual trades
        if cost_bps_list:
            avg_cost = np.mean(cost_bps_list)
            median_cost = np.median(cost_bps_list)
            total_trades = len(cost_bps_list)
            
            print(f"✓ Transaction cost analysis: {total_trades} actual trades analyzed")
            print(f"✓ Average cost: {avg_cost:.1f} basis points")
            print(f"✓ Median cost: {median_cost:.1f} basis points")
            
            # Generate comprehensive cost analysis
            analysis = cost_analyzer.analyze_execution_quality()
            
            return {
                'actual_trades': True,
                'executions': all_executions,
                'cost_bps_list': cost_bps_list,
                'analysis': analysis,
                'average_cost_bps': avg_cost,
                'median_cost_bps': median_cost,
                'total_trades': total_trades,
                'cost_report': cost_analyzer.generate_cost_analysis_report()
            }
        else:
            print("⚠️ No valid cost data, falling back to simulation")
            return self._simulate_cost_analysis(all_results, cost_analyzer)
    
    def _simulate_cost_analysis(self, all_results: list, cost_analyzer) -> dict:
        """Fallback simulation-based cost analysis"""
        
        simulated_costs = []
        total_simulated_trades = 0
        
        # Determine minimum number of trades for robust analysis
        min_trades_per_result = 30
        max_total_trades = 1000  # Cap to prevent excessive computation
        
        for result in all_results:
            # Get trading frequency from backtest result
            num_trades = getattr(result, 'num_trades', min_trades_per_result)
            
            # Simulate more trades based on result characteristics
            if hasattr(result, 'total_return') and abs(result.total_return) > 0.1:
                # More active strategies get more simulated trades
                num_trades = max(num_trades, 50)
            
            trades_to_simulate = min(num_trades, 100)  # Per result limit
            
            for i in range(trades_to_simulate):
                if total_simulated_trades >= max_total_trades:
                    break
                    
                try:
                    # More realistic market data simulation
                    base_price = 50000 * np.random.uniform(0.8, 1.2)  # Price variation
                    spread_bps = np.random.uniform(1, 8)  # 1-8 bps spread
                    spread_amount = base_price * spread_bps / 10000
                    
                    market_data = {
                        'bid': base_price - spread_amount/2,
                        'ask': base_price + spread_amount/2,
                        'volume': np.random.uniform(500, 5000),  # Larger volume range
                        'volatility': np.random.uniform(0.01, 0.05),
                        'mid': base_price
                    }
                    
                    # Simulate execution with varying order characteristics
                    from transaction_cost_analyzer import OrderSide, OrderType
                    
                    # Realistic order size distribution (log-normal)
                    order_size = np.random.lognormal(mean=0, sigma=0.5)
                    order_size = np.clip(order_size, 0.01, 10.0)  # Reasonable bounds
                    
                    execution = cost_analyzer.calculate_execution_costs(
                        order_side=OrderSide.BUY if np.random.random() > 0.5 else OrderSide.SELL,
                        order_type=OrderType.MARKET if np.random.random() > 0.3 else OrderType.LIMIT,
                        quantity=order_size,
                        target_price=base_price,
                        market_data=market_data,
                        order_id=f"sim_trade_{total_simulated_trades}"
                    )
                    
                    cost_bps = cost_analyzer.calculate_cost_basis_points(execution)
                    if not np.isnan(cost_bps) and cost_bps < 1000:  # Sanity check
                        simulated_costs.append(cost_bps)
                        total_simulated_trades += 1
                        
                except Exception as e:
                    print(f"⚠️ Error in cost simulation: {e}")
                    continue
            
            if total_simulated_trades >= max_total_trades:
                break
        
        # Analyze costs with improved statistics
        if simulated_costs:
            costs_array = np.array(simulated_costs)
            
            # Calculate comprehensive cost statistics
            cost_stats = {
                'mean': np.mean(costs_array),
                'median': np.median(costs_array),
                'std': np.std(costs_array),
                'min': np.min(costs_array),
                'max': np.max(costs_array),
                'p25': np.percentile(costs_array, 25),
                'p75': np.percentile(costs_array, 75),
                'p95': np.percentile(costs_array, 95)
            }
            
            print(f"✓ Transaction cost analysis: {len(simulated_costs)} simulated trades")
            print(f"✓ Average cost: {cost_stats['mean']:.1f} basis points")
            print(f"✓ Median cost: {cost_stats['median']:.1f} basis points")
            print(f"✓ 95th percentile: {cost_stats['p95']:.1f} basis points")
            
            # Generate cost analysis
            analysis = cost_analyzer.analyze_execution_quality()
            
            return {
                'simulated': True,
                'simulated_costs': simulated_costs,
                'cost_statistics': cost_stats,
                'analysis': analysis,
                'average_cost_bps': cost_stats['mean'],
                'median_cost_bps': cost_stats['median'],
                'total_simulated_trades': len(simulated_costs),
                'cost_report': cost_analyzer.generate_cost_analysis_report()
            }
        else:
            print("⚠️ No cost data available for analysis")
            return {'error': 'No cost data'}
    
    def _calculate_advanced_metrics(self, backtest_results: dict) -> dict:
        """Calculate advanced performance metrics"""
        
        metrics_calc = self.components['metrics']
        all_results = backtest_results['all_results']
        
        # Aggregate all returns
        all_returns = []
        all_prices = []
        
        for result in all_results:
            if hasattr(result, 'equity_curve') and len(result.equity_curve) > 1:
                denominator = result.equity_curve[:-1]
                # Avoid division by zero
                non_zero_mask = denominator != 0
                if np.any(non_zero_mask):
                    returns = np.zeros_like(denominator)
                    returns[non_zero_mask] = np.diff(result.equity_curve)[non_zero_mask] / denominator[non_zero_mask]
                    all_returns.extend(returns[non_zero_mask])  # Only add valid returns
                all_prices.extend(result.equity_curve)
        
        if not all_returns:
            print("⚠️ No returns available for advanced metrics")
            return {'error': 'No returns data'}
        
        print("Calculating comprehensive performance metrics...")
        
        # Calculate all metrics
        advanced_metrics = metrics_calc.calculate_all_metrics(
            np.array(all_returns), 
            np.array(all_prices)
        )
        
        print(f"✓ Calculated {len(advanced_metrics)} advanced metrics")
        print(f"✓ Sharpe ratio: {advanced_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"✓ Maximum drawdown: {advanced_metrics.get('max_drawdown', 0):.2%}")
        print(f"✓ VaR (95%): {advanced_metrics.get('var_95', 0):.2%}")
        
        return advanced_metrics
    
    def _generate_visualizations(self, backtest_results: dict) -> dict:
        """Generate visualizations"""
        
        visualizer = self.components['visualizer']
        all_results = backtest_results['all_results']
        
        # Create visualization output directory
        viz_dir = "results/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        print(f"Generating comprehensive dashboard in {viz_dir}...")
        
        try:
            # Generate static dashboard
            static_path = os.path.join(viz_dir, "comprehensive_backtest_dashboard.png")
            fig = visualizer.create_comprehensive_dashboard(
                all_results,
                save_path=static_path
            )
            
            print(f"✓ Static dashboard saved to '{static_path}'")
            
            # Generate interactive dashboard
            interactive_path = os.path.join(viz_dir, "interactive_backtest_dashboard.html")
            interactive_fig = create_interactive_dashboard(all_results)
            interactive_fig.write_html(interactive_path)
            
            print(f"✓ Interactive dashboard saved to '{interactive_path}'")
            
            return {
                'static_dashboard': static_path,
                'interactive_dashboard': interactive_path,
                'output_directory': viz_dir
            }
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
            return {'error': str(e)}
    
    def _generate_reports(self, backtest_results: dict) -> dict:
        """Generate comprehensive reports"""
        
        report_generator = self.components['report_generator']
        all_results = backtest_results['all_results']
        
        print("Generating comprehensive reports...")
        
        try:
            # Get additional data for reports
            benchmark_returns = self._load_benchmark_data()
            market_conditions = self.results.get('market_analysis', {}).get('market_conditions')
            transaction_costs = self.results.get('cost_analysis', {}).get('simulated_costs')
            
            # Generate reports
            output_files = report_generator.generate_comprehensive_report(
                backtest_results=all_results,
                strategy_name="FinRL Contest 2024 Trading Strategy",
                benchmark_returns=benchmark_returns,
                market_conditions=market_conditions,
                transaction_costs=transaction_costs,
                output_formats=self.config['output_formats']
            )
            
            print("✓ Reports generated:")
            for format_type, filename in output_files.items():
                print(f"  - {format_type.upper()}: {filename}")
            
            return output_files
            
        except Exception as e:
            print(f"⚠️ Report generation error: {e}")
            return {'error': str(e)}
    
    def _generate_final_summary(self) -> dict:
        """Generate final analysis summary"""
        
        print("Generating final summary...")
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'components_used': list(self.components.keys()),
            'results_generated': list(self.results.keys())
        }
        
        # Extract key metrics
        if 'backtest_results' in self.results:
            backtest_summary = self.results['backtest_results'].get('summary', {})
            summary['performance'] = backtest_summary.get('overall_assessment', {})
        
        if 'advanced_metrics' in self.results:
            summary['key_metrics'] = {
                'sharpe_ratio': self.results['advanced_metrics'].get('sharpe_ratio', 0),
                'max_drawdown': self.results['advanced_metrics'].get('max_drawdown', 0),
                'total_return': self.results['advanced_metrics'].get('total_return', 0),
                'win_rate': self.results['advanced_metrics'].get('win_rate', 0)
            }
        
        if 'validation_results' in self.results:
            validation = self.results['validation_results']['validation_result']
            summary['statistical_validation'] = {
                'overall_score': validation.overall_score,
                'risk_level': validation.risk_level,
                'reliability': validation.reliability_assessment
            }
        
        # Generate recommendations using configurable thresholds
        recommendations = []
        
        min_sharpe_threshold = self.config.get('min_sharpe', 0.5)
        max_drawdown_threshold = self.config.get('max_drawdown_threshold', 0.05)
        min_validation_score = self.config.get('min_validation_score', 70)
        
        if summary.get('key_metrics', {}).get('sharpe_ratio', 0) < min_sharpe_threshold:
            recommendations.append(f"Improve risk-adjusted returns through better signal generation (current < {min_sharpe_threshold})")
        
        if abs(summary.get('key_metrics', {}).get('max_drawdown', 0)) > max_drawdown_threshold:
            recommendations.append(f"Implement stronger risk controls to reduce drawdowns (current > {max_drawdown_threshold:.1%})")
        
        if summary.get('statistical_validation', {}).get('overall_score', 0) < min_validation_score:
            recommendations.append(f"Address statistical validation issues before deployment (current < {min_validation_score}%)")
        
        summary['recommendations'] = recommendations
        
        print("✓ Final summary generated")
        
        return summary
    
    def _serialize_complex_object(self, obj):
        """Convert complex objects to JSON-serializable format"""
        
        if hasattr(obj, '__dict__'):
            # For objects with attributes, extract key information
            result = {'_type': type(obj).__name__}
            
            for key, value in obj.__dict__.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    result[key] = value
                elif isinstance(value, (list, tuple)):
                    result[key] = [self._serialize_complex_object(item) for item in value[:10]]  # Limit for size
                elif isinstance(value, dict):
                    result[key] = {k: self._serialize_complex_object(v) for k, v in list(value.items())[:10]}
                elif isinstance(value, np.ndarray):
                    if value.size < 100:  # Small arrays only
                        result[key] = value.tolist()
                    else:
                        result[key] = {
                            '_type': 'numpy_array',
                            'shape': value.shape,
                            'dtype': str(value.dtype),
                            'size': value.size,
                            'sample': value.flatten()[:10].tolist() if value.size > 0 else []
                        }
                elif hasattr(value, '__dict__'):
                    result[key] = self._serialize_complex_object(value)
                else:
                    result[key] = str(value)
                    
            return result
            
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_complex_object(item) for item in obj[:10]]
        elif isinstance(obj, dict):
            return {k: self._serialize_complex_object(v) for k, v in list(obj.items())[:10]}
        elif isinstance(obj, np.ndarray):
            if obj.size < 100:
                return obj.tolist()
            else:
                return {
                    '_type': 'numpy_array',
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size': obj.size,
                    'sample': obj.flatten()[:10].tolist() if obj.size > 0 else []
                }
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
            
    def _save_results(self):
        """Save all results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            import json
            import pickle
            
            # Save as JSON (for human readable data)
            json_filename = f"backtest_results_{timestamp}.json"
            
            # Create comprehensive JSON-serializable version
            json_results = {}
            for key, value in self.results.items():
                try:
                    json_results[key] = self._serialize_complex_object(value)
                except Exception as e:
                    print(f"⚠️ Error serializing {key}: {e}")
                    json_results[key] = f"Serialization error - {type(value).__name__}"
            
            with open(json_filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"✓ JSON results saved to {json_filename}")
            
            # Save complete results as pickle (for Python objects)
            pickle_filename = f"backtest_results_complete_{timestamp}.pkl"
            with open(pickle_filename, 'wb') as f:
                pickle.dump(self.results, f)
            
            print(f"✓ Complete results saved to {pickle_filename}")
            
        except Exception as e:
            print(f"⚠️ Error saving results: {e}")

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Comprehensive Backtesting Framework')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--ensemble-path', type=str, default='ensemble_optimized_phase2',
                       help='Path to ensemble models')
    parser.add_argument('--monte-carlo-runs', type=int, default=100,
                       help='Number of Monte Carlo runs')
    parser.add_argument('--walk-forward-window', type=int, default=3000,
                       help='Walk-forward window size')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip report generation')
    parser.add_argument('--output-formats', nargs='+', default=['html', 'markdown'],
                       choices=['html', 'markdown', 'json', 'pdf'],
                       help='Report output formats')
    parser.add_argument('--benchmark-file', type=str,
                       help='Path to benchmark data file (CSV)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("🚀 FinRL Contest 2024 - Comprehensive Backtesting Framework")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Override config with command line arguments
    config_overrides = {
        'ensemble_path': args.ensemble_path,
        'monte_carlo_runs': args.monte_carlo_runs,
        'walk_forward_window': args.walk_forward_window,
        'generate_visualizations': not args.no_visualizations,
        'generate_reports': not args.no_reports,
        'output_formats': args.output_formats,
        'benchmark_file': args.benchmark_file,
        'verbose': args.verbose
    }
    
    # Initialize runner
    runner = ComprehensiveBacktestRunner(args.config)
    
    # Apply command line overrides
    runner.config.update(config_overrides)
    
    # Run comprehensive analysis
    results = runner.run_comprehensive_analysis()
    
    if 'error' not in results:
        print("\n" + "=" * 70)
        print("📋 FINAL SUMMARY")
        print("=" * 70)
        
        summary = results.get('summary', {})
        
        # Performance summary
        if 'key_metrics' in summary:
            metrics = summary['key_metrics']
            print(f"📊 Performance Metrics:")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        
        # Statistical validation
        if 'statistical_validation' in summary:
            validation = summary['statistical_validation']
            print(f"\n🧪 Statistical Validation:")
            print(f"   Overall Score: {validation.get('overall_score', 0):.1f}%")
            print(f"   Risk Level: {validation.get('risk_level', 'Unknown')}")
            print(f"   Reliability: {validation.get('reliability', 'Unknown')}")
        
        # Recommendations
        if 'recommendations' in summary and summary['recommendations']:
            print(f"\n💡 Key Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Generated outputs
        if 'reports' in results:
            print(f"\n📄 Generated Reports:")
            for format_type, filename in results['reports'].items():
                print(f"   {format_type.upper()}: {filename}")
        
        if 'visualizations' in results:
            print(f"\n🎨 Generated Visualizations:")
            for viz_type, filename in results['visualizations'].items():
                print(f"   {viz_type}: {filename}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"⏱️ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    else:
        print(f"\n❌ Analysis failed: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()