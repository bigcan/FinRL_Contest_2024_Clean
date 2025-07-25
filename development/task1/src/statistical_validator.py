"""
Statistical Validation Framework for Backtesting
Rigorous statistical tests and validation methods for trading strategies
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest, ttest_1samp, mannwhitneyu
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StatisticalTest:
    """Statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    passed: bool
    interpretation: str
    confidence_level: float = 0.95

@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    strategy_name: str
    test_results: List[StatisticalTest]
    overall_score: float
    risk_level: str
    reliability_assessment: str
    recommendations: List[str]
    
class StatisticalValidator:
    """Comprehensive statistical validation framework"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def validate_strategy(self, returns: np.ndarray, 
                         benchmark_returns: np.ndarray = None,
                         strategy_name: str = "Strategy") -> ValidationResult:
        """Comprehensive strategy validation"""
        
        test_results = []
        
        # 1. Normality Tests
        test_results.extend(self._test_normality(returns))
        
        # 2. Stationarity Tests
        test_results.extend(self._test_stationarity(returns))
        
        # 3. Independence Tests (Serial Correlation)
        test_results.extend(self._test_independence(returns))
        
        # 4. Performance Significance Tests
        test_results.extend(self._test_performance_significance(returns))
        
        # 5. Risk Tests
        test_results.extend(self._test_risk_characteristics(returns))
        
        # 6. Benchmark Comparison Tests (if benchmark provided)
        if benchmark_returns is not None:
            test_results.extend(self._test_benchmark_comparison(returns, benchmark_returns))
        
        # 7. Stability Tests
        test_results.extend(self._test_stability(returns))
        
        # 8. Outlier Tests
        test_results.extend(self._test_outliers(returns))
        
        # Calculate overall assessment
        overall_score = self._calculate_overall_score(test_results)
        risk_level = self._assess_risk_level(test_results, overall_score)
        reliability = self._assess_reliability(test_results, overall_score)
        recommendations = self._generate_recommendations(test_results)
        
        return ValidationResult(
            strategy_name=strategy_name,
            test_results=test_results,
            overall_score=overall_score,
            risk_level=risk_level,
            reliability_assessment=reliability,
            recommendations=recommendations
        )
    
    def _test_normality(self, returns: np.ndarray) -> List[StatisticalTest]:
        """Test return distribution normality"""
        tests = []
        
        if len(returns) < 8:
            return tests
        
        # Jarque-Bera test
        try:
            jb_stat, jb_p = jarque_bera(returns)
            tests.append(StatisticalTest(
                test_name="Jarque-Bera Normality",
                statistic=jb_stat,
                p_value=jb_p,
                critical_value=None,
                passed=jb_p > self.alpha,
                interpretation="Returns follow normal distribution" if jb_p > self.alpha else "Returns are not normally distributed",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        # Shapiro-Wilk test (for smaller samples)
        if len(returns) <= 5000:
            try:
                sw_stat, sw_p = shapiro(returns)
                tests.append(StatisticalTest(
                    test_name="Shapiro-Wilk Normality",
                    statistic=sw_stat,
                    p_value=sw_p,
                    critical_value=None,
                    passed=sw_p > self.alpha,
                    interpretation="Returns follow normal distribution" if sw_p > self.alpha else "Returns are not normally distributed",
                    confidence_level=self.confidence_level
                ))
            except Exception as e:
                pass
        
        # Kolmogorov-Smirnov test against normal
        try:
            ks_stat, ks_p = kstest(returns, 'norm', args=(np.mean(returns), np.std(returns)))
            tests.append(StatisticalTest(
                test_name="Kolmogorov-Smirnov Normality",
                statistic=ks_stat,
                p_value=ks_p,
                critical_value=None,
                passed=ks_p > self.alpha,
                interpretation="Returns follow normal distribution" if ks_p > self.alpha else "Returns deviate from normal distribution",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        return tests
    
    def _test_stationarity(self, returns: np.ndarray) -> List[StatisticalTest]:
        """Test return series stationarity"""
        tests = []
        
        if len(returns) < 12:
            return tests
        
        # Augmented Dickey-Fuller test
        try:
            adf_stat, adf_p, _, _, critical_values, _ = adfuller(returns, autolag='AIC')
            critical_1 = critical_values['1%']
            
            tests.append(StatisticalTest(
                test_name="Augmented Dickey-Fuller Stationarity",
                statistic=adf_stat,
                p_value=adf_p,
                critical_value=critical_1,
                passed=adf_p < self.alpha,
                interpretation="Returns are stationary" if adf_p < self.alpha else "Returns may have unit root (non-stationary)",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        return tests
    
    def _test_independence(self, returns: np.ndarray) -> List[StatisticalTest]:
        """Test return independence (serial correlation)"""
        tests = []
        
        if len(returns) < 20:
            return tests
        
        # Ljung-Box test for serial correlation
        try:
            lags = min(10, len(returns) // 4)
            lb_result = acorr_ljungbox(returns, lags=lags, return_df=True)
            
            # Use the overall test statistic
            lb_stat = lb_result['lb_stat'].iloc[-1]
            lb_p = lb_result['lb_pvalue'].iloc[-1]
            
            tests.append(StatisticalTest(
                test_name="Ljung-Box Independence",
                statistic=lb_stat,
                p_value=lb_p,
                critical_value=None,
                passed=lb_p > self.alpha,
                interpretation="Returns are independent" if lb_p > self.alpha else "Returns show serial correlation",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        # Durbin-Watson test (simplified)
        try:
            dw_stat = self._durbin_watson_statistic(returns)
            # DW test interpretation: values around 2 indicate no autocorrelation
            dw_passed = 1.5 < dw_stat < 2.5
            
            tests.append(StatisticalTest(
                test_name="Durbin-Watson Autocorrelation",
                statistic=dw_stat,
                p_value=None,
                critical_value=2.0,
                passed=dw_passed,
                interpretation="No significant autocorrelation" if dw_passed else "Significant autocorrelation detected",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        return tests
    
    def _test_performance_significance(self, returns: np.ndarray) -> List[StatisticalTest]:
        """Test statistical significance of performance"""
        tests = []
        
        if len(returns) < 3:
            return tests
        
        # t-test for mean return significance
        try:
            t_stat, t_p = ttest_1samp(returns, 0)
            
            tests.append(StatisticalTest(
                test_name="Mean Return Significance",
                statistic=t_stat,
                p_value=t_p,
                critical_value=None,
                passed=t_p < self.alpha and t_stat > 0,
                interpretation="Mean return is significantly positive" if (t_p < self.alpha and t_stat > 0) else "Mean return not significantly different from zero",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        # Sharpe ratio significance test
        try:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            sharpe_t_stat = sharpe * np.sqrt(len(returns))
            sharpe_p = 2 * (1 - stats.t.cdf(abs(sharpe_t_stat), len(returns) - 1))
            
            tests.append(StatisticalTest(
                test_name="Sharpe Ratio Significance",
                statistic=sharpe_t_stat,
                p_value=sharpe_p,
                critical_value=None,
                passed=sharpe_p < self.alpha and sharpe > 0,
                interpretation="Sharpe ratio is significantly positive" if (sharpe_p < self.alpha and sharpe > 0) else "Sharpe ratio not significantly different from zero",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        return tests
    
    def _test_risk_characteristics(self, returns: np.ndarray) -> List[StatisticalTest]:
        """Test risk characteristics"""
        tests = []
        
        if len(returns) < 10:
            return tests
        
        # Test for excessive kurtosis (fat tails)
        try:
            kurtosis_val = stats.kurtosis(returns)
            # Test if kurtosis is significantly different from normal (3)
            # Using approximate standard error
            n = len(returns)
            se_kurt = np.sqrt(24 / n)
            kurt_z = (kurtosis_val - 0) / se_kurt  # Excess kurtosis should be 0 for normal
            kurt_p = 2 * (1 - stats.norm.cdf(abs(kurt_z)))
            
            tests.append(StatisticalTest(
                test_name="Excess Kurtosis Test",
                statistic=kurt_z,
                p_value=kurt_p,
                critical_value=None,
                passed=kurt_p > self.alpha,
                interpretation="Normal tail behavior" if kurt_p > self.alpha else "Fat tails detected (higher risk)",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        # Test for skewness
        try:
            skewness_val = stats.skew(returns)
            # Test if skewness is significantly different from 0
            n = len(returns)
            se_skew = np.sqrt(6 / n)
            skew_z = skewness_val / se_skew
            skew_p = 2 * (1 - stats.norm.cdf(abs(skew_z)))
            
            tests.append(StatisticalTest(
                test_name="Skewness Test",
                statistic=skew_z,
                p_value=skew_p,
                critical_value=None,
                passed=skew_p > self.alpha,
                interpretation="Symmetric returns" if skew_p > self.alpha else "Asymmetric returns detected",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        return tests
    
    def _test_benchmark_comparison(self, returns: np.ndarray, 
                                 benchmark_returns: np.ndarray) -> List[StatisticalTest]:
        """Test strategy vs benchmark performance"""
        tests = []
        
        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        if len(returns) < 3:
            return tests
        
        # Two-sample t-test
        try:
            t_stat, t_p = stats.ttest_ind(returns, benchmark_returns)
            
            tests.append(StatisticalTest(
                test_name="Strategy vs Benchmark t-test",
                statistic=t_stat,
                p_value=t_p,
                critical_value=None,
                passed=t_p < self.alpha and t_stat > 0,
                interpretation="Strategy significantly outperforms benchmark" if (t_p < self.alpha and t_stat > 0) else "No significant outperformance vs benchmark",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p = mannwhitneyu(returns, benchmark_returns, alternative='greater')
            
            tests.append(StatisticalTest(
                test_name="Mann-Whitney U Test vs Benchmark",
                statistic=u_stat,
                p_value=u_p,
                critical_value=None,
                passed=u_p < self.alpha,
                interpretation="Strategy significantly outperforms benchmark (non-parametric)" if u_p < self.alpha else "No significant outperformance vs benchmark (non-parametric)",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        return tests
    
    def _test_stability(self, returns: np.ndarray) -> List[StatisticalTest]:
        """Test performance stability over time"""
        tests = []
        
        if len(returns) < 20:
            return tests
        
        # Split returns into periods and test consistency
        try:
            n_periods = min(5, len(returns) // 10)  # At least 10 observations per period
            period_size = len(returns) // n_periods
            
            period_means = []
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(returns)
                period_returns = returns[start_idx:end_idx]
                period_means.append(np.mean(period_returns))
            
            # Test if period means are significantly different
            if len(period_means) > 2:
                f_stat, f_p = stats.f_oneway(*[returns[i*period_size:(i+1)*period_size] 
                                             for i in range(n_periods-1)])
                
                tests.append(StatisticalTest(
                    test_name="Performance Stability (ANOVA)",
                    statistic=f_stat,
                    p_value=f_p,
                    critical_value=None,
                    passed=f_p > self.alpha,
                    interpretation="Stable performance across periods" if f_p > self.alpha else "Performance varies significantly across periods",
                    confidence_level=self.confidence_level
                ))
        except Exception as e:
            pass
        
        return tests
    
    def _test_outliers(self, returns: np.ndarray) -> List[StatisticalTest]:
        """Test for outliers in returns"""
        tests = []
        
        if len(returns) < 10:
            return tests
        
        # Grubbs test for outliers (simplified)
        try:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            
            # Find most extreme value
            max_deviation = np.max(np.abs(returns - mean_ret))
            grubbs_stat = max_deviation / std_ret
            
            # Critical value for Grubbs test (approximate)
            n = len(returns)
            t_critical = stats.t.ppf(1 - self.alpha/(2*n), n-2)
            grubbs_critical = ((n-1) * np.sqrt(t_critical**2 / (n-2+t_critical**2))) / np.sqrt(n)
            
            tests.append(StatisticalTest(
                test_name="Grubbs Outlier Test",
                statistic=grubbs_stat,
                p_value=None,
                critical_value=grubbs_critical,
                passed=grubbs_stat < grubbs_critical,
                interpretation="No significant outliers" if grubbs_stat < grubbs_critical else "Significant outliers detected",
                confidence_level=self.confidence_level
            ))
        except Exception as e:
            pass
        
        return tests
    
    def _durbin_watson_statistic(self, returns: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic"""
        if len(returns) < 2:
            return 2.0
        
        diff = np.diff(returns)
        return np.sum(diff**2) / np.sum(returns**2)
    
    def _calculate_overall_score(self, test_results: List[StatisticalTest]) -> float:
        """Calculate overall validation score (0-100)"""
        if not test_results:
            return 0
        
        passed_tests = sum(1 for test in test_results if test.passed)
        return (passed_tests / len(test_results)) * 100
    
    def _assess_risk_level(self, test_results: List[StatisticalTest], overall_score: float) -> str:
        """Assess overall risk level"""
        
        # Check for specific risk indicators
        risk_indicators = []
        
        for test in test_results:
            if "kurtosis" in test.test_name.lower() and not test.passed:
                risk_indicators.append("fat_tails")
            elif "outlier" in test.test_name.lower() and not test.passed:
                risk_indicators.append("outliers")
            elif "normality" in test.test_name.lower() and not test.passed:
                risk_indicators.append("non_normal")
            elif "stability" in test.test_name.lower() and not test.passed:
                risk_indicators.append("unstable")
        
        # Determine risk level
        if overall_score >= 80 and len(risk_indicators) == 0:
            return "Low"
        elif overall_score >= 60 and len(risk_indicators) <= 1:
            return "Medium"
        elif overall_score >= 40:
            return "High"
        else:
            return "Very High"
    
    def _assess_reliability(self, test_results: List[StatisticalTest], overall_score: float) -> str:
        """Assess strategy reliability"""
        
        # Check for key reliability indicators
        has_significant_performance = any(
            "significance" in test.test_name.lower() and test.passed 
            for test in test_results
        )
        
        has_stable_performance = any(
            "stability" in test.test_name.lower() and test.passed 
            for test in test_results
        )
        
        has_good_statistical_properties = any(
            "independence" in test.test_name.lower() and test.passed 
            for test in test_results
        )
        
        # Determine reliability
        if overall_score >= 80 and has_significant_performance and has_stable_performance:
            return "Highly Reliable"
        elif overall_score >= 60 and has_significant_performance:
            return "Moderately Reliable"
        elif overall_score >= 40:
            return "Low Reliability"
        else:
            return "Unreliable"
    
    def _generate_recommendations(self, test_results: List[StatisticalTest]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        failed_tests = [test for test in test_results if not test.passed]
        
        # Specific recommendations based on failed tests
        for test in failed_tests:
            if "normality" in test.test_name.lower():
                recommendations.append("Consider using non-parametric performance measures due to non-normal returns")
            elif "significance" in test.test_name.lower():
                recommendations.append("Improve strategy parameters to achieve statistically significant performance")
            elif "stability" in test.test_name.lower():
                recommendations.append("Investigate regime-dependent behavior and consider adaptive strategies")
            elif "independence" in test.test_name.lower():
                recommendations.append("Address serial correlation in returns through improved signal processing")
            elif "kurtosis" in test.test_name.lower():
                recommendations.append("Implement risk controls to manage fat tail events")
            elif "outlier" in test.test_name.lower():
                recommendations.append("Review outlier periods and improve risk management")
            elif "benchmark" in test.test_name.lower():
                recommendations.append("Enhance strategy to achieve consistent benchmark outperformance")
        
        # General recommendations
        if len(failed_tests) / len(test_results) > 0.5:
            recommendations.append("Consider fundamental strategy redesign due to multiple statistical issues")
        elif len(failed_tests) > 0:
            recommendations.append("Address specific statistical issues before production deployment")
        
        return list(set(recommendations))  # Remove duplicates

class CrossValidationFramework:
    """Time series cross-validation for strategy validation"""
    
    def __init__(self, n_splits: int = 5, test_size: int = None):
        self.n_splits = n_splits
        self.test_size = test_size
        
    def validate_strategy_performance(self, returns: np.ndarray, 
                                    features: np.ndarray = None) -> Dict[str, Any]:
        """Cross-validate strategy performance"""
        
        if features is None:
            # Simple time series split on returns
            return self._validate_returns_only(returns)
        else:
            # More sophisticated validation with features
            return self._validate_with_features(returns, features)
    
    def _validate_returns_only(self, returns: np.ndarray) -> Dict[str, Any]:
        """Validate using returns only"""
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(returns)):
            train_returns = returns[train_idx]
            test_returns = returns[test_idx]
            
            # Calculate performance metrics for each fold
            fold_result = {
                'fold': fold,
                'train_size': len(train_returns),
                'test_size': len(test_returns),
                'train_sharpe': self._calculate_sharpe(train_returns),
                'test_sharpe': self._calculate_sharpe(test_returns),
                'train_return': np.mean(train_returns) * 252,
                'test_return': np.mean(test_returns) * 252,
                'train_volatility': np.std(train_returns) * np.sqrt(252),
                'test_volatility': np.std(test_returns) * np.sqrt(252)
            }
            
            fold_results.append(fold_result)
        
        # Aggregate results
        test_sharpes = [fold['test_sharpe'] for fold in fold_results]
        test_returns = [fold['test_return'] for fold in fold_results]
        
        return {
            'fold_results': fold_results,
            'mean_cv_sharpe': np.mean(test_sharpes),
            'std_cv_sharpe': np.std(test_sharpes),
            'mean_cv_return': np.mean(test_returns),
            'std_cv_return': np.std(test_returns),
            'cv_consistency': 1 - (np.std(test_sharpes) / np.mean(test_sharpes)) if np.mean(test_sharpes) != 0 else 0
        }
    
    def _validate_with_features(self, returns: np.ndarray, features: np.ndarray) -> Dict[str, Any]:
        """Validate with feature analysis"""
        # This would include more sophisticated validation
        # For now, fall back to returns-only validation
        return self._validate_returns_only(returns)
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

def main():
    """Example usage of statistical validator"""
    
    print("🚀 Statistical Validation Demo")
    print("=" * 50)
    
    # Generate sample returns
    np.random.seed(42)
    
    # Strategy returns (slightly positive with some structure)
    strategy_returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
    strategy_returns[100:200] += 0.005  # Add some regime change
    
    # Benchmark returns
    benchmark_returns = np.random.normal(0, 0.015, 1000)
    
    # Initialize validator
    validator = StatisticalValidator(confidence_level=0.95)
    
    # Run comprehensive validation
    print("🔍 Running statistical validation...")
    validation_result = validator.validate_strategy(
        strategy_returns, benchmark_returns, "Sample Strategy"
    )
    
    # Display results
    print(f"\n📊 Validation Results for {validation_result.strategy_name}")
    print("=" * 60)
    print(f"Overall Score: {validation_result.overall_score:.1f}/100")
    print(f"Risk Level: {validation_result.risk_level}")
    print(f"Reliability: {validation_result.reliability_assessment}")
    
    print(f"\n🧪 Test Results ({len(validation_result.test_results)} tests):")
    print("-" * 60)
    
    for test in validation_result.test_results:
        status = "✅ PASS" if test.passed else "❌ FAIL"
        print(f"{status} {test.test_name}")
        print(f"    Statistic: {test.statistic:.4f}, p-value: {test.p_value:.4f}" if test.p_value else f"    Statistic: {test.statistic:.4f}")
        print(f"    {test.interpretation}")
        print()
    
    if validation_result.recommendations:
        print("💡 Recommendations:")
        print("-" * 30)
        for i, rec in enumerate(validation_result.recommendations, 1):
            print(f"{i}. {rec}")
    
    # Cross-validation example
    print("\n🔄 Cross-Validation Analysis:")
    print("-" * 40)
    
    cv_framework = CrossValidationFramework(n_splits=5)
    cv_results = cv_framework.validate_strategy_performance(strategy_returns)
    
    print(f"Mean CV Sharpe: {cv_results['mean_cv_sharpe']:.3f} ± {cv_results['std_cv_sharpe']:.3f}")
    print(f"Mean CV Return: {cv_results['mean_cv_return']:.1%} ± {cv_results['std_cv_return']:.1%}")
    print(f"CV Consistency: {cv_results['cv_consistency']:.3f}")
    
    print("\n✅ Statistical validation completed!")

if __name__ == "__main__":
    main()