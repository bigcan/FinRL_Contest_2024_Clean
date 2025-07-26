"""
Comprehensive test runner for the FinRL Contest 2024 refactored framework.

This script runs all test suites and provides detailed reporting on test results,
coverage, and performance metrics.
"""

import unittest
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Add src_refactored to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import test modules
from . import TEST_CONFIG
from .test_agents import *
from .test_ensemble import *
from .test_integration import *


class TestRunner:
    """
    Comprehensive test runner with detailed reporting.
    
    Provides options for running specific test suites, performance testing,
    and detailed result analysis.
    """
    
    def __init__(self, verbosity: int = 2):
        self.verbosity = verbosity
        self.results = {}
        self.total_time = 0.0
        
        # Test suite registry
        self.test_suites = {
            'agents': {
                'description': 'Agent implementation tests',
                'modules': [
                    'test_agents.TestAgentRegistry',
                    'test_agents.TestBaseAgentInterface',
                    'test_agents.TestDoubleDQNAgent',
                    'test_agents.TestD3QNAgent',
                    'test_agents.TestPrioritizedDQNAgent',
                    'test_agents.TestNoisyDQNAgent',
                    'test_agents.TestRainbowDQNAgent',
                    'test_agents.TestAdaptiveDQNAgent',
                    'test_agents.TestAgentIntegration',
                    'test_agents.TestAgentPerformance'
                ]
            },
            'ensemble': {
                'description': 'Ensemble strategy tests',
                'modules': [
                    'test_ensemble.TestEnsembleMetrics',
                    'test_ensemble.TestMetaLearnerNetwork',
                    'test_ensemble.TestVotingEnsemble',
                    'test_ensemble.TestStackingEnsemble',
                    'test_ensemble.TestEnsembleFactory',
                    'test_ensemble.TestEnsembleEvaluation',
                    'test_ensemble.TestEnsembleIntegration',
                    'test_ensemble.TestEnsembleCheckpoints',
                    'test_ensemble.TestEnsemblePerformance'
                ]
            },
            'integration': {
                'description': 'System integration tests',
                'modules': [
                    'test_integration.TestAgentEnsembleIntegration',
                    'test_integration.TestTrainingConfig',
                    'test_integration.TestTrainingResults',
                    'test_integration.TestEnsembleTrainerIntegration',
                    'test_integration.TestSystemIntegration',
                    'test_integration.TestPerformanceIntegration'
                ]
            }
        }
        
        # Performance benchmarks
        self.performance_benchmarks = {
            'agent_action_selection': 0.5,  # seconds for 100 actions
            'ensemble_action_selection': 1.0,  # seconds for 50 actions
            'agent_update': 2.0,  # seconds for 10 updates
            'ensemble_training': 5.0,  # seconds for short training sequence
        }
    
    def run_all_tests(self) -> Dict[str, any]:
        """
        Run all test suites and return comprehensive results.
        
        Returns:
            Dictionary containing test results and statistics
        """
        print("=" * 80)
        print("FinRL Contest 2024 - Comprehensive Test Suite")
        print("=" * 80)
        print(f"Python version: {sys.version}")
        print(f"Test configuration: {TEST_CONFIG}")
        print()
        
        start_time = time.time()
        overall_success = True
        
        # Run each test suite
        for suite_name, suite_info in self.test_suites.items():
            print(f"\n🔍 Running {suite_name.upper()} tests: {suite_info['description']}")
            print("-" * 60)
            
            suite_result = self._run_test_suite(suite_name, suite_info['modules'])
            self.results[suite_name] = suite_result
            
            if not suite_result['success']:
                overall_success = False
        
        self.total_time = time.time() - start_time
        
        # Generate summary report
        self._print_summary_report(overall_success)
        
        return {
            'overall_success': overall_success,
            'total_time': self.total_time,
            'suite_results': self.results,
            'summary': self._generate_summary_stats()
        }
    
    def run_suite(self, suite_name: str) -> Dict[str, any]:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of test suite to run
            
        Returns:
            Test results for the suite
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite_info = self.test_suites[suite_name]
        print(f"Running {suite_name} tests: {suite_info['description']}")
        
        start_time = time.time()
        result = self._run_test_suite(suite_name, suite_info['modules'])
        result['elapsed_time'] = time.time() - start_time
        
        return result
    
    def run_performance_tests(self) -> Dict[str, any]:
        """
        Run performance-focused tests with benchmarking.
        
        Returns:
            Performance test results
        """
        print("\n🚀 Running Performance Tests")
        print("-" * 40)
        
        performance_modules = [
            'test_agents.TestAgentPerformance',
            'test_ensemble.TestEnsemblePerformance',
            'test_integration.TestPerformanceIntegration'
        ]
        
        start_time = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for module_name in performance_modules:
            try:
                # Parse module and class name
                parts = module_name.split('.')
                if len(parts) == 2:
                    module_path, class_name = parts
                    module = sys.modules.get(f'__{module_path}__') or sys.modules[__name__]
                    if hasattr(module, class_name):
                        test_class = getattr(module, class_name)
                        suite.addTests(loader.loadTestsFromTestCase(test_class))
            except Exception as e:
                print(f"Warning: Could not load performance test {module_name}: {e}")
        
        # Run performance tests
        runner = unittest.TextTestRunner(
            verbosity=self.verbosity,
            stream=sys.stdout,
            buffer=True
        )
        
        result = runner.run(suite)
        elapsed_time = time.time() - start_time
        
        performance_result = {
            'success': result.wasSuccessful(),
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'elapsed_time': elapsed_time,
            'failure_details': result.failures,
            'error_details': result.errors
        }
        
        # Check against benchmarks
        if elapsed_time > 30.0:  # 30 seconds for all performance tests
            print(f"⚠️  Performance tests took {elapsed_time:.2f}s (expected < 30s)")
        else:
            print(f"✅ Performance tests completed in {elapsed_time:.2f}s")
        
        return performance_result
    
    def _run_test_suite(self, suite_name: str, module_names: List[str]) -> Dict[str, any]:
        """Run a specific test suite and return results."""
        start_time = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Load tests from modules
        loaded_modules = 0
        for module_name in module_names:
            try:
                # Parse module and class name
                parts = module_name.split('.')
                if len(parts) == 2:
                    module_path, class_name = parts
                    # Get the test class from current module namespace
                    if module_path == 'test_agents':
                        import test_agents
                        test_class = getattr(test_agents, class_name)
                    elif module_path == 'test_ensemble':
                        import test_ensemble
                        test_class = getattr(test_ensemble, class_name)
                    elif module_path == 'test_integration':
                        import test_integration
                        test_class = getattr(test_integration, class_name)
                    else:
                        continue
                    
                    suite.addTests(loader.loadTestsFromTestCase(test_class))
                    loaded_modules += 1
                    
            except Exception as e:
                print(f"Warning: Could not load test module {module_name}: {e}")
        
        print(f"Loaded {loaded_modules}/{len(module_names)} test modules")
        
        # Run tests with custom result handler
        runner = unittest.TextTestRunner(
            verbosity=self.verbosity,
            stream=sys.stdout,
            buffer=True
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings during testing
            result = runner.run(suite)
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        suite_result = {
            'success': result.wasSuccessful(),
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'elapsed_time': elapsed_time,
            'loaded_modules': loaded_modules,
            'total_modules': len(module_names)
        }
        
        # Print suite summary
        status = "✅ PASSED" if result.wasSuccessful() else "❌ FAILED"
        print(f"\n{status} - {suite_name}: {result.testsRun} tests, "
              f"{len(result.failures)} failures, {len(result.errors)} errors "
              f"({elapsed_time:.2f}s)")
        
        if result.failures:
            print(f"Failures in {suite_name}:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print(f"Errors in {suite_name}:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
        
        return suite_result
    
    def _print_summary_report(self, overall_success: bool):
        """Print comprehensive summary report."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY REPORT")
        print("=" * 80)
        
        # Overall status
        overall_status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
        print(f"\n{overall_status}")
        print(f"Total execution time: {self.total_time:.2f} seconds")
        
        # Suite breakdown
        print(f"\nTest Suite Breakdown:")
        print("-" * 40)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        
        for suite_name, result in self.results.items():
            status = "PASS" if result['success'] else "FAIL"
            print(f"{suite_name:12} | {status:4} | "
                  f"{result['tests_run']:3} tests | "
                  f"{result['failures']:2} failures | "
                  f"{result['errors']:2} errors | "
                  f"{result['elapsed_time']:5.2f}s")
            
            total_tests += result['tests_run']
            total_failures += result['failures']
            total_errors += result['errors']
            total_skipped += result['skipped']
        
        print("-" * 40)
        print(f"{'TOTAL':12} |      | "
              f"{total_tests:3} tests | "
              f"{total_failures:2} failures | "
              f"{total_errors:2} errors")
        
        # Success rate
        if total_tests > 0:
            success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
            print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        # Recommendations
        print(f"\nRecommendations:")
        if overall_success:
            print("🎉 All tests passed! The refactored framework is working correctly.")
            print("✅ Consider running performance benchmarks for optimization.")
        else:
            print("🔧 Some tests failed. Please review and fix the issues above.")
            print("🔍 Focus on the failing test suites first.")
            print("📝 Consider adding more specific unit tests for edge cases.")
        
        # Performance insights
        avg_time_per_test = self.total_time / max(total_tests, 1)
        if avg_time_per_test > 1.0:
            print(f"⚠️  Average test time is {avg_time_per_test:.2f}s per test (consider optimization)")
        
        print("=" * 80)
    
    def _generate_summary_stats(self) -> Dict[str, any]:
        """Generate summary statistics."""
        total_tests = sum(r['tests_run'] for r in self.results.values())
        total_failures = sum(r['failures'] for r in self.results.values())
        total_errors = sum(r['errors'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())
        
        return {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'success_rate': ((total_tests - total_failures - total_errors) / max(total_tests, 1)) * 100,
            'avg_time_per_test': self.total_time / max(total_tests, 1),
            'suites_passed': sum(1 for r in self.results.values() if r['success']),
            'suites_total': len(self.results)
        }


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description='FinRL Contest 2024 Test Runner')
    parser.add_argument('--suite', choices=['agents', 'ensemble', 'integration', 'all'], 
                       default='all', help='Test suite to run')
    parser.add_argument('--performance', action='store_true', 
                       help='Run performance tests only')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity level')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (minimal output)')
    
    args = parser.parse_args()
    
    # Set verbosity
    verbosity = 0 if args.quiet else min(args.verbose, 2)
    
    # Create test runner
    runner = TestRunner(verbosity=verbosity)
    
    try:
        if args.performance:
            # Run performance tests only
            results = runner.run_performance_tests()
            success = results['success']
        elif args.suite == 'all':
            # Run all test suites
            results = runner.run_all_tests()
            success = results['overall_success']
        else:
            # Run specific suite
            results = runner.run_suite(args.suite)
            success = results['success']
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()