"""
Test Runner - Comprehensive testing framework for FinRL Contest 2024
"""

import unittest
import sys
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime
import argparse

class TestRunner:
    """Comprehensive test runner for all project components."""
    
    def __init__(self, project_root: str = None):
        """Initialize test runner."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = Path(project_root)
        self.test_results = {}
        
    def discover_tests(self, test_dir: str) -> list:
        """Discover all test files in a directory."""
        test_dir_path = self.project_root / "testing" / test_dir
        test_files = []
        
        if test_dir_path.exists():
            for test_file in test_dir_path.rglob("test_*.py"):
                test_files.append(test_file)
                
        return test_files
    
    def run_unit_tests(self, task: str = None) -> dict:
        """Run unit tests for specified task or all tasks."""
        print("🧪 Running Unit Tests...")
        results = {"passed": 0, "failed": 0, "errors": [], "duration": 0}
        
        start_time = datetime.now()
        
        if task:
            test_dirs = [f"unit_tests/{task}"]
        else:
            test_dirs = ["unit_tests/task1", "unit_tests/task2"]
            
        for test_dir in test_dirs:
            test_files = self.discover_tests(test_dir)
            
            for test_file in test_files:
                try:
                    # Run individual test file
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", str(test_file), "-v"
                    ], capture_output=True, text=True, cwd=self.project_root)
                    
                    if result.returncode == 0:
                        results["passed"] += 1
                        print(f"✅ {test_file.name}")
                    else:
                        results["failed"] += 1
                        results["errors"].append({
                            "file": str(test_file),
                            "error": result.stderr
                        })
                        print(f"❌ {test_file.name}")
                        
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": str(test_file),
                        "error": str(e)
                    })
                    print(f"❌ {test_file.name} - Exception: {e}")
        
        results["duration"] = (datetime.now() - start_time).total_seconds()
        return results
    
    def run_integration_tests(self, task: str = None) -> dict:
        """Run integration tests for specified task or all tasks."""
        print("🔗 Running Integration Tests...")
        results = {"passed": 0, "failed": 0, "errors": [], "duration": 0}
        
        start_time = datetime.now()
        
        if task:
            test_dirs = [f"integration_tests/{task}"]
        else:
            test_dirs = ["integration_tests/task1", "integration_tests/task2"]
            
        for test_dir in test_dirs:
            test_files = self.discover_tests(test_dir)
            
            for test_file in test_files:
                try:
                    # Run integration test
                    result = subprocess.run([
                        sys.executable, str(test_file)
                    ], capture_output=True, text=True, cwd=self.project_root)
                    
                    if result.returncode == 0:
                        results["passed"] += 1
                        print(f"✅ {test_file.name}")
                    else:
                        results["failed"] += 1
                        results["errors"].append({
                            "file": str(test_file),
                            "error": result.stderr
                        })
                        print(f"❌ {test_file.name}")
                        
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": str(test_file),
                        "error": str(e)
                    })
                    print(f"❌ {test_file.name} - Exception: {e}")
        
        results["duration"] = (datetime.now() - start_time).total_seconds()
        return results
    
    def run_performance_tests(self, task: str = None) -> dict:
        """Run performance benchmarks."""
        print("⚡ Running Performance Tests...")
        results = {"benchmarks": [], "duration": 0}
        
        start_time = datetime.now()
        
        if task:
            test_dirs = [f"performance_tests/{task}"]
        else:
            test_dirs = ["performance_tests/task1", "performance_tests/task2"]
            
        for test_dir in test_dirs:
            test_files = self.discover_tests(test_dir)
            
            for test_file in test_files:
                try:
                    # Run performance test
                    result = subprocess.run([
                        sys.executable, str(test_file)
                    ], capture_output=True, text=True, cwd=self.project_root)
                    
                    if result.returncode == 0:
                        # Parse performance results if available
                        try:
                            perf_data = json.loads(result.stdout)
                            results["benchmarks"].append({
                                "test": test_file.name,
                                "metrics": perf_data
                            })
                            print(f"📊 {test_file.name}")
                        except:
                            results["benchmarks"].append({
                                "test": test_file.name,
                                "status": "completed"
                            })
                            print(f"✅ {test_file.name}")
                    else:
                        print(f"❌ {test_file.name}")
                        
                except Exception as e:
                    print(f"❌ {test_file.name} - Exception: {e}")
        
        results["duration"] = (datetime.now() - start_time).total_seconds()
        return results
    
    def run_data_validation(self) -> dict:
        """Run data validation tests."""
        print("📊 Running Data Validation...")
        results = {"status": "success", "issues": [], "duration": 0}
        
        start_time = datetime.now()
        
        try:
            # Import and use data inspector
            sys.path.append(str(self.project_root / "debugging" / "debug_tools" / "data_inspectors"))
            from data_inspector import DataInspector
            
            inspector = DataInspector()
            
            # Check Task 1 data
            btc_csv = self.project_root / "data" / "raw" / "task1" / "BTC_1sec.csv"
            btc_npy = self.project_root / "data" / "raw" / "task1" / "BTC_1sec_predict.npy"
            
            if btc_csv.exists():
                btc_results = inspector.validate_btc_data(str(btc_csv), str(btc_npy) if btc_npy.exists() else None)
                if btc_results["issues"]:
                    results["issues"].extend(btc_results["issues"])
                print("✅ Task 1 data validation completed")
            else:
                results["issues"].append("Task 1 Bitcoin data not found")
                print("⚠️ Task 1 data not found")
            
            # Check Task 2 data
            news_train = self.project_root / "development" / "task2" / "src" / "task2_dsets" / "train" / "task2_news_train.csv"
            stocks_train = self.project_root / "development" / "task2" / "src" / "task2_dsets" / "train" / "task2_stocks_train.csv"
            
            if news_train.exists() and stocks_train.exists():
                news_results = inspector.validate_news_data(str(news_train), str(stocks_train))
                if news_results["issues"]:
                    results["issues"].extend(news_results["issues"])
                print("✅ Task 2 data validation completed")
            else:
                results["issues"].append("Task 2 data not found")
                print("⚠️ Task 2 data not found")
                
        except Exception as e:
            results["status"] = "error"
            results["issues"].append(f"Data validation error: {str(e)}")
            print(f"❌ Data validation failed: {e}")
        
        results["duration"] = (datetime.now() - start_time).total_seconds()
        return results
    
    def generate_test_report(self, output_path: str = None) -> str:
        """Generate comprehensive test report."""
        if output_path is None:
            output_path = self.project_root / "testing" / "test_reports" / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        html_content = [
            "<html><head><title>FinRL Contest 2024 - Test Report</title></head><body>",
            "<h1>FinRL Contest 2024 - Test Report</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "<h2>Test Summary</h2>"
        ]
        
        total_passed = 0
        total_failed = 0
        total_duration = 0
        
        for test_type, results in self.test_results.items():
            html_content.append(f"<h3>{test_type.replace('_', ' ').title()}</h3>")
            
            if "passed" in results and "failed" in results:
                passed = results["passed"]
                failed = results["failed"]
                duration = results.get("duration", 0)
                
                total_passed += passed
                total_failed += failed
                total_duration += duration
                
                html_content.append(f"<p>✅ Passed: {passed} | ❌ Failed: {failed} | ⏱️ Duration: {duration:.2f}s</p>")
                
                if results.get("errors"):
                    html_content.append("<h4>Errors:</h4><ul>")
                    for error in results["errors"]:
                        html_content.append(f"<li><strong>{error['file']}</strong>: {error['error']}</li>")
                    html_content.append("</ul>")
            
            elif "issues" in results:
                issues = len(results["issues"])
                duration = results.get("duration", 0)
                total_duration += duration
                
                html_content.append(f"<p>Issues: {issues} | ⏱️ Duration: {duration:.2f}s</p>")
                
                if results["issues"]:
                    html_content.append("<h4>Issues Found:</h4><ul>")
                    for issue in results["issues"]:
                        html_content.append(f"<li>{issue}</li>")
                    html_content.append("</ul>")
            
            elif "benchmarks" in results:
                benchmarks = len(results["benchmarks"])
                duration = results.get("duration", 0)
                total_duration += duration
                
                html_content.append(f"<p>Benchmarks: {benchmarks} | ⏱️ Duration: {duration:.2f}s</p>")
        
        # Overall summary
        html_content.insert(4, f"<p><strong>Overall: ✅ {total_passed} passed, ❌ {total_failed} failed, ⏱️ {total_duration:.2f}s total</strong></p>")
        
        html_content.append("</body></html>")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(html_content))
        
        print(f"📄 Test report saved to: {output_path}")
        return str(output_path)
    
    def run_all_tests(self, task: str = None, skip_performance: bool = False) -> dict:
        """Run all test suites."""
        print("🚀 Starting Comprehensive Test Suite...")
        print("=" * 50)
        
        # Run all test types
        self.test_results["unit_tests"] = self.run_unit_tests(task)
        self.test_results["integration_tests"] = self.run_integration_tests(task)
        self.test_results["data_validation"] = self.run_data_validation()
        
        if not skip_performance:
            self.test_results["performance_tests"] = self.run_performance_tests(task)
        
        print("=" * 50)
        print("📄 Generating test report...")
        
        report_path = self.generate_test_report()
        
        return {
            "results": self.test_results,
            "report_path": report_path
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="FinRL Contest 2024 Test Runner")
    parser.add_argument("--task", choices=["task1", "task2"], help="Run tests for specific task only")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--data-only", action="store_true", help="Run only data validation")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.unit_only:
        runner.test_results["unit_tests"] = runner.run_unit_tests(args.task)
    elif args.integration_only:
        runner.test_results["integration_tests"] = runner.run_integration_tests(args.task)
    elif args.performance_only:
        runner.test_results["performance_tests"] = runner.run_performance_tests(args.task)
    elif args.data_only:
        runner.test_results["data_validation"] = runner.run_data_validation()
    else:
        runner.run_all_tests(args.task, args.skip_performance)

if __name__ == "__main__":
    main()