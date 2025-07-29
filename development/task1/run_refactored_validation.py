#!/usr/bin/env python3
"""
Quick validation runner for the refactored framework that sets up proper Python paths.
"""

import sys
import os
from pathlib import Path

# Add the project root and src_refactored to Python path
current_dir = Path(__file__).parent
src_refactored_dir = current_dir / "src_refactored"
project_root = current_dir

# Ensure paths are in sys.path
for path in [str(project_root), str(src_refactored_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Change to src_refactored directory for proper module resolution
os.chdir(str(src_refactored_dir))

try:
    # Now run the validation by importing as a module
    import sys
    sys.path.insert(0, '.')
    
    # Import and run validation functions directly
    from tests.validate_framework import (
        validate_imports, validate_agent_creation, validate_ensemble_creation,
        validate_mock_environment, validate_training_config, run_integration_test
    )
    
    print("================================================================================")
    print("FinRL Contest 2024 - Framework Validation (Quick Fix)")
    print("================================================================================")
    
    results = []
    
    # Run each validation
    tests = [
        ("Imports", validate_imports),
        ("Agent Creation", validate_agent_creation), 
        ("Ensemble Creation", validate_ensemble_creation),
        ("Mock Environment", validate_mock_environment),
        ("Training Config", validate_training_config),
        ("Integration", run_integration_test)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            results.append(False)
            print(f"{test_name}: ❌ FAIL - {e}")
    
    # Summary
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Status: {'✅ PASS' if passed == total else '❌ PARTIAL' if passed > 0 else '❌ FAILED'}")
    print(f"Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if passed == total:
        print("🎉 All validations passed! Refactored framework is ready.")
        exit(0)
    elif passed > 0:
        print("⚠️  Some validations failed. Framework partially working.")
        exit(1)
    else:
        print("❌ All validations failed. Check setup and dependencies.")
        exit(2)

except ImportError as e:
    print(f"❌ Failed to import validation framework: {e}")
    print("The refactored framework still has import issues.")
    exit(3)
except Exception as e:
    print(f"❌ Unexpected error during validation: {e}")
    exit(4)