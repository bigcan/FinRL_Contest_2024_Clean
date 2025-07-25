"""
Quick test script to validate HPO integration
Tests core functionality without running full optimization
"""

import os
import sys
import tempfile
import shutil
from typing import Dict, Any

def test_imports():
    """Test that all HPO modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import optuna
        print("✅ Optuna imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Optuna: {e}")
        return False
    
    try:
        # Test Task 1 imports
        sys.path.append('task1/src')
        from hpo_config import HPOConfig, Task1HPOSearchSpace, HPOResultsManager
        print("✅ Task 1 HPO modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Task 1 HPO modules: {e}")
        return False
    
    try:
        # Test shared utilities
        sys.path.append('shared')
        from hpo_utils import HPOAnalyzer
        print("✅ Shared HPO utilities imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import shared HPO utilities: {e}")
        return False
    
    return True


def test_hpo_config():
    """Test HPO configuration creation"""
    print("\n🔧 Testing HPO configuration...")
    
    try:
        from hpo_config import HPOConfig, create_sqlite_storage
        
        # Test basic configuration
        config = HPOConfig(
            study_name="test_study",
            n_trials=5,
            n_jobs=1,
            timeout=60
        )
        
        # Test study creation
        study = config.create_study()
        print(f"✅ Created study: {study.study_name}")
        
        # Test SQLite storage
        storage_url = create_sqlite_storage("test_hpo.db")
        print(f"✅ Created storage URL: {storage_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ HPO configuration test failed: {e}")
        return False


def test_search_spaces():
    """Test parameter search space definitions"""
    print("\n🎯 Testing search spaces...")
    
    try:
        import optuna
        from hpo_config import Task1HPOSearchSpace, Task2HPOSearchSpace
        
        # Test Task 1 search space
        study = optuna.create_study()
        trial = study.ask()
        
        task1_params = Task1HPOSearchSpace.suggest_parameters(trial)
        print(f"✅ Task 1 parameters suggested: {len(task1_params)} params")
        
        # Test parameter conversion
        task1_config = Task1HPOSearchSpace.convert_to_config(task1_params)
        print(f"✅ Task 1 config converted: net_dims = {task1_config.get('net_dims', 'N/A')}")
        
        # Test Task 2 search space  
        task2_params = Task2HPOSearchSpace.suggest_parameters(trial)
        print(f"✅ Task 2 parameters suggested: {len(task2_params)} params")
        
        return True
        
    except Exception as e:
        print(f"❌ Search space test failed: {e}")
        return False


def test_results_manager():
    """Test HPO results management"""
    print("\n📊 Testing results manager...")
    
    try:
        from hpo_config import HPOResultsManager
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            results_manager = HPOResultsManager(temp_dir)
            
            # Create mock study data
            import optuna
            study = optuna.create_study(direction='maximize')
            
            # Add some mock trials
            study.enqueue_trial({'param1': 1.0, 'param2': 0.5})
            study.enqueue_trial({'param1': 2.0, 'param2': 0.8})
            
            def objective(trial):
                p1 = trial.suggest_float('param1', 0, 3)
                p2 = trial.suggest_float('param2', 0, 1)
                return p1 * p2
            
            study.optimize(objective, n_trials=2)
            
            # Test saving results
            results_manager.save_study_results(study, "test_task")
            print("✅ Study results saved successfully")
            
            # Test loading results
            best_params = results_manager.load_best_parameters("test_task")
            print(f"✅ Best parameters loaded: {len(best_params)} params")
            
            # Test report generation
            report = results_manager.generate_optimization_report(study, "test_task")
            print("✅ Optimization report generated")
            
        return True
        
    except Exception as e:
        print(f"❌ Results manager test failed: {e}")
        return False


def test_analyzer():
    """Test HPO analyzer functionality"""
    print("\n📈 Testing HPO analyzer...")
    
    try:
        from hpo_utils import HPOAnalyzer
        
        # Create temporary directory with mock results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock results files
            import json
            
            mock_best_params = {
                'learning_rate': 0.001,
                'batch_size': 256,
                'net_dims_0': 128,
                'net_dims_1': 128,
                'net_dims_2': 128
            }
            
            mock_stats = {
                'best_value': 0.85,
                'best_trial_number': 15,
                'n_trials': 50,
                'study_name': 'test_study',
                'direction': 'MAXIMIZE'
            }
            
            mock_trials = [
                {
                    'number': i,
                    'value': 0.5 + i * 0.01,
                    'params': {'param1': i * 0.1, 'param2': i * 0.05},
                    'state': 'COMPLETE',
                    'datetime_start': '2023-11-20T10:00:00',
                    'datetime_complete': '2023-11-20T10:05:00',
                    'duration': 300
                }
                for i in range(10)
            ]
            
            # Save mock files
            with open(os.path.join(temp_dir, 'task1_best_params.json'), 'w') as f:
                json.dump(mock_best_params, f)
            
            with open(os.path.join(temp_dir, 'task1_study_stats.json'), 'w') as f:
                json.dump(mock_stats, f)
            
            with open(os.path.join(temp_dir, 'task1_trials_history.json'), 'w') as f:
                json.dump(mock_trials, f)
            
            # Test analyzer
            analyzer = HPOAnalyzer(temp_dir)
            print("✅ HPO analyzer initialized")
            
            # Test report generation
            report = analyzer.generate_comparison_report()
            print("✅ Comparison report generated")
            
            # Test configuration export
            export_dir = os.path.join(temp_dir, 'export')
            analyzer.export_best_configurations(export_dir)
            print("✅ Best configurations exported")
            
        return True
        
    except Exception as e:
        print(f"❌ HPO analyzer test failed: {e}")
        return False


def test_integration():
    """Test HPO integration with training scripts"""
    print("\n🔗 Testing training integration...")
    
    try:
        sys.path.append('task1/src')
        from task1_ensemble_hpo_integrated import HPOIntegratedTrainer
        
        # Test trainer initialization
        trainer = HPOIntegratedTrainer(hpo_results_dir=None)
        print("✅ HPO integrated trainer initialized")
        
        # Test configuration generation
        config = trainer.get_training_configuration()
        print(f"✅ Training configuration generated: {len(config)} params")
        
        # Test agent selection
        agents = trainer.get_ensemble_agents()
        print(f"✅ Ensemble agents selected: {[a.__name__ for a in agents]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting HPO Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("HPO Config Test", test_hpo_config),
        ("Search Spaces Test", test_search_spaces),
        ("Results Manager Test", test_results_manager),
        ("HPO Analyzer Test", test_analyzer),
        ("Integration Test", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HPO integration is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)