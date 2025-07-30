"""
Test script to verify HPO GPU setup and functionality
Runs a quick test with minimal trials
"""

import torch as th
import numpy as np
from pathlib import Path
import sys
import json
sys.path.append('.')

from hpo_optimization import HPOObjective, run_hpo_study

def test_gpu_availability():
    """Test GPU availability and configuration"""
    print("="*60)
    print("Testing GPU Configuration")
    print("="*60)
    
    if th.cuda.is_available():
        print(f"✓ GPU Available: {th.cuda.get_device_name(0)}")
        print(f"  Device Count: {th.cuda.device_count()}")
        print(f"  Current Device: {th.cuda.current_device()}")
        print(f"  Memory Allocated: {th.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Memory Reserved: {th.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"  Total Memory: {th.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test tensor creation
        try:
            test_tensor = th.randn(1000, 1000).cuda()
            print("✓ Can create GPU tensors")
            del test_tensor
            th.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU tensor creation failed: {e}")
            return False
            
        return True
    else:
        print("✗ No GPU available")
        return False

def test_hpo_objective():
    """Test HPO objective function"""
    print("\n" + "="*60)
    print("Testing HPO Objective")
    print("="*60)
    
    # Create dummy data
    data_dir = Path(__file__).parent.parent / "task1_data"
    
    # Create synthetic data if real data doesn't exist
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        dummy_data = np.random.randn(10000, 15).astype(np.float32)
        np.save(data_dir / "BTC_1sec_predict_reduced.npy", dummy_data)
        print("Created synthetic data for testing")
    
    try:
        # Initialize objective
        objective = HPOObjective(
            data_path=data_dir / "BTC_1sec_predict.npy",
            num_episodes=2,
            samples_per_episode=1000,
            device="cuda" if th.cuda.is_available() else "cpu",
            use_regime_detection=True
        )
        print("✓ HPO Objective initialized successfully")
        
        # Test parameter suggestion
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        
        params = objective._suggest_parameters(trial)
        print(f"✓ Parameter suggestion working: {len(params)} parameter groups")
        
        # Show sample parameters
        print("\nSample parameters:")
        for key, value in params.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} parameters")
            else:
                print(f"  {key}: {value}")
                
        return True
        
    except Exception as e:
        print(f"✗ HPO Objective test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_hpo_study():
    """Run a minimal HPO study"""
    print("\n" + "="*60)
    print("Testing Mini HPO Study")
    print("="*60)
    
    if not th.cuda.is_available():
        print("⚠ Skipping HPO study test - GPU not available")
        return False
    
    try:
        # Run mini study
        data_path = Path(__file__).parent.parent / "task1_data" / "BTC_1sec_predict.npy"
        
        study = run_hpo_study(
            data_path=data_path,
            n_trials=2,  # Just 2 trials for testing
            n_jobs=1,
            study_name="test_hpo_gpu",
            use_gpu=True
        )
        
        print("✓ Mini HPO study completed successfully")
        print(f"  Completed trials: {len(study.trials)}")
        print(f"  Best value: {-study.best_value:.3f}")
        
        # Clean up test database
        import os
        if os.path.exists("test_hpo_gpu.db"):
            os.remove("test_hpo_gpu.db")
            
        return True
        
    except Exception as e:
        print(f"✗ Mini HPO study failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_management():
    """Test GPU memory management"""
    print("\n" + "="*60)
    print("Testing GPU Memory Management")
    print("="*60)
    
    if not th.cuda.is_available():
        print("⚠ Skipping memory test - GPU not available")
        return True
    
    try:
        # Initial memory
        initial_memory = th.cuda.memory_allocated(0)
        print(f"Initial memory: {initial_memory / 1e6:.2f} MB")
        
        # Allocate some tensors
        tensors = []
        for i in range(5):
            t = th.randn(1000, 1000).cuda()
            tensors.append(t)
            print(f"After allocation {i+1}: {th.cuda.memory_allocated(0) / 1e6:.2f} MB")
        
        # Clear tensors
        del tensors
        th.cuda.empty_cache()
        
        final_memory = th.cuda.memory_allocated(0)
        print(f"After cleanup: {final_memory / 1e6:.2f} MB")
        
        if final_memory <= initial_memory + 1e6:  # Allow 1MB tolerance
            print("✓ Memory management working correctly")
            return True
        else:
            print("⚠ Memory not fully released")
            return False
            
    except Exception as e:
        print(f"✗ Memory management test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("HPO GPU Test Suite")
    print("=" * 60)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("HPO Objective", test_hpo_objective),
        ("Memory Management", test_memory_management),
        ("Mini HPO Study", test_mini_hpo_study)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Ready for HPO optimization.")
        print("\nTo run full HPO:")
        print("  python src/hpo_optimization.py")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()