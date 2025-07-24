#!/usr/bin/env python3
"""
Test Enhanced Features Integration

Simple test script to verify that the enhanced features integrate properly
with the existing trade_simulator and training pipeline.

Usage:
    python test_enhanced_integration.py
"""

import sys
import os
import numpy as np

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared/features'))

def test_data_config():
    """Test data_config import and paths"""
    print("Testing data_config...")
    try:
        from data_config import ConfigData
        config = ConfigData()
        
        print(f"✓ ConfigData imported successfully")
        print(f"  CSV path: {config.csv_path}")
        print(f"  Predict path: {config.predict_ary_path}")
        
        # Check if files exist
        if os.path.exists(config.csv_path):
            print(f"✓ CSV file exists")
        else:
            print(f"⚠ CSV file not found: {config.csv_path}")
            
        if os.path.exists(config.predict_ary_path):
            print(f"✓ Original predict file exists")
        else:
            print(f"⚠ Original predict file not found: {config.predict_ary_path}")
            
        return config
        
    except Exception as e:
        print(f"✗ Error importing data_config: {e}")
        return None

def test_feature_processor():
    """Test feature processor import"""
    print("\nTesting feature_processor...")
    try:
        from feature_processor import FeatureProcessor
        processor = FeatureProcessor()
        
        print(f"✓ FeatureProcessor imported successfully")
        print(f"✓ Components initialized:")
        print(f"  - TechnicalIndicators: {type(processor.tech_indicators).__name__}")
        print(f"  - LOBFeatures: {type(processor.lob_features).__name__}")
        print(f"  - DataTransformer: {type(processor.data_transformer).__name__}")
        print(f"  - FeatureSelector: {type(processor.feature_selector).__name__}")
        
        return processor
        
    except Exception as e:
        print(f"✗ Error importing feature_processor: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trade_simulator_compatibility(config):
    """Test trade_simulator compatibility"""
    print("\nTesting trade_simulator compatibility...")
    try:
        from trade_simulator import TradeSimulator
        
        # Initialize with default parameters
        simulator = TradeSimulator(num_sims=2, gpu_id=-1)
        
        print(f"✓ TradeSimulator imported and initialized")
        print(f"  State dim: {simulator.state_dim}")
        print(f"  Action dim: {simulator.action_dim}")
        print(f"  Factor array shape: {simulator.factor_ary.shape}")
        print(f"  Price array shape: {simulator.price_ary.shape}")
        
        # Check if enhanced features are loaded
        enhanced_path = config.predict_ary_path.replace('.npy', '_enhanced_v2.npy')
        if os.path.exists(enhanced_path):
            print(f"✓ Enhanced features v2 detected: {enhanced_path}")
            
            # Test loading enhanced features
            enhanced_data = np.load(enhanced_path)
            print(f"  Enhanced features shape: {enhanced_data.shape}")
            
            # Check metadata
            metadata_path = enhanced_path.replace('.npy', '_metadata.npy')
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True).item()
                print(f"  Feature names: {len(metadata['feature_names'])}")
                print(f"  First few features: {metadata['feature_names'][:5]}")
        else:
            print(f"⚠ Enhanced features v2 not found: {enhanced_path}")
        
        return simulator
        
    except Exception as e:
        print(f"✗ Error testing trade_simulator: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_enhanced_features_generation():
    """Test if enhanced features can be generated"""
    print("\nTesting enhanced features generation...")
    try:
        from feature_processor import FeatureProcessor
        from data_config import ConfigData
        
        config = ConfigData()
        processor = FeatureProcessor(use_cache=True)
        
        print("Testing raw feature computation (small sample)...")
        
        # Load a small sample of data for testing
        import pandas as pd
        lob_data = pd.read_csv(config.csv_path, nrows=1000)  # Small sample
        print(f"Loaded sample data: {len(lob_data)} rows")
        
        # Test LOB features
        lob_features = processor.lob_features.compute_lob_features(lob_data)
        print(f"✓ LOB features computed: {len(lob_features)} features")
        
        # Test data transformation features  
        transform_features = processor.data_transformer.transform_data(lob_data, apply_normalization=False)
        print(f"✓ Transform features computed: {len(transform_features)} features")
        
        # Test technical indicators
        price_data = lob_data[['bids_distance_3', 'asks_distance_3', 'midpoint']].values
        tech_features = processor.tech_indicators.compute_indicators(price_data)
        print(f"✓ Technical features computed: {len(tech_features)} features")
        
        total_features = len(lob_features) + len(transform_features) + len(tech_features)
        print(f"✓ Total features computed: {total_features}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing feature generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("ENHANCED FEATURES INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Data config
    config = test_data_config()
    if config is None:
        print("\n✗ Critical error: Could not load data config")
        return 1
    
    # Test 2: Feature processor
    processor = test_feature_processor()
    if processor is None:
        print("\n✗ Critical error: Could not load feature processor")
        return 1
    
    # Test 3: Trade simulator compatibility
    simulator = test_trade_simulator_compatibility(config)
    if simulator is None:
        print("\n✗ Critical error: Trade simulator compatibility failed")
        return 1
    
    # Test 4: Feature generation
    generation_success = test_enhanced_features_generation()
    if not generation_success:
        print("\n⚠ Warning: Feature generation test failed")
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    if config is not None:
        tests_passed += 1
        print("✓ Data config test: PASSED")
    else:
        print("✗ Data config test: FAILED")
        
    if processor is not None:
        tests_passed += 1
        print("✓ Feature processor test: PASSED")
    else:
        print("✗ Feature processor test: FAILED")
        
    if simulator is not None:
        tests_passed += 1
        print("✓ Trade simulator test: PASSED")
    else:
        print("✗ Trade simulator test: FAILED")
        
    if generation_success:
        tests_passed += 1
        print("✓ Feature generation test: PASSED")
    else:
        print("✗ Feature generation test: FAILED")
    
    print(f"\nTests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 All integration tests PASSED!")
        print("Enhanced features are ready for use with the training pipeline.")
        return 0
    else:
        print("\n⚠ Some integration tests FAILED!")
        print("Please review the errors above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())