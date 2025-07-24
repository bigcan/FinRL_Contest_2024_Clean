#!/usr/bin/env python3
"""
Enhanced Feature Generation Script v2.0

Production-ready script for generating enhanced features using the complete
feature engineering pipeline including:
- Market microstructure features (LOB)
- Data transformation features (log returns, rolling stats, time features)
- Technical indicators with enhanced volatility and regime detection
- Comprehensive feature validation and selection

Usage:
    python generate_enhanced_features_v2.py [--gpu-id GPU_ID] [--max-features MAX_FEATURES]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add the shared features directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared/features'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from feature_processor import FeatureProcessor
from data_config import ConfigData

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate enhanced features for FinRL Contest 2024 Task 1')
    parser.add_argument('--gpu-id', type=int, default=-1, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--max-features', type=int, default=30, help='Maximum number of features to select')
    parser.add_argument('--force-recompute', action='store_true', help='Force recomputation even if cache exists')
    parser.add_argument('--validate-only', action='store_true', help='Only run feature validation on existing features')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for features')
    return parser.parse_args()

def setup_paths(args):
    """Setup input and output paths"""
    config = ConfigData()
    
    paths = {
        'csv_path': config.csv_path,
        'predict_path': config.predict_ary_path,
        'cache_dir': os.path.join(os.path.dirname(config.csv_path), 'processed', 'task1'),
        'output_dir': args.output_dir or os.path.dirname(config.predict_ary_path)
    }
    
    # Create directories if they don't exist
    os.makedirs(paths['cache_dir'], exist_ok=True)
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    return paths, config

def main():
    """Main feature generation pipeline"""
    args = parse_arguments()
    paths, config = setup_paths(args)
    
    print("=" * 80)
    print("ENHANCED FEATURE GENERATION v2.0")
    print("=" * 80)
    print(f"GPU ID: {args.gpu_id}")
    print(f"Max features: {args.max_features}")
    print(f"Force recompute: {args.force_recompute}")
    print(f"CSV path: {paths['csv_path']}")
    print(f"Predict path: {paths['predict_path']}")
    print(f"Cache directory: {paths['cache_dir']}")
    print(f"Output directory: {paths['output_dir']}")
    print("=" * 80)
    
    # Verify input files exist
    if not os.path.exists(paths['csv_path']):
        print(f"ERROR: CSV file not found: {paths['csv_path']}")
        return 1
    
    if not os.path.exists(paths['predict_path']):
        print(f"WARNING: Predict file not found: {paths['predict_path']}")
        print("Will generate features without original predict array.")
    
    # Initialize feature processor
    print("\\nInitializing feature processor...")
    processor = FeatureProcessor(cache_dir=paths['cache_dir'], use_cache=not args.force_recompute)
    
    # Enhanced feature output path
    enhanced_features_path = os.path.join(paths['output_dir'], 'BTC_1sec_predict_enhanced_v2.npy')
    
    # Validation only mode
    if args.validate_only:
        if not os.path.exists(enhanced_features_path):
            print(f"ERROR: Enhanced features not found for validation: {enhanced_features_path}")
            return 1
            
        print("\\nLoading existing enhanced features for validation...")
        feature_array, feature_names = processor.load_processed_features(enhanced_features_path)
        
        print("\\nRunning feature validation...")
        validation_results = processor.validate_features(feature_array, feature_names, verbose=True)
        
        # Save validation report
        report_path = enhanced_features_path.replace('.npy', '_validation_report.txt')
        with open(report_path, 'w') as f:
            f.write("ENHANCED FEATURES VALIDATION REPORT\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"Total features: {validation_results['total_features']}\\n")
            f.write(f"Array shape: {validation_results['array_shape']}\\n")
            f.write(f"NaN values: {validation_results['nan_count']}\\n")
            f.write(f"Inf values: {validation_results['inf_count']}\\n")
            f.write(f"Constant features: {len(validation_results['constant_features'])}\\n")
            f.write(f"High correlation pairs: {len(validation_results['high_correlation_pairs'])}\\n")
            
        print(f"\\nValidation report saved to: {report_path}")
        return 0
    
    # Phase 1: Compute all raw features
    print("\\n" + "=" * 60)
    print("PHASE 1: COMPUTING RAW FEATURES")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        feature_array, feature_names = processor.compute_all_features(
            csv_path=paths['csv_path'],
            predict_path=paths['predict_path'],
            force_recompute=args.force_recompute
        )
        
        compute_time = time.time() - start_time
        print(f"\\nRaw feature computation completed in {compute_time:.2f} seconds")
        print(f"Generated {feature_array.shape[1]} raw features from {feature_array.shape[0]} data points")
        
    except Exception as e:
        print(f"ERROR during raw feature computation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Phase 2: Feature validation
    print("\\n" + "=" * 60)
    print("PHASE 2: RAW FEATURE VALIDATION")
    print("=" * 60)
    
    validation_results = processor.validate_features(feature_array, feature_names, verbose=True)
    
    # Phase 3: Feature selection and processing
    print("\\n" + "=" * 60)
    print("PHASE 3: FEATURE SELECTION AND PROCESSING")
    print("=" * 60)
    
    try:
        # Create target returns for feature selection
        print("Creating target returns for feature selection...")
        
        # Load midpoint prices for target creation
        lob_data = pd.read_csv(paths['csv_path'])
        midpoint_prices = lob_data['midpoint'].values
        
        # Align with feature array length
        min_length = min(len(midpoint_prices), len(feature_array))
        midpoint_aligned = midpoint_prices[-min_length:]
        feature_array_aligned = feature_array[-min_length:]
        
        # Create forward-looking returns as target (1-step ahead)
        target_returns = np.diff(midpoint_aligned) / midpoint_aligned[:-1]
        feature_array_for_selection = feature_array_aligned[:-1]  # Align lengths
        
        print(f"Target returns shape: {target_returns.shape}")
        print(f"Feature array for selection shape: {feature_array_for_selection.shape}")
        
        # Feature selection and processing
        processed_array, selected_names = processor.select_and_process_features(
            feature_array=feature_array_for_selection,
            target_returns=target_returns,
            max_features=args.max_features,
            save_path=enhanced_features_path
        )
        
        selection_time = time.time() - start_time - compute_time
        print(f"\\nFeature selection completed in {selection_time:.2f} seconds")
        print(f"Selected {len(selected_names)} features from {feature_array.shape[1]} candidates")
        
    except Exception as e:
        print(f"ERROR during feature selection: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Phase 4: Final validation
    print("\\n" + "=" * 60)
    print("PHASE 4: FINAL FEATURE VALIDATION")
    print("=" * 60)
    
    final_validation = processor.validate_features(processed_array, selected_names, verbose=True)
    
    # Phase 5: Feature importance analysis
    print("\\n" + "=" * 60)
    print("PHASE 5: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    importance_report = processor.get_feature_importance_report()
    print(importance_report)
    
    # Save comprehensive report
    report_path = enhanced_features_path.replace('.npy', '_generation_report.txt')
    total_time = time.time() - start_time
    
    with open(report_path, 'w') as f:
        f.write("ENHANCED FEATURES GENERATION REPORT v2.0\\n")
        f.write("=" * 60 + "\\n\\n")
        
        f.write("CONFIGURATION:\\n")
        f.write(f"GPU ID: {args.gpu_id}\\n")
        f.write(f"Max features: {args.max_features}\\n")
        f.write(f"Force recompute: {args.force_recompute}\\n")
        f.write(f"Total time: {total_time:.2f} seconds\\n\\n")
        
        f.write("INPUT FILES:\\n")
        f.write(f"CSV path: {paths['csv_path']}\\n")
        f.write(f"Predict path: {paths['predict_path']}\\n\\n")
        
        f.write("OUTPUT FILES:\\n")
        f.write(f"Enhanced features: {enhanced_features_path}\\n")
        f.write(f"Metadata: {enhanced_features_path.replace('.npy', '_metadata.npy')}\\n\\n")
        
        f.write("FEATURE STATISTICS:\\n")
        f.write(f"Raw features computed: {feature_array.shape[1]}\\n")
        f.write(f"Final features selected: {len(selected_names)}\\n")
        f.write(f"Data points: {processed_array.shape[0]}\\n\\n")
        
        f.write("FEATURE CATEGORIES:\\n")
        metadata = processor.get_feature_metadata()
        if metadata and 'feature_categories' in metadata:
            for category, features in metadata['feature_categories'].items():
                f.write(f"{category.upper()}: {len(features)} features\\n")
                for feature in features[:3]:
                    f.write(f"  - {feature}\\n")
                if len(features) > 3:
                    f.write(f"  ... and {len(features) - 3} more\\n")
                f.write("\\n")
        
        f.write("VALIDATION RESULTS:\\n")
        f.write(f"NaN values: {final_validation['nan_count']}\\n")
        f.write(f"Inf values: {final_validation['inf_count']}\\n")
        f.write(f"Constant features: {len(final_validation['constant_features'])}\\n")
        f.write(f"High correlation pairs: {len(final_validation['high_correlation_pairs'])}\\n\\n")
        
        f.write("FEATURE IMPORTANCE (Top 10):\\n")
        if hasattr(processor.feature_selector, 'feature_importance') and processor.feature_selector.feature_importance is not None:
            rankings = processor.feature_selector.get_feature_rankings()
            for i, (name, importance) in enumerate(rankings[:10], 1):
                f.write(f"{i:2d}. {name:<30} {importance:.4f}\\n")
        
        f.write("\\n" + "=" * 60 + "\\n")
        f.write("Generation completed successfully!\\n")
    
    print(f"\\nGeneration report saved to: {report_path}")
    
    # Final summary
    print("\\n" + "=" * 80)
    print("FEATURE GENERATION SUMMARY")
    print("=" * 80)
    print(f"✓ Raw features computed: {feature_array.shape[1]}")
    print(f"✓ Final features selected: {len(selected_names)}")
    print(f"✓ Data points processed: {processed_array.shape[0]}")
    print(f"✓ Total processing time: {total_time:.2f} seconds")
    print(f"✓ Enhanced features saved to: {enhanced_features_path}")
    print(f"✓ Metadata saved to: {enhanced_features_path.replace('.npy', '_metadata.npy')}")
    print(f"✓ Report saved to: {report_path}")
    
    # Check integration compatibility
    print("\\n" + "=" * 60)
    print("INTEGRATION COMPATIBILITY CHECK")
    print("=" * 60)
    
    try:
        # Verify the enhanced features can be loaded by trade_simulator
        print("Testing compatibility with trade_simulator...")
        
        # Check that the array dimensions match expectations
        expected_features = args.max_features
        actual_features = processed_array.shape[1]
        
        if actual_features == expected_features:
            print(f"✓ Feature count matches expectation: {actual_features}")
        else:
            print(f"⚠ Feature count mismatch: expected {expected_features}, got {actual_features}")
        
        # Check for position features (first two columns should be position/holding)
        if selected_names[0] == 'position_norm' and selected_names[1] == 'holding_norm':
            print("✓ Position features correctly positioned at indices 0, 1")
        else:
            print("⚠ Position features not found at expected positions")
            
        # Verify no NaN or Inf values in final array
        if final_validation['nan_count'] == 0 and final_validation['inf_count'] == 0:
            print("✓ No NaN or Inf values in final features")
        else:
            print(f"⚠ Found {final_validation['nan_count']} NaN and {final_validation['inf_count']} Inf values")
        
        print("Integration compatibility check completed.")
        
    except Exception as e:
        print(f"⚠ Compatibility check failed: {str(e)}")
    
    print("\\n" + "=" * 80)
    print("ENHANCED FEATURE GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())