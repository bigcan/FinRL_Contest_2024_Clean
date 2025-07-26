#!/usr/bin/env python3
"""
Generate Enhanced Features V3 with Validation Framework
Comprehensive feature generation with proper train/validation/test splitting
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime

# Import our modules
from data_config import ConfigData
from enhanced_features_v3 import EnhancedFeatureEngineering
from validation_framework import ValidationFramework

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"enhanced_features_v3_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def validate_raw_data(csv_path: str) -> Dict[str, Any]:
    """
    Validate raw data quality before feature generation
    
    Args:
        csv_path: Path to raw CSV data
        
    Returns:
        Dictionary of validation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Validating raw data quality...")
    
    # Load data
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return {'valid': False, 'error': str(e)}
    
    validation_results = {
        'valid': True,
        'n_rows': len(data),
        'n_columns': len(data.columns),
        'columns': list(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'data_types': data.dtypes.to_dict(),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Check for required columns
    required_columns = ['midpoint', 'spread']
    missing_required = [col for col in required_columns if col not in data.columns]
    
    if missing_required:
        validation_results['valid'] = False
        validation_results['missing_required_columns'] = missing_required
        logger.error(f"Missing required columns: {missing_required}")
        return validation_results
    
    # Check data quality
    if validation_results['missing_values'] > 0:
        logger.warning(f"Found {validation_results['missing_values']} missing values")
    
    # Check for reasonable price data
    midpoint = data['midpoint'].values
    if np.any(midpoint <= 0):
        logger.warning("Found non-positive midpoint prices")
    
    price_changes = np.diff(midpoint) / midpoint[:-1]
    extreme_changes = np.abs(price_changes) > 0.1  # 10% threshold
    if np.any(extreme_changes):
        logger.warning(f"Found {np.sum(extreme_changes)} extreme price changes (>10%)")
    
    logger.info(f"Raw data validation completed: {data.shape} with {validation_results['missing_values']} missing values")
    
    return validation_results

def generate_and_validate_features():
    """Main function to generate enhanced features with validation"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting enhanced feature generation v3 with validation framework")
    
    try:
        # Load configuration
        args = ConfigData()
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"CSV path: {args.csv_path}")
        logger.info(f"Predict path: {args.predict_ary_path}")
        
        # Validate raw data
        validation_results = validate_raw_data(args.csv_path)
        if not validation_results['valid']:
            logger.error("Raw data validation failed. Stopping.")
            return False
        
        logger.info(f"Raw data validated: {validation_results['n_rows']} rows, "
                   f"{validation_results['n_columns']} columns")
        
        # Initialize feature engineering with validation
        feature_engine = EnhancedFeatureEngineering(
            use_cache=True,
            cache_dir=args.data_dir,
            validation_split=True
        )
        
        # Generate enhanced features v3
        logger.info("Generating enhanced features v3...")
        
        feature_array, feature_names, metadata = feature_engine.generate_enhanced_features_v3(
            csv_path=args.csv_path,
            predict_path=args.predict_ary_path,
            split_ratios=(0.7, 0.15, 0.15),
            save_path=args.predict_ary_path
        )
        
        logger.info(f"Enhanced features generated: {feature_array.shape}")
        logger.info(f"Total features: {len(feature_names)}")
        
        # Feature category breakdown
        categories = metadata['feature_categories']
        for category, features in categories.items():
            if features:
                logger.info(f"  {category}: {len(features)} features")
        
        # Setup validation framework
        logger.info("Setting up validation framework...")
        
        validation_framework = ValidationFramework(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            cv_folds=5,
            purge_gap=60
        )
        
        # Create timestamps for validation framework
        timestamps = pd.date_range(
            start='2020-01-01', 
            periods=len(feature_array), 
            freq='1min'
        )
        
        # Setup validation with feature data
        setup_info = validation_framework.setup_validation(
            data=feature_array,
            timestamps=timestamps.values,
            save_splits=True,
            save_path=os.path.join(args.data_dir, "validation_splits")
        )
        
        logger.info("Validation framework setup completed:")
        logger.info(f"  Train samples: {setup_info['train_samples']}")
        logger.info(f"  Validation samples: {setup_info['val_samples']}")
        logger.info(f"  Test samples: {setup_info['test_samples']}")
        logger.info(f"  CV folds: {setup_info['cv_folds']}")
        
        # Validate feature quality
        logger.info("Validating feature quality...")
        
        feature_validation = validation_framework.metrics_calculator.validate_features(
            feature_array=feature_array,
            feature_names=feature_names,
            verbose=True
        )
        
        # Check for issues
        issues = []
        if feature_validation['nan_count'] > 0:
            issues.append(f"NaN values: {feature_validation['nan_count']}")
        
        if feature_validation['inf_count'] > 0:
            issues.append(f"Infinite values: {feature_validation['inf_count']}")
        
        if len(feature_validation['constant_features']) > 0:
            issues.append(f"Constant features: {len(feature_validation['constant_features'])}")
        
        if len(feature_validation['high_correlation_pairs']) > 10:
            issues.append(f"High correlation pairs: {len(feature_validation['high_correlation_pairs'])}")
        
        if issues:
            logger.warning("Feature quality issues detected:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Feature quality validation passed")
        
        # Create comprehensive report
        logger.info("Creating comprehensive feature report...")
        
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'raw_data_validation': validation_results,
            'feature_metadata': metadata,
            'validation_setup': {
                'total_samples': setup_info['n_samples'],
                'train_samples': setup_info['train_samples'],
                'val_samples': setup_info['val_samples'],
                'test_samples': setup_info['test_samples'],
                'cv_folds': setup_info['cv_folds']
            },
            'feature_quality': feature_validation,
            'issues_detected': issues,
            'files_generated': [
                f"{args.predict_ary_path.replace('.npy', '')}_enhanced_v3.npy",
                f"{args.predict_ary_path.replace('.npy', '')}_enhanced_v3_metadata.npy"
            ]
        }
        
        # Save report
        report_path = os.path.join(args.data_dir, "enhanced_features_v3_report.npy")
        np.save(report_path, report)
        logger.info(f"Comprehensive report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("ENHANCED FEATURES V3 GENERATION SUMMARY")
        print("="*80)
        print(f"Status: SUCCESS")
        print(f"Total Features: {len(feature_names)}")
        print(f"Data Shape: {feature_array.shape}")
        print(f"Train/Val/Test Split: {setup_info['train_samples']}/{setup_info['val_samples']}/{setup_info['test_samples']}")
        print(f"Cross-Validation Folds: {setup_info['cv_folds']}")
        
        print(f"\nFeature Categories:")
        for category, features in categories.items():
            if features:
                print(f"  {category.capitalize()}: {len(features)}")
        
        if issues:
            print(f"\nIssues Detected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\nFeature Quality: EXCELLENT")
        
        print(f"\nFiles Generated:")
        for file_path in report['files_generated']:
            print(f"  - {file_path}")
        
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in feature generation: {str(e)}", exc_info=True)
        return False

def main():
    """Main entry point"""
    success = generate_and_validate_features()
    
    if success:
        print("\n✅ Enhanced features v3 generated successfully!")
        print("Ready for training with validation framework.")
        sys.exit(0)
    else:
        print("\n❌ Enhanced feature generation failed!")
        print("Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()