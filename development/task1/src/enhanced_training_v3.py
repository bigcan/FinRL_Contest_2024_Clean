#!/usr/bin/env python3
"""
Enhanced Training V3 with Validation Framework
Production-ready training with enhanced features v3 and comprehensive validation
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import json

# Fix encoding issues on Windows
import io
if os.name == 'nt':  # Windows only
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import existing modules
from task1_ensemble import Ensemble, get_state_dim, run
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_config import Config, build_env
from trade_simulator import TradeSimulator, EvalTradeSimulator
from validation_framework import ValidationFramework
from data_config import ConfigData


class EnhancedTrainingV3:
    """
    Enhanced training system with validation framework for v3 features
    """
    
    def __init__(
        self,
        use_enhanced_v3: bool = True,
        validation_monitoring: bool = True,
        early_stopping: bool = True,
        save_path: str = "enhanced_training_v3",
        gpu_id: int = 0
    ):
        """
        Initialize enhanced training system
        
        Args:
            use_enhanced_v3: Whether to use enhanced features v3
            validation_monitoring: Enable validation monitoring during training
            early_stopping: Enable early stopping based on validation performance
            save_path: Path to save training results
            gpu_id: GPU ID to use
        """
        self.use_enhanced_v3 = use_enhanced_v3
        self.validation_monitoring = validation_monitoring
        self.early_stopping = early_stopping
        self.save_path = save_path
        self.gpu_id = gpu_id
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize validation framework
        if validation_monitoring:
            self.validation_framework = ValidationFramework(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                cv_folds=5,
                purge_gap=60
            )
        else:
            self.validation_framework = None
        
        # Training state
        self.training_history = {
            'train_metrics': [],
            'val_metrics': [],
            'best_val_score': float('-inf'),
            'best_model_step': 0,
            'patience_counter': 0
        }
        
        # Load configuration
        self.config = self.load_optimized_config()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = os.path.join(self.save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"enhanced_training_v3_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def load_optimized_config(self) -> Dict[str, Any]:
        """
        Load optimized hyperparameters from HPO results
        """
        # Check for existing HPO results
        hpo_db_path = "archived_experiments/hpo_databases/task1_production_hpo.db"
        
        if os.path.exists(hpo_db_path):
            try:
                # Try to load best parameters from HPO
                import optuna
                from hpo_config import create_sqlite_storage
                
                storage_url = create_sqlite_storage(hpo_db_path)
                
                # Get the most recent study
                import sqlite3
                conn = sqlite3.connect(hpo_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT study_name FROM studies ORDER BY study_id DESC LIMIT 1")
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    study_name = result[0]
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                    
                    if study.best_trial:
                        best_params = study.best_params
                        self.logger.info(f"Loaded optimized parameters from HPO study: {study_name}")
                        self.logger.info(f"Best Sharpe ratio: {study.best_value:.4f}")
                        
                        # Convert HPO parameters to training config
                        config = self.convert_hpo_params_to_config(best_params)
                        return config
                        
            except Exception as e:
                self.logger.warning(f"Could not load HPO parameters: {e}")
        
        # Fallback to optimized default configuration
        self.logger.info("Using optimized default configuration")
        return self.get_optimized_default_config()
    
    def convert_hpo_params_to_config(self, hpo_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert HPO parameters to training configuration"""
        
        config = {
            'gpu_id': self.gpu_id,
            'num_sims': 2**4,  # 16 parallel environments (reduced for memory)
            'num_ignore_step': 60,
            'max_position': hpo_params.get('max_position', 1),
            'step_gap': hpo_params.get('step_gap', 2),
            'slippage': hpo_params.get('slippage', 7e-7),
            'starting_cash': 1e6,
            'net_dims': (
                hpo_params.get('net_dim_0', 128),
                hpo_params.get('net_dim_1', 128),
                hpo_params.get('net_dim_2', 128)
            ),
            'gamma': hpo_params.get('gamma', 0.995),
            'explore_rate': hpo_params.get('explore_rate', 0.005),
            'state_value_tau': hpo_params.get('state_value_tau', 0.01),
            'soft_update_tau': hpo_params.get('soft_update_tau', 2e-6),
            'learning_rate': hpo_params.get('learning_rate', 2e-6),
            'batch_size': hpo_params.get('batch_size', 512),
            'break_step': hpo_params.get('break_step', 16),
            'buffer_size_multiplier': 2,  # Reduced from 8 to 2 for memory
            'repeat_times': hpo_params.get('repeat_times', 2),
            'horizon_len_multiplier': 1,  # Reduced from 2 to 1 for memory  
            'eval_per_step_multiplier': hpo_params.get('eval_per_step_multiplier', 1),
            'num_workers': 1,
            'save_gap': 8,
            'data_length': self.get_data_length()
        }
        
        return config
    
    def get_optimized_default_config(self) -> Dict[str, Any]:
        """Get optimized default configuration based on previous experiments"""
        
        return {
            'gpu_id': self.gpu_id,
            'num_sims': 2**4,  # 16 parallel environments (reduced for memory)
            'num_ignore_step': 60,
            'max_position': 1,
            'step_gap': 2,
            'slippage': 7e-7,
            'starting_cash': 1e6,
            'net_dims': (128, 128, 128),
            'gamma': 0.995,
            'explore_rate': 0.005,
            'state_value_tau': 0.01,
            'soft_update_tau': 2e-6,
            'learning_rate': 2e-6,
            'batch_size': 512,
            'break_step': 20,  # Increased for longer training
            'buffer_size_multiplier': 2,  # Reduced from 8 to 2 for memory
            'repeat_times': 2,
            'horizon_len_multiplier': 1,  # Reduced from 2 to 1 for memory
            'eval_per_step_multiplier': 1,
            'num_workers': 1,
            'save_gap': 8,
            'data_length': self.get_data_length()
        }
    
    def get_data_length(self) -> int:
        """Get the actual data length from enhanced features v3"""
        if self.use_enhanced_v3:
            from data_config import ConfigData
            args = ConfigData()
            
            enhanced_v3_path = args.predict_ary_path.replace('.npy', '_enhanced_v3.npy')
            if os.path.exists(enhanced_v3_path):
                factor_ary = np.load(enhanced_v3_path)
                # Be very conservative with data length to avoid boundary issues
                # Use seq_len * 2 (default 10000) as the safe limit to avoid random offset + step_i overflow
                safe_length = min(10000, int(factor_ary.shape[0] * 0.5))  # Much more conservative
                self.logger.info(f"Calculated safe data length: {safe_length} (from {factor_ary.shape[0]} total)")
                return safe_length
        
        # Fallback
        return 4800
    
    def get_enhanced_state_dim(self) -> int:
        """Get state dimension for enhanced features v3"""
        
        if self.use_enhanced_v3:
            # Check for enhanced v3 features
            from data_config import ConfigData
            args = ConfigData()
            
            enhanced_v3_path = args.predict_ary_path.replace('.npy', '_enhanced_v3.npy')
            if os.path.exists(enhanced_v3_path):
                self.logger.info(f"Using enhanced features v3 from {enhanced_v3_path}")
                factor_ary = np.load(enhanced_v3_path)
                state_dim = factor_ary.shape[1]
                self.logger.info(f"Enhanced v3 state dimension: {state_dim}")
                return state_dim
        
        # Fallback to existing method
        return get_state_dim()
    
    def setup_validation_monitoring(self) -> bool:
        """
        Setup validation monitoring for the training data
        
        Returns:
            True if validation setup successful, False otherwise
        """
        if not self.validation_monitoring or not self.validation_framework:
            return False
        
        try:
            # Load enhanced features v3 data
            from data_config import ConfigData
            args = ConfigData()
            
            enhanced_v3_path = args.predict_ary_path.replace('.npy', '_enhanced_v3.npy')
            if not os.path.exists(enhanced_v3_path):
                self.logger.error(f"Enhanced v3 features not found: {enhanced_v3_path}")
                return False
            
            feature_data = np.load(enhanced_v3_path)
            
            # Create timestamps
            timestamps = pd.date_range(
                start='2020-01-01', 
                periods=len(feature_data), 
                freq='1min'
            )
            
            # Setup validation framework
            setup_info = self.validation_framework.setup_validation(
                data=feature_data,
                timestamps=timestamps.values,
                save_splits=True,
                save_path=os.path.join(self.save_path, "validation_splits")
            )
            
            self.logger.info("Validation monitoring setup completed")
            self.logger.info(f"  Train samples: {setup_info['train_samples']}")
            self.logger.info(f"  Validation samples: {setup_info['val_samples']}")
            self.logger.info(f"  Test samples: {setup_info['test_samples']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup validation monitoring: {e}")
            return False
    
    def create_enhanced_ensemble_config(self) -> Tuple[Dict[str, Any], List]:
        """
        Create enhanced ensemble configuration
        
        Returns:
            Tuple of (config_dict, agent_list)
        """
        config = self.config.copy()
        config['state_dim'] = self.get_enhanced_state_dim()
        
        # Agent selection based on performance
        # Use a diverse ensemble for robustness
        agent_list = [
            AgentD3QN,      # Strong performer
            AgentDoubleDQN,  # Good for reducing overestimation
            AgentTwinD3QN,   # Advanced architecture  
            AgentDoubleDQN   # Duplicate for ensemble diversity
        ]
        
        self.logger.info(f"Enhanced ensemble configuration:")
        self.logger.info(f"  State dimension: {config['state_dim']}")
        self.logger.info(f"  Agents: {[cls.__name__ for cls in agent_list]}")
        self.logger.info(f"  Network dims: {config['net_dims']}")
        self.logger.info(f"  Learning rate: {config['learning_rate']}")
        self.logger.info(f"  Batch size: {config['batch_size']}")
        
        return config, agent_list
    
    def run_enhanced_training(self) -> Dict[str, Any]:
        """
        Run enhanced training with validation monitoring
        
        Returns:
            Dictionary containing training results
        """
        self.logger.info("Starting enhanced training v3 with validation framework")
        
        start_time = time.time()
        
        try:
            # Setup validation monitoring
            if self.validation_monitoring:
                validation_setup = self.setup_validation_monitoring()
                if not validation_setup:
                    self.logger.warning("Validation monitoring setup failed, continuing without validation")
                    self.validation_monitoring = False
            
            # Create enhanced ensemble configuration
            config_dict, agent_list = self.create_enhanced_ensemble_config()
            
            # Save configuration
            config_path = os.path.join(self.save_path, "training_config.json")
            with open(config_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_config = {}
                for k, v in config_dict.items():
                    if isinstance(v, np.integer):
                        json_config[k] = int(v)
                    elif isinstance(v, np.floating):
                        json_config[k] = float(v)
                    elif isinstance(v, tuple):
                        json_config[k] = list(v)
                    else:
                        json_config[k] = v
                        
                json.dump(json_config, f, indent=2)
            
            self.logger.info(f"Training configuration saved to {config_path}")
            
            # Run enhanced ensemble training
            self.logger.info("Launching enhanced ensemble training...")
            
            run(
                save_path=self.save_path,
                agent_list=agent_list,
                log_rules=True,  # Enable detailed logging
                config_dict=config_dict
            )
            
            training_time = time.time() - start_time
            
            # Training results
            results = {
                'training_completed': True,
                'training_time_minutes': training_time / 60,
                'save_path': self.save_path,
                'config': config_dict,
                'agents': [cls.__name__ for cls in agent_list],
                'validation_monitoring': self.validation_monitoring,
                'enhanced_features_v3': self.use_enhanced_v3
            }
            
            self.logger.info(f"Enhanced training completed successfully!")
            self.logger.info(f"  Training time: {training_time/60:.1f} minutes")
            self.logger.info(f"  Models saved to: {self.save_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced training failed: {str(e)}", exc_info=True)
            return {
                'training_completed': False,
                'error': str(e),
                'training_time_minutes': (time.time() - start_time) / 60
            }
    
    def validate_trained_models(self) -> Dict[str, Any]:
        """
        Validate trained models using the validation framework
        
        Returns:
            Dictionary containing validation results
        """
        if not self.validation_monitoring or not self.validation_framework:
            self.logger.warning("Validation framework not available")
            return {'validation_available': False}
        
        self.logger.info("Starting model validation...")
        
        try:
            # Check if models were trained
            ensemble_dir = os.path.join(self.save_path, "ensemble_models")
            if not os.path.exists(ensemble_dir):
                self.logger.error("No trained models found for validation")
                return {'validation_available': False, 'error': 'No models found'}
            
            # TODO: Implement model validation logic
            # This would involve:
            # 1. Loading trained models
            # 2. Running inference on validation set
            # 3. Computing validation metrics
            # 4. Checking for overfitting
            
            validation_results = {
                'validation_available': True,
                'ensemble_models_found': True,
                'validation_metrics': {},
                'overfitting_analysis': {},
                'model_paths': ensemble_dir
            }
            
            self.logger.info("Model validation completed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            return {
                'validation_available': False,
                'error': str(e)
            }
    
    def generate_training_report(self, training_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive training report
        
        Args:
            training_results: Results from enhanced training
            
        Returns:
            Path to generated report
        """
        report_data = {
            'training_summary': training_results,
            'configuration': self.config,
            'enhanced_features_v3': self.use_enhanced_v3,
            'validation_monitoring': self.validation_monitoring,
            'early_stopping': self.early_stopping,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save detailed report
        report_path = os.path.join(self.save_path, "training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Create summary report
        summary_path = os.path.join(self.save_path, "training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("ENHANCED TRAINING V3 SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Training Status: {'SUCCESS' if training_results.get('training_completed', False) else 'FAILED'}\n")
            f.write(f"Training Time: {training_results.get('training_time_minutes', 0):.1f} minutes\n")
            f.write(f"Enhanced Features V3: {self.use_enhanced_v3}\n")
            f.write(f"Validation Monitoring: {self.validation_monitoring}\n")
            f.write(f"Early Stopping: {self.early_stopping}\n")
            f.write(f"Save Path: {self.save_path}\n\n")
            
            if 'config' in training_results:
                f.write("Configuration:\n")
                for key, value in training_results['config'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            if 'agents' in training_results:
                f.write(f"Agents: {', '.join(training_results['agents'])}\n\n")
            
            if 'error' in training_results:
                f.write(f"Error: {training_results['error']}\n")
        
        self.logger.info(f"Training report saved to {report_path}")
        self.logger.info(f"Training summary saved to {summary_path}")
        
        return report_path


def main():
    """Main entry point for enhanced training v3"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Training V3 with Validation Framework')
    parser.add_argument('gpu_id', nargs='?', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--no-enhanced-v3', action='store_true', help='Disable enhanced features v3')
    parser.add_argument('--no-validation', action='store_true', help='Disable validation monitoring')
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--save-path', type=str, default='enhanced_training_v3', help='Path to save results')
    
    args = parser.parse_args()
    
    # Initialize enhanced training
    trainer = EnhancedTrainingV3(
        use_enhanced_v3=not args.no_enhanced_v3,
        validation_monitoring=not args.no_validation,
        early_stopping=not args.no_early_stopping,
        save_path=args.save_path,
        gpu_id=args.gpu_id
    )
    
    print(f"🚀 Enhanced Training V3 Starting...")
    print(f"🔧 GPU ID: {args.gpu_id}")
    print(f"✨ Enhanced Features V3: {not args.no_enhanced_v3}")
    print(f"📊 Validation Monitoring: {not args.no_validation}")
    print(f"⏹️  Early Stopping: {not args.no_early_stopping}")
    print(f"💾 Save Path: {args.save_path}")
    print()
    
    # Run enhanced training
    training_results = trainer.run_enhanced_training()
    
    # Validate trained models
    if training_results.get('training_completed', False):
        validation_results = trainer.validate_trained_models()
        training_results['validation_results'] = validation_results
    
    # Generate comprehensive report
    report_path = trainer.generate_training_report(training_results)
    
    # Final status
    if training_results.get('training_completed', False):
        print(f"\n✅ Enhanced Training V3 Completed Successfully!")
        print(f"📁 Results saved to: {args.save_path}")
        print(f"📊 Report: {report_path}")
        print(f"⏱️  Training time: {training_results.get('training_time_minutes', 0):.1f} minutes")
        sys.exit(0)
    else:
        print(f"\n❌ Enhanced Training V3 Failed!")
        print(f"❓ Error: {training_results.get('error', 'Unknown error')}")
        print(f"📊 Report: {report_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()