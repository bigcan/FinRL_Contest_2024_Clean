"""
Task 1 Hyperparameter Optimization using Optuna
Systematic hyperparameter tuning for ensemble cryptocurrency trading models
"""

import optuna
import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime

# Fix encoding issues on Windows
import io
import os
if os.name == 'nt':  # Windows only
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import existing modules
from task1_ensemble import run, Ensemble
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from hpo_config import (
    HPOConfig, 
    Task1HPOSearchSpace, 
    HPOResultsManager, 
    create_sqlite_storage,
    suggest_ensemble_agents
)
from metrics import *


class Task1HPOOptimizer:
    """Hyperparameter optimizer for Task 1 ensemble models"""
    
    def __init__(
        self, 
        hpo_config: HPOConfig,
        base_save_path: str = "hpo_experiments",
        evaluation_metric: str = "sharpe_ratio",
        use_pruning: bool = True,
        intermediate_reporting: bool = True
    ):
        """
        Initialize Task 1 HPO optimizer
        
        Args:
            hpo_config: HPO configuration object
            base_save_path: Base path for saving HPO experiment results
            evaluation_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'romad')
            use_pruning: Whether to enable trial pruning
            intermediate_reporting: Whether to report intermediate results
        """
        self.hpo_config = hpo_config
        self.base_save_path = base_save_path
        self.evaluation_metric = evaluation_metric
        self.use_pruning = use_pruning
        self.intermediate_reporting = intermediate_reporting
        
        # Create results manager
        self.results_manager = HPOResultsManager(
            save_dir=os.path.join(base_save_path, "task1_hpo_results")
        )
        
        # Setup logging
        self.setup_logging()
        
        # Agent mapping
        self.agent_mapping = {
            'AgentD3QN': AgentD3QN,
            'AgentDoubleDQN': AgentDoubleDQN,
            'AgentTwinD3QN': AgentTwinD3QN
        }
    
    def setup_logging(self):
        """Setup logging for HPO process"""
        log_dir = os.path.join(self.base_save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"task1_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to optimize
        """
        try:
            # Suggest hyperparameters
            params = Task1HPOSearchSpace.suggest_parameters(trial)
            
            # Suggest ensemble configuration
            agent_names = suggest_ensemble_agents(trial)
            agent_classes = [self.agent_mapping[name] for name in agent_names]
            
            # Convert parameters to configuration
            config_dict = Task1HPOSearchSpace.convert_to_config(params)
            
            # Create unique save path for this trial
            trial_save_path = os.path.join(
                self.base_save_path, 
                f"trial_{trial.number}_{int(time.time())}"
            )
            os.makedirs(trial_save_path, exist_ok=True)
            
            self.logger.info(f"Starting trial {trial.number} with config: {config_dict}")
            self.logger.info(f"Agent ensemble: {agent_names}")
            
            # Run training with suggested parameters
            success, metrics = self.run_training_trial(
                trial_save_path, 
                agent_classes, 
                config_dict,
                trial
            )
            
            if not success:
                self.logger.warning(f"Trial {trial.number} failed during training")
                raise optuna.TrialPruned()
            
            # Extract objective value
            objective_value = metrics.get(self.evaluation_metric, 0.0)
            
            self.logger.info(f"Trial {trial.number} completed with {self.evaluation_metric}: {objective_value:.4f}")
            
            # Report intermediate result for pruning
            if self.use_pruning and self.intermediate_reporting:
                trial.report(objective_value, step=0)
                
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned")
                    raise optuna.TrialPruned()
            
            # Clean up trial directory to save space (keep only best few)
            self.cleanup_trial_directory(trial_save_path, objective_value, trial.number)
            
            return objective_value
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            return float('-inf') if self.hpo_config.direction == 'maximize' else float('inf')
    
    def run_training_trial(
        self, 
        save_path: str, 
        agent_classes: List, 
        config_dict: Dict[str, Any],
        trial: optuna.Trial
    ) -> tuple:
        """
        Run training for a single trial
        
        Args:
            save_path: Path to save trial results
            agent_classes: List of agent classes for ensemble
            config_dict: Configuration dictionary
            trial: Optuna trial object
            
        Returns:
            Tuple of (success, metrics)
        """
        try:
            # Set GPU ID (use trial number modulo available GPUs)
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                gpu_id = trial.number % n_gpus
                config_dict['gpu_id'] = gpu_id
            else:
                config_dict['gpu_id'] = -1  # CPU
            
            # Reduce training steps for HPO (faster trials)
            config_dict['break_step'] = min(config_dict.get('break_step', 16), 8)
            
            # Run ensemble training with sys.argv workaround
            original_argv = sys.argv.copy()
            try:
                # Temporarily modify sys.argv to avoid parsing issues
                sys.argv = ['task1_ensemble.py', str(config_dict.get('gpu_id', 0))]
                
                run(
                    save_path=save_path,
                    agent_list=agent_classes,
                    log_rules=False,
                    config_dict=config_dict
                )
            finally:
                # Restore original sys.argv
                sys.argv = original_argv
            
            # Evaluate the trained ensemble
            metrics = self.evaluate_ensemble(save_path)
            
            return True, metrics
            
        except Exception as e:
            self.logger.error(f"Training trial failed: {str(e)}")
            return False, {}
    
    def evaluate_ensemble(self, ensemble_path: str) -> Dict[str, float]:
        """
        Evaluate trained ensemble and compute metrics
        
        Args:
            ensemble_path: Path to trained ensemble
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Look for evaluation results or run evaluation
            eval_results_path = os.path.join(ensemble_path, "evaluation_results.json")
            
            if os.path.exists(eval_results_path):
                import json
                with open(eval_results_path, 'r') as f:
                    return json.load(f)
            
            # If no evaluation results, run quick evaluation
            from task1_eval import run_evaluation
            
            metrics = run_evaluation(
                ensemble_path=ensemble_path,
                quick_eval=True  # Faster evaluation for HPO
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ensemble evaluation failed: {str(e)}")
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'romad': 0.0
            }
    
    def cleanup_trial_directory(self, trial_path: str, objective_value: float, trial_number: int):
        """
        Clean up trial directory to manage disk space
        
        Args:
            trial_path: Path to trial directory
            objective_value: Objective value achieved
            trial_number: Trial number
        """
        try:
            # Keep only top 10 trials
            keep_threshold = 10
            
            # Simple cleanup strategy: remove if not in top performers
            # (In practice, you might want more sophisticated cleanup)
            if trial_number > keep_threshold:
                import shutil
                shutil.rmtree(trial_path, ignore_errors=True)
                self.logger.info(f"Cleaned up trial directory: {trial_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cleanup trial directory: {str(e)}")
    
    def run_optimization(self) -> optuna.Study:
        """
        Run the hyperparameter optimization process
        
        Returns:
            Completed Optuna study
        """
        self.logger.info("Starting Task 1 Hyperparameter Optimization")
        self.logger.info(f"Configuration: {self.hpo_config.n_trials} trials, metric: {self.evaluation_metric}")
        
        # Create study
        study = self.hpo_config.create_study()
        
        # Add custom attributes
        study.set_user_attr("evaluation_metric", self.evaluation_metric)
        study.set_user_attr("start_time", datetime.now().isoformat())
        
        try:
            # Run optimization
            study.optimize(
                self.objective, 
                n_trials=self.hpo_config.n_trials,
                timeout=self.hpo_config.timeout,
                n_jobs=self.hpo_config.n_jobs,
                catch=(Exception,)
            )
            
            # Save results
            self.results_manager.save_study_results(study, "task1")
            
            # Generate and display report
            report = self.results_manager.generate_optimization_report(study, "task1")
            self.logger.info("Optimization completed!")
            self.logger.info(f"Best value: {study.best_value:.6f}")
            self.logger.info(f"Best parameters: {study.best_params}")
            
            return study
            
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
            return study
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return study
    
    def get_best_configuration(self) -> Dict[str, Any]:
        """
        Get the best configuration from completed optimization
        
        Returns:
            Best configuration dictionary
        """
        try:
            return self.results_manager.load_best_parameters("task1")
        except FileNotFoundError:
            self.logger.warning("No HPO results found. Run optimization first.")
            return {}


def create_hpo_configs() -> Dict[str, HPOConfig]:
    """Create different HPO configurations for various scenarios"""
    
    configs = {
        # Quick exploration (for testing)
        "quick": HPOConfig(
            study_name="task1_quick_hpo",
            n_trials=20,
            n_jobs=1,
            timeout=3600,  # 1 hour
            sampler_type="random"
        ),
        
        # Thorough optimization
        "thorough": HPOConfig(
            study_name="task1_thorough_hpo",
            n_trials=200,
            n_jobs=2,
            timeout=86400,  # 24 hours
            sampler_type="tpe",
            storage_url=create_sqlite_storage("task1_hpo.db")
        ),
        
        # Production optimization
        "production": HPOConfig(
            study_name="task1_production_hpo", 
            n_trials=500,
            n_jobs=4,
            timeout=259200,  # 72 hours
            sampler_type="tpe",
            storage_url=create_sqlite_storage("task1_production_hpo.db")
        )
    }
    
    return configs


def main():
    """Main function for Task 1 HPO"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Task 1 Hyperparameter Optimization')
    parser.add_argument('--config', type=str, default='quick', 
                       choices=['quick', 'thorough', 'production'],
                       help='HPO configuration type')
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'total_return', 'romad'],
                       help='Metric to optimize')
    parser.add_argument('--base-path', type=str, default='hpo_experiments',
                       help='Base path for HPO experiments')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing study')
    
    args = parser.parse_args()
    
    # Create HPO configuration
    hpo_configs = create_hpo_configs()
    hpo_config = hpo_configs[args.config]
    
    if not args.resume:
        # Create new study name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hpo_config.study_name = f"{hpo_config.study_name}_{timestamp}"
        hpo_config.load_if_exists = False
    
    # Create optimizer
    optimizer = Task1HPOOptimizer(
        hpo_config=hpo_config,
        base_save_path=args.base_path,
        evaluation_metric=args.metric,
        use_pruning=True,
        intermediate_reporting=True
    )
    
    # Run optimization
    study = optimizer.run_optimization()
    
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("="*80)
    print(f"Best {args.metric}: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Total trials: {len(study.trials)}")
    print("\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print("="*80)


if __name__ == "__main__":
    main()