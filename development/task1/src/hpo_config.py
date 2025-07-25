"""
Hyperparameter Optimization Configuration for FinRL Contest 2024
Integrates Optuna for systematic hyperparameter tuning
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import Dict, Any, Tuple, List, Optional
import json
import os


class HPOConfig:
    """Configuration class for hyperparameter optimization using Optuna"""
    
    def __init__(
        self,
        study_name: str = "finrl_contest_2024",
        storage_url: Optional[str] = None,
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: Optional[int] = None,
        sampler_type: str = "tpe",
        pruner_type: str = "median",
        direction: str = "maximize",
        load_if_exists: bool = True
    ):
        """
        Initialize HPO configuration
        
        Args:
            study_name: Name of the Optuna study
            storage_url: Database URL for persistent storage (None for in-memory)
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds for optimization
            sampler_type: Type of sampler ('tpe', 'random', 'cmaes')
            pruner_type: Type of pruner ('median', 'successive_halving', 'hyperband')
            direction: Optimization direction ('maximize' or 'minimize')
            load_if_exists: Whether to load existing study
        """
        self.study_name = study_name
        self.storage_url = storage_url
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.sampler_type = sampler_type
        self.pruner_type = pruner_type
        self.direction = direction
        self.load_if_exists = load_if_exists
        
        # Initialize sampler
        self.sampler = self._create_sampler()
        
        # Initialize pruner
        self.pruner = self._create_pruner()
    
    def _create_sampler(self):
        """Create Optuna sampler based on configuration"""
        if self.sampler_type == "tpe":
            return TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=42
            )
        elif self.sampler_type == "random":
            return optuna.samplers.RandomSampler(seed=42)
        elif self.sampler_type == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")
    
    def _create_pruner(self):
        """Create Optuna pruner based on configuration"""
        if self.pruner_type == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        elif self.pruner_type == "successive_halving":
            return optuna.pruners.SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4,
                min_early_stopping_rate=0
            )
        elif self.pruner_type == "hyperband":
            return optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3
            )
        else:
            return optuna.pruners.NopPruner()
    
    def create_study(self) -> optuna.Study:
        """Create or load Optuna study"""
        return optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=self.direction,
            load_if_exists=self.load_if_exists
        )


class Task1HPOSearchSpace:
    """Defines search space for Task 1 hyperparameters"""
    
    @staticmethod
    def suggest_parameters(trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Task 1 ensemble training
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        return {
            # Network architecture
            'net_dims_0': trial.suggest_int('net_dims_0', 64, 512, step=64),
            'net_dims_1': trial.suggest_int('net_dims_1', 64, 512, step=64),
            'net_dims_2': trial.suggest_int('net_dims_2', 64, 512, step=64),
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.99, 0.999),
            'explore_rate': trial.suggest_float('explore_rate', 0.001, 0.1, log=True),
            
            # Training parameters
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
            'buffer_size_multiplier': trial.suggest_int('buffer_size_multiplier', 4, 16),
            'horizon_len_multiplier': trial.suggest_int('horizon_len_multiplier', 1, 4),
            'repeat_times': trial.suggest_int('repeat_times', 1, 4),
            
            # Regularization
            'soft_update_tau': trial.suggest_float('soft_update_tau', 1e-6, 1e-3, log=True),
            'state_value_tau': trial.suggest_float('state_value_tau', 0.005, 0.02),
            
            # Environment parameters
            'num_sims': trial.suggest_categorical('num_sims', [32, 64, 128, 256]),
            'step_gap': trial.suggest_int('step_gap', 1, 5),
            'slippage': trial.suggest_float('slippage', 1e-8, 1e-5, log=True),
            'max_position': trial.suggest_int('max_position', 1, 3),
        }
    
    @staticmethod
    def convert_to_config(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Optuna parameters to training configuration"""
        config = params.copy()
        
        # Convert network dimensions to tuple
        config['net_dims'] = (
            config.pop('net_dims_0'),
            config.pop('net_dims_1'),
            config.pop('net_dims_2')
        )
        
        return config


class Task2HPOSearchSpace:
    """Defines search space for Task 2 hyperparameters"""
    
    @staticmethod
    def suggest_parameters(trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Task 2 LLM fine-tuning
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        return {
            # LoRA parameters
            'lora_r': trial.suggest_int('lora_r', 8, 64, step=8),
            'lora_alpha': trial.suggest_int('lora_alpha', 8, 32, step=8),
            'lora_dropout': trial.suggest_float('lora_dropout', 0.05, 0.3),
            
            # Training parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
            'max_train_steps': trial.suggest_int('max_train_steps', 20, 100),
            
            # Signal generation parameters
            'signal_strength': trial.suggest_int('signal_strength', 5, 20),
            'lookahead': trial.suggest_int('lookahead', 1, 7),
            
            # Model quantization
            'use_4bit': trial.suggest_categorical('use_4bit', [True, False]),
            'quantization_type': trial.suggest_categorical('quantization_type', ['fp4', 'nf4']),
            
            # Environment parameters
            'max_env_steps': trial.suggest_int('max_env_steps', 200, 300),
            'reward_scaling': trial.suggest_float('reward_scaling', 0.1, 2.0),
        }


class HPOResultsManager:
    """Manages HPO results and analysis"""
    
    def __init__(self, save_dir: str = "hpo_results"):
        """
        Initialize results manager
        
        Args:
            save_dir: Directory to save HPO results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_study_results(self, study: optuna.Study, task_name: str):
        """
        Save study results to files
        
        Args:
            study: Completed Optuna study
            task_name: Name of the task (task1 or task2)
        """
        # Save best parameters
        best_params_path = os.path.join(self.save_dir, f"{task_name}_best_params.json")
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save study statistics
        stats = {
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'n_trials': len(study.trials),
            'study_name': study.study_name,
            'direction': study.direction.name
        }
        
        stats_path = os.path.join(self.save_dir, f"{task_name}_study_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save trial history
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            trials_data.append(trial_data)
        
        trials_path = os.path.join(self.save_dir, f"{task_name}_trials_history.json")
        with open(trials_path, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        print(f"📊 HPO results saved to {self.save_dir}")
        print(f"   Best value: {study.best_value:.6f}")
        print(f"   Best trial: {study.best_trial.number}")
        print(f"   Total trials: {len(study.trials)}")
    
    def load_best_parameters(self, task_name: str) -> Dict[str, Any]:
        """
        Load best parameters from previous HPO run
        
        Args:
            task_name: Name of the task (task1 or task2)
            
        Returns:
            Dictionary of best parameters
        """
        best_params_path = os.path.join(self.save_dir, f"{task_name}_best_params.json")
        
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No HPO results found for {task_name}")
    
    def generate_optimization_report(self, study: optuna.Study, task_name: str) -> str:
        """
        Generate a comprehensive optimization report
        
        Args:
            study: Completed Optuna study
            task_name: Name of the task
            
        Returns:
            Report as string
        """
        report = f"""
# Hyperparameter Optimization Report: {task_name.upper()}

## Study Overview
- **Study Name**: {study.study_name}
- **Direction**: {study.direction.name}
- **Total Trials**: {len(study.trials)}
- **Best Value**: {study.best_value:.6f}
- **Best Trial**: {study.best_trial.number}

## Best Parameters
"""
        
        for param, value in study.best_params.items():
            report += f"- **{param}**: {value}\n"
        
        report += "\n## Trial Statistics\n"
        
        # Calculate completion rate
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completion_rate = len(completed_trials) / len(study.trials) * 100
        
        report += f"- **Completion Rate**: {completion_rate:.1f}%\n"
        report += f"- **Completed Trials**: {len(completed_trials)}\n"
        report += f"- **Failed Trials**: {len(study.trials) - len(completed_trials)}\n"
        
        if completed_trials:
            values = [t.value for t in completed_trials if t.value is not None]
            if values:
                import numpy as np
                report += f"- **Mean Objective**: {np.mean(values):.6f}\n"
                report += f"- **Std Objective**: {np.std(values):.6f}\n"
                report += f"- **Min Objective**: {np.min(values):.6f}\n"
                report += f"- **Max Objective**: {np.max(values):.6f}\n"
        
        # Save report
        report_path = os.path.join(self.save_dir, f"{task_name}_optimization_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report


def create_sqlite_storage(db_path: str = "hpo_studies.db") -> str:
    """
    Create SQLite storage URL for persistent study storage
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLite storage URL
    """
    return f"sqlite:///{db_path}"


def suggest_ensemble_agents(trial: optuna.Trial) -> List[str]:
    """
    Suggest ensemble agent configuration
    
    Args:
        trial: Optuna trial object
        
    Returns:
        List of agent class names
    """
    # Define available agents
    available_agents = ['AgentD3QN', 'AgentDoubleDQN', 'AgentTwinD3QN']
    
    # Suggest number of agents in ensemble
    n_agents = trial.suggest_int('n_ensemble_agents', 2, 4)
    
    # Suggest specific agents
    selected_agents = []
    for i in range(n_agents):
        agent = trial.suggest_categorical(f'agent_{i}', available_agents)
        selected_agents.append(agent)
    
    return selected_agents