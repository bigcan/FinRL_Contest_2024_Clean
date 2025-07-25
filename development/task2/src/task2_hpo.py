"""
Task 2 Hyperparameter Optimization using Optuna
Systematic hyperparameter tuning for LLM-based signal generation with RLMF
"""

import optuna
import os
import sys
import time
import torch
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.optim import Adam

# Import existing modules
from task2_config import Task2Config
from task2_env import Task2Env
from task2_news import get_news
from task2_signal import generate_signal
from task2_train import setup_and_initialize_model, plot_training_metrics

# Import HPO configuration
import sys
sys.path.append('../task1/src')
from hpo_config import (
    HPOConfig, 
    Task2HPOSearchSpace, 
    HPOResultsManager, 
    create_sqlite_storage
)


class Task2HPOOptimizer:
    """Hyperparameter optimizer for Task 2 LLM fine-tuning"""
    
    def __init__(
        self,
        hpo_config: HPOConfig,
        base_save_path: str = "hpo_experiments",
        evaluation_metric: str = "cumulative_return",
        use_pruning: bool = True,
        intermediate_reporting: bool = True,
        stock_data_path: str = "task2_dsets/test/task2_stocks_test.csv",
        news_data_path: str = "task2_dsets/test/task2_news_test.csv"
    ):
        """
        Initialize Task 2 HPO optimizer
        
        Args:
            hpo_config: HPO configuration object
            base_save_path: Base path for saving HPO experiment results
            evaluation_metric: Metric to optimize
            use_pruning: Whether to enable trial pruning
            intermediate_reporting: Whether to report intermediate results
            stock_data_path: Path to stock data CSV
            news_data_path: Path to news data CSV
        """
        self.hpo_config = hpo_config
        self.base_save_path = base_save_path
        self.evaluation_metric = evaluation_metric
        self.use_pruning = use_pruning
        self.intermediate_reporting = intermediate_reporting
        self.stock_data_path = stock_data_path
        self.news_data_path = news_data_path
        
        # Create results manager
        self.results_manager = HPOResultsManager(
            save_dir=os.path.join(base_save_path, "task2_hpo_results")
        )
        
        # Setup logging
        self.setup_logging()
        
        # Load data once
        self.stock_data = pd.read_csv(stock_data_path)
        
        # Stock tickers
        self.stock_tickers = [
            "AAPL", "NVDA", "GOOG", "AMZN", "MSFT", "XOM", "WMT"
        ]
        
        # Training constants
        self.END_DATE = "2022-10-31"
        self.START_DATE = "2022-10-10"
    
    def setup_logging(self):
        """Setup logging for HPO process"""
        log_dir = os.path.join(self.base_save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"task2_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
            params = Task2HPOSearchSpace.suggest_parameters(trial)
            
            # Create unique save path for this trial
            trial_save_path = os.path.join(
                self.base_save_path,
                f"trial_{trial.number}_{int(time.time())}"
            )
            os.makedirs(trial_save_path, exist_ok=True)
            
            self.logger.info(f"Starting trial {trial.number} with params: {params}")
            
            # Run training with suggested parameters
            success, metrics = self.run_training_trial(trial_save_path, params, trial)
            
            if not success:
                self.logger.warning(f"Trial {trial.number} failed during training")
                raise optuna.TrialPruned()
            
            # Extract objective value
            objective_value = metrics.get(self.evaluation_metric, 0.0)
            
            self.logger.info(f"Trial {trial.number} completed with {self.evaluation_metric}: {objective_value:.4f}")
            
            # Report intermediate results for pruning
            if self.use_pruning and self.intermediate_reporting:
                # Report multiple steps for better pruning decisions
                for step, value in enumerate(metrics.get('intermediate_rewards', [objective_value])):
                    trial.report(value, step=step)
                    
                    if trial.should_prune():
                        self.logger.info(f"Trial {trial.number} pruned at step {step}")
                        raise optuna.TrialPruned()
            
            # Clean up trial directory
            self.cleanup_trial_directory(trial_save_path, objective_value, trial.number)
            
            return objective_value
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            return float('-inf') if self.hpo_config.direction == 'maximize' else float('inf')
    
    def run_training_trial(
        self,
        save_path: str,
        params: Dict[str, Any],
        trial: optuna.Trial
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run training for a single trial
        
        Args:
            save_path: Path to save trial results
            params: Hyperparameter dictionary
            trial: Optuna trial object
            
        Returns:
            Tuple of (success, metrics)
        """
        try:
            # Create training configuration
            train_config = self.create_train_config(params)
            
            # Setup model and environment
            device, tokenizer, model = self.setup_model_for_trial(train_config, params)
            
            # Create environment
            task2env = Task2Env(
                model,
                tokenizer,
                self.stock_tickers,
                self.stock_data,
                (-2, 2),
                max_steps=params.get('max_env_steps', 250),
                lookahead=params['lookahead'],
            )
            
            # Run training loop
            metrics = self.run_training_loop(
                task2env, model, tokenizer, device, train_config, params, trial, save_path
            )
            
            return True, metrics
            
        except Exception as e:
            self.logger.error(f"Training trial failed: {str(e)}")
            return False, {}
    
    def create_train_config(self, params: Dict[str, Any]) -> Task2Config:
        """Create training configuration from parameters"""
        return Task2Config(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            bnb_config=BitsAndBytesConfig(
                load_in_4bit=params['use_4bit'],
                load_in_8bit=not params['use_4bit'],
                bnb_4bit_quant_type=params.get('quantization_type', 'fp4')
            ),
            tickers=self.stock_tickers,
            end_date=self.END_DATE,
            start_date=self.START_DATE,
            lookahead=params['lookahead'],
            signal_strength=params['signal_strength'],
            max_train_steps=params['max_train_steps'],
        )
    
    def setup_model_for_trial(self, config: Task2Config, params: Dict[str, Any]):
        """Setup model with trial-specific parameters"""
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Setup quantization config
        if params['use_4bit']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type=params['quantization_type'],
            )
        else:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        # Configure model settings
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        
        # Setup LoRA configuration with trial parameters
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=params['lora_r'],
            target_modules=["q_proj", "v_proj"],
            lora_alpha=params['lora_alpha'],
            lora_dropout=params['lora_dropout'],
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        
        return device, tokenizer, model
    
    def run_training_loop(
        self,
        task2env,
        model,
        tokenizer,
        device,
        train_config,
        params,
        trial,
        save_path
    ) -> Dict[str, Any]:
        """Run the training loop and return metrics"""
        
        # Initialize training metrics
        state = task2env.reset()
        rewards = []
        returns = []
        running_eval = []
        losses = []
        intermediate_rewards = []
        
        optimizer = Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        for step in range(train_config.max_train_steps):
            date, prices = state
            date = pd.Timestamp(date)
            ticker_actions = {}
            log_probs = []
            
            for ticker in prices.Ticker:
                news = get_news(
                    ticker,
                    (date - timedelta(days=1))._date_repr,
                    (date - timedelta(days=11))._date_repr,
                    self.news_data_path,
                )
                
                sentiment_score, log_prob = generate_signal(
                    tokenizer,
                    model,
                    device,
                    news,
                    prices.copy().drop("Future_Close", axis=1)[prices["Ticker"] == ticker],
                    train_config.signal_strength,
                    train_config.threshold,
                )
                
                ticker_actions[ticker] = sentiment_score
                log_probs.append(log_prob)
            
            state, reward, done, info = task2env.step(ticker_actions)
            
            # Apply reward scaling
            scaled_reward = reward * params.get('reward_scaling', 1.0)
            
            # Update metrics
            rewards.append(scaled_reward)
            returns.append(info["price change"])
            running_eval.append(info["running eval"])
            
            # Compute and apply gradients
            loss = -torch.stack(log_probs) * torch.tensor(scaled_reward)
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Store intermediate rewards for pruning
            if step % 5 == 0:  # Every 5 steps
                intermediate_rewards.append(np.mean(rewards[-5:]) if len(rewards) >= 5 else rewards[-1])
            
            # Report intermediate results for pruning
            if self.use_pruning and step % 10 == 0 and step > 0:
                intermediate_value = np.mean(rewards[-10:])
                trial.report(intermediate_value, step=step // 10)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            if done:
                break
        
        # Save model for this trial
        model_save_path = os.path.join(save_path, "model")
        model.save_pretrained(model_save_path)
        
        # Calculate final metrics
        final_metrics = {
            'cumulative_return': running_eval[-1] if running_eval else 0.0,
            'mean_reward': np.mean(rewards) if rewards else 0.0,
            'total_reward': np.sum(rewards) if rewards else 0.0,
            'mean_loss': np.mean(losses) if losses else 0.0,
            'final_portfolio_value': running_eval[-1] if running_eval else 0.0,
            'volatility': np.std(returns) if returns else 0.0,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) if returns else 0.0,
            'intermediate_rewards': intermediate_rewards,
            'n_steps': len(rewards)
        }
        
        return final_metrics
    
    def cleanup_trial_directory(self, trial_path: str, objective_value: float, trial_number: int):
        """Clean up trial directory to manage disk space"""
        try:
            # Keep only top 10 trials
            keep_threshold = 10
            
            if trial_number > keep_threshold:
                import shutil
                shutil.rmtree(trial_path, ignore_errors=True)
                self.logger.info(f"Cleaned up trial directory: {trial_path}")
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup trial directory: {str(e)}")
    
    def run_optimization(self) -> optuna.Study:
        """Run the hyperparameter optimization process"""
        self.logger.info("🚀 Starting Task 2 Hyperparameter Optimization")
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
            self.results_manager.save_study_results(study, "task2")
            
            # Generate and display report
            report = self.results_manager.generate_optimization_report(study, "task2")
            self.logger.info("📊 Optimization completed!")
            self.logger.info(f"Best value: {study.best_value:.6f}")
            self.logger.info(f"Best parameters: {study.best_params}")
            
            return study
            
        except KeyboardInterrupt:
            self.logger.info("⚠️ Optimization interrupted by user")
            return study
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return study
    
    def get_best_configuration(self) -> Dict[str, Any]:
        """Get the best configuration from completed optimization"""
        try:
            return self.results_manager.load_best_parameters("task2")
        except FileNotFoundError:
            self.logger.warning("No HPO results found. Run optimization first.")
            return {}


def create_hpo_configs() -> Dict[str, HPOConfig]:
    """Create different HPO configurations for various scenarios"""
    
    configs = {
        # Quick exploration (for testing)
        "quick": HPOConfig(
            study_name="task2_quick_hpo",
            n_trials=15,
            n_jobs=1,
            timeout=7200,  # 2 hours
            sampler_type="random"
        ),
        
        # Thorough optimization
        "thorough": HPOConfig(
            study_name="task2_thorough_hpo",
            n_trials=100,
            n_jobs=1,  # LLM training is memory intensive
            timeout=86400,  # 24 hours
            sampler_type="tpe",
            storage_url=create_sqlite_storage("task2_hpo.db")
        ),
        
        # Production optimization
        "production": HPOConfig(
            study_name="task2_production_hpo",
            n_trials=200,
            n_jobs=1,
            timeout=259200,  # 72 hours
            sampler_type="tpe",
            storage_url=create_sqlite_storage("task2_production_hpo.db")
        )
    }
    
    return configs


def main():
    """Main function for Task 2 HPO"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Task 2 Hyperparameter Optimization')
    parser.add_argument('--config', type=str, default='quick',
                       choices=['quick', 'thorough', 'production'],
                       help='HPO configuration type')
    parser.add_argument('--metric', type=str, default='cumulative_return',
                       choices=['cumulative_return', 'mean_reward', 'sharpe_ratio'],
                       help='Metric to optimize')
    parser.add_argument('--base-path', type=str, default='hpo_experiments',
                       help='Base path for HPO experiments')
    parser.add_argument('--stock-data', type=str, default='task2_dsets/test/task2_stocks_test.csv',
                       help='Path to stock data CSV')
    parser.add_argument('--news-data', type=str, default='task2_dsets/test/task2_news_test.csv',
                       help='Path to news data CSV')
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
    optimizer = Task2HPOOptimizer(
        hpo_config=hpo_config,
        base_save_path=args.base_path,
        evaluation_metric=args.metric,
        use_pruning=True,
        intermediate_reporting=True,
        stock_data_path=args.stock_data,
        news_data_path=args.news_data
    )
    
    # Run optimization
    study = optimizer.run_optimization()
    
    print("\n" + "="*80)
    print("🎯 HYPERPARAMETER OPTIMIZATION COMPLETED")
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