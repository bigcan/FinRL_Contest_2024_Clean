"""
Enhanced HPO Runner with Conservative Trading Solution
Integrates new reward systems, exploration strategies, and monitoring
"""

import optuna
import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import json

# Import enhanced modules
from hpo_config import HPOConfig, Task1HPOSearchSpace, HPOResultsManager
from task1_ensemble import run, Ensemble
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN
from training_monitor import ActionDiversityMonitor, TrainingValidator
from reward_functions import create_reward_calculator
from trade_simulator import TradeSimulator


class EnhancedHPOOptimizer:
    """
    Enhanced HPO optimizer with conservative trading solution
    """
    
    def __init__(self,
                 n_trials: int = 50,
                 study_name: str = "enhanced_conservative_solution",
                 gpu_id: int = 0,
                 results_dir: str = "enhanced_hpo_results"):
        """
        Initialize enhanced HPO optimizer
        
        Args:
            n_trials: Number of HPO trials
            study_name: Name for the study
            gpu_id: GPU device ID
            results_dir: Directory for results
        """
        self.n_trials = n_trials
        self.study_name = study_name
        self.gpu_id = gpu_id
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize HPO components
        self.hpo_config = HPOConfig(
            study_name=study_name,
            n_trials=n_trials,
            direction="maximize",
            storage_url=f"sqlite:///{results_dir}/enhanced_hpo.db"
        )
        
        self.results_manager = HPOResultsManager(save_dir=results_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Trial tracking
        self.trial_results = []
        self.best_trial_info = None
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_file = os.path.join(self.results_dir, f"hpo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
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
        Enhanced objective function with action diversity monitoring
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value (Sharpe ratio with diversity bonus)
        """
        try:
            # Get hyperparameters from enhanced search space
            params = Task1HPOSearchSpace.suggest_parameters(trial)
            config = Task1HPOSearchSpace.convert_to_config(params)
            
            # Extract reward configuration
            reward_config = {
                'reward_type': params.get('reward_type', 'adaptive_multi_objective'),
                'reward_weights': {
                    'conservatism_penalty_weight': params.get('conservatism_penalty_weight', 0.2),
                    'action_diversity_weight': params.get('action_diversity_weight', 0.15),
                    'transaction_cost_weight': params.get('transaction_cost_weight', 0.5),
                    'risk_adjusted_return_weight': params.get('risk_adjusted_return_weight', 0.7),
                    'conservatism_escalation_rate': params.get('conservatism_escalation_rate', 1.5),
                    'activity_threshold': params.get('activity_threshold', 0.3),
                    'regime_sensitivity': params.get('regime_sensitivity', 1.0)
                }
            }
            
            self.logger.info(f"\n🎯 Trial {trial.number}: Testing enhanced parameters")
            self.logger.info(f"   Reward type: {reward_config['reward_type']}")
            self.logger.info(f"   Explore rate: {params['explore_rate']:.4f}")
            self.logger.info(f"   Min explore rate: {params['min_explore_rate']:.4f}")
            self.logger.info(f"   Conservatism penalty weight: {reward_config['reward_weights']['conservatism_penalty_weight']:.3f}")
            
            # Initialize action diversity monitor
            monitor = ActionDiversityMonitor(
                window_size=1000,
                diversity_threshold=0.3,
                conservatism_threshold=params.get('activity_threshold', 0.3),
                checkpoint_dir=os.path.join(self.results_dir, f"trial_{trial.number}_checkpoints")
            )
            
            # Enhanced training with monitoring
            results = self._run_enhanced_training(config, reward_config, monitor, trial.number)
            
            # Calculate enhanced objective with diversity metrics
            objective_value = self._calculate_enhanced_objective(results, monitor, trial)
            
            # Report intermediate values for pruning
            trial.report(objective_value, step=0)
            
            # Save trial results
            self._save_trial_results(trial, params, results, monitor, objective_value)
            
            self.logger.info(f"✅ Trial {trial.number} completed - Objective: {objective_value:.6f}")
            
            return objective_value
            
        except Exception as e:
            self.logger.error(f"❌ Trial {trial.number} failed: {str(e)}")
            return -float('inf')  # Return very low value for failed trials
            
    def _run_enhanced_training(self,
                              config: Dict[str, Any],
                              reward_config: Dict[str, Any],
                              monitor: ActionDiversityMonitor,
                              trial_number: int) -> Dict[str, Any]:
        """
        Run training with enhanced monitoring and early stopping
        
        Args:
            config: Training configuration
            reward_config: Reward system configuration
            monitor: Action diversity monitor
            trial_number: Current trial number
            
        Returns:
            Training results
        """
        try:
            # Create enhanced ensemble with monitoring
            ensemble = self._create_enhanced_ensemble(config, reward_config)
            
            # Custom training loop with monitoring
            results = self._monitored_training_loop(ensemble, config, monitor, trial_number)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed in trial {trial_number}: {str(e)}")
            return {
                'sharpe_ratio': -10.0,
                'total_return': -1.0,
                'max_drawdown': 1.0,
                'action_diversity': 0.0,
                'training_failed': True
            }
            
    def _create_enhanced_ensemble(self,
                                 config: Dict[str, Any],
                                 reward_config: Dict[str, Any]) -> Ensemble:
        """Create ensemble with enhanced configuration"""
        
        # Enhanced agent configuration
        enhanced_agents = [
            {
                'class': AgentDoubleDQN,
                'kwargs': {
                    'net_dims': config['net_dims'],
                    'learning_rate': config['learning_rate'],
                    'explore_rate': config['explore_rate'],
                    'min_explore_rate': config.get('min_explore_rate', 0.01),
                    'exploration_decay_rate': config.get('exploration_decay_rate', 0.995),
                    'exploration_warmup_steps': config.get('exploration_warmup_steps', 5000),
                    'force_exploration_probability': config.get('force_exploration_probability', 0.05)
                }
            },
            {
                'class': AgentD3QN,
                'kwargs': {
                    'net_dims': config['net_dims'],
                    'learning_rate': config['learning_rate'],
                    'explore_rate': config['explore_rate'] * 0.8,  # Slightly different for diversity
                    'min_explore_rate': config.get('min_explore_rate', 0.01),
                }
            },
            {
                'class': AgentPrioritizedDQN,
                'kwargs': {
                    'net_dims': config['net_dims'],
                    'learning_rate': config['learning_rate'],
                    'explore_rate': config['explore_rate'] * 1.2,  # Higher exploration
                    'min_explore_rate': config.get('min_explore_rate', 0.01),
                }
            }
        ]
        
        # Create enhanced trading environment
        env_config = {
            'num_sims': config.get('num_sims', 64),
            'slippage': config.get('slippage', 5e-5),
            'step_gap': config.get('step_gap', 1),
            'gpu_id': self.gpu_id,
            'data_length': 50000  # Limit for faster HPO
        }
        
        # Initialize enhanced environment with reward configuration
        env = TradeSimulator(**env_config)
        env.set_reward_type(
            reward_type=reward_config['reward_type'],
            reward_weights=reward_config['reward_weights']
        )
        
        return Ensemble(
            agents=enhanced_agents,
            env=env,
            gpu_id=self.gpu_id
        )
        
    def _monitored_training_loop(self,
                                ensemble: Ensemble,
                                config: Dict[str, Any],
                                monitor: ActionDiversityMonitor,
                                trial_number: int) -> Dict[str, Any]:
        """
        Custom training loop with action diversity monitoring
        
        Args:
            ensemble: Enhanced ensemble
            config: Training configuration
            monitor: Action diversity monitor
            trial_number: Trial number
            
        Returns:
            Training results with diversity metrics
        """
        # Training parameters
        max_episodes = 200  # Reduced for faster HPO
        episode_returns = []
        early_stop_patience = 20
        episodes_without_improvement = 0
        best_performance = -float('inf')
        
        self.logger.info(f"🚀 Starting monitored training for trial {trial_number}")
        
        for episode in range(max_episodes):
            # Run episode
            episode_return = self._run_episode(ensemble, monitor, config)
            episode_returns.append(episode_return)
            
            # Check for improvement
            recent_performance = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else episode_return
            
            if recent_performance > best_performance * 1.05:  # 5% improvement threshold
                best_performance = recent_performance
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1
                
            # Check action diversity and early stopping
            diversity_check = monitor.check_diversity()
            
            if episode % 20 == 0:
                self.logger.info(f"   Episode {episode}: Return={episode_return:.4f}, "
                               f"Status={diversity_check.get('status', 'N/A')}")
                
                if 'metrics' in diversity_check:
                    metrics = diversity_check['metrics']
                    self.logger.info(f"      Hold ratio: {metrics.get('hold_ratio', 0):.2%}, "
                                   f"Buy ratio: {metrics.get('buy_ratio', 0):.2%}")
                    
            # Early stopping conditions
            should_intervene, reason = monitor.should_intervene()
            
            if should_intervene and episode > 50:
                self.logger.warning(f"⚠️ Early stopping due to: {reason}")
                break
                
            if episodes_without_improvement >= early_stop_patience and episode > 50:
                self.logger.info(f"📈 Early stopping due to no improvement ({early_stop_patience} episodes)")
                break
                
        # Calculate final metrics
        final_metrics = self._calculate_training_metrics(
            episode_returns, monitor, ensemble, trial_number
        )
        
        return final_metrics
        
    def _run_episode(self,
                    ensemble: Ensemble,
                    monitor: ActionDiversityMonitor,
                    config: Dict[str, Any]) -> float:
        """Run a single training episode with monitoring"""
        
        # Reset environment
        state = ensemble.env.reset()
        episode_return = 0.0
        episode_actions = []
        
        # Run episode steps
        for step in range(1000):  # Limit steps for faster training
            # Get ensemble action
            action = ensemble.get_ensemble_action(state)
            
            # Step environment
            next_state, reward, done, _ = ensemble.env.step(action)
            
            # Track for monitoring
            if hasattr(action, 'item'):
                action_int = action.item()
            else:
                action_int = int(action[0]) if hasattr(action, '__getitem__') else int(action)
                
            episode_actions.append(action_int)
            episode_return += reward.mean().item() if hasattr(reward, 'mean') else float(reward)
            
            # Update monitor
            monitor.update(action_int, float(reward.mean().item() if hasattr(reward, 'mean') else reward), done.any() if hasattr(done, 'any') else done)
            
            state = next_state
            
            if (done.any() if hasattr(done, 'any') else done):
                break
                
        return episode_return
        
    def _calculate_training_metrics(self,
                                   episode_returns: List[float],
                                   monitor: ActionDiversityMonitor,
                                   ensemble: Ensemble,
                                   trial_number: int) -> Dict[str, Any]:
        """Calculate comprehensive training metrics"""
        
        if len(episode_returns) == 0:
            return {
                'sharpe_ratio': -10.0,
                'total_return': -1.0,
                'max_drawdown': 1.0,
                'action_diversity': 0.0,
                'training_failed': True
            }
            
        # Basic performance metrics
        returns_array = np.array(episode_returns)
        mean_return = np.mean(returns_array)
        return_std = np.std(returns_array)
        
        # Sharpe ratio (handle division by zero)
        sharpe_ratio = mean_return / max(return_std, 1e-8)
        
        # Drawdown calculation
        cumulative = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / np.maximum(running_max, 1e-8)
        max_drawdown = np.max(drawdowns)
        
        # Action diversity metrics
        diversity_check = monitor.check_diversity()
        action_diversity = 0.0
        
        if 'metrics' in diversity_check:
            metrics = diversity_check['metrics']
            action_diversity = metrics.get('entropy', 0.0)
            
        # Calculate final metrics
        total_return = np.sum(returns_array)
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'action_diversity': float(action_diversity),
            'mean_return': float(mean_return),
            'return_volatility': float(return_std),
            'num_episodes': len(episode_returns),
            'diversity_metrics': diversity_check.get('metrics', {}),
            'training_failed': False
        }
        
    def _calculate_enhanced_objective(self,
                                    results: Dict[str, Any],
                                    monitor: ActionDiversityMonitor,
                                    trial: optuna.Trial) -> float:
        """
        Calculate enhanced objective function with diversity bonus
        
        Args:
            results: Training results
            monitor: Action diversity monitor
            trial: Optuna trial
            
        Returns:
            Enhanced objective value
        """
        if results.get('training_failed', False):
            return -10.0
            
        # Base objective (Sharpe ratio)
        base_sharpe = results.get('sharpe_ratio', -10.0)
        
        # Handle infinite/invalid Sharpe ratios
        if not np.isfinite(base_sharpe) or base_sharpe > 100:
            base_sharpe = -5.0
            
        # Action diversity bonus
        diversity_score = results.get('action_diversity', 0.0)
        diversity_bonus = min(2.0, diversity_score * 3.0)  # Max bonus of 2.0
        
        # Conservative penalty
        diversity_metrics = results.get('diversity_metrics', {})
        hold_ratio = diversity_metrics.get('hold_ratio', 0.5)
        buy_ratio = diversity_metrics.get('buy_ratio', 0.3)
        
        # Penalty for excessive conservatism
        conservatism_penalty = 0.0
        if hold_ratio > 0.7:  # More than 70% hold
            conservatism_penalty = (hold_ratio - 0.7) * 10.0
            
        # Penalty for never buying
        if buy_ratio < 0.05:  # Less than 5% buy actions
            conservatism_penalty += 2.0
            
        # Return penalty
        if results.get('total_return', 0) <= 0:
            conservatism_penalty += 5.0
            
        # Combined objective
        enhanced_objective = base_sharpe + diversity_bonus - conservatism_penalty
        
        self.logger.info(f"   📊 Objective calculation:")
        self.logger.info(f"      Base Sharpe: {base_sharpe:.3f}")
        self.logger.info(f"      Diversity bonus: {diversity_bonus:.3f}")
        self.logger.info(f"      Conservatism penalty: {conservatism_penalty:.3f}")
        self.logger.info(f"      Final objective: {enhanced_objective:.3f}")
        
        return float(enhanced_objective)
        
    def _save_trial_results(self,
                           trial: optuna.Trial,
                           params: Dict[str, Any],
                           results: Dict[str, Any],
                           monitor: ActionDiversityMonitor,
                           objective_value: float):
        """Save detailed trial results"""
        
        trial_info = {
            'trial_number': trial.number,
            'objective_value': objective_value,
            'parameters': params,
            'results': results,
            'diversity_metrics': monitor.check_diversity(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save individual trial
        trial_file = os.path.join(self.results_dir, f"trial_{trial.number}_results.json")
        with open(trial_file, 'w') as f:
            json.dump(trial_info, f, indent=2)
            
        # Update best trial if needed
        if self.best_trial_info is None or objective_value > self.best_trial_info['objective_value']:
            self.best_trial_info = trial_info
            
        # Add to trial results
        self.trial_results.append(trial_info)
        
        # Save diversity plots
        try:
            plot_path = os.path.join(self.results_dir, f"trial_{trial.number}_diversity.png")
            monitor.plot_diversity_metrics(plot_path)
        except Exception as e:
            self.logger.warning(f"Could not save diversity plot: {e}")
            
    def run_optimization(self) -> optuna.Study:
        """
        Run the enhanced HPO optimization
        
        Returns:
            Completed Optuna study
        """
        self.logger.info(f"🚀 Starting Enhanced HPO Optimization")
        self.logger.info(f"   Study: {self.study_name}")
        self.logger.info(f"   Trials: {self.n_trials}")
        self.logger.info(f"   Results dir: {self.results_dir}")
        
        # Create study
        study = self.hpo_config.create_study()
        
        # Add study start time
        start_time = time.time()
        
        try:
            # Run optimization
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=None,  # No timeout, let it run
                show_progress_bar=True
            )
            
            # Calculate total time
            total_time = time.time() - start_time
            
            self.logger.info(f"✅ HPO Optimization completed!")
            self.logger.info(f"   Total time: {total_time/3600:.2f} hours")
            self.logger.info(f"   Best value: {study.best_value:.6f}")
            self.logger.info(f"   Best trial: {study.best_trial.number}")
            
            # Save comprehensive results
            self.results_manager.save_study_results(study, "enhanced_conservative_solution")
            self.results_manager.generate_optimization_report(study, "enhanced_conservative_solution")
            
            # Save additional analysis
            self._save_comprehensive_analysis(study)
            
            return study
            
        except Exception as e:
            self.logger.error(f"❌ HPO Optimization failed: {str(e)}")
            raise
            
    def _save_comprehensive_analysis(self, study: optuna.Study):
        """Save comprehensive analysis of the optimization"""
        
        # Collect all trial data
        analysis = {
            'study_summary': {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'best_trial_number': study.best_trial.number
            },
            'parameter_importance': {},
            'trial_results': self.trial_results,
            'best_trial_info': self.best_trial_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate parameter importance if enough trials
        if len(study.trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                analysis['parameter_importance'] = importance
            except Exception as e:
                self.logger.warning(f"Could not calculate parameter importance: {e}")
                
        # Save analysis
        analysis_file = os.path.join(self.results_dir, "comprehensive_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        self.logger.info(f"📊 Comprehensive analysis saved to {analysis_file}")


def main():
    """Main execution function"""
    print("🚀 Enhanced HPO Runner for Conservative Trading Solution")
    print("=" * 60)
    
    # Configuration
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print(f"   GPU ID: {gpu_id}")
    print(f"   Number of trials: {n_trials}")
    
    # Create optimizer
    optimizer = EnhancedHPOOptimizer(
        n_trials=n_trials,
        study_name=f"enhanced_conservative_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        gpu_id=gpu_id,
        results_dir=f"enhanced_hpo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run optimization
    try:
        study = optimizer.run_optimization()
        
        print("\n✅ Enhanced HPO Optimization completed!")
        print(f"   Best objective value: {study.best_value:.6f}")
        print(f"   Best parameters:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")
            
        print(f"\n📊 Results saved to: {optimizer.results_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()