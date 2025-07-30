"""
Hyperparameter Optimization (HPO) for Profit Maximization
Phase 5 of the profitability enhancement plan
GPU-accelerated Optuna study for systematic parameter search
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch as th
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profit_focused_rewards import MetaRewardCalculator
from advanced_market_regime import AdvancedMarketRegimeDetector, RegimeAwareEnvironment
from reward_functions import create_reward_calculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure GPU usage
if not th.cuda.is_available():
    logger.warning("GPU not available! HPO will run on CPU (much slower)")
else:
    logger.info(f"GPU detected: {th.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {th.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class HPOObjective:
    """
    Objective function for HPO optimization
    Evaluates different parameter combinations
    """
    
    def __init__(self, 
                 data_path: Path,
                 num_episodes: int = 10,
                 samples_per_episode: int = 5000,
                 device: str = "cuda",
                 use_regime_detection: bool = True):
        """
        Initialize HPO objective
        
        Args:
            data_path: Path to feature data
            num_episodes: Episodes per trial
            samples_per_episode: Samples per episode
            device: PyTorch device (cuda/cpu)
            use_regime_detection: Whether to use market regime detection
        """
        self.data_path = data_path
        self.num_episodes = num_episodes
        self.samples_per_episode = samples_per_episode
        self.device = device if th.cuda.is_available() else "cpu"
        self.use_regime_detection = use_regime_detection
        
        # Load data once
        self.features = self._load_features()
        self.train_samples = int(len(self.features) * 0.8)
        
        logger.info(f"HPO Objective initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Episodes per trial: {num_episodes}")
        logger.info(f"  Samples per episode: {samples_per_episode}")
        logger.info(f"  Total samples: {len(self.features)}")
        
    def _load_features(self) -> np.ndarray:
        """Load feature data"""
        # Try reduced features first
        reduced_path = self.data_path.parent / "BTC_1sec_predict_reduced.npy"
        if reduced_path.exists():
            features = np.load(reduced_path)
            logger.info(f"Loaded reduced features: {features.shape}")
        else:
            features = np.load(self.data_path)
            if features.shape[1] > 15:
                features = features[:, :15]
                logger.info(f"Truncated features to: {features.shape}")
        
        return features
        
    def __call__(self, trial: Trial) -> float:
        """
        Evaluate a trial with suggested parameters
        
        Returns:
            Negative Sharpe ratio (for minimization)
        """
        # Suggest hyperparameters
        params = self._suggest_parameters(trial)
        
        # Create environment and agent
        env = self._create_environment(params)
        agent = self._create_agent(params, env)
        
        # Run training episodes
        episode_returns = []
        episode_sharpes = []
        
        for episode in range(self.num_episodes):
            # Random start for diversity
            start_idx = np.random.randint(0, self.train_samples - self.samples_per_episode)
            end_idx = start_idx + self.samples_per_episode
            
            # Run episode
            episode_return, episode_sharpe = self._run_episode(
                agent, env, 
                self.features[start_idx:end_idx],
                episode, trial
            )
            
            episode_returns.append(episode_return)
            episode_sharpes.append(episode_sharpe)
            
            # Report intermediate results for pruning
            trial.report(episode_sharpe, episode)
            
            # Check if trial should be pruned
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at episode {episode}")
                raise optuna.TrialPruned()
        
        # Calculate final metrics
        avg_return = np.mean(episode_returns)
        avg_sharpe = np.mean(episode_sharpes)
        
        # Log trial results
        logger.info(f"Trial {trial.number} completed:")
        logger.info(f"  Avg return: {avg_return:.4%}")
        logger.info(f"  Avg Sharpe: {avg_sharpe:.3f}")
        
        # Return negative Sharpe for minimization
        return -avg_sharpe
        
    def _suggest_parameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for trial"""
        params = {}
        
        # Reward function parameters
        params['reward_weights'] = {
            'profit_amplifier': trial.suggest_float('profit_amplifier', 2.0, 10.0),
            'loss_multiplier': trial.suggest_float('loss_multiplier', 0.5, 1.5),
            'trade_completion_bonus': trial.suggest_float('trade_completion_bonus', 0.01, 0.05),
            'opportunity_cost_penalty': trial.suggest_float('opportunity_cost_penalty', 0.0005, 0.005),
            'momentum_bonus': trial.suggest_float('momentum_bonus', 0.2, 1.0),
            'action_bonus': trial.suggest_float('action_bonus', 0.0005, 0.002),
            'blend_factor': trial.suggest_float('blend_factor', 0.6, 0.95)
        }
        
        # Profit speed parameters
        params['profit_speed'] = {
            'enabled': True,  # Always enabled
            'max_speed_multiplier': trial.suggest_float('max_speed_multiplier', 3.0, 10.0),
            'speed_decay_rate': trial.suggest_float('speed_decay_rate', 0.01, 0.05),
            'min_holding_time': trial.suggest_int('min_holding_time', 3, 10)
        }
        
        # Agent hyperparameters
        params['agent'] = {
            'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            'horizon_len': trial.suggest_categorical('horizon_len', [1024, 2048, 4096]),
            'explore_rate': trial.suggest_float('explore_rate', 0.05, 0.2),
            'explore_decay': trial.suggest_float('explore_decay', 0.98, 0.995),
            'explore_min': trial.suggest_float('explore_min', 0.005, 0.02),
            'clip_grad_norm': trial.suggest_float('clip_grad_norm', 1.0, 10.0),
            'soft_update_tau': trial.suggest_float('soft_update_tau', 0.001, 0.02),
            'gamma': trial.suggest_float('gamma', 0.99, 0.999),
            'lambda_gae': trial.suggest_float('lambda_gae', 0.9, 0.99),
            'entropy_coef': trial.suggest_float('entropy_coef', 0.001, 0.1, log=True)
        }
        
        # Network architecture
        network_type = trial.suggest_categorical('network_type', ['medium', 'large', 'xlarge'])
        if network_type == 'medium':
            params['agent']['net_dims'] = [256, 256, 256]
        elif network_type == 'large':
            params['agent']['net_dims'] = [512, 512, 256]
        else:  # xlarge
            params['agent']['net_dims'] = [512, 512, 512, 256]
        
        # Environment parameters
        params['environment'] = {
            'max_position': trial.suggest_int('max_position', 2, 5),
            'transaction_cost': trial.suggest_float('transaction_cost', 0.0005, 0.002),
            'slippage': trial.suggest_float('slippage', 1e-5, 1e-4, log=True),
            'max_holding_time': trial.suggest_int('max_holding_time', 600, 3600, step=300)
        }
        
        # Regime detection parameters (if enabled)
        if self.use_regime_detection:
            params['regime'] = {
                'short_lookback': trial.suggest_int('regime_short_lookback', 10, 30),
                'medium_lookback': trial.suggest_int('regime_medium_lookback', 30, 70),
                'long_lookback': trial.suggest_int('regime_long_lookback', 70, 150)
            }
        
        return params
        
    def _create_environment(self, params: Dict) -> Any:
        """Create trading environment with suggested parameters"""
        from trading_environment import LOBEnvironment
        
        # Base environment
        env_params = params['environment']
        base_env = LOBEnvironment(
            data=self.features[:self.train_samples],
            max_position=env_params['max_position'],
            lookback_window=10,
            step_gap=1,
            delay_step=1
        )
        
        base_env.transaction_cost = env_params['transaction_cost']
        base_env.slippage = env_params['slippage']
        
        # Add regime detection if enabled
        if self.use_regime_detection:
            regime_params = params['regime']
            regime_detector = AdvancedMarketRegimeDetector(
                short_lookback=regime_params['short_lookback'],
                medium_lookback=regime_params['medium_lookback'],
                long_lookback=regime_params['long_lookback'],
                device=self.device
            )
            env = RegimeAwareEnvironment(base_env, regime_detector)
        else:
            env = base_env
            
        return env
        
    def _create_agent(self, params: Dict, env: Any) -> Any:
        """Create agent with suggested parameters"""
        from elegantrl.agents.AgentPPO import AgentPPO
        
        agent_params = params['agent']
        
        # Create args object
        class Args:
            def __init__(self):
                self.net_dims = agent_params['net_dims']
                self.learning_rate = agent_params['learning_rate']
                self.batch_size = agent_params['batch_size']
                self.horizon_len = agent_params['horizon_len']
                self.gamma = agent_params['gamma']
                self.lambda_gae = agent_params['lambda_gae']
                self.entropy_coef = agent_params['entropy_coef']
                self.clip_grad_norm = agent_params['clip_grad_norm']
                self.soft_update_tau = agent_params['soft_update_tau']
                self.device = th.device(self.device)
                self.state_dim = env.state_dim
                self.action_dim = env.action_dim
                self.if_discrete = env.if_discrete
                self.max_step = 10000
                self.if_off_policy = False
                
        args = Args()
        agent = AgentPPO(args.net_dims, args.state_dim, args.action_dim, args)
        
        # Set exploration parameters
        agent.explore_rate = agent_params['explore_rate']
        
        # Create profit-focused reward calculator
        self._setup_reward_calculator(env, params)
        
        return agent
        
    def _setup_reward_calculator(self, env: Any, params: Dict) -> None:
        """Setup profit-focused reward calculator"""
        reward_weights = params['reward_weights']
        profit_speed = params['profit_speed']
        
        # Create meta reward calculator
        from profit_focused_rewards import integrate_profit_rewards
        
        # Create base calculator
        reward_calc = create_reward_calculator(
            reward_type="multi_objective",
            device=self.device
        )
        
        # Integrate profit-focused rewards
        reward_calc = integrate_profit_rewards(reward_calc)
        
        # Update parameters
        if hasattr(reward_calc, 'profit_calculator'):
            calc = reward_calc.profit_calculator
            calc.profit_amplifier = reward_weights['profit_amplifier']
            calc.loss_multiplier = reward_weights['loss_multiplier']
            calc.trade_completion_bonus = reward_weights['trade_completion_bonus']
            calc.opportunity_cost_penalty = reward_weights['opportunity_cost_penalty']
            calc.profit_speed_enabled = profit_speed['enabled']
            calc.max_speed_multiplier = profit_speed['max_speed_multiplier']
            calc.speed_decay_rate = profit_speed['speed_decay_rate']
            calc.min_holding_time = profit_speed['min_holding_time']
        
        # Override environment reward calculation
        original_step = env.step
        
        def profit_step(action):
            state, reward, done, info = original_step(action)
            
            # Recalculate reward
            if hasattr(env, 'previous_total_value'):
                old_asset = th.tensor([env.previous_total_value], device=self.device)
                new_asset = th.tensor([env.current_total_value], device=self.device)
                action_tensor = th.tensor([action], device=self.device)
                price_tensor = th.tensor([env.current_price], device=self.device)
                
                new_reward = reward_calc.calculate_reward(
                    old_asset, new_asset, action_tensor,
                    price_tensor, env.slippage
                )
                reward = new_reward.item()
            
            return state, reward, done, info
        
        env.step = profit_step
        
    def _run_episode(self, agent: Any, env: Any, 
                    episode_data: np.ndarray,
                    episode_num: int, trial: Trial) -> Tuple[float, float]:
        """Run single training episode"""
        # Update environment data
        env.data = episode_data
        if hasattr(env, 'env'):  # RegimeAwareEnvironment
            env.env.data = episode_data
        
        # Reset environment
        state = env.reset()
        
        # Episode metrics
        episode_reward = 0
        episode_returns = []
        actions_taken = []
        
        # Training loop
        step = 0
        max_steps = min(len(episode_data) - 100, 10000)
        
        while step < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update metrics
            episode_reward += reward
            actions_taken.append(action)
            
            if hasattr(env, 'current_total_value'):
                current_return = (env.current_total_value - env.initial_total_value) / env.initial_total_value
                episode_returns.append(current_return)
            
            # Update agent
            if step % agent.horizon_len == 0:
                agent.update_net()
                
                # Decay exploration
                if hasattr(agent, 'explore_rate'):
                    decay = trial.params['explore_decay']
                    min_rate = trial.params['explore_min']
                    agent.explore_rate = max(min_rate, agent.explore_rate * decay)
            
            state = next_state
            step += 1
            
            if done:
                break
        
        # Calculate episode metrics
        final_return = episode_returns[-1] if episode_returns else 0
        
        # Simple Sharpe calculation
        if len(episode_returns) > 10:
            returns_array = np.array(episode_returns)
            returns_diff = np.diff(returns_array)
            if np.std(returns_diff) > 0:
                sharpe = np.mean(returns_diff) / np.std(returns_diff) * np.sqrt(252 * 86400)
            else:
                sharpe = 0
        else:
            sharpe = 0
            
        # Action diversity bonus
        action_counts = np.bincount(actions_taken, minlength=3)
        action_diversity = 1 - np.max(action_counts) / len(actions_taken)
        sharpe += action_diversity * 0.1  # Small bonus for diversity
        
        return final_return, sharpe


def run_hpo_study(
    data_path: Path,
    n_trials: int = 100,
    n_jobs: int = 1,
    study_name: str = "profit_maximization_hpo",
    use_gpu: bool = True) -> optuna.Study:
    """
    Run HPO study for profit maximization
    
    Args:
        data_path: Path to feature data
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (use 1 for GPU)
        study_name: Name of the study
        use_gpu: Whether to use GPU
        
    Returns:
        Completed Optuna study
    """
    logger.info("="*60)
    logger.info("Starting HPO Study for Profit Maximization")
    logger.info("="*60)
    
    # Device setup
    device = "cuda" if use_gpu and th.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Using GPU: {th.cuda.get_device_name(0)}")
        # Set GPU memory growth
        th.cuda.empty_cache()
    else:
        logger.warning("Running on CPU - this will be slow!")
    
    # Create objective
    objective = HPOObjective(
        data_path=data_path,
        num_episodes=5,  # Reduced for faster HPO
        samples_per_episode=5000,
        device=device,
        use_regime_detection=True
    )
    
    # Create study
    sampler = TPESampler(seed=42)
    pruner = HyperbandPruner(min_resource=1, max_resource=5, reduction_factor=3)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # Minimizing negative Sharpe
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True
    )
    
    # Add default trial (current best known parameters)
    study.enqueue_trial({
        'profit_amplifier': 5.0,
        'loss_multiplier': 0.8,
        'trade_completion_bonus': 0.03,
        'opportunity_cost_penalty': 0.002,
        'momentum_bonus': 0.5,
        'action_bonus': 0.001,
        'blend_factor': 0.85,
        'max_speed_multiplier': 7.0,
        'speed_decay_rate': 0.015,
        'min_holding_time': 3,
        'learning_rate': 0.0001,
        'batch_size': 256,
        'horizon_len': 2048,
        'explore_rate': 0.15,
        'explore_decay': 0.99,
        'explore_min': 0.005,
        'clip_grad_norm': 5.0,
        'soft_update_tau': 0.01,
        'gamma': 0.995,
        'lambda_gae': 0.97,
        'entropy_coef': 0.02,
        'network_type': 'large',
        'max_position': 3,
        'transaction_cost': 0.0008,
        'slippage': 3e-5,
        'max_holding_time': 1800,
        'regime_short_lookback': 20,
        'regime_medium_lookback': 50,
        'regime_long_lookback': 100
    })
    
    # Run optimization
    logger.info(f"Running {n_trials} trials with {n_jobs} parallel jobs")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,  # Use 1 for GPU to avoid memory issues
        show_progress_bar=True,
        gc_after_trial=True  # Important for GPU memory
    )
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("HPO Study Completed!")
    logger.info("="*60)
    
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best Sharpe: {-study.best_value:.3f}")
    logger.info("\nBest parameters:")
    
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    return study


def analyze_hpo_results(study: optuna.Study, output_dir: Path) -> Dict[str, Any]:
    """
    Analyze and visualize HPO results
    
    Args:
        study: Completed Optuna study
        output_dir: Directory to save results
        
    Returns:
        Analysis results dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract results
    results = {
        'best_trial': study.best_trial.number,
        'best_sharpe': -study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        results['param_importance'] = importance
        
        # Save importance plot
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(output_dir / "param_importance.html")
        
        logger.info("\nParameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {param}: {imp:.3f}")
    except Exception as e:
        logger.warning(f"Could not calculate parameter importance: {e}")
    
    # Optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(output_dir / "optimization_history.html")
    except Exception as e:
        logger.warning(f"Could not create optimization history plot: {e}")
    
    # Parallel coordinate plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(output_dir / "parallel_coordinate.html")
    except Exception as e:
        logger.warning(f"Could not create parallel coordinate plot: {e}")
    
    # Save best parameters
    best_params_path = output_dir / "best_parameters.json"
    with open(best_params_path, 'w') as f:
        json.dump(results['best_params'], f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Best parameters: {best_params_path}")
    
    return results


def create_production_config(best_params: Dict[str, Any], output_path: Path) -> None:
    """
    Create production configuration from best HPO parameters
    
    Args:
        best_params: Best parameters from HPO
        output_path: Path to save configuration
    """
    config = {
        "experiment_name": "hpo_optimized_production",
        "description": "Production configuration optimized by HPO",
        
        "reward_config": {
            "reward_type": "profit_focused",
            "profit_amplifier": best_params.get('profit_amplifier', 5.0),
            "loss_multiplier": best_params.get('loss_multiplier', 0.8),
            "trade_completion_bonus": best_params.get('trade_completion_bonus', 0.03),
            "opportunity_cost_penalty": best_params.get('opportunity_cost_penalty', 0.002),
            "blend_factor": best_params.get('blend_factor', 0.85),
            "regime_sensitivity": True,
            "profit_speed_enabled": True,
            "max_speed_multiplier": best_params.get('max_speed_multiplier', 7.0),
            "speed_decay_rate": best_params.get('speed_decay_rate', 0.015),
            "min_holding_time": best_params.get('min_holding_time', 3)
        },
        
        "agent_config": {
            "net_dims": {
                'medium': [256, 256, 256],
                'large': [512, 512, 256],
                'xlarge': [512, 512, 512, 256]
            }.get(best_params.get('network_type', 'large'), [512, 512, 256]),
            "learning_rate": best_params.get('learning_rate', 1e-4),
            "batch_size": best_params.get('batch_size', 256),
            "horizon_len": best_params.get('horizon_len', 2048),
            "explore_rate": best_params.get('explore_rate', 0.15),
            "explore_decay": best_params.get('explore_decay', 0.99),
            "explore_min": best_params.get('explore_min', 0.005),
            "clip_grad_norm": best_params.get('clip_grad_norm', 5.0),
            "soft_update_tau": best_params.get('soft_update_tau', 0.01),
            "gamma": best_params.get('gamma', 0.995),
            "lambda_gae": best_params.get('lambda_gae', 0.97),
            "entropy_coef": best_params.get('entropy_coef', 0.02)
        },
        
        "environment_config": {
            "max_position": best_params.get('max_position', 3),
            "slippage": best_params.get('slippage', 3e-5),
            "transaction_cost": best_params.get('transaction_cost', 0.0008),
            "max_holding_time": best_params.get('max_holding_time', 1800)
        },
        
        "regime_config": {
            "short_lookback": best_params.get('regime_short_lookback', 20),
            "medium_lookback": best_params.get('regime_medium_lookback', 50),
            "long_lookback": best_params.get('regime_long_lookback', 100)
        },
        
        "training_config": {
            "num_episodes": 100,
            "samples_per_episode": 15000,
            "use_gpu": True
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Production configuration saved to: {output_path}")


if __name__ == "__main__":
    # Setup paths
    data_dir = Path(__file__).parent.parent / "task1_data"
    data_path = data_dir / "BTC_1sec_predict.npy"
    
    # Check GPU
    if not th.cuda.is_available():
        logger.error("GPU not available! HPO requires GPU for reasonable performance.")
        logger.error("Please run on a machine with CUDA-capable GPU.")
        sys.exit(1)
    
    # Run HPO study
    study = run_hpo_study(
        data_path=data_path,
        n_trials=50,  # Adjust based on time/resources
        n_jobs=1,     # Use 1 for GPU to avoid memory issues
        study_name="profit_maximization_hpo",
        use_gpu=True
    )
    
    # Analyze results
    results_dir = Path(__file__).parent.parent / "hpo_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results = analyze_hpo_results(study, results_dir)
    
    # Create production config
    prod_config_path = results_dir / "production_config.json"
    create_production_config(study.best_params, prod_config_path)
    
    print("\n" + "="*60)
    print("HPO Optimization Complete!")
    print(f"Best Sharpe Ratio: {results['best_sharpe']:.3f}")
    print(f"Results saved to: {results_dir}")
    print("="*60)