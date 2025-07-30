"""
Training script with aggressive hyperparameters for profit maximization
Implements Phase 3 of the profitability enhancement plan
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch as th
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aggressive_training.log')
    ]
)
logger = logging.getLogger(__name__)

def load_aggressive_config():
    """Load the aggressive profit configuration"""
    config_path = Path(__file__).parent.parent / "configs" / "aggressive_profit_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def prepare_reduced_features(data_path):
    """Load and prepare the reduced feature set"""
    try:
        # Try loading reduced features first
        reduced_path = data_path.parent / "BTC_1sec_predict_reduced.npy"
        if reduced_path.exists():
            logger.info(f"Loading reduced features from {reduced_path}")
            features = np.load(reduced_path)
            logger.info(f"Loaded reduced features with shape: {features.shape}")
            return features
    except Exception as e:
        logger.warning(f"Could not load reduced features: {e}")
    
    # Fallback to original features
    logger.info("Loading original features as fallback")
    features = np.load(data_path)
    
    # If original has more than 15 features, select first 15
    if features.shape[1] > 15:
        logger.warning(f"Original features have {features.shape[1]} columns, selecting first 15")
        features = features[:, :15]
    
    return features

def setup_aggressive_agent(config, env):
    """Initialize agent with aggressive hyperparameters"""
    from elegantrl.agents.AgentPPO import AgentPPO
    
    # Extract agent config
    agent_config = config["agent_config"]
    
    # Create args object for agent
    class Args:
        def __init__(self):
            # Network architecture
            self.net_dims = agent_config["net_dims"]
            self.learning_rate = agent_config["learning_rate"]
            self.batch_size = agent_config["batch_size"]
            self.horizon_len = agent_config["horizon_len"]
            self.buffer_size = agent_config["buffer_size"]
            
            # Exploration
            self.explore_rate = agent_config["explore_rate"]
            self.explore_decay = agent_config["explore_decay"]
            self.explore_min = agent_config["explore_min"]
            
            # PPO specific
            self.gamma = agent_config.get("gamma", 0.995)
            self.lambda_gae = agent_config.get("lambda_gae", 0.97)
            self.entropy_coef = agent_config.get("entropy_coef", 0.02)
            
            # Optimization
            self.clip_grad_norm = agent_config["clip_grad_norm"]
            self.soft_update_tau = agent_config["soft_update_tau"]
            self.weight_decay = agent_config.get("weight_decay", 1e-5)
            
            # Learning rate scheduling
            self.lr_scheduler = agent_config.get("lr_scheduler", None)
            self.lr_warmup_steps = agent_config.get("lr_warmup_steps", 1000)
            
            # Device
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
            
            # Environment
            self.env_name = "AggressiveProfitTrading"
            self.state_dim = env.state_dim
            self.action_dim = env.action_dim
            self.if_discrete = env.if_discrete
            
            # Training
            self.max_step = config["training_config"].get("max_steps_per_episode", 20000)
            self.eval_times = config["training_config"].get("eval_episodes", 5)
            
            # Logging
            self.if_use_per = False
            self.if_off_policy = False
    
    args = Args()
    
    # Initialize agent
    agent = AgentPPO(args.net_dims, args.state_dim, args.action_dim, args)
    agent.learning_rate = args.learning_rate
    
    # Set aggressive exploration
    if hasattr(agent, 'explore_rate'):
        agent.explore_rate = args.explore_rate
    
    logger.info(f"Initialized aggressive agent with:")
    logger.info(f"  - Network: {args.net_dims}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Exploration: {args.explore_rate} → {args.explore_min}")
    logger.info(f"  - Device: {args.device}")
    
    return agent, args

def create_profit_focused_env(config, features):
    """Create environment with profit-focused rewards"""
    from task1_optuna_hpo import LOBEnvironment
    from reward_functions import create_reward_calculator
    
    # Create base environment
    env_config = config["environment_config"]
    env = LOBEnvironment(
        data=features,
        max_position=env_config["max_position"],
        lookback_window=10,
        step_gap=env_config["step_gap"],
        delay_step=env_config["delay_step"]
    )
    
    # Set transaction costs
    env.slippage = env_config["slippage"]
    env.transaction_cost = env_config["transaction_cost"]
    
    # Create profit-focused reward calculator
    reward_calc = create_reward_calculator(
        reward_type="profit_focused",
        device="cuda" if th.cuda.is_available() else "cpu"
    )
    
    # Override environment's reward calculation
    original_step = env.step
    
    def profit_focused_step(action):
        """Step with profit-focused rewards"""
        # Get original step results
        state, reward, done, info = original_step(action)
        
        # Recalculate reward using profit-focused calculator
        if hasattr(env, 'previous_price') and hasattr(env, 'current_price'):
            # Calculate profit-focused reward
            old_asset = th.tensor([env.previous_total_value])
            new_asset = th.tensor([env.current_total_value])
            action_tensor = th.tensor([action])
            price_tensor = th.tensor([env.current_price])
            
            profit_reward = reward_calc.calculate_reward(
                old_asset, new_asset, action_tensor, 
                price_tensor, env.slippage
            )
            
            # Use profit-focused reward
            reward = profit_reward.item()
        
        return state, reward, done, info
    
    # Replace step method
    env.step = profit_focused_step
    
    logger.info("Created profit-focused environment with aggressive rewards")
    
    return env

def train_aggressive_model(config):
    """Main training loop with aggressive hyperparameters"""
    logger.info("Starting aggressive profit-focused training")
    logger.info(f"Configuration: {config['experiment_name']}")
    
    # Load data
    data_dir = Path(__file__).parent.parent / "task1_data"
    features = prepare_reduced_features(data_dir / "BTC_1sec_predict.npy")
    
    # Split data for training
    train_config = config["training_config"]
    num_episodes = train_config["num_episodes"]
    samples_per_episode = train_config["samples_per_episode"]
    
    # Calculate split
    total_samples = features.shape[0]
    train_samples = int(total_samples * 0.8)
    
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(f"Training samples: {train_samples:,}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Samples per episode: {samples_per_episode:,}")
    
    # Create environment
    env = create_profit_focused_env(config, features[:train_samples])
    
    # Setup agent
    agent, args = setup_aggressive_agent(config, env)
    
    # Training metrics
    episode_rewards = []
    episode_profits = []
    episode_sharpe_ratios = []
    best_sharpe = -np.inf
    
    # Create results directory
    results_dir = Path(__file__).parent.parent / "aggressive_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Training loop
    for episode in range(num_episodes):
        # Random episode start for diversity
        max_start = train_samples - samples_per_episode - 1000
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + samples_per_episode
        
        # Create episode environment
        episode_features = features[start_idx:end_idx]
        episode_env = create_profit_focused_env(config, episode_features)
        
        # Collect experience
        logger.info(f"\nEpisode {episode + 1}/{num_episodes}")
        logger.info(f"Data range: {start_idx:,} to {end_idx:,}")
        
        # Reset environment
        state = episode_env.reset()
        episode_reward = 0
        episode_trades = 0
        episode_wins = 0
        step = 0
        
        # Episode loop
        while step < args.max_step:
            # Get action from agent
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = episode_env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update metrics
            episode_reward += reward
            if info.get('trade_completed', False):
                episode_trades += 1
                if info.get('trade_return', 0) > 0:
                    episode_wins += 1
            
            # Update state
            state = next_state
            step += 1
            
            # Update agent if buffer ready
            if step % args.horizon_len == 0:
                agent.update_net()
                
                # Decay exploration
                if hasattr(agent, 'explore_rate'):
                    agent.explore_rate = max(
                        args.explore_min,
                        agent.explore_rate * args.explore_decay
                    )
            
            if done:
                break
        
        # Calculate episode metrics
        win_rate = episode_wins / max(episode_trades, 1)
        final_value = episode_env.current_total_value
        initial_value = episode_env.initial_total_value
        episode_return = (final_value - initial_value) / initial_value
        
        # Simple Sharpe approximation
        if len(episode_rewards) > 10:
            returns = np.array(episode_rewards[-10:])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        
        episode_rewards.append(episode_reward)
        episode_profits.append(episode_return)
        episode_sharpe_ratios.append(sharpe)
        
        logger.info(f"Episode Summary:")
        logger.info(f"  - Total reward: {episode_reward:.4f}")
        logger.info(f"  - Portfolio return: {episode_return:.4%}")
        logger.info(f"  - Trades: {episode_trades}, Win rate: {win_rate:.2%}")
        logger.info(f"  - Sharpe ratio: {sharpe:.3f}")
        logger.info(f"  - Exploration rate: {agent.explore_rate:.4f}")
        
        # Evaluation and checkpointing
        if (episode + 1) % train_config["eval_frequency"] == 0:
            logger.info("\nRunning evaluation...")
            
            # Evaluate on validation data
            eval_rewards = []
            eval_returns = []
            
            for eval_ep in range(train_config["eval_episodes"]):
                eval_start = train_samples + eval_ep * 5000
                eval_end = eval_start + 5000
                eval_features = features[eval_start:eval_end]
                
                eval_env = create_profit_focused_env(config, eval_features)
                eval_state = eval_env.reset()
                eval_reward = 0
                
                while True:
                    eval_action = agent.select_action(eval_state, if_train=False)
                    eval_state, reward, done, _ = eval_env.step(eval_action)
                    eval_reward += reward
                    if done:
                        break
                
                eval_rewards.append(eval_reward)
                eval_returns.append((eval_env.current_total_value - eval_env.initial_total_value) / eval_env.initial_total_value)
            
            avg_eval_reward = np.mean(eval_rewards)
            avg_eval_return = np.mean(eval_returns)
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  - Avg reward: {avg_eval_reward:.4f}")
            logger.info(f"  - Avg return: {avg_eval_return:.4%}")
            
            # Save best model
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                agent.save_agent(results_dir / "best_model")
                logger.info(f"New best model saved! Sharpe: {best_sharpe:.3f}")
        
        # Regular checkpointing
        if (episode + 1) % train_config["save_frequency"] == 0:
            checkpoint_dir = results_dir / f"checkpoint_ep{episode + 1}"
            agent.save_agent(checkpoint_dir)
            
            # Save metrics
            metrics = {
                "episode_rewards": episode_rewards,
                "episode_profits": episode_profits,
                "episode_sharpe_ratios": episode_sharpe_ratios,
                "best_sharpe": best_sharpe
            }
            
            with open(results_dir / "training_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Early stopping check
        if len(episode_profits) > train_config["early_stopping_patience"]:
            recent_profits = episode_profits[-train_config["early_stopping_patience"]:]
            if all(p > train_config["min_profit_threshold"] for p in recent_profits[-5:]):
                logger.info("Early stopping: Consistent profitability achieved!")
                break
    
    # Final save
    agent.save_agent(results_dir / "final_model")
    
    # Generate summary report
    summary = {
        "experiment": config["experiment_name"],
        "total_episodes": len(episode_rewards),
        "final_sharpe": episode_sharpe_ratios[-1],
        "best_sharpe": best_sharpe,
        "avg_profit": np.mean(episode_profits),
        "profit_episodes": sum(1 for p in episode_profits if p > 0),
        "final_exploration": agent.explore_rate
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Best Sharpe Ratio: {best_sharpe:.3f}")
    logger.info(f"Profitable Episodes: {summary['profit_episodes']}/{summary['total_episodes']}")
    logger.info(f"Average Profit: {summary['avg_profit']:.4%}")
    logger.info("="*60)
    
    return results_dir

if __name__ == "__main__":
    # Load configuration
    config = load_aggressive_config()
    
    # Run training
    results_dir = train_aggressive_model(config)
    
    print(f"\nTraining complete! Results saved to: {results_dir}")