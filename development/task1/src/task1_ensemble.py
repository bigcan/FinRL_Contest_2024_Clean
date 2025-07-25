import os
import time
import torch
import numpy as np
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter

from metrics import *
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class TrainingLogger:
    """Enhanced training progress visualization and logging"""
    
    def __init__(self, save_path, agent_name):
        self.save_path = save_path
        self.agent_name = agent_name
        self.log_dir = os.path.join(save_path, "training_logs", agent_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Training metrics storage
        self.training_metrics = {
            'steps': [],
            'episode_rewards': [],
            'action_counts': [],
            'position_counts': [],
            'losses': [],
            'exploration_rates': [],
            'evaluation_scores': [],
            'timestamps': []
        }
        
        # Real-time plotting setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f'Training Progress: {agent_name}', fontsize=16)
        plt.ion()  # Interactive mode
        
        # CSV logging
        self.csv_path = os.path.join(self.log_dir, 'training_metrics.csv')
        
    def log_training_step(self, step, exp_r, action_count, position_count, logging_tuple=None, eval_score=None):
        """Log metrics for a training step"""
        
        timestamp = datetime.now()
        
        # Store metrics
        self.training_metrics['steps'].append(step)
        self.training_metrics['episode_rewards'].append(exp_r)
        self.training_metrics['action_counts'].append(action_count.tolist() if hasattr(action_count, 'tolist') else action_count)
        self.training_metrics['position_counts'].append(position_count.tolist() if hasattr(position_count, 'tolist') else position_count)
        self.training_metrics['timestamps'].append(timestamp)
        
        # Extract loss information from logging_tuple if available
        if logging_tuple and len(logging_tuple) > 0:
            loss = logging_tuple[0] if isinstance(logging_tuple[0], (int, float)) else 0.0
            self.training_metrics['losses'].append(loss)
        else:
            self.training_metrics['losses'].append(0.0)
            
        # Store evaluation score
        if eval_score is not None:
            self.training_metrics['evaluation_scores'].append(eval_score)
        else:
            self.training_metrics['evaluation_scores'].append(None)
        
        # Update real-time plots every 10 steps
        if step % 10 == 0:
            self.update_plots()
            
        # Save to CSV every 50 steps
        if step % 50 == 0:
            self.save_to_csv()
    
    def update_plots(self):
        """Update real-time training plots"""
        
        try:
            steps = self.training_metrics['steps']
            rewards = self.training_metrics['episode_rewards']
            losses = self.training_metrics['losses']
            
            if len(steps) < 2:
                return
                
            # Clear and update plots
            for ax in self.axes.flat:
                ax.clear()
            
            # Plot 1: Episode Rewards
            self.axes[0, 0].plot(steps, rewards, 'b-', alpha=0.7, linewidth=1)
            if len(rewards) > 20:
                # Add moving average
                rewards_ma = pd.Series(rewards).rolling(window=20).mean()
                self.axes[0, 0].plot(steps, rewards_ma, 'r-', linewidth=2, label='MA(20)')
                self.axes[0, 0].legend()
            self.axes[0, 0].set_title('Episode Rewards')
            self.axes[0, 0].set_xlabel('Steps')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Training Loss
            if any(l != 0 for l in losses):
                self.axes[0, 1].plot(steps, losses, 'g-', alpha=0.7, linewidth=1)
                if len(losses) > 20:
                    loss_ma = pd.Series(losses).rolling(window=20).mean()
                    self.axes[0, 1].plot(steps, loss_ma, 'r-', linewidth=2, label='MA(20)')
                    self.axes[0, 1].legend()
            self.axes[0, 1].set_title('Training Loss')
            self.axes[0, 1].set_xlabel('Steps')
            self.axes[0, 1].set_ylabel('Loss')
            self.axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Action Distribution (latest)
            if self.training_metrics['action_counts']:
                latest_actions = self.training_metrics['action_counts'][-1]
                if isinstance(latest_actions, list) and len(latest_actions) > 0:
                    action_labels = ['Sell', 'Hold', 'Buy'] if len(latest_actions) == 3 else [f'Action_{i}' for i in range(len(latest_actions))]
                    self.axes[1, 0].bar(action_labels, latest_actions, alpha=0.7)
                    self.axes[1, 0].set_title('Latest Action Distribution')
                    self.axes[1, 0].set_ylabel('Count')
                    
            # Plot 4: Position Distribution (latest)
            if self.training_metrics['position_counts']:
                latest_positions = self.training_metrics['position_counts'][-1]
                if isinstance(latest_positions, list) and len(latest_positions) > 0:
                    pos_labels = [f'Pos_{i-2}' for i in range(len(latest_positions))]
                    self.axes[1, 1].bar(pos_labels, latest_positions, alpha=0.7)
                    self.axes[1, 1].set_title('Latest Position Distribution')
                    self.axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.pause(0.01)  # Small pause for update
            
        except Exception as e:
            print(f"⚠️ Error updating plots: {e}")
    
    def save_to_csv(self):
        """Save training metrics to CSV"""
        
        try:
            # Prepare data for CSV
            csv_data = []
            for i in range(len(self.training_metrics['steps'])):
                row = {
                    'step': self.training_metrics['steps'][i],
                    'episode_reward': self.training_metrics['episode_rewards'][i],
                    'loss': self.training_metrics['losses'][i],
                    'timestamp': self.training_metrics['timestamps'][i],
                }
                
                # Add action counts
                if i < len(self.training_metrics['action_counts']):
                    actions = self.training_metrics['action_counts'][i]
                    if isinstance(actions, list):
                        for j, count in enumerate(actions):
                            row[f'action_{j}_count'] = count
                            
                # Add position counts
                if i < len(self.training_metrics['position_counts']):
                    positions = self.training_metrics['position_counts'][i]
                    if isinstance(positions, list):
                        for j, count in enumerate(positions):
                            row[f'position_{j}_count'] = count
                
                # Add evaluation score
                if i < len(self.training_metrics['evaluation_scores']) and self.training_metrics['evaluation_scores'][i] is not None:
                    row['evaluation_score'] = self.training_metrics['evaluation_scores'][i]
                
                csv_data.append(row)
            
            # Save to CSV
            df = pd.DataFrame(csv_data)
            df.to_csv(self.csv_path, index=False)
            
        except Exception as e:
            print(f"⚠️ Error saving CSV: {e}")
    
    def save_final_plots(self):
        """Save final training plots"""
        
        try:
            # Save the current figure
            plot_path = os.path.join(self.log_dir, 'training_progress.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Create summary plot
            self.create_summary_plot()
            
            print(f"✅ Training plots saved to {self.log_dir}")
            
        except Exception as e:
            print(f"⚠️ Error saving plots: {e}")
    
    def create_summary_plot(self):
        """Create a comprehensive summary plot"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Summary: {self.agent_name}', fontsize=16)
        
        steps = self.training_metrics['steps']
        rewards = self.training_metrics['episode_rewards']
        losses = self.training_metrics['losses']
        
        if len(steps) < 2:
            return
        
        # Rewards over time
        axes[0, 0].plot(steps, rewards, alpha=0.6, linewidth=1)
        if len(rewards) > 20:
            rewards_ma = pd.Series(rewards).rolling(window=20).mean()
            axes[0, 0].plot(steps, rewards_ma, 'r-', linewidth=2, label='MA(20)')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss over time
        if any(l != 0 for l in losses):
            axes[0, 1].plot(steps, losses, 'g-', alpha=0.6, linewidth=1)
            if len(losses) > 20:
                loss_ma = pd.Series(losses).rolling(window=20).mean()
                axes[0, 1].plot(steps, loss_ma, 'r-', linewidth=2, label='MA(20)')
                axes[0, 1].legend()
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward distribution
        axes[0, 2].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Reward Distribution')
        axes[0, 2].set_xlabel('Reward')
        axes[0, 2].set_ylabel('Frequency')
        
        # Action distribution over time (if available)
        if self.training_metrics['action_counts']:
            action_data = []
            for i, actions in enumerate(self.training_metrics['action_counts']):
                if isinstance(actions, list) and len(actions) >= 3:
                    action_data.append([steps[i], actions[0], actions[1], actions[2]])
            
            if action_data:
                action_df = pd.DataFrame(action_data, columns=['Step', 'Sell', 'Hold', 'Buy'])
                axes[1, 0].plot(action_df['Step'], action_df['Sell'], label='Sell', alpha=0.7)
                axes[1, 0].plot(action_df['Step'], action_df['Hold'], label='Hold', alpha=0.7)
                axes[1, 0].plot(action_df['Step'], action_df['Buy'], label='Buy', alpha=0.7)
                axes[1, 0].set_title('Action Counts Over Time')
                axes[1, 0].set_xlabel('Steps')
                axes[1, 0].set_ylabel('Action Count')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Training statistics
        axes[1, 1].text(0.1, 0.9, 'Training Statistics:', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
        stats_text = f"""
        Total Steps: {len(steps)}
        Mean Reward: {np.mean(rewards):.4f}
        Std Reward: {np.std(rewards):.4f}
        Max Reward: {np.max(rewards):.4f}
        Min Reward: {np.min(rewards):.4f}
        Final Reward: {rewards[-1]:.4f}
        """
        axes[1, 1].text(0.1, 0.7, stats_text, fontsize=10, transform=axes[1, 1].transAxes, verticalalignment='top')
        axes[1, 1].axis('off')
        
        # Performance trend
        if len(rewards) > 100:
            # Split into chunks and calculate means
            chunk_size = len(rewards) // 10
            chunk_means = []
            chunk_steps = []
            for i in range(0, len(rewards), chunk_size):
                chunk = rewards[i:i+chunk_size]
                if chunk:
                    chunk_means.append(np.mean(chunk))
                    chunk_steps.append(steps[i + len(chunk)//2])
            
            axes[1, 2].plot(chunk_steps, chunk_means, 'o-', linewidth=2, markersize=6)
            axes[1, 2].set_title('Performance Trend (Chunked Means)')
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Mean Reward')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_path = os.path.join(self.log_dir, 'training_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def load_ensemble_config(config_file=None):
    """
    Load ensemble configuration from file or return defaults
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if config_file and os.path.exists(config_file):
        import json
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading config file {config_file}: {e}")
            print("Using default configuration")
            return None
    return None


def can_buy(action, mid_price, cash, current_btc):
    if action == 1 and cash > mid_price:  # can buy
        last_cash = cash
        new_cash = last_cash - mid_price
        current_btc += 1
    elif action == -1 and current_btc > 0:  # can sell
        last_cash = cash
        new_cash = last_cash + mid_price
        current_btc -= 1
    else:
        new_cash = cash

    return new_cash, current_btc


def winloss(action, last_price, mid_price):
    if action > 0:
        if last_price < mid_price:
            correct_pred = 1
        elif last_price > mid_price:
            correct_pred = -1
        else:
            correct_pred = 0
    elif action < 0:
        if last_price < mid_price:
            correct_pred = -1
        elif last_price > mid_price:
            correct_pred = 1
        else:
            correct_pred = 0
    else:
        correct_pred = 0
    return correct_pred


class Ensemble:
    def __init__(self, log_rules, save_path, starting_cash, agent_classes, args: Config):

        self.log_rules = log_rules

        # ensemble configs
        self.save_path = save_path
        self.starting_cash = starting_cash
        self.current_btc = 0
        self.position = [0]
        self.btc_assets = [0]
        self.net_assets = [starting_cash]
        self.cash = [starting_cash]
        self.agent_classes = agent_classes

        self.from_env_step_is = None

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        # Get state_dim from TradeSimulator (supports both original and enhanced features)
        temp_sim = TradeSimulator(num_sims=1)
        self.state_dim = temp_sim.state_dim
        print(f"Using state_dim: {self.state_dim}")
        # gpu_id = 0
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        eval_env_class = args.eval_env_class
        eval_env_class.num_envs = 1

        eval_env_args = args.eval_env_args
        eval_env_args["num_envs"] = 1
        eval_env_args["num_sims"] = 1

        self.trade_env = build_env(eval_env_class, eval_env_args, gpu_id=args.gpu_id)

        self.actions = []

        self.firstbpi = True

    def save_ensemble(self):
        """Saves the ensemble of agents to a directory."""
        ensemble_dir = os.path.join(self.save_path, "ensemble_models")
        os.makedirs(ensemble_dir, exist_ok=True)
        for idx, agent in enumerate(self.agents):
            agent_name = self.agent_classes[idx].__name__
            agent_dir = os.path.join(ensemble_dir, agent_name)
            os.makedirs(agent_dir, exist_ok=True)
            agent.save_or_load_agent(agent_dir, if_save=True)
        print(f"Ensemble models saved in directory: {ensemble_dir}")

    def ensemble_train(self):
        args = self.args

        for agent_class in self.agent_classes:

            args.agent_class = agent_class

            agent = self.train_agent(args=args)
            self.agents.append(agent)

        self.save_ensemble()

    def _majority_vote(self, actions):
        """handles tie breaks by returning first element of the most common ones"""
        count = Counter(actions)
        majority_action, _ = count.most_common(1)[0]
        return majority_action

    def train_agent(self, args: Config):
        """
        Trains agent
        Builds env inside
        """
        args.init_before_training()
        torch.set_grad_enabled(False)

        """init environment"""
        env = build_env(args.env_class, args.env_args, args.gpu_id)

        """init agent"""
        agent = args.agent_class(
            args.net_dims,
            args.state_dim,
            args.action_dim,
            gpu_id=args.gpu_id,
            args=args,
        )
        agent.save_or_load_agent(args.cwd, if_save=False)

        state = env.reset()

        if args.num_envs == 1:
            assert state.shape == (args.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            if state.shape != (args.num_envs, args.state_dim):
                raise ValueError(f"state.shape == (num_envs, state_dim): {state.shape, args.num_envs, args.state_dim}")
            if not isinstance(state, torch.Tensor):
                raise TypeError(f"isinstance(state, torch.Tensor): {repr(state)}")
            state = state.to(agent.device)
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        agent.last_state = state.detach()

        """init buffer"""

        if args.if_off_policy:
            buffer = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs,
                max_size=args.buffer_size,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
            )
            buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
            buffer.update(buffer_items)  # warm up for ReplayBuffer
        else:
            buffer = []

        """init evaluator"""
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)
        
        """init training logger"""
        agent_name = args.agent_class.__name__
        training_logger = TrainingLogger(save_path=self.save_path, agent_name=agent_name)
        print(f"📊 Training logger initialized for {agent_name}")

        """train loop"""
        cwd = args.cwd
        break_step = args.break_step
        horizon_len = args.horizon_len
        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer
        del args

        import torch as th

        if_train = True
        training_step = 0
        while if_train:
            buffer_items = agent.explore_env(env, horizon_len)

            action = buffer_items[1].flatten()
            action_count = th.bincount(action).data.cpu().numpy() / action.shape[0]
            action_count = np.ceil(action_count * 998).astype(int)

            position = buffer_items[0][:, :, 0].long().flatten()
            position = position.float()  # TODO Only if on cpu
            position_count = torch.histc(position, bins=env.max_position * 2 + 1, min=-2, max=2)
            position_count = position_count.data.cpu().numpy() / position.shape[0]
            position_count = np.ceil(position_count * 998).astype(int)

            print(";;;", " " * 70, action_count, position_count)

            exp_r = buffer_items[2].mean().item()
            if if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            # Log training progress
            training_logger.log_training_step(
                step=training_step,
                exp_r=exp_r,
                action_count=action_count,
                position_count=position_count,
                logging_tuple=logging_tuple
            )

            evaluator.evaluate_and_save(
                actor=agent.act,
                steps=horizon_len,
                exp_r=exp_r,
                logging_tuple=logging_tuple,
            )
            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            training_step += 1

        # Save final training plots and logs
        training_logger.save_final_plots()
        print(f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}")

        env.close() if hasattr(env, "close") else None
        evaluator.save_training_curve_jpg()
        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, "save_or_load_history"):
            buffer.save_or_load_history(cwd, if_save=True)

        self.from_env_step_is = env.step_is
        return agent


def run(save_path, agent_list, log_rules=False, config_dict=None):
    """
    Run ensemble training with configurable parameters
    
    Args:
        save_path: Path to save ensemble models
        agent_list: List of agent classes to use in ensemble
        log_rules: Whether to log trading rules
        config_dict: Dictionary of configuration parameters to override defaults
    """
    import sys

    # Default configuration
    default_config = {
        'gpu_id': int(sys.argv[1]) if len(sys.argv) > 1 else 0,
        'num_sims': 2**6,  # Number of parallel environments
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
        'break_step': 16,
        'buffer_size_multiplier': 8,  # buffer_size = max_step * this
        'repeat_times': 2,
        'horizon_len_multiplier': 2,  # horizon_len = max_step * this
        'eval_per_step_multiplier': 1,  # eval_per_step = max_step * this
        'num_workers': 1,
        'save_gap': 8,
        'data_length': 4800  # Total data length for max_step calculation
    }
    
    # Override defaults with provided config
    if config_dict:
        default_config.update(config_dict)
    
    config = default_config
    
    from erl_agent import AgentD3QN

    # Calculate derived parameters
    max_step = (config['data_length'] - config['num_ignore_step']) // config['step_gap']

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": config['num_sims'],
        "max_step": max_step,
        "state_dim": TradeSimulator(num_sims=1).state_dim,  # Dynamic detection of enhanced features
        "action_dim": 3,  # long, 0, short
        "if_discrete": True,
        "max_position": config['max_position'],
        "slippage": config['slippage'],
        "num_sims": config['num_sims'],
        "step_gap": config['step_gap'],
    }
    
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = config['gpu_id']
    args.random_seed = config['gpu_id']
    args.net_dims = config['net_dims']

    args.gamma = config['gamma']
    args.explore_rate = config['explore_rate']
    args.state_value_tau = config['state_value_tau']
    args.soft_update_tau = config['soft_update_tau']
    args.learning_rate = config['learning_rate']
    args.batch_size = config['batch_size']
    args.break_step = int(config['break_step'])
    args.buffer_size = int(max_step * config['buffer_size_multiplier'])
    args.repeat_times = config['repeat_times']
    args.horizon_len = int(max_step * config['horizon_len_multiplier'])
    args.eval_per_step = int(max_step * config['eval_per_step_multiplier'])
    args.num_workers = config['num_workers']
    args.save_gap = config['save_gap']

    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = env_args.copy()
    
    print(f"🔧 Ensemble Configuration:")
    print(f"   GPU ID: {config['gpu_id']}")
    print(f"   Parallel Environments: {config['num_sims']}")
    print(f"   Max Steps: {max_step}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Network Dims: {config['net_dims']}")

    ensemble_env = Ensemble(
        log_rules,
        save_path,
        config['starting_cash'],
        agent_list,
        args,
    )
    ensemble_env.ensemble_train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FinRL Contest 2024 - Ensemble Training')
    parser.add_argument('gpu_id', nargs='?', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--save-path', type=str, default='ensemble_teamname', help='Path to save ensemble models')
    parser.add_argument('--log-rules', action='store_true', help='Enable trading rules logging')
    
    args = parser.parse_args()
    
    # Load configuration from file if specified
    config_dict = load_ensemble_config(args.config)
    if config_dict is None:
        config_dict = {}
    
    # Override GPU ID from command line
    config_dict['gpu_id'] = args.gpu_id
    
    print(f"🚀 FinRL Contest 2024 - Ensemble Training")
    print(f"🔧 Using GPU: {args.gpu_id}")
    if args.config:
        print(f"📁 Config file: {args.config}")
    print(f"💾 Save path: {args.save_path}")
    
    run(
        args.save_path,
        [AgentD3QN, AgentDoubleDQN, AgentDoubleDQN, AgentTwinD3QN],
        log_rules=args.log_rules,
        config_dict=config_dict
    )
