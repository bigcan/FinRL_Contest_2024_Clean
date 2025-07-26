#!/usr/bin/env python3
"""
Improved Ensemble Training for FinRL Contest 2024
Optimized for achieving Sharpe ratio > 1.0
"""
import os
import sys
import time

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

import torch
import numpy as np
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from task1_ensemble import Ensemble, get_state_dim, TrainingLogger


def get_improved_ensemble_config():
    """
    Get improved configuration optimized for higher Sharpe ratios
    """
    return {
        # Aggressive learning parameters
        'gpu_id': int(sys.argv[1]) if len(sys.argv) > 1 else 0,
        'num_sims': 128,  # More parallel environments for better sampling
        'num_ignore_step': 60,
        'max_position': 5,  # Allow larger positions for more alpha
        'step_gap': 1,     # Higher frequency trading
        'slippage': 5e-7,  # Reduced slippage
        'starting_cash': 1e6,
        
        # Larger, deeper networks
        'net_dims': (256, 128, 64),
        'gamma': 0.999,    # Longer-term focus
        'explore_rate': 0.01,  # More exploration
        'state_value_tau': 0.02,
        'soft_update_tau': 5e-6,  # Slower target updates
        'learning_rate': 5e-5,    # Faster learning
        'batch_size': 1024,       # Larger batches
        
        # Extended training
        'break_step': 32,         # Train longer
        'buffer_size_multiplier': 16,  # Larger replay buffer
        'repeat_times': 4,        # More gradient updates
        'horizon_len_multiplier': 4,   # Longer episodes
        'eval_per_step_multiplier': 1,
        'num_workers': 1,
        'save_gap': 4,           # Save more frequently
        'data_length': 4800
    }


def run_improved_ensemble(save_path="ensemble_improved_sharpe", agent_list=None):
    """
    Run improved ensemble training optimized for Sharpe ratio > 1.0
    """
    if agent_list is None:
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    
    config = get_improved_ensemble_config()
    
    print("="*80)
    print("🚀 IMPROVED ENSEMBLE TRAINING FOR HIGH SHARPE RATIO")
    print("="*80)
    print(f"🎯 Target: Sharpe Ratio > 1.0")
    print(f"🔧 Using GPU: {config['gpu_id']}")
    print(f"📊 Parallel Environments: {config['num_sims']}")
    print(f"🧠 Network Architecture: {config['net_dims']}")
    print(f"📈 Learning Rate: {config['learning_rate']}")
    print(f"💰 Max Position Size: {config['max_position']}")
    print(f"⚡ Step Gap: {config['step_gap']} (higher frequency)")
    print("="*80)
    
    # Calculate derived parameters
    max_step = (config['data_length'] - config['num_ignore_step']) // config['step_gap']
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": config['num_sims'],
        "max_step": max_step,
        "state_dim": get_state_dim(),
        "action_dim": 3,
        "if_discrete": True,
        "max_position": config['max_position'],
        "slippage": config['slippage'],
        "num_sims": config['num_sims'],
        "step_gap": config['step_gap'],
    }
    
    # Create config object
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    
    # Apply improved hyperparameters
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
    
    print(f"\n📋 Training Configuration:")
    print(f"   Max Steps per Episode: {max_step}")
    print(f"   Buffer Size: {args.buffer_size:,}")
    print(f"   Horizon Length: {args.horizon_len}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Repeat Times: {args.repeat_times}")
    print(f"   Break Step: {args.break_step}")
    
    # Initialize ensemble with improved reward function
    ensemble_env = ImprovedEnsemble(
        log_rules=False,
        save_path=save_path,
        starting_cash=config['starting_cash'],
        agent_list=agent_list,
        args=args,
    )
    
    # Start training
    print(f"\n🏁 Starting improved ensemble training...")
    start_time = time.time()
    ensemble_env.ensemble_train()
    end_time = time.time()
    
    print(f"\n✅ Training completed in {end_time - start_time:.0f} seconds")
    print(f"💾 Models saved to: {save_path}")
    print(f"🎯 Ready for evaluation to measure Sharpe ratio improvement!")


class ImprovedEnsemble(Ensemble):
    """
    Enhanced ensemble class with improved reward functions and training
    """
    
    def train_agent(self, args: Config):
        """
        Enhanced agent training with improved reward function
        """
        args.init_before_training()
        torch.set_grad_enabled(False)

        # Build environment
        env = build_env(args.env_class, args.env_args, args.gpu_id)
        
        # CRITICAL: Set the trade simulator to use the improved reward function
        if hasattr(env, 'simulator') and hasattr(env.simulator, 'set_reward_type'):
            env.simulator.set_reward_type("profit_maximizing")
            print(f"🎯 Using 'profit_maximizing' reward function for {args.agent_class.__name__}")
        elif hasattr(env, 'set_reward_type'):
            env.set_reward_type("profit_maximizing") 
            print(f"🎯 Using 'profit_maximizing' reward function for {args.agent_class.__name__}")
        else:
            print(f"⚠️  Warning: Could not set reward function for {args.agent_class.__name__}")

        # Initialize agent
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

        # Initialize buffer
        if args.if_off_policy:
            buffer = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs,
                max_size=args.buffer_size,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
            )
            buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
            buffer.update(buffer_items)
        else:
            buffer = []

        # Initialize evaluator
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)
        
        # Initialize training logger
        agent_name = args.agent_class.__name__
        training_logger = TrainingLogger(save_path=self.save_path, agent_name=agent_name)
        print(f"✅ Training logger initialized for {agent_name}")

        # Enhanced training loop with Sharpe ratio monitoring
        cwd = args.cwd
        break_step = args.break_step
        horizon_len = args.horizon_len
        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer

        if_train = True
        training_step = 0
        best_sharpe = -float('inf')
        
        print(f"🏁 Starting training for {agent_name}...")
        
        while if_train:
            buffer_items = agent.explore_env(env, horizon_len)

            action = buffer_items[1].flatten()
            action_count = torch.bincount(action).data.cpu().numpy() / action.shape[0]
            action_count = np.ceil(action_count * 998).astype(int)

            position = buffer_items[0][:, :, 0].long().flatten()
            position = position.float()
            position_count = torch.histc(position, bins=env.max_position * 2 + 1, min=-2, max=2)
            position_count = position_count.data.cpu().numpy() / position.shape[0]
            position_count = np.ceil(position_count * 998).astype(int)

            exp_r = buffer_items[2].mean().item()
            
            if if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            # Enhanced logging with Sharpe ratio tracking
            training_logger.log_training_step(
                step=training_step,
                exp_r=exp_r,
                action_count=action_count,
                position_count=position_count,
                logging_tuple=logging_tuple
            )

            # Evaluate and track Sharpe ratio
            evaluator.evaluate_and_save(
                actor=agent.act,
                steps=horizon_len,
                exp_r=exp_r,
                logging_tuple=logging_tuple,
            )
            
            # Check for Sharpe ratio improvement
            if hasattr(env, 'get_reward_metrics'):
                metrics = env.get_reward_metrics()
                current_sharpe = metrics.get('sharpe_ratio', -float('inf'))
                if current_sharpe > best_sharpe:
                    best_sharpe = current_sharpe
                    print(f"🎯 New best Sharpe ratio: {best_sharpe:.6f} at step {training_step}")
            
            # Print progress periodically
            if training_step % 5 == 0:
                print(f"📊 Step {training_step}: exp_r={exp_r:.4f}, actions={action_count}")
            
            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            training_step += 1

        # Save final results
        training_logger.save_final_plots()
        print(f"✅ {agent_name} training completed. Best Sharpe: {best_sharpe:.6f}")
        print(f"📊 Total steps: {evaluator.total_step}")
        print(f"💾 Saved to: {cwd}")

        env.close() if hasattr(env, "close") else None
        evaluator.save_training_curve_jpg()
        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, "save_or_load_history"):
            buffer.save_or_load_history(cwd, if_save=True)

        self.from_env_step_is = env.step_is
        return agent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FinRL Contest 2024 - Improved Ensemble Training')
    parser.add_argument('gpu_id', nargs='?', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--save-path', type=str, default='ensemble_improved_sharpe', 
                       help='Path to save improved ensemble models')
    
    args = parser.parse_args()
    
    print(f"🚀 FinRL Contest 2024 - Improved Ensemble Training")
    print(f"🎯 Target: Achieve Sharpe Ratio > 1.0")
    print(f"🔧 Using GPU: {args.gpu_id}")
    print(f"💾 Save path: {args.save_path}")
    
    # Run improved training
    run_improved_ensemble(
        save_path=args.save_path,
        agent_list=[AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    )