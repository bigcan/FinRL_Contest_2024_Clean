"""
Quick training test with ultra-aggressive reward configuration
"""
import os
import torch
import numpy as np
from erl_config import Config, build_env
from reward_shaped_training_simulator import RewardShapedTrainingSimulator
from training_reward_config import TrainingRewardConfig
from erl_agent import AgentD3QN
from erl_run import train_agent
import time


def quick_ultra_aggressive_training():
    """Train single agent with ultra-aggressive reward configuration"""
    
    print("="*60)
    print("QUICK ULTRA-AGGRESSIVE TRAINING TEST")
    print("="*60)
    
    # Create ultra-aggressive reward configuration
    reward_config = TrainingRewardConfig.ultra_aggressive_training_config()
    training_step_tracker = {'step': 0}
    
    # Quick training parameters  
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 64,  # Match actual environment
        "max_step": 200,  # Shorter episodes for quick test
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 64,  # Match actual environment
        "step_gap": 2,
        "dataset_path": "data/BTC_1sec_predict.npy",
        "reward_config": reward_config,
        "training_step_tracker": training_step_tracker
    }
    
    # Create agent configuration
    args = Config(agent_class=AgentD3QN, 
                  env_class=RewardShapedTrainingSimulator, 
                  env_args=env_args)
    args.gpu_id = -1  # Use CPU for quick test
    args.random_seed = 42
    args.net_dims = (64, 64)  # Smaller networks for speed
    args.starting_cash = 1e6
    
    # Training hyperparameters - optimized for quick results
    args.gamma = 0.99
    args.explore_rate = 0.05  # Lower exploration, let reward shaping guide
    args.state_value_tau = 0.01
    args.target_step = 512   # Small target steps
    args.eval_times = 32     # Quick evaluation
    args.break_step = 4096   # Early stopping
    args.if_allow_break = True
    args.if_remove = True
    args.cwd = "quick_ultra_aggressive_test"
    
    # Print configuration summary
    print("Ultra-Aggressive Configuration:")
    reward_config.print_training_config()
    
    print(f"\nTraining parameters:")
    print(f"  Environments: {env_args['num_sims']}")
    print(f"  Max steps per episode: {env_args['max_step']}")
    print(f"  Target steps: {args.target_step}")
    print(f"  Break steps: {args.break_step}")
    print(f"  Network size: {args.net_dims}")
    
    # Create save directory
    os.makedirs(args.cwd, exist_ok=True)
    
    try:
        print(f"\nStarting ultra-aggressive training...")
        start_time = time.time()
        
        # Train the agent
        train_agent(args)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"✓ Ultra-aggressive training completed in {training_time:.1f} seconds")
        print(f"  Total training steps: {training_step_tracker['step']:,}")
        
        # Quick evaluation
        return evaluate_ultra_aggressive_agent(args)
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_ultra_aggressive_agent(args):
    """Quick evaluation of the ultra-aggressive agent"""
    
    print(f"\n" + "="*40)
    print("EVALUATING ULTRA-AGGRESSIVE AGENT")
    print("="*40)
    
    try:
        from task1_eval import EnsembleEvaluator
        from trade_simulator import EvalTradeSimulator
        
        # Create evaluation environment (no reward shaping)
        eval_env_args = args.env_args.copy()
        eval_env_args.pop('reward_config', None)
        eval_env_args.pop('training_step_tracker', None)
        eval_env_args['num_envs'] = 1
        eval_env_args['num_sims'] = 1
        
        eval_args = Config(agent_class=AgentD3QN, 
                          env_class=EvalTradeSimulator, 
                          env_args=eval_env_args)
        eval_args.gpu_id = args.gpu_id
        eval_args.random_seed = args.random_seed
        eval_args.net_dims = args.net_dims
        eval_args.starting_cash = args.starting_cash
        
        # Create evaluator
        evaluator = EnsembleEvaluator(
            save_path=args.cwd,
            agent_classes=[AgentD3QN],
            args=eval_args
        )
        
        # Load and evaluate
        evaluator.load_agents()
        evaluator.multi_trade()
        
        # Analyze results
        net_assets = np.load('evaluation_net_assets.npy')
        positions = np.load('evaluation_positions.npy')
        
        initial_value = net_assets[0]
        final_value = net_assets[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        # Count trades
        position_changes = np.diff(positions)
        trades = np.sum(np.abs(position_changes) > 0)
        
        print(f"Results:")
        print(f"  Total return: {total_return:.2f}%")
        print(f"  Total trades: {trades}")
        print(f"  Final value: ${final_value:,.2f}")
        
        if trades > 0:
            print(f"🎉 SUCCESS! Ultra-aggressive agent makes {trades} trades!")
            print(f"   Return: {total_return:.2f}%")
        else:
            print(f"😐 Still conservative: {trades} trades, {total_return:.2f}% return")
        
        return {
            'trades': trades,
            'return': total_return,
            'final_value': final_value
        }
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return None


if __name__ == "__main__":
    print("Starting quick ultra-aggressive training test...")
    results = quick_ultra_aggressive_training()
    
    if results:
        print(f"\n" + "="*60)
        print("ULTRA-AGGRESSIVE TRAINING TEST COMPLETE")
        print("="*60)
        print(f"Trades: {results['trades']}")
        print(f"Return: {results['return']:.2f}%")
        print(f"Success: {'Yes' if results['trades'] > 0 else 'No'}")
    else:
        print(f"\n❌ Ultra-aggressive training test failed")