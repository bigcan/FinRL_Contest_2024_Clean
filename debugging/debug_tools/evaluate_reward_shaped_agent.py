"""
Quick evaluation of reward-shaped trained agent
"""
import os
import numpy as np
import torch
from task1_eval import EnsembleEvaluator
from erl_config import Config
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN


def evaluate_reward_shaped_agent():
    """Evaluate the reward-shaped trained agent"""
    
    print("="*60)
    print("EVALUATING REWARD-SHAPED AGENT")
    print("="*60)
    
    # Configuration matching the training setup
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": (4800 - 60) // 2,  # Same as training
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 1,
        "step_gap": 2,
        "dataset_path": "data/BTC_1sec_predict.npy"
    }
    
    args = Config(agent_class=AgentD3QN, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = -1
    args.random_seed = 0
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6
    
    # Path to the trained reward-shaped agent
    agent_path = "ensemble_reward_shaped_aggressive/AgentD3QN"
    
    if not os.path.exists(agent_path):
        print(f"❌ Agent path not found: {agent_path}")
        return None
    
    print(f"Loading agent from: {agent_path}")
    
    try:
        # Create evaluator  
        evaluator = EnsembleEvaluator(
            save_path="ensemble_reward_shaped_aggressive",
            agent_classes=[AgentD3QN],
            args=args
        )
        
        # Load and evaluate the agent
        evaluator.load_agents()
        print("✓ Agent loaded successfully")
        
        # Run evaluation
        print("Running evaluation...")
        evaluator.multi_trade()
        print("✓ Evaluation completed")
        
        # Load and analyze results
        try:
            net_assets = np.load('evaluation_net_assets.npy')
            positions = np.load('evaluation_positions.npy')
            
            initial_value = net_assets[0]
            final_value = net_assets[-1]
            total_return = (final_value / initial_value - 1) * 100
            
            # Count trades
            position_changes = np.diff(positions)
            trades = np.sum(np.abs(position_changes) > 0)
            
            print(f"\n=== REWARD-SHAPED AGENT RESULTS ===")
            print(f"Initial portfolio value: ${initial_value:,.2f}")
            print(f"Final portfolio value: ${final_value:,.2f}")
            print(f"Total return: {total_return:.2f}%")
            print(f"Total trades executed: {trades}")
            print(f"Average position: {np.mean(positions):.3f}")
            print(f"Position std: {np.std(positions):.3f}")
            
            # Basic performance metrics
            returns = np.diff(net_assets) / net_assets[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252*24*60) if np.std(returns) > 0 else 0
            max_drawdown = np.max((np.maximum.accumulate(net_assets) - net_assets) / np.maximum.accumulate(net_assets))
            
            print(f"Sharpe ratio: {sharpe_ratio:.3f}")
            print(f"Max drawdown: {max_drawdown:.1%}")
            
            return {
                'total_return': total_return,
                'trades': trades,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': final_value
            }
            
        except FileNotFoundError as e:
            print(f"❌ Could not load evaluation results: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_baseline():
    """Compare reward-shaped agent with original conservative baseline"""
    
    print(f"\n" + "="*60)
    print("COMPARING WITH CONSERVATIVE BASELINE")  
    print("="*60)
    
    # Load baseline results
    baseline_path = "ensemble_teamname"
    if os.path.exists(f"{baseline_path}/ensemble_models_net_assets.npy"):
        baseline_assets = np.load(f"{baseline_path}/ensemble_models_net_assets.npy")
        baseline_positions = np.load(f"{baseline_path}/ensemble_models_positions.npy")
        
        baseline_return = (baseline_assets[-1] / baseline_assets[0] - 1) * 100
        baseline_trades = np.sum(np.abs(np.diff(baseline_positions)) > 0)
        
        print(f"BASELINE (Conservative Agents):")
        print(f"  Total return: {baseline_return:.2f}%")
        print(f"  Total trades: {baseline_trades}")
        
        # Evaluate reward-shaped agent
        reward_shaped_results = evaluate_reward_shaped_agent()
        
        if reward_shaped_results:
            print(f"\nREWARD-SHAPED AGENT:")
            print(f"  Total return: {reward_shaped_results['total_return']:.2f}%")
            print(f"  Total trades: {reward_shaped_results['trades']}")
            
            print(f"\n=== COMPARISON ===")
            return_improvement = reward_shaped_results['total_return'] - baseline_return
            trade_improvement = reward_shaped_results['trades'] - baseline_trades
            
            print(f"Return improvement: {return_improvement:+.2f}%")
            print(f"Trade activity improvement: {trade_improvement:+d} trades")
            
            if reward_shaped_results['trades'] > 0 and baseline_trades == 0:
                print("✅ SUCCESS: Reward shaping overcame conservative training!")
                print("   - Agent now makes trades instead of just holding")
                print(f"   - Achieved {reward_shaped_results['total_return']:.2f}% returns with active trading")
            elif reward_shaped_results['trades'] > baseline_trades:
                print("✅ IMPROVEMENT: Increased trading activity")
            else:
                print("⚠️  Mixed results: Check if further tuning needed")
                
            return reward_shaped_results, baseline_return, baseline_trades
    else:
        print("❌ Baseline results not found - running standalone evaluation")
        return evaluate_reward_shaped_agent(), None, None


if __name__ == "__main__":
    results = compare_with_baseline()
    print(f"\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)