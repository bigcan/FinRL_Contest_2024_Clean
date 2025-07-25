#!/usr/bin/env python3
"""
A/B Testing Framework for Reward Functions
Systematically compares different reward functions with fixed Q-value calculation
"""

import os
import sys
import torch
import numpy as np
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from erl_agent import AgentD3QN
from enhanced_training_config import EnhancedConfig
from optimized_hyperparameters import get_optimized_hyperparameters, apply_optimized_hyperparameters
from erl_config import build_env
from erl_replay_buffer import ReplayBuffer


@dataclass
class RewardTestResult:
    """Results from testing a specific reward function"""
    reward_type: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_reward_per_step: float
    final_q_values: List[float]
    training_time: float
    convergence_step: int
    action_distribution: Dict[int, int]
    performance_curve: List[float]


class RewardFunctionTester:
    """
    A/B testing framework for reward functions
    """
    
    def __init__(self, 
                 test_steps: int = 100,
                 num_environments: int = 16,
                 random_seed: int = 42,
                 parallel_testing: bool = True):
        """
        Initialize reward function tester
        
        Args:
            test_steps: Number of training steps for each test
            num_environments: Number of parallel environments
            random_seed: Random seed for reproducibility
            parallel_testing: Whether to run tests in parallel
        """
        self.test_steps = test_steps
        self.num_environments = num_environments
        self.random_seed = random_seed
        self.parallel_testing = parallel_testing
        
        # Available reward functions to test
        self.reward_functions = [
            "simple",
            "transaction_cost_adjusted", 
            "multi_objective"
        ]
        
        # Test results storage
        self.test_results = {}
        
        print(f"🧪 Reward Function A/B Tester initialized:")
        print(f"   Test steps: {test_steps}")
        print(f"   Environments: {num_environments}")
        print(f"   Reward functions: {', '.join(self.reward_functions)}")
        print(f"   Parallel testing: {parallel_testing}")
    
    def create_test_configuration(self, reward_type: str) -> EnhancedConfig:
        """Create optimized configuration for testing a reward function"""
        
        # Get basic simulator for state dimension
        temp_sim = TradeSimulator(num_sims=1)
        state_dim = temp_sim.state_dim
        temp_sim.set_reward_type(reward_type)
        
        # Environment configuration
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": self.num_environments,
            "max_step": 1000,  # Reasonable for testing
            "state_dim": state_dim,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "num_sims": self.num_environments,
            "step_gap": 2,
        }
        
        # Create enhanced configuration
        config = EnhancedConfig(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
        config.gpu_id = 0 if torch.cuda.is_available() else -1
        config.random_seed = self.random_seed
        config.state_dim = state_dim
        
        # Apply optimized hyperparameters for this reward type
        optimized_params = get_optimized_hyperparameters(reward_type)
        config = apply_optimized_hyperparameters(config, optimized_params, env_args)
        
        # Override for testing
        config.break_step = self.test_steps
        config.eval_per_step = max(1, self.test_steps // 10)
        config.horizon_len = 200
        config.buffer_size = 2000
        config.early_stopping_enabled = False  # Complete all steps
        
        return config
    
    def test_single_reward_function(self, reward_type: str) -> RewardTestResult:
        """Test a single reward function and return comprehensive results"""
        
        print(f"\n🧪 Testing reward function: {reward_type}")
        test_start = time.time()
        
        try:
            # Create configuration
            config = self.create_test_configuration(reward_type)
            
            # Build environment
            env = build_env(config.env_class, config.env_args, config.gpu_id)
            env.set_reward_type(reward_type)
            
            # Create agent
            agent = AgentD3QN(
                config.net_dims,
                config.state_dim,
                config.action_dim,
                gpu_id=config.gpu_id,
                args=config,
            )
            
            # Initialize
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state = state.to(agent.device)
            agent.last_state = state.detach()
            
            # Create buffer
            buffer = ReplayBuffer(
                gpu_id=config.gpu_id,
                num_seqs=config.num_envs,
                max_size=config.buffer_size,
                state_dim=config.state_dim,
                action_dim=1,
            )
            
            # Warm up buffer
            buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
            buffer.update(buffer_items)
            
            # Training metrics
            rewards_history = []
            q_values_history = []
            action_counts = {0: 0, 1: 0, 2: 0}
            performance_curve = []
            
            convergence_step = self.test_steps  # Default to end if no convergence
            convergence_threshold = 0.001  # Threshold for convergence detection
            
            print(f"   Training {self.test_steps} steps...")
            
            # Training loop
            for step in range(self.test_steps):
                # Collect experience
                buffer_items = agent.explore_env(env, config.horizon_len)
                exp_r = buffer_items[2].mean().item()
                rewards_history.append(exp_r)
                
                # Update buffer
                buffer.update(buffer_items)
                
                # Update network
                torch.set_grad_enabled(True)
                logging_tuple = agent.update_net(buffer)
                torch.set_grad_enabled(False)
                
                # Track Q-values and actions
                with torch.no_grad():
                    test_state = state[:1]  # First environment
                    q_values = agent.act(test_state)
                    q_values_history.append(q_values[0].cpu().numpy().copy())
                    
                    # Sample actions for distribution
                    action = q_values.argmax(dim=1, keepdim=True)[0].item()
                    action_counts[action] += 1
                
                # Performance tracking
                if step >= 10:  # After some warmup
                    recent_performance = np.mean(rewards_history[-10:])
                    performance_curve.append(recent_performance)
                    
                    # Check for convergence (stable performance)
                    if len(performance_curve) >= 20:
                        recent_std = np.std(performance_curve[-20:])
                        if recent_std < convergence_threshold and convergence_step == self.test_steps:
                            convergence_step = step
                
                # Progress update
                if step % max(1, self.test_steps // 5) == 0:
                    obj_critic, obj_actor = logging_tuple[:2] if logging_tuple else (0, 0)
                    avg_q = np.mean(q_values_history[-1]) if q_values_history else 0
                    print(f"     Step {step}: Reward={exp_r:.4f}, Q-avg={avg_q:.3f}")
            
            # Final evaluation
            print(f"   Running final evaluation...")
            eval_returns = []
            eval_actions = []
            eval_steps = 100
            
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state = state.to(agent.device)
            
            for _ in range(eval_steps):
                with torch.no_grad():
                    q_values = agent.act(state)
                    action = q_values.argmax(dim=1, keepdim=True)
                    eval_actions.extend(action.cpu().numpy().flatten())
                
                next_state, reward, done, _ = env.step(action)
                eval_returns.append(reward.mean().item())
                
                if done.any():
                    state = env.reset()
                    if not isinstance(state, torch.Tensor):
                        state = torch.tensor(state, dtype=torch.float32)
                    state = state.to(agent.device)
                else:
                    state = next_state
            
            # Calculate metrics
            total_return = sum(eval_returns)
            
            # Sharpe ratio
            if len(eval_returns) > 1 and np.std(eval_returns) > 1e-6:
                sharpe_ratio = np.mean(eval_returns) / np.std(eval_returns)
            else:
                sharpe_ratio = 0.0
            
            # Max drawdown (simplified)
            cumulative_returns = np.cumsum(eval_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            
            # Win rate
            win_rate = np.mean(np.array(eval_returns) > 0)
            
            # Trading statistics
            total_trades = len([a for a in eval_actions if a != 0])
            avg_reward_per_step = np.mean(rewards_history)
            
            # Action distribution in evaluation
            eval_action_counts = {0: 0, 1: 0, 2: 0}
            for action in eval_actions:
                eval_action_counts[int(action)] += 1
            
            training_time = time.time() - test_start
            
            # Create result
            result = RewardTestResult(
                reward_type=reward_type,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                avg_reward_per_step=avg_reward_per_step,
                final_q_values=q_values_history[-1] if q_values_history else [0, 0, 0],
                training_time=training_time,
                convergence_step=convergence_step,
                action_distribution=eval_action_counts,
                performance_curve=performance_curve
            )
            
            print(f"   ✅ {reward_type} completed in {training_time:.1f}s")
            print(f"     Total return: {total_return:.4f}")
            print(f"     Sharpe ratio: {sharpe_ratio:.3f}")
            print(f"     Win rate: {win_rate:.1%}")
            
            env.close() if hasattr(env, "close") else None
            return result
            
        except Exception as e:
            print(f"   ❌ Error testing {reward_type}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return failed result
            return RewardTestResult(
                reward_type=reward_type,
                total_return=-1000.0,
                sharpe_ratio=-10.0,
                max_drawdown=-1.0,
                win_rate=0.0,
                total_trades=0,
                avg_reward_per_step=0.0,
                final_q_values=[0, 0, 0],
                training_time=time.time() - test_start,
                convergence_step=self.test_steps,
                action_distribution={0: 0, 1: 0, 2: 0},
                performance_curve=[]
            )
    
    def run_ab_test(self) -> Dict[str, RewardTestResult]:
        """Run A/B test comparing all reward functions"""
        
        print(f"\\n🚀 Starting Reward Function A/B Test")
        print("=" * 70)
        
        test_start = time.time()
        
        if self.parallel_testing:
            # Run tests in parallel
            print(f"📊 Running {len(self.reward_functions)} tests in parallel...")
            
            with ThreadPoolExecutor(max_workers=len(self.reward_functions)) as executor:
                # Submit all tests
                future_to_reward = {
                    executor.submit(self.test_single_reward_function, reward_type): reward_type 
                    for reward_type in self.reward_functions
                }
                
                # Collect results
                for future in as_completed(future_to_reward):
                    reward_type = future_to_reward[future]
                    try:
                        result = future.result()
                        self.test_results[reward_type] = result
                    except Exception as e:
                        print(f"❌ Failed to get result for {reward_type}: {e}")
        else:
            # Run tests sequentially
            print(f"📊 Running {len(self.reward_functions)} tests sequentially...")
            
            for reward_type in self.reward_functions:
                result = self.test_single_reward_function(reward_type)
                self.test_results[reward_type] = result
        
        total_time = time.time() - test_start
        
        print(f"\\n🎉 A/B Test completed in {total_time:.1f}s")
        return self.test_results
    
    def analyze_results(self) -> Dict[str, any]:
        """Analyze and rank the test results"""
        
        if not self.test_results:
            print("❌ No test results to analyze")
            return {}
        
        print(f"\\n📊 REWARD FUNCTION A/B TEST ANALYSIS")
        print("=" * 70)
        
        # Create comparison table
        results_data = []
        
        for reward_type, result in self.test_results.items():
            results_data.append({
                'Reward Function': reward_type,
                'Total Return': result.total_return,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown': result.max_drawdown,
                'Win Rate': result.win_rate,
                'Avg Reward/Step': result.avg_reward_per_step,
                'Total Trades': result.total_trades,
                'Training Time': result.training_time,
                'Convergence Step': result.convergence_step,
                'Hold %': result.action_distribution[0] / sum(result.action_distribution.values()) * 100 if sum(result.action_distribution.values()) > 0 else 0,
                'Buy %': result.action_distribution[1] / sum(result.action_distribution.values()) * 100 if sum(result.action_distribution.values()) > 0 else 0,
                'Sell %': result.action_distribution[2] / sum(result.action_distribution.values()) * 100 if sum(result.action_distribution.values()) > 0 else 0
            })
        
        df = pd.DataFrame(results_data)
        
        # Print detailed comparison
        print(f"\\n📈 Performance Comparison:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Ranking system (weighted score)
        print(f"\\n🏆 Overall Ranking (Weighted Score):")
        
        ranking_scores = {}
        
        for reward_type, result in self.test_results.items():
            # Multi-objective scoring
            score = (
                0.3 * result.sharpe_ratio +        # Risk-adjusted returns (most important)
                0.25 * result.total_return +       # Total returns
                0.2 * result.win_rate +           # Consistency
                0.15 * (-abs(result.max_drawdown)) + # Risk control (negative is better)
                0.1 * (1.0 - result.convergence_step / self.test_steps)  # Faster learning
            )
            
            ranking_scores[reward_type] = score
            
            # Calculate trading activity
            total_actions = sum(result.action_distribution.values())
            trading_activity = (result.action_distribution[1] + result.action_distribution[2]) / max(1, total_actions) * 100
            
            print(f"   {reward_type:25}: Score={score:6.3f} (Return={result.total_return:6.3f}, Sharpe={result.sharpe_ratio:5.2f}, Win={result.win_rate:5.1%}, Trading={trading_activity:4.1f}%)")
        
        # Best reward function
        best_reward = max(ranking_scores, key=ranking_scores.get)
        best_score = ranking_scores[best_reward]
        
        print(f"\\n🥇 WINNER: {best_reward} (Score: {best_score:.3f})")
        
        # Recommendations
        print(f"\\n💡 RECOMMENDATIONS:")
        best_result = self.test_results[best_reward]
        
        print(f"   ✅ Use '{best_reward}' reward function for production")
        print(f"   📊 Expected performance: {best_result.total_return:.3f} return, {best_result.sharpe_ratio:.2f} Sharpe")
        print(f"   🎯 Training efficiency: Converged at step {best_result.convergence_step}/{self.test_steps}")
        
        # Trading behavior analysis
        total_actions = sum(best_result.action_distribution.values())
        if total_actions > 0:
            hold_pct = best_result.action_distribution[0] / total_actions * 100
            buy_pct = best_result.action_distribution[1] / total_actions * 100
            sell_pct = best_result.action_distribution[2] / total_actions * 100
            
            print(f"   🎪 Trading behavior: {hold_pct:.1f}% Hold, {buy_pct:.1f}% Buy, {sell_pct:.1f}% Sell")
            
            if hold_pct > 80:
                print(f"   ⚠️  WARNING: High hold percentage may indicate conservative bias")
            elif buy_pct + sell_pct < 20:
                print(f"   ⚠️  WARNING: Low trading activity may limit profit potential")
            else:
                print(f"   ✅ Good trading activity balance")
        
        analysis_result = {
            'best_reward_function': best_reward,
            'best_score': best_score,
            'ranking_scores': ranking_scores,
            'results_dataframe': df,
            'detailed_results': self.test_results
        }
        
        return analysis_result


def main():
    """Main execution for reward function A/B testing"""
    
    print("🧪 REWARD FUNCTION A/B TESTING")
    print("=" * 70)
    
    # Create tester
    tester = RewardFunctionTester(
        test_steps=20,  # Very quick test to avoid timeout
        num_environments=4,
        parallel_testing=False  # Sequential for better output clarity
    )
    
    # Run A/B test
    results = tester.run_ab_test()
    
    # Analyze results
    analysis = tester.analyze_results()
    
    print(f"\\n📋 NEXT STEPS:")
    if analysis:
        best_reward = analysis['best_reward_function']
        print(f"   1. Configure production system with '{best_reward}' reward")
        print(f"   2. Run full 200-step training with optimal reward function")
        print(f"   3. Validate results with longer evaluation period")
        print(f"   4. Consider ensemble approach with top 2 reward functions")
    else:
        print(f"   1. Debug test issues and retry")
        print(f"   2. Check system configuration")
    
    return analysis


if __name__ == "__main__":
    analysis = main()