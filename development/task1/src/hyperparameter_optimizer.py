"""
Hyperparameter Optimization for Profitability Improvements
Addresses learning rate and exploration issues identified in baseline
"""

import numpy as np
import torch
import itertools
from typing import Dict, List, Tuple, Any
import time
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from enhanced_training_config import EnhancedConfig
from erl_agent import AgentD3QN


class HyperparameterOptimizer:
    """
    Systematic hyperparameter optimization for improved profitability
    """
    
    def __init__(self, reward_type="multi_objective"):
        self.reward_type = reward_type
        self.results = []
        
        print(f"🎯 Hyperparameter Optimizer for Profitability")
        print(f"   Reward type: {reward_type}")
        print(f"   Focus: Learning rate, exploration, architecture optimization")
    
    def define_search_space(self) -> Dict[str, List[Any]]:
        """Define hyperparameter search space based on profitability analysis"""
        
        # Based on analysis: current LR (2e-6) too low, exploration (0.005) too conservative
        search_space = {
            "learning_rate": [
                5e-5,   # 25x increase from baseline (2e-6)
                1e-4,   # 50x increase from baseline  
                2e-4,   # 100x increase from baseline
                5e-4,   # 250x increase from baseline
            ],
            "exploration_rate": [
                0.01,   # 2x increase from baseline (0.005)
                0.05,   # 10x increase from baseline
                0.1,    # 20x increase from baseline
                0.2,    # 40x increase from baseline
            ],
            "batch_size": [
                256,    # Smaller for faster updates
                512,    # Baseline
                1024,   # Larger for stability
            ],
            "network_architecture": [
                (64, 32),       # Smaller for 8 features
                (128, 64, 32),  # Current optimized
                (256, 128, 64), # Larger capacity
            ],
            "gamma": [
                0.99,   # Standard discount
                0.995,  # Current (good)
                0.999,  # High future value
            ]
        }
        
        print(f"📊 Search Space Defined:")
        for param, values in search_space.items():
            print(f"   {param}: {values}")
        
        return search_space
    
    def create_config_combinations(self, search_space: Dict[str, List[Any]], 
                                 max_combinations: int = 20) -> List[Dict[str, Any]]:
        """Create smart combinations prioritizing high-impact changes"""
        
        # Prioritize key parameters that address profitability issues
        high_impact_combos = []
        
        # High-impact combinations: Higher LR + Higher exploration
        priority_lr = [1e-4, 2e-4]  # Focus on significantly higher learning rates
        priority_exploration = [0.05, 0.1]  # Focus on much higher exploration
        
        # Generate priority combinations
        for lr in priority_lr:
            for exploration in priority_exploration:
                for arch in search_space["network_architecture"]:
                    combo = {
                        "learning_rate": lr,
                        "exploration_rate": exploration,
                        "batch_size": 512,  # Keep stable
                        "network_architecture": arch,
                        "gamma": 0.995,  # Keep stable
                        "priority": "high"
                    }
                    high_impact_combos.append(combo)
        
        # Add some diverse combinations
        if len(high_impact_combos) < max_combinations:
            remaining = max_combinations - len(high_impact_combos)
            
            # Generate additional combinations
            param_names = list(search_space.keys())
            param_values = list(search_space.values())
            
            additional_combos = []
            for combo in itertools.product(*param_values):
                if len(additional_combos) >= remaining:
                    break
                    
                combo_dict = {
                    param_names[i]: combo[i] for i in range(len(param_names))
                }
                combo_dict["priority"] = "medium"
                
                # Avoid duplicates
                if combo_dict not in high_impact_combos:
                    additional_combos.append(combo_dict)
            
            high_impact_combos.extend(additional_combos[:remaining])
        
        print(f"🔬 Generated {len(high_impact_combos)} hyperparameter combinations")
        print(f"   High priority (LR + exploration focus): {len([c for c in high_impact_combos if c.get('priority') == 'high'])}")
        
        return high_impact_combos[:max_combinations]
    
    def evaluate_hyperparameters(self, params: Dict[str, Any], 
                                training_steps: int = 20) -> Dict[str, float]:
        """Evaluate a specific hyperparameter combination"""
        
        print(f"\n🧪 Testing hyperparameters:")
        for key, value in params.items():
            if key != "priority":
                print(f"   {key}: {value}")
        
        try:
            # Setup configuration with test parameters
            temp_sim = TradeSimulator(num_sims=1)
            state_dim = temp_sim.state_dim
            temp_sim.set_reward_type(self.reward_type)
            
            env_args = {
                "env_name": "TradeSimulator-v0",
                "num_envs": 4,  # Small for quick testing
                "max_step": 200,  # Reasonable for testing
                "state_dim": state_dim,
                "action_dim": 3,
                "if_discrete": True,
                "max_position": 1,
                "slippage": 7e-7,
                "num_sims": 4,
                "step_gap": 2,
            }
            
            config = EnhancedConfig(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
            config.gpu_id = -1  # Use CPU for speed
            config.state_dim = state_dim
            
            # Apply test hyperparameters
            config.break_step = training_steps
            config.learning_rate = params["learning_rate"]
            config.explore_rate = params["exploration_rate"] 
            config.initial_exploration = params["exploration_rate"]
            config.batch_size = params["batch_size"]
            config.net_dims = params["network_architecture"]
            config.gamma = params["gamma"]
            config.horizon_len = 100  # Smaller for speed
            config.buffer_size = 500   # Smaller for speed
            
            # Quick training test
            from erl_config import build_env
            from erl_replay_buffer import ReplayBuffer
            
            # Build environment
            env = build_env(config.env_class, config.env_args, config.gpu_id)
            
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
            
            # Buffer
            buffer = ReplayBuffer(
                gpu_id=config.gpu_id,
                num_seqs=config.num_envs,
                max_size=config.buffer_size,
                state_dim=config.state_dim,
                action_dim=1,
            )
            
            # Warm up
            buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
            buffer.update(buffer_items)
            
            # Training metrics
            rewards = []
            losses_critic = []
            losses_actor = []
            training_start = time.time()
            
            # Quick training
            for step in range(training_steps):
                # Collect experience
                buffer_items = agent.explore_env(env, config.horizon_len)
                exp_r = buffer_items[2].mean().item()
                rewards.append(exp_r)
                
                # Update buffer
                buffer.update(buffer_items)
                
                # Update network
                torch.set_grad_enabled(True)
                logging_tuple = agent.update_net(buffer)
                torch.set_grad_enabled(False)
                
                if logging_tuple:
                    obj_critic, obj_actor = logging_tuple[:2]
                    losses_critic.append(obj_critic)
                    losses_actor.append(obj_actor)
            
            training_time = time.time() - training_start
            
            # Evaluation
            eval_rewards = []
            action_counts = {0: 0, 1: 0, 2: 0}
            
            for _ in range(3):  # Quick evaluation
                state = env.reset()
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                state = state.to(agent.device)
                
                episode_reward = 0
                for _ in range(20):
                    with torch.no_grad():
                        q_values = agent.act(state)
                        action = q_values.argmax(dim=1, keepdim=True)
                        action_counts[action[0].item()] += 1
                    
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward.mean().item()
                    
                    if done.any():
                        break
                        
                    state = next_state
                
                eval_rewards.append(episode_reward)
            
            env.close() if hasattr(env, "close") else None
            
            # Calculate metrics
            avg_reward = np.mean(rewards) if rewards else 0
            avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
            avg_critic_loss = np.mean(losses_critic) if losses_critic else 0
            avg_actor_loss = np.mean(losses_actor) if losses_actor else 0
            action_variety = len([v for v in action_counts.values() if v > 0])
            
            # Profitability score (key metric)
            diversity_bonus = action_variety / 3.0 * 0.5  # Bonus for trading diversity
            stability_bonus = -abs(avg_critic_loss) * 0.01 if avg_critic_loss != 0 else 0  # Penalty for instability
            profitability_score = avg_eval_reward + diversity_bonus + stability_bonus
            
            results = {
                "avg_reward": avg_reward,
                "avg_eval_reward": avg_eval_reward,
                "avg_critic_loss": avg_critic_loss,
                "avg_actor_loss": avg_actor_loss,
                "action_variety": action_variety,
                "training_time": training_time,
                "profitability_score": profitability_score,
                "action_counts": action_counts
            }
            
            print(f"   📊 Results:")
            print(f"      Profitability Score: {profitability_score:.4f}")
            print(f"      Avg Eval Reward: {avg_eval_reward:.4f}")
            print(f"      Action Variety: {action_variety}/3")
            print(f"      Training Time: {training_time:.1f}s")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error evaluating hyperparameters: {e}")
            return {
                "avg_reward": -1000,
                "avg_eval_reward": -1000,
                "profitability_score": -1000,
                "action_variety": 0,
                "training_time": float('inf'),
                "error": str(e)
            }
    
    def run_optimization(self, max_combinations: int = 10, training_steps: int = 20) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        print(f"\n🚀 Starting Hyperparameter Optimization")
        print(f"   Max combinations: {max_combinations}")
        print(f"   Training steps per test: {training_steps}")
        print("=" * 60)
        
        # Define search space
        search_space = self.define_search_space()
        
        # Generate combinations
        combinations = self.create_config_combinations(search_space, max_combinations)
        
        # Test each combination
        optimization_start = time.time()
        
        for i, params in enumerate(combinations):
            print(f"\n🧪 Testing combination {i+1}/{len(combinations)}")
            print(f"   Priority: {params.get('priority', 'unknown')}")
            
            results = self.evaluate_hyperparameters(params, training_steps)
            results["combination_id"] = i
            results["params"] = params
            
            self.results.append(results)
        
        optimization_time = time.time() - optimization_start
        
        # Analyze results
        best_result = max(self.results, key=lambda x: x["profitability_score"])
        
        print(f"\n🎉 Hyperparameter Optimization Complete!")
        print(f"⏱️  Total time: {optimization_time:.1f}s")
        print("=" * 60)
        
        print(f"\n🏆 BEST HYPERPARAMETERS:")
        best_params = best_result["params"]
        for key, value in best_params.items():
            if key != "priority":
                print(f"   {key}: {value}")
        
        print(f"\n📊 BEST PERFORMANCE:")
        print(f"   Profitability Score: {best_result['profitability_score']:.4f}")
        print(f"   Avg Eval Reward: {best_result['avg_eval_reward']:.4f}")
        print(f"   Action Variety: {best_result['action_variety']}/3")
        print(f"   Action Distribution: {best_result.get('action_counts', {})}")
        
        # Summary of improvements
        baseline_lr = 2e-6
        baseline_exploration = 0.005
        best_lr = best_params["learning_rate"]
        best_exploration = best_params["exploration_rate"]
        
        print(f"\n📈 IMPROVEMENTS vs BASELINE:")
        print(f"   Learning Rate: {baseline_lr:.2e} → {best_lr:.2e} ({best_lr/baseline_lr:.0f}x increase)")
        print(f"   Exploration Rate: {baseline_exploration:.3f} → {best_exploration:.3f} ({best_exploration/baseline_exploration:.0f}x increase)")
        print(f"   Network Architecture: {best_params['network_architecture']}")
        
        return {
            "best_params": best_params,
            "best_results": best_result,
            "all_results": self.results,
            "optimization_time": optimization_time
        }
    
    def save_results(self, optimization_results: Dict[str, Any], 
                    filename: str = "hyperparameter_optimization_results.txt"):
        """Save optimization results to file"""
        
        with open(filename, 'w') as f:
            f.write("🎯 Hyperparameter Optimization Results\n")
            f.write("=" * 60 + "\n\n")
            
            best_params = optimization_results["best_params"]
            best_results = optimization_results["best_results"]
            
            f.write("🏆 BEST HYPERPARAMETERS:\n")
            for key, value in best_params.items():
                if key != "priority":
                    f.write(f"   {key}: {value}\n")
            
            f.write(f"\n📊 BEST PERFORMANCE:\n")
            f.write(f"   Profitability Score: {best_results['profitability_score']:.4f}\n")
            f.write(f"   Avg Eval Reward: {best_results['avg_eval_reward']:.4f}\n")
            f.write(f"   Action Variety: {best_results['action_variety']}/3\n")
            
            f.write(f"\n📋 ALL RESULTS:\n")
            for i, result in enumerate(optimization_results["all_results"]):
                f.write(f"\nCombination {i+1}:\n")
                f.write(f"   Params: {result['params']}\n")
                f.write(f"   Profitability Score: {result['profitability_score']:.4f}\n")
                f.write(f"   Action Variety: {result['action_variety']}/3\n")
        
        print(f"📄 Results saved to: {filename}")


def main():
    """Main hyperparameter optimization execution"""
    
    print("🎯 Hyperparameter Optimization for Profitability")
    print("=" * 60)
    
    # Get parameters
    reward_type = sys.argv[1] if len(sys.argv) > 1 else "multi_objective"
    max_combinations = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    training_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    print(f"🎯 Reward type: {reward_type}")
    print(f"🔬 Max combinations: {max_combinations}")
    print(f"🏋️  Training steps per test: {training_steps}")
    
    # Run optimization
    optimizer = HyperparameterOptimizer(reward_type=reward_type)
    results = optimizer.run_optimization(
        max_combinations=max_combinations,
        training_steps=training_steps
    )
    
    # Save results
    optimizer.save_results(results)
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Use best hyperparameters in extended training")
    print(f"   2. Run full 200-step training with optimized params")
    print(f"   3. Compare against baseline performance")
    
    return results


if __name__ == "__main__":
    results = main()