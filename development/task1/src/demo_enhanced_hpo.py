"""
Demo Enhanced HPO with Conservative Trading Solution
Runs a small demonstration of the enhanced HPO system
"""

import os
import sys
import time
import numpy as np
import json
from datetime import datetime
import optuna

# Import our components
from hpo_config import Task1HPOSearchSpace
from training_monitor import ActionDiversityMonitor
from trade_simulator import TradeSimulator
from erl_agent import AgentDoubleDQN
from erl_config import Config


class DemoHPORunner:
    """Demonstration HPO runner with action diversity monitoring"""
    
    def __init__(self, results_dir="demo_hpo_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Trial tracking
        self.trial_results = []
        
    def objective(self, trial):
        """Demo objective function with action diversity monitoring"""
        
        # Get enhanced parameters
        params = Task1HPOSearchSpace.suggest_parameters(trial)
        
        print(f"\n🎯 Trial {trial.number}: Testing enhanced parameters")
        print(f"   Explore rate: {params['explore_rate']:.4f}")
        print(f"   Min explore rate: {params['min_explore_rate']:.4f}")
        print(f"   Reward type: {params['reward_type']}")
        print(f"   Conservatism penalty: {params['conservatism_penalty_weight']:.3f}")
        
        try:
            # Create enhanced simulator
            simulator = TradeSimulator(
                num_sims=8,  # Small for demo
                gpu_id=-1,   # CPU
                data_length=2000  # Limited data for speed
            )
            
            # Set reward system
            reward_weights = {
                'conservatism_penalty_weight': params['conservatism_penalty_weight'],
                'action_diversity_weight': params['action_diversity_weight'],
                'transaction_cost_weight': params['transaction_cost_weight'],
                'risk_adjusted_return_weight': params['risk_adjusted_return_weight']
            }
            
            simulator.set_reward_type(params['reward_type'], reward_weights)
            
            # Create enhanced agent
            args = Config()
            args.explore_rate = params['explore_rate']
            args.min_explore_rate = params['min_explore_rate']
            args.exploration_decay_rate = params['exploration_decay_rate']
            args.exploration_warmup_steps = params['exploration_warmup_steps']
            args.force_exploration_probability = params['force_exploration_probability']
            
            agent = AgentDoubleDQN(
                net_dims=(64, 32, 16),  # Small network for demo
                state_dim=simulator.state_dim,
                action_dim=simulator.action_dim,
                gpu_id=-1,
                args=args
            )
            
            # Initialize action diversity monitor
            monitor = ActionDiversityMonitor(
                window_size=500,
                diversity_threshold=0.3,
                conservatism_threshold=0.7,
                checkpoint_dir=os.path.join(self.results_dir, f"trial_{trial.number}")
            )
            
            # Run demo training
            results = self._run_demo_training(simulator, agent, monitor, trial.number)
            
            # Calculate objective
            objective_value = self._calculate_objective(results, monitor)
            
            # Save results
            self._save_trial(trial, params, results, monitor, objective_value)
            
            print(f"✅ Trial {trial.number} completed - Objective: {objective_value:.6f}")
            
            return objective_value
            
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {str(e)}")
            return -10.0
            
    def _run_demo_training(self, simulator, agent, monitor, trial_number):
        """Run demo training with monitoring"""
        
        print(f"🚀 Running demo training for trial {trial_number}")
        
        episode_returns = []
        
        # Run training episodes
        for episode in range(20):  # Short demo
            state = simulator.reset()
            episode_return = 0.0
            episode_actions = []
            
            # Run episode
            for step in range(200):  # Short episodes
                # Get action from agent (simplified)
                action = np.random.choice([0, 1, 2], size=(simulator.num_sims, 1))
                action = agent.exploration_orchestrator.get_masked_action(
                    q_values=np.random.randn(simulator.num_sims, 3),
                    temperature=0.1
                ).unsqueeze(1)
                
                # Step environment
                next_state, reward, done, _ = simulator.step(action)
                
                # Track for monitoring
                action_int = action[0].item()
                episode_actions.append(action_int)
                episode_return += reward.mean().item()
                
                # Update monitor
                monitor.update(action_int, reward.mean().item(), done.any())
                
                state = next_state
                
                if done.any():
                    break
                    
            episode_returns.append(episode_return)
            
            if episode % 5 == 0:
                diversity_check = monitor.check_diversity()
                print(f"   Episode {episode}: Return={episode_return:.2f}, "
                      f"Status={diversity_check.get('status', 'N/A')}")
        
        # Calculate results
        return {
            'episode_returns': episode_returns,
            'mean_return': np.mean(episode_returns),
            'total_return': np.sum(episode_returns),
            'return_std': np.std(episode_returns),
            'sharpe_ratio': np.mean(episode_returns) / max(np.std(episode_returns), 1e-8)
        }
        
    def _calculate_objective(self, results, monitor):
        """Calculate enhanced objective with diversity bonus"""
        
        base_sharpe = results['sharpe_ratio']
        
        # Handle invalid Sharpe ratios
        if not np.isfinite(base_sharpe) or abs(base_sharpe) > 100:
            base_sharpe = -5.0
            
        # Get diversity metrics
        diversity_check = monitor.check_diversity()
        diversity_metrics = diversity_check.get('metrics', {})
        
        # Action diversity bonus
        entropy = diversity_metrics.get('entropy', 0.0)
        diversity_bonus = min(2.0, entropy * 3.0)
        
        # Conservative penalty
        hold_ratio = diversity_metrics.get('hold_ratio', 0.5)
        buy_ratio = diversity_metrics.get('buy_ratio', 0.3)
        
        conservatism_penalty = 0.0
        if hold_ratio > 0.7:
            conservatism_penalty += (hold_ratio - 0.7) * 5.0
        if buy_ratio < 0.1:
            conservatism_penalty += 2.0
        if results['total_return'] <= 0:
            conservatism_penalty += 3.0
            
        # Combined objective
        objective = base_sharpe + diversity_bonus - conservatism_penalty
        
        print(f"   📊 Objective breakdown:")
        print(f"      Base Sharpe: {base_sharpe:.3f}")
        print(f"      Diversity bonus: {diversity_bonus:.3f}")
        print(f"      Conservatism penalty: {conservatism_penalty:.3f}")
        print(f"      Final objective: {objective:.3f}")
        
        return objective
        
    def _save_trial(self, trial, params, results, monitor, objective_value):
        """Save trial results"""
        
        trial_data = {
            'trial_number': trial.number,
            'objective_value': objective_value,
            'parameters': params,
            'results': results,
            'diversity_metrics': monitor.check_diversity(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.trial_results.append(trial_data)
        
        # Save individual trial
        trial_file = os.path.join(self.results_dir, f"trial_{trial.number}.json")
        with open(trial_file, 'w') as f:
            json.dump(trial_data, f, indent=2)
            
    def run_demo(self, n_trials=5):
        """Run demo HPO"""
        
        print("🚀 Enhanced HPO Demo with Conservative Trading Solution")
        print("=" * 60)
        print(f"   Number of trials: {n_trials}")
        print(f"   Results directory: {self.results_dir}")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name="demo_enhanced_conservative_solution"
        )
        
        start_time = time.time()
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        duration = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 60)
        print("✅ Demo HPO Completed!")
        print("=" * 60)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Best objective: {study.best_value:.6f}")
        print(f"Best trial: {study.best_trial.number}")
        
        print(f"\n📊 Best Parameters:")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                print(f"   {param}: {value:.6f}")
            else:
                print(f"   {param}: {value}")
                
        # Analyze conservative behavior
        self._analyze_conservative_behavior()
        
        return study
        
    def _analyze_conservative_behavior(self):
        """Analyze conservative behavior across trials"""
        
        if not self.trial_results:
            return
            
        print(f"\n📊 Conservative Behavior Analysis:")
        print("-" * 40)
        
        conservative_count = 0
        total_trials = len(self.trial_results)
        
        hold_ratios = []
        buy_ratios = []
        entropies = []
        
        for trial in self.trial_results:
            diversity_metrics = trial.get('diversity_metrics', {}).get('metrics', {})
            
            if diversity_metrics:
                hold_ratio = diversity_metrics.get('hold_ratio', 0)
                buy_ratio = diversity_metrics.get('buy_ratio', 0)
                entropy = diversity_metrics.get('entropy', 0)
                
                hold_ratios.append(hold_ratio)
                buy_ratios.append(buy_ratio)
                entropies.append(entropy)
                
                if hold_ratio > 0.7 or buy_ratio < 0.1:
                    conservative_count += 1
                    
        if hold_ratios:
            print(f"Conservative trials: {conservative_count}/{total_trials} ({conservative_count/total_trials:.1%})")
            print(f"Average hold ratio: {np.mean(hold_ratios):.1%}")
            print(f"Average buy ratio: {np.mean(buy_ratios):.1%}")
            print(f"Average entropy: {np.mean(entropies):.3f}")
            
            if conservative_count / total_trials > 0.5:
                print("⚠️ High conservative behavior detected!")
                print("   Recommendation: Increase exploration parameters")
            else:
                print("✅ Good action diversity maintained")


def main():
    """Run demo"""
    
    # Create demo runner
    demo = DemoHPORunner(
        results_dir=f"demo_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run demo with small number of trials
    study = demo.run_demo(n_trials=8)
    
    print(f"\n📁 Demo results saved to: {demo.results_dir}")


if __name__ == "__main__":
    main()