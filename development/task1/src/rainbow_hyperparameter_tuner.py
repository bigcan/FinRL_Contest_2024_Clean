#!/usr/bin/env python3
"""
Rainbow DQN Hyperparameter Tuning and Stability Monitoring
Addresses the critical need for proper Rainbow DQN configuration and monitoring
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json
from collections import deque

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from erl_agent_rainbow import AgentRainbow, PrioritizedReplayBuffer
from enhanced_training_config import EnhancedConfig
from optimized_hyperparameters import get_optimized_hyperparameters, apply_optimized_hyperparameters
from erl_config import build_env


@dataclass
class RainbowHyperparams:
    """Rainbow DQN hyperparameter configuration"""
    learning_rate: float = 1e-4
    n_step: int = 3
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 10000
    noisy_std: float = 0.017
    target_update_freq: int = 4
    soft_update_tau: float = 0.005
    gradient_clip: float = 10.0
    batch_size: int = 512
    buffer_size: int = 50000


@dataclass 
class RainbowMetrics:
    """Comprehensive Rainbow DQN training metrics"""
    step: int
    episode_return: float
    distributional_loss: float
    td_error_mean: float
    td_error_std: float
    q_value_mean: float
    q_value_std: float
    priority_weight_mean: float
    noisy_weight_std: float
    exploration_entropy: float
    target_q_diff: float
    gradient_norm: float
    learning_rate: float
    training_time: float
    
    # Stability indicators
    loss_variance: float
    q_value_drift: float
    priority_saturation: float
    exploration_decay: float


class RainbowStabilityMonitor:
    """
    Real-time stability monitoring for Rainbow DQN
    Detects training instabilities and suggests corrections
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.stability_alerts = []
        
        # Stability thresholds
        self.loss_explosion_threshold = 10.0
        self.q_drift_threshold = 5.0
        self.gradient_explosion_threshold = 50.0
        self.priority_saturation_threshold = 0.95
        self.exploration_collapse_threshold = 0.01
        
        print(f"🔍 Rainbow Stability Monitor initialized:")
        print(f"   Window size: {window_size}")
        print(f"   Monitoring: loss, Q-values, gradients, priorities, exploration")
    
    def add_metrics(self, metrics: RainbowMetrics):
        """Add new metrics and check for stability issues"""
        self.metrics_history.append(metrics)
        
        if len(self.metrics_history) >= 10:  # Need some history
            self._check_stability()
    
    def _check_stability(self):
        """Check for various stability issues"""
        recent_metrics = list(self.metrics_history)[-10:]
        
        # 1. Loss explosion detection
        recent_losses = [m.distributional_loss for m in recent_metrics]
        if any(loss > self.loss_explosion_threshold for loss in recent_losses):
            self._add_alert("LOSS_EXPLOSION", 
                          f"Distributional loss exploded: {max(recent_losses):.2f}")
        
        # 2. Q-value drift detection  
        recent_q_means = [m.q_value_mean for m in recent_metrics]
        q_drift = abs(recent_q_means[-1] - recent_q_means[0])
        if q_drift > self.q_drift_threshold:
            self._add_alert("Q_VALUE_DRIFT", 
                          f"Q-values drifting: {q_drift:.2f}")
        
        # 3. Gradient explosion detection
        recent_grad_norms = [m.gradient_norm for m in recent_metrics]
        if any(grad > self.gradient_explosion_threshold for grad in recent_grad_norms):
            self._add_alert("GRADIENT_EXPLOSION", 
                          f"Gradient norm exploded: {max(recent_grad_norms):.2f}")
        
        # 4. Priority saturation detection
        recent_priority_weights = [m.priority_weight_mean for m in recent_metrics]
        if any(w > self.priority_saturation_threshold for w in recent_priority_weights):
            self._add_alert("PRIORITY_SATURATION", 
                          f"Priority weights saturated: {max(recent_priority_weights):.3f}")
        
        # 5. Exploration collapse detection
        recent_entropy = [m.exploration_entropy for m in recent_metrics]
        if any(e < self.exploration_collapse_threshold for e in recent_entropy):
            self._add_alert("EXPLORATION_COLLAPSE", 
                          f"Exploration entropy collapsed: {min(recent_entropy):.4f}")
        
        # 6. Loss variance (instability indicator)
        loss_variance = np.var(recent_losses)
        if loss_variance > 5.0:
            self._add_alert("HIGH_LOSS_VARIANCE", 
                          f"Training unstable, loss variance: {loss_variance:.2f}")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add stability alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': time.time(),
            'step': self.metrics_history[-1].step if self.metrics_history else 0
        }
        self.stability_alerts.append(alert)
        print(f"🚨 STABILITY ALERT [{alert_type}]: {message}")
    
    def get_stability_report(self) -> Dict:
        """Generate comprehensive stability report"""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data", "alerts": self.stability_alerts}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 steps
        
        # Calculate stability indicators
        losses = [m.distributional_loss for m in recent_metrics]
        q_values = [m.q_value_mean for m in recent_metrics]
        grad_norms = [m.gradient_norm for m in recent_metrics]
        
        stability_score = self._calculate_stability_score(recent_metrics)
        
        report = {
            "status": "stable" if stability_score > 0.7 else "unstable" if stability_score < 0.3 else "moderate",
            "stability_score": stability_score,
            "alerts": self.stability_alerts[-10:],  # Recent alerts
            "metrics": {
                "loss_trend": "increasing" if losses[-1] > losses[0] else "decreasing",
                "loss_variance": float(np.var(losses)),
                "q_value_drift": float(abs(q_values[-1] - q_values[0])),
                "gradient_stability": float(np.mean(grad_norms)),
                "recent_alerts": len([a for a in self.stability_alerts if time.time() - a['timestamp'] < 300])  # Last 5 min
            },
            "recommendations": self._get_stability_recommendations(stability_score)
        }
        
        return report
    
    def _calculate_stability_score(self, metrics: List[RainbowMetrics]) -> float:
        """Calculate overall stability score (0-1)"""
        scores = []
        
        # Loss stability (lower variance = higher score)
        losses = [m.distributional_loss for m in metrics]
        loss_stability = max(0, 1 - np.var(losses) / 10.0)
        scores.append(loss_stability)
        
        # Q-value stability  
        q_values = [m.q_value_mean for m in metrics]
        q_drift = abs(q_values[-1] - q_values[0])
        q_stability = max(0, 1 - q_drift / 10.0)
        scores.append(q_stability)
        
        # Gradient stability
        grad_norms = [m.gradient_norm for m in metrics]
        grad_stability = max(0, 1 - max(grad_norms) / 50.0)
        scores.append(grad_stability)
        
        # Exploration maintenance
        entropies = [m.exploration_entropy for m in metrics]
        exploration_stability = min(1, np.mean(entropies) / 0.5)
        scores.append(exploration_stability)
        
        return np.mean(scores)
    
    def _get_stability_recommendations(self, stability_score: float) -> List[str]:
        """Get recommendations based on stability score"""
        recommendations = []
        
        if stability_score < 0.3:
            recommendations.extend([
                "CRITICAL: Reduce learning rate by 50%",
                "CRITICAL: Increase gradient clipping (reduce max norm)",
                "CRITICAL: Reduce priority alpha to 0.4",
                "Consider reducing n_atoms to 21 for simpler distribution"
            ])
        elif stability_score < 0.7:
            recommendations.extend([
                "Moderate instability detected",
                "Consider reducing learning rate by 25%",
                "Monitor loss curves closely",
                "May need priority beta adjustment"
            ])
        else:
            recommendations.extend([
                "Training appears stable",
                "Continue current configuration",
                "Monitor for any sudden changes"
            ])
        
        return recommendations


class RainbowHyperparameterTuner:
    """
    Systematic hyperparameter tuning for Rainbow DQN
    Uses grid search and performance-based selection
    """
    
    def __init__(self, reward_type: str = "simple", gpu_id: int = -1):
        self.reward_type = reward_type
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")
        
        # Get state dimension
        temp_sim = TradeSimulator(num_sims=1)
        self.state_dim = temp_sim.state_dim
        temp_sim.set_reward_type(reward_type)
        
        # Tuning results
        self.tuning_results = []
        
        print(f"🎛️ Rainbow Hyperparameter Tuner initialized:")
        print(f"   Reward type: {reward_type}")
        print(f"   Device: {self.device}")
        print(f"   State dimension: {self.state_dim}")
    
    def get_hyperparameter_grid(self) -> List[RainbowHyperparams]:
        """Generate grid of hyperparameters to test"""
        
        # Conservative grid focusing on most impactful parameters
        learning_rates = [5e-5, 1e-4, 2e-4]  # Most critical
        n_steps = [1, 3, 5]  # Bias-variance tradeoff
        n_atoms = [21, 51, 101]  # Distributional complexity
        priority_alphas = [0.4, 0.6, 0.8]  # Priority strength
        
        grid = []
        for lr in learning_rates:
            for n_step in n_steps:
                for n_atom in n_atoms:
                    for alpha in priority_alphas:
                        # Skip extreme combinations
                        if lr == 2e-4 and n_atom == 101:  # Too aggressive
                            continue
                        if lr == 5e-5 and n_step == 1:  # Too conservative
                            continue
                            
                        params = RainbowHyperparams(
                            learning_rate=lr,
                            n_step=n_step,
                            n_atoms=n_atom,
                            priority_alpha=alpha,
                            # Keep other params at reasonable defaults
                            v_min=-5.0,  # Tighter range for crypto
                            v_max=5.0,
                            priority_beta_start=0.4,
                            noisy_std=0.017,
                            target_update_freq=4,
                            batch_size=256,  # Smaller for faster tuning
                            buffer_size=10000
                        )
                        grid.append(params)
        
        print(f"📊 Generated hyperparameter grid: {len(grid)} configurations")
        return grid
    
    def test_hyperparameter_config(self, 
                                  params: RainbowHyperparams,
                                  test_steps: int = 20) -> Dict:
        """Test a single hyperparameter configuration"""
        
        print(f"\n🧪 Testing config: LR={params.learning_rate:.0e}, N-step={params.n_step}, "
              f"Atoms={params.n_atoms}, Alpha={params.priority_alpha}")
        
        test_start = time.time()
        
        try:
            # Setup environment
            env_args = {
                "env_name": "TradeSimulator-v0",
                "num_envs": 4,  # Small for fast tuning
                "max_step": 200,
                "state_dim": self.state_dim,
                "action_dim": 3,
                "if_discrete": True,
                "max_position": 1,
                "slippage": 7e-7,
                "num_sims": 4,
                "step_gap": 2,
            }
            
            config = EnhancedConfig(agent_class=AgentRainbow, env_class=TradeSimulator, env_args=env_args)
            config.gpu_id = self.gpu_id
            config.state_dim = self.state_dim
            
            # Apply hyperparameters
            config.learning_rate = params.learning_rate
            config.rainbow_n_step = params.n_step
            config.rainbow_n_atoms = params.n_atoms
            config.rainbow_v_min = params.v_min
            config.rainbow_v_max = params.v_max
            config.rainbow_use_noisy = True
            config.rainbow_use_prioritized = True
            config.batch_size = params.batch_size
            config.buffer_size = params.buffer_size
            config.break_step = test_steps
            config.net_dims = (64, 32)  # Smaller for faster tuning
            
            # Build environment and agent
            env = build_env(config.env_class, env_args, config.gpu_id)
            env.set_reward_type(self.reward_type)
            
            agent = AgentRainbow(
                config.net_dims,
                config.state_dim,
                config.action_dim,
                gpu_id=config.gpu_id,
                args=config,
            )
            
            # Create prioritized buffer
            buffer = PrioritizedReplayBuffer(
                max_size=params.buffer_size,
                state_dim=config.state_dim,
                action_dim=1,
                gpu_id=config.gpu_id,
                alpha=params.priority_alpha,
                beta_start=params.priority_beta_start,
                beta_frames=test_steps * 200
            )
            
            # Initialize
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state = state.to(agent.device)
            agent.last_state = state.detach()
            
            # Warm up
            buffer_items = agent.explore_env(env, 50, if_random=True)
            buffer.update(buffer_items)
            
            # Training with monitoring
            monitor = RainbowStabilityMonitor(window_size=test_steps)
            rewards = []
            losses = []
            
            for step in range(test_steps):
                # Experience collection
                buffer_items = agent.explore_env(env, 50)
                exp_r = buffer_items[2].mean().item()
                rewards.append(exp_r)
                
                # Update buffer and agent
                buffer.update(buffer_items)
                loss, q_mean = agent.update_net(buffer)
                losses.append(loss)
                
                # Monitor stability
                metrics = RainbowMetrics(
                    step=step,
                    episode_return=exp_r,
                    distributional_loss=loss,
                    td_error_mean=0.0,  # Simplified for tuning
                    td_error_std=0.0,
                    q_value_mean=q_mean,
                    q_value_std=0.0,
                    priority_weight_mean=buffer.get_beta(),
                    noisy_weight_std=0.1,  # Placeholder
                    exploration_entropy=0.5,  # Placeholder
                    target_q_diff=0.0,
                    gradient_norm=1.0,  # Placeholder
                    learning_rate=params.learning_rate,
                    training_time=time.time() - test_start,
                    loss_variance=np.var(losses[-10:]) if len(losses) >= 10 else 0.0,
                    q_value_drift=0.0,
                    priority_saturation=0.0,
                    exploration_decay=0.0
                )
                monitor.add_metrics(metrics)
            
            # Calculate performance metrics
            avg_reward = np.mean(rewards)
            final_reward = rewards[-1]
            reward_improvement = final_reward - rewards[0] if len(rewards) > 1 else 0
            loss_stability = 1.0 / (1.0 + np.var(losses))  # Inverse of loss variance
            
            # Get stability report
            stability_report = monitor.get_stability_report()
            
            # Overall score combining performance and stability
            performance_score = avg_reward
            stability_score = stability_report['stability_score']
            overall_score = 0.7 * performance_score + 0.3 * stability_score
            
            result = {
                'hyperparams': params,
                'performance': {
                    'avg_reward': avg_reward,
                    'final_reward': final_reward,
                    'improvement': reward_improvement,
                    'loss_stability': loss_stability
                },
                'stability': stability_report,
                'overall_score': overall_score,
                'training_time': time.time() - test_start,
                'success': True
            }
            
            print(f"   ✅ Score: {overall_score:.3f} (Perf: {performance_score:.3f}, Stab: {stability_score:.3f})")
            
            env.close() if hasattr(env, "close") else None
            return result
            
        except Exception as e:
            print(f"   ❌ Config failed: {e}")
            return {
                'hyperparams': params,
                'performance': {'avg_reward': -1000, 'final_reward': -1000, 'improvement': -1000, 'loss_stability': 0},
                'stability': {'status': 'failed', 'stability_score': 0.0},
                'overall_score': -1000,
                'training_time': time.time() - test_start,
                'success': False,
                'error': str(e)
            }
    
    def run_hyperparameter_tuning(self, 
                                 max_configs: int = 10,
                                 test_steps: int = 20) -> Dict:
        """Run systematic hyperparameter tuning"""
        
        print(f"🎛️ STARTING RAINBOW HYPERPARAMETER TUNING")
        print("=" * 60)
        
        tuning_start = time.time()
        
        # Get hyperparameter grid
        param_grid = self.get_hyperparameter_grid()
        
        # Limit to max_configs for practical runtime
        if len(param_grid) > max_configs:
            print(f"📊 Testing {max_configs} of {len(param_grid)} configurations")
            # Sample diverse configurations
            indices = np.linspace(0, len(param_grid)-1, max_configs, dtype=int)
            param_grid = [param_grid[i] for i in indices]
        
        # Test each configuration
        self.tuning_results = []
        for i, params in enumerate(param_grid, 1):
            print(f"\n[{i}/{len(param_grid)}] Testing configuration...")
            result = self.test_hyperparameter_config(params, test_steps)
            self.tuning_results.append(result)
        
        total_time = time.time() - tuning_start
        
        # Analyze results
        return self._analyze_tuning_results(total_time)
    
    def _analyze_tuning_results(self, total_time: float) -> Dict:
        """Analyze hyperparameter tuning results"""
        
        print(f"\n📊 RAINBOW HYPERPARAMETER TUNING RESULTS")
        print("=" * 60)
        
        successful_results = [r for r in self.tuning_results if r['success']]
        failed_results = [r for r in self.tuning_results if not r['success']]
        
        print(f"⏱️ Total tuning time: {total_time:.1f}s")
        print(f"✅ Successful configs: {len(successful_results)}/{len(self.tuning_results)}")
        
        if not successful_results:
            print("❌ No successful configurations found!")
            return {'status': 'failed', 'results': self.tuning_results}
        
        # Rank by overall score
        ranked_results = sorted(successful_results, key=lambda x: x['overall_score'], reverse=True)
        
        print(f"\n🏆 TOP CONFIGURATIONS:")
        for i, result in enumerate(ranked_results[:5], 1):
            params = result['hyperparams']
            score = result['overall_score']
            perf = result['performance']['avg_reward']
            stab = result['stability']['stability_score']
            
            print(f"   {i}. Score={score:.3f} | LR={params.learning_rate:.0e}, N-step={params.n_step}, "
                  f"Atoms={params.n_atoms}, Alpha={params.priority_alpha:.1f} | Perf={perf:.3f}, Stab={stab:.3f}")
        
        # Best configuration analysis
        best_config = ranked_results[0]
        best_params = best_config['hyperparams']
        
        print(f"\n🥇 OPTIMAL CONFIGURATION:")
        print(f"   Learning Rate: {best_params.learning_rate:.0e}")
        print(f"   N-step: {best_params.n_step}")
        print(f"   N-atoms: {best_params.n_atoms}")
        print(f"   Priority Alpha: {best_params.priority_alpha}")
        print(f"   V-range: [{best_params.v_min}, {best_params.v_max}]")
        print(f"   Overall Score: {best_config['overall_score']:.3f}")
        print(f"   Performance: {best_config['performance']['avg_reward']:.3f}")
        print(f"   Stability: {best_config['stability']['stability_score']:.3f}")
        
        # Parameter sensitivity analysis
        print(f"\n🔍 PARAMETER SENSITIVITY ANALYSIS:")
        self._analyze_parameter_sensitivity(successful_results)
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        stability_status = best_config['stability']['status']
        if stability_status == 'stable':
            print(f"   ✅ Use optimal configuration for production training")
            print(f"   📈 Expected stable learning with good performance")
        elif stability_status == 'moderate':
            print(f"   ⚠️ Monitor training closely with optimal configuration")
            print(f"   🔧 Consider reducing learning rate if instability occurs")
        else:
            print(f"   🚨 All configurations showed instability")
            print(f"   🛠️ Implement more conservative hyperparameters")
        
        analysis_result = {
            'status': 'success',
            'total_time': total_time,
            'successful_configs': len(successful_results),
            'failed_configs': len(failed_results),
            'best_config': best_config,
            'top_5_configs': ranked_results[:5],
            'all_results': self.tuning_results
        }
        
        return analysis_result
    
    def _analyze_parameter_sensitivity(self, results: List[Dict]):
        """Analyze sensitivity of each parameter"""
        
        # Group results by parameter values
        param_effects = {
            'learning_rate': {},
            'n_step': {},
            'n_atoms': {},
            'priority_alpha': {}
        }
        
        for result in results:
            params = result['hyperparams']
            score = result['overall_score']
            
            # Group by each parameter
            for param_name in param_effects.keys():
                param_value = getattr(params, param_name)
                if param_value not in param_effects[param_name]:
                    param_effects[param_name][param_value] = []
                param_effects[param_name][param_value].append(score)
        
        # Calculate average performance for each parameter value
        for param_name, value_scores in param_effects.items():
            print(f"   {param_name}:")
            avg_scores = {v: np.mean(scores) for v, scores in value_scores.items()}
            best_value = max(avg_scores, key=avg_scores.get)
            
            for value, avg_score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
                marker = "🥇" if value == best_value else "  "
                print(f"     {marker} {value}: {avg_score:.3f}")


def main():
    """Main execution for Rainbow hyperparameter tuning"""
    
    print("🎛️ RAINBOW DQN HYPERPARAMETER TUNING & STABILITY MONITORING")
    print("=" * 80)
    
    # Create tuner
    tuner = RainbowHyperparameterTuner(
        reward_type="simple",  # Best reward from A/B testing
        gpu_id=-1  # CPU for broad compatibility
    )
    
    # Run tuning
    results = tuner.run_hyperparameter_tuning(
        max_configs=9,  # 3x3 grid for practical runtime
        test_steps=15   # Quick but informative
    )
    
    if results['status'] == 'success':
        # Save results
        results_file = "rainbow_tuning_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if key == 'best_config':
                    json_results[key] = {
                        'overall_score': float(value['overall_score']),
                        'hyperparams': {
                            'learning_rate': float(value['hyperparams'].learning_rate),
                            'n_step': int(value['hyperparams'].n_step),
                            'n_atoms': int(value['hyperparams'].n_atoms),
                            'priority_alpha': float(value['hyperparams'].priority_alpha)
                        }
                    }
                elif key not in ['top_5_configs', 'all_results']:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
        print(f"🎯 Use optimal configuration for production Rainbow training")
    else:
        print(f"\n❌ Hyperparameter tuning failed - check system configuration")
    
    return results


if __name__ == "__main__":
    results = main()