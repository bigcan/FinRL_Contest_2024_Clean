#!/usr/bin/env python3
"""
Rainbow DQN Training Monitor and Visualization
Real-time monitoring of Rainbow training with comprehensive metrics and alerts
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RainbowTrainingMonitor:
    """
    Comprehensive real-time monitoring for Rainbow DQN training
    Tracks stability, exploration, and performance metrics
    """
    
    def __init__(self, save_dir: str = "rainbow_monitoring", plot_freq: int = 20):
        self.save_dir = save_dir
        self.plot_freq = plot_freq
        os.makedirs(save_dir, exist_ok=True)
        
        # Metric storage
        self.metrics = {
            'steps': [],
            'episode_returns': [],
            'distributional_losses': [],
            'q_value_means': [],
            'q_value_stds': [],
            'td_errors': [],
            'priority_weights': [],
            'gradient_norms': [],
            'learning_rates': [],
            'exploration_entropies': [],
            'noisy_weights_std': [],
            'target_q_diffs': [],
            'training_times': [],
            
            # Stability indicators
            'loss_smoothness': [],
            'q_value_stability': [],
            'exploration_consistency': [],
            'priority_diversity': []
        }
        
        # Alert system
        self.alerts = []
        self.stability_score_history = deque(maxlen=100)
        
        # Plotting setup
        self.fig = None
        self.axes = None
        
        print(f"📊 Rainbow Training Monitor initialized:")
        print(f"   Save directory: {save_dir}")
        print(f"   Plot frequency: every {plot_freq} steps")
        print(f"   Monitoring: 15+ critical Rainbow metrics")
    
    def log_training_step(self, 
                         step: int,
                         agent,
                         buffer,
                         episode_return: float,
                         distributional_loss: float,
                         training_time: float):
        """Log comprehensive metrics for a training step"""
        
        # Basic metrics
        self.metrics['steps'].append(step)
        self.metrics['episode_returns'].append(episode_return)
        self.metrics['distributional_losses'].append(distributional_loss)
        self.metrics['training_times'].append(training_time)
        
        # Agent-specific metrics
        with torch.no_grad():
            # Q-value statistics
            if hasattr(agent, 'act') and hasattr(agent.act, 'support'):
                # Sample some states for Q-value analysis
                sample_states = torch.randn(32, agent.state_dim, device=agent.device)
                q_dists = agent.act(sample_states)
                q_values = torch.sum(q_dists * agent.act.support, dim=-1)
                
                q_mean = q_values.mean().item()
                q_std = q_values.std().item()
                self.metrics['q_value_means'].append(q_mean)
                self.metrics['q_value_stds'].append(q_std)
                
                # Exploration entropy from action probabilities
                action_probs = q_dists.sum(dim=-1)  # Sum over atoms
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean().item()
                self.metrics['exploration_entropies'].append(entropy)
            else:
                self.metrics['q_value_means'].append(0.0)
                self.metrics['q_value_stds'].append(0.0)
                self.metrics['exploration_entropies'].append(0.0)
            
            # Noisy network analysis
            if hasattr(agent.act, 'use_noisy') and agent.act.use_noisy:
                # Analyze noise magnitude in first noisy layer
                if hasattr(agent.act, 'advantage_head') and hasattr(agent.act.advantage_head, 'weight_sigma'):
                    noise_std = agent.act.advantage_head.weight_sigma.std().item()
                    self.metrics['noisy_weights_std'].append(noise_std)
                else:
                    self.metrics['noisy_weights_std'].append(0.0)
            else:
                self.metrics['noisy_weights_std'].append(0.0)
            
            # Target network divergence
            if hasattr(agent, 'act_target'):
                sample_states = torch.randn(16, agent.state_dim, device=agent.device)
                current_q = agent.act.get_q_values(sample_states)
                target_q = agent.act_target.get_q_values(sample_states)
                target_diff = (current_q - target_q).abs().mean().item()
                self.metrics['target_q_diffs'].append(target_diff)
            else:
                self.metrics['target_q_diffs'].append(0.0)
        
        # Buffer-specific metrics
        if hasattr(buffer, 'get_beta'):
            priority_weight = buffer.get_beta()
            self.metrics['priority_weights'].append(priority_weight)
        else:
            self.metrics['priority_weights'].append(0.0)
        
        # Gradient norm (placeholder - would need to be passed from training)
        self.metrics['gradient_norms'].append(1.0)  # Would be actual gradient norm
        
        # Learning rate
        if hasattr(agent, 'optimizer'):
            lr = agent.optimizer.param_groups[0]['lr']
            self.metrics['learning_rates'].append(lr)
        else:
            self.metrics['learning_rates'].append(0.0)
        
        # TD error (simplified approximation)
        td_error = abs(distributional_loss)  # Simplified
        self.metrics['td_errors'].append(td_error)
        
        # Calculate stability indicators
        self._calculate_stability_indicators(step)
        
        # Check for alerts
        self._check_training_alerts(step)
        
        # Generate plots periodically
        if step % self.plot_freq == 0 and step > 0:
            self._generate_monitoring_plots(step)
    
    def _calculate_stability_indicators(self, step: int):
        """Calculate rolling stability indicators"""
        
        window_size = min(20, len(self.metrics['distributional_losses']))
        
        if window_size >= 5:
            # Loss smoothness (inverse of variance)
            recent_losses = self.metrics['distributional_losses'][-window_size:]
            loss_variance = np.var(recent_losses)
            loss_smoothness = 1.0 / (1.0 + loss_variance)
            self.metrics['loss_smoothness'].append(loss_smoothness)
            
            # Q-value stability
            recent_q_means = self.metrics['q_value_means'][-window_size:]
            q_drift = abs(recent_q_means[-1] - recent_q_means[0]) if len(recent_q_means) > 1 else 0
            q_stability = max(0, 1 - q_drift / 5.0)  # Normalize by expected range
            self.metrics['q_value_stability'].append(q_stability)
            
            # Exploration consistency
            recent_entropies = self.metrics['exploration_entropies'][-window_size:]
            entropy_variance = np.var(recent_entropies)
            exploration_consistency = 1.0 / (1.0 + entropy_variance * 10)
            self.metrics['exploration_consistency'].append(exploration_consistency)
            
            # Priority diversity
            recent_priorities = self.metrics['priority_weights'][-window_size:]
            priority_range = max(recent_priorities) - min(recent_priorities) if recent_priorities else 0
            priority_diversity = min(1.0, priority_range / 0.5)  # Normalize
            self.metrics['priority_diversity'].append(priority_diversity)
            
            # Overall stability score
            stability_score = np.mean([loss_smoothness, q_stability, exploration_consistency, priority_diversity])
            self.stability_score_history.append(stability_score)
        else:
            # Not enough data yet
            self.metrics['loss_smoothness'].append(1.0)
            self.metrics['q_value_stability'].append(1.0)
            self.metrics['exploration_consistency'].append(1.0)
            self.metrics['priority_diversity'].append(1.0)
    
    def _check_training_alerts(self, step: int):
        """Check for training issues and generate alerts"""
        
        if len(self.metrics['distributional_losses']) < 5:
            return
        
        recent_losses = self.metrics['distributional_losses'][-5:]
        recent_q_means = self.metrics['q_value_means'][-5:]
        recent_entropies = self.metrics['exploration_entropies'][-5:]
        
        # Loss explosion
        if any(loss > 20.0 for loss in recent_losses):
            self._add_alert(step, "LOSS_EXPLOSION", 
                          f"Distributional loss exploded: {max(recent_losses):.2f}")
        
        # Loss plateau
        if len(recent_losses) >= 5 and np.var(recent_losses) < 0.001 and recent_losses[-1] > 5.0:
            self._add_alert(step, "LOSS_PLATEAU", 
                          f"Loss stuck at high value: {recent_losses[-1]:.2f}")
        
        # Q-value explosion
        if any(abs(q) > 50.0 for q in recent_q_means):
            self._add_alert(step, "Q_VALUE_EXPLOSION", 
                          f"Q-values exploded: {max(recent_q_means):.2f}")
        
        # Exploration collapse
        if any(entropy < 0.01 for entropy in recent_entropies):
            self._add_alert(step, "EXPLORATION_COLLAPSE", 
                          f"Exploration entropy collapsed: {min(recent_entropies):.4f}")
        
        # Stability degradation
        if len(self.stability_score_history) >= 10:
            recent_stability = list(self.stability_score_history)[-10:]
            if recent_stability[-1] < 0.3 and np.mean(recent_stability) < 0.4:
                self._add_alert(step, "STABILITY_DEGRADATION", 
                              f"Training stability degraded: {recent_stability[-1]:.3f}")
    
    def _add_alert(self, step: int, alert_type: str, message: str):
        """Add training alert"""
        alert = {
            'step': step,
            'type': alert_type,
            'message': message,
            'timestamp': time.time()
        }
        self.alerts.append(alert)
        print(f"🚨 [{step:04d}] {alert_type}: {message}")
    
    def _generate_monitoring_plots(self, step: int):
        """Generate comprehensive monitoring plots"""
        
        if len(self.metrics['steps']) < 10:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Rainbow DQN Training Monitor - Step {step}', fontsize=16)
        
        steps = self.metrics['steps']
        
        # 1. Episode Returns
        axes[0, 0].plot(steps, self.metrics['episode_returns'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distributional Loss
        axes[0, 1].plot(steps, self.metrics['distributional_losses'], 'r-', alpha=0.7)
        axes[0, 1].set_title('Distributional Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Values
        axes[0, 2].plot(steps, self.metrics['q_value_means'], 'g-', alpha=0.7, label='Mean')
        axes[0, 2].fill_between(steps, 
                               np.array(self.metrics['q_value_means']) - np.array(self.metrics['q_value_stds']),
                               np.array(self.metrics['q_value_means']) + np.array(self.metrics['q_value_stds']),
                               alpha=0.3, color='g')
        axes[0, 2].set_title('Q-Values (Mean ± Std)')
        axes[0, 2].set_ylabel('Q-Value')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Exploration Entropy
        axes[1, 0].plot(steps, self.metrics['exploration_entropies'], 'm-', alpha=0.7)
        axes[1, 0].set_title('Exploration Entropy')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Priority Weights
        axes[1, 1].plot(steps, self.metrics['priority_weights'], 'c-', alpha=0.7)
        axes[1, 1].set_title('Priority Weights (Beta)')
        axes[1, 1].set_ylabel('Beta')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Noisy Network Weights
        axes[1, 2].plot(steps, self.metrics['noisy_weights_std'], 'orange', alpha=0.7)
        axes[1, 2].set_title('Noisy Weights Std')
        axes[1, 2].set_ylabel('Std')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Stability Indicators
        if len(self.metrics['loss_smoothness']) > 0:
            axes[2, 0].plot(steps, self.metrics['loss_smoothness'], 'purple', alpha=0.7, label='Loss Smoothness')
            axes[2, 0].plot(steps, self.metrics['q_value_stability'], 'brown', alpha=0.7, label='Q-Value Stability')
            axes[2, 0].set_title('Stability Indicators')
            axes[2, 0].set_ylabel('Stability Score')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Target Network Divergence
        axes[2, 1].plot(steps, self.metrics['target_q_diffs'], 'red', alpha=0.7)
        axes[2, 1].set_title('Target Network Divergence')
        axes[2, 1].set_ylabel('Abs Difference')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Learning Rate
        axes[2, 2].plot(steps, self.metrics['learning_rates'], 'black', alpha=0.7)
        axes[2, 2].set_title('Learning Rate')
        axes[2, 2].set_ylabel('LR')
        axes[2, 2].grid(True, alpha=0.3)
        
        # Add X-labels to bottom row
        for ax in axes[2, :]:
            ax.set_xlabel('Training Step')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'rainbow_monitor_step_{step:04d}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Monitoring plots saved: {plot_path}")
    
    def generate_training_report(self, step: int) -> Dict:
        """Generate comprehensive training report"""
        
        if len(self.metrics['steps']) < 10:
            return {"status": "insufficient_data"}
        
        # Calculate summary statistics
        recent_window = min(50, len(self.metrics['steps']))
        
        recent_returns = self.metrics['episode_returns'][-recent_window:]
        recent_losses = self.metrics['distributional_losses'][-recent_window:]
        recent_q_means = self.metrics['q_value_means'][-recent_window:]
        recent_entropies = self.metrics['exploration_entropies'][-recent_window:]
        
        # Performance metrics
        avg_return = np.mean(recent_returns)
        return_trend = "increasing" if recent_returns[-1] > recent_returns[0] else "decreasing"
        
        # Stability metrics
        current_stability = self.stability_score_history[-1] if self.stability_score_history else 0.5
        loss_variance = np.var(recent_losses)
        q_stability = 1.0 - abs(recent_q_means[-1] - recent_q_means[0]) / 10.0
        
        # Exploration health
        avg_entropy = np.mean(recent_entropies)
        entropy_trend = "increasing" if recent_entropies[-1] > recent_entropies[0] else "decreasing"
        
        # Alert summary
        recent_alerts = [a for a in self.alerts if step - a['step'] <= 50]
        critical_alerts = [a for a in recent_alerts if a['type'] in ['LOSS_EXPLOSION', 'Q_VALUE_EXPLOSION', 'STABILITY_DEGRADATION']]
        
        # Overall assessment
        if current_stability > 0.7 and len(critical_alerts) == 0:
            training_status = "healthy"
        elif current_stability > 0.4 and len(critical_alerts) <= 1:
            training_status = "moderate"
        else:
            training_status = "unstable"
        
        report = {
            "status": training_status,
            "step": step,
            "performance": {
                "avg_return": avg_return,
                "return_trend": return_trend,
                "latest_return": recent_returns[-1]
            },
            "stability": {
                "current_score": current_stability,
                "loss_variance": loss_variance,
                "q_value_stability": q_stability
            },
            "exploration": {
                "avg_entropy": avg_entropy,
                "entropy_trend": entropy_trend,
                "latest_entropy": recent_entropies[-1]
            },
            "alerts": {
                "total_alerts": len(self.alerts),
                "recent_alerts": len(recent_alerts),
                "critical_alerts": len(critical_alerts),
                "latest_alerts": recent_alerts[-3:] if recent_alerts else []
            },
            "recommendations": self._get_training_recommendations(training_status, current_stability, critical_alerts)
        }
        
        return report
    
    def _get_training_recommendations(self, status: str, stability_score: float, critical_alerts: List) -> List[str]:
        """Generate training recommendations based on current state"""
        
        recommendations = []
        
        if status == "healthy":
            recommendations.extend([
                "✅ Training is proceeding well",
                "Continue with current hyperparameters",
                "Monitor for any sudden changes"
            ])
        elif status == "moderate":
            if stability_score < 0.6:
                recommendations.append("⚠️ Consider reducing learning rate by 25%")
            if len(critical_alerts) > 0:
                recommendations.append("⚠️ Monitor loss curves closely")
            recommendations.append("📊 Increase monitoring frequency")
        else:  # unstable
            recommendations.extend([
                "🚨 URGENT: Reduce learning rate by 50%",
                "🚨 URGENT: Increase gradient clipping",
                "🚨 Consider reducing priority alpha to 0.4",
                "🚨 May need to restart with more conservative hyperparameters"
            ])
            
            # Specific recommendations based on alert types
            alert_types = set(a['type'] for a in critical_alerts)
            if 'LOSS_EXPLOSION' in alert_types:
                recommendations.append("🔥 Loss explosion detected - reduce LR immediately")
            if 'Q_VALUE_EXPLOSION' in alert_types:
                recommendations.append("💥 Q-value explosion - check network initialization")
            if 'EXPLORATION_COLLAPSE' in alert_types:
                recommendations.append("🔍 Exploration collapsed - verify noisy network settings")
        
        return recommendations
    
    def save_monitoring_data(self, filename: str = None):
        """Save all monitoring data to file"""
        
        if filename is None:
            filename = f"rainbow_monitoring_data_{int(time.time())}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Prepare data for JSON serialization
        data = {
            'metrics': self.metrics,
            'alerts': self.alerts,
            'stability_scores': list(self.stability_score_history),
            'monitoring_config': {
                'save_dir': self.save_dir,
                'plot_freq': self.plot_freq
            },
            'timestamp': time.time()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Monitoring data saved to {filepath}")
        return filepath


def create_rainbow_monitoring_dashboard(monitor: RainbowTrainingMonitor, step: int):
    """Create comprehensive monitoring dashboard"""
    
    if len(monitor.metrics['steps']) < 10:
        print("📊 Not enough data for dashboard")
        return
    
    # Create dashboard with multiple visualizations
    fig = plt.figure(figsize=(20, 14))
    
    # Define grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    steps = monitor.metrics['steps']
    
    # 1. Main performance plot (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(steps, monitor.metrics['episode_returns'], 'b-', linewidth=2, label='Episode Returns')
    line2 = ax1_twin.plot(steps, monitor.metrics['distributional_losses'], 'r-', linewidth=2, label='Distributional Loss', alpha=0.7)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Episode Return', color='b')
    ax1_twin.set_ylabel('Distributional Loss', color='r')
    ax1.set_title('Rainbow DQN Training Progress', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 2. Q-Value Distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    if len(monitor.metrics['q_value_means']) > 0:
        ax2.plot(steps, monitor.metrics['q_value_means'], 'g-', linewidth=2, label='Q-Value Mean')
        ax2.fill_between(steps, 
                        np.array(monitor.metrics['q_value_means']) - np.array(monitor.metrics['q_value_stds']),
                        np.array(monitor.metrics['q_value_means']) + np.array(monitor.metrics['q_value_stds']),
                        alpha=0.3, color='g', label='±1 Std')
    ax2.set_title('Q-Value Statistics')
    ax2.set_ylabel('Q-Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Exploration Metrics (second row, left)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(steps, monitor.metrics['exploration_entropies'], 'm-', linewidth=2, label='Action Entropy')
    if len(monitor.metrics['noisy_weights_std']) > 0:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(steps, monitor.metrics['noisy_weights_std'], 'orange', linewidth=2, label='Noisy Weights Std', alpha=0.7)
        ax3_twin.set_ylabel('Noisy Weights Std', color='orange')
    ax3.set_title('Exploration Health')
    ax3.set_ylabel('Action Entropy', color='m')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Priority and Buffer Metrics (second row, right)
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.plot(steps, monitor.metrics['priority_weights'], 'c-', linewidth=2, label='Priority Beta')
    ax4.plot(steps, monitor.metrics['target_q_diffs'], 'red', linewidth=2, label='Target Q Diff', alpha=0.7)
    ax4.set_title('Buffer and Target Network')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Stability Dashboard (third row, spans full width)
    ax5 = fig.add_subplot(gs[2, :])
    if len(monitor.metrics['loss_smoothness']) > 0:
        ax5.plot(steps, monitor.metrics['loss_smoothness'], 'purple', linewidth=2, label='Loss Smoothness')
        ax5.plot(steps, monitor.metrics['q_value_stability'], 'brown', linewidth=2, label='Q-Value Stability')
        ax5.plot(steps, monitor.metrics['exploration_consistency'], 'pink', linewidth=2, label='Exploration Consistency')
        ax5.plot(steps, monitor.metrics['priority_diversity'], 'gray', linewidth=2, label='Priority Diversity')
        
        # Add stability zones
        ax5.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Stable Zone')
        ax5.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Unstable Zone')
    
    ax5.set_title('Stability Indicators', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Stability Score')
    ax5.set_ylim(0, 1.1)
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 6. Recent Alerts Summary (bottom left)
    ax6 = fig.add_subplot(gs[3, :2])
    ax6.axis('off')
    
    recent_alerts = [a for a in monitor.alerts if step - a['step'] <= 20]
    alert_text = "Recent Alerts (Last 20 steps):\n"
    
    if recent_alerts:
        for alert in recent_alerts[-5:]:  # Show last 5 alerts
            alert_text += f"• Step {alert['step']}: {alert['type']}\n"
    else:
        alert_text += "• No recent alerts ✅"
    
    ax6.text(0.05, 0.95, alert_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 7. Training Summary (bottom right)
    ax7 = fig.add_subplot(gs[3, 2:])
    ax7.axis('off')
    
    # Generate current report
    report = monitor.generate_training_report(step)
    
    summary_text = f"Training Status: {report['status'].upper()}\n"
    summary_text += f"Current Step: {step}\n"
    summary_text += f"Avg Return: {report['performance']['avg_return']:.3f}\n"
    summary_text += f"Stability Score: {report['stability']['current_score']:.3f}\n"
    summary_text += f"Exploration Entropy: {report['exploration']['avg_entropy']:.3f}\n"
    summary_text += f"Total Alerts: {report['alerts']['total_alerts']}\n"
    
    # Color based on status
    status_color = {'healthy': 'lightgreen', 'moderate': 'lightyellow', 'unstable': 'lightcoral'}
    bg_color = status_color.get(report['status'], 'lightgray')
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))
    
    plt.suptitle(f'Rainbow DQN Comprehensive Dashboard - Step {step}', fontsize=16, fontweight='bold')
    
    # Save dashboard
    dashboard_path = os.path.join(monitor.save_dir, f'rainbow_dashboard_step_{step:04d}.png')
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comprehensive dashboard saved: {dashboard_path}")
    return dashboard_path


def test_rainbow_monitor():
    """Test the Rainbow monitoring system"""
    
    print("🧪 Testing Rainbow Training Monitor")
    print("=" * 60)
    
    # Create monitor
    monitor = RainbowTrainingMonitor(save_dir="test_rainbow_monitoring", plot_freq=5)
    
    # Simulate training data
    print("📊 Simulating training data...")
    
    for step in range(25):
        # Simulate realistic training metrics
        episode_return = 0.5 + 0.1 * np.sin(step * 0.3) + np.random.normal(0, 0.05)
        
        # Simulate initial high loss that decreases
        distributional_loss = 5.0 * np.exp(-step * 0.1) + np.random.normal(0, 0.2)
        
        training_time = 0.1 + np.random.normal(0, 0.02)
        
        # Mock agent and buffer
        class MockAgent:
            def __init__(self):
                self.state_dim = 8
                self.device = torch.device('cpu')
                
                # Mock networks
                self.act = MockNetwork()
                self.act_target = MockNetwork()
                self.optimizer = MockOptimizer()
        
        class MockNetwork:
            def __init__(self):
                self.support = torch.linspace(-5, 5, 21)
                self.use_noisy = True
                self.advantage_head = MockNoisyLayer()
            
            def __call__(self, states):
                # Return mock Q-distributions
                batch_size = states.shape[0]
                return torch.softmax(torch.randn(batch_size, 3, 21), dim=-1)
            
            def get_q_values(self, states):
                # Return mock Q-values
                batch_size = states.shape[0]
                return torch.randn(batch_size, 3)
        
        class MockNoisyLayer:
            def __init__(self):
                self.weight_sigma = torch.randn(16, 8) * 0.1
        
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{'lr': 1e-4}]
        
        class MockBuffer:
            def get_beta(self):
                return 0.4 + 0.6 * (step / 25)  # Beta annealing
        
        # Log metrics
        mock_agent = MockAgent()
        mock_buffer = MockBuffer()
        
        monitor.log_training_step(
            step=step,
            agent=mock_agent,
            buffer=mock_buffer,
            episode_return=episode_return,
            distributional_loss=distributional_loss,
            training_time=training_time
        )
        
        # Add some alerts for testing
        if step == 10:
            monitor._add_alert(step, "TEST_ALERT", "This is a test alert")
        
        if step % 5 == 0:
            print(f"   Step {step}: Return={episode_return:.3f}, Loss={distributional_loss:.3f}")
    
    # Generate final dashboard
    create_rainbow_monitoring_dashboard(monitor, 24)
    
    # Generate final report
    final_report = monitor.generate_training_report(24)
    print(f"\n📋 Final Training Report:")
    print(f"   Status: {final_report['status']}")
    print(f"   Stability Score: {final_report['stability']['current_score']:.3f}")
    print(f"   Total Alerts: {final_report['alerts']['total_alerts']}")
    
    # Save data
    monitor.save_monitoring_data("test_monitoring_data.json")
    
    print(f"\n✅ Rainbow monitoring system test completed!")
    print(f"   Generated {len(monitor.metrics['steps'])} data points")
    print(f"   Created monitoring plots and dashboard")
    print(f"   Detected {len(monitor.alerts)} alerts")


if __name__ == "__main__":
    test_rainbow_monitor()