"""
Training Monitor with Action Diversity Validation
Monitors training progress and prevents conservative convergence
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class ActionDiversityMonitor:
    """
    Monitors action diversity during training to detect and prevent conservative convergence
    """
    
    def __init__(self,
                 window_size: int = 1000,
                 diversity_threshold: float = 0.3,
                 conservatism_threshold: float = 0.7,
                 num_actions: int = 3,
                 checkpoint_dir: str = "training_checkpoints"):
        """
        Initialize action diversity monitor
        
        Args:
            window_size: Window for tracking recent actions
            diversity_threshold: Minimum acceptable action diversity
            conservatism_threshold: Maximum acceptable hold ratio
            num_actions: Number of possible actions
            checkpoint_dir: Directory for saving checkpoints
        """
        self.window_size = window_size
        self.diversity_threshold = diversity_threshold
        self.conservatism_threshold = conservatism_threshold
        self.num_actions = num_actions
        self.checkpoint_dir = checkpoint_dir
        
        # Action tracking
        self.action_history = deque(maxlen=window_size * 10)
        self.episode_actions = []
        
        # Performance tracking
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.sharpe_ratios = deque(maxlen=100)
        
        # Diversity metrics over time
        self.diversity_history = []
        self.hold_ratio_history = []
        self.action_entropy_history = []
        
        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        self.conservatism_warnings = 0
        self.last_checkpoint_step = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def update(self, action: int, reward: float, done: bool):
        """Update monitor with new action and reward"""
        self.action_history.append(action)
        self.episode_actions.append(action)
        self.total_steps += 1
        
        if done:
            self._process_episode()
            
    def _process_episode(self):
        """Process completed episode"""
        if len(self.episode_actions) == 0:
            return
            
        # Calculate episode metrics
        action_counts = np.bincount(self.episode_actions, minlength=self.num_actions)
        action_probs = action_counts / len(self.episode_actions)
        
        # Action entropy
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        normalized_entropy = entropy / np.log(self.num_actions)
        
        # Hold ratio (assuming action 1 is hold)
        hold_ratio = action_probs[1] if len(action_probs) > 1 else 0.0
        
        # Update histories
        self.action_entropy_history.append(normalized_entropy)
        self.hold_ratio_history.append(hold_ratio)
        
        # Reset episode tracking
        self.episode_actions = []
        self.total_episodes += 1
        
    def check_diversity(self) -> Dict[str, Any]:
        """
        Check current action diversity metrics
        
        Returns:
            Dictionary with diversity metrics and warnings
        """
        if len(self.action_history) < 100:
            return {
                'status': 'warming_up',
                'total_steps': self.total_steps
            }
            
        # Recent action distribution
        recent_actions = list(self.action_history)[-self.window_size:]
        action_counts = np.bincount(recent_actions, minlength=self.num_actions)
        action_probs = action_counts / len(recent_actions)
        
        # Calculate metrics
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        normalized_entropy = entropy / np.log(self.num_actions)
        
        hold_ratio = action_probs[1] if len(action_probs) > 1 else 0.0
        buy_ratio = action_probs[2] if len(action_probs) > 2 else 0.0
        sell_ratio = action_probs[0] if len(action_probs) > 0 else 0.0
        
        # Check for issues
        is_conservative = hold_ratio > self.conservatism_threshold
        is_low_diversity = normalized_entropy < self.diversity_threshold
        never_buys = buy_ratio < 0.01
        
        # Warning level
        warning_level = 0
        warnings = []
        
        if is_conservative:
            warning_level += 2
            warnings.append(f"High conservatism: {hold_ratio:.1%} hold actions")
            self.conservatism_warnings += 1
            
        if is_low_diversity:
            warning_level += 1
            warnings.append(f"Low action diversity: entropy={normalized_entropy:.3f}")
            
        if never_buys:
            warning_level += 3
            warnings.append(f"Never buys: {buy_ratio:.1%} buy actions")
            
        return {
            'status': 'warning' if warning_level > 0 else 'healthy',
            'warning_level': warning_level,
            'warnings': warnings,
            'metrics': {
                'entropy': normalized_entropy,
                'hold_ratio': hold_ratio,
                'buy_ratio': buy_ratio,
                'sell_ratio': sell_ratio,
                'action_distribution': action_probs.tolist(),
                'total_steps': self.total_steps,
                'total_episodes': self.total_episodes,
                'conservatism_warnings': self.conservatism_warnings
            }
        }
        
    def should_intervene(self) -> Tuple[bool, str]:
        """
        Determine if training intervention is needed
        
        Returns:
            (should_intervene, reason)
        """
        diversity_check = self.check_diversity()
        
        if diversity_check['status'] == 'warming_up':
            return False, ""
            
        warning_level = diversity_check.get('warning_level', 0)
        
        # Critical intervention thresholds
        if warning_level >= 5:
            return True, "Critical: Severe conservative convergence detected"
            
        if self.conservatism_warnings > 10:
            return True, "Critical: Persistent conservative behavior"
            
        # Check if stuck in local optimum
        if len(self.hold_ratio_history) >= 20:
            recent_hold_ratios = self.hold_ratio_history[-20:]
            if all(r > 0.8 for r in recent_hold_ratios):
                return True, "Critical: Stuck in conservative local optimum"
                
        return False, ""
        
    def save_checkpoint(self, agent, optimizer, additional_info: Optional[Dict] = None):
        """Save training checkpoint with diversity metrics"""
        checkpoint = {
            'step': self.total_steps,
            'episode': self.total_episodes,
            'diversity_metrics': self.check_diversity(),
            'action_history_sample': list(self.action_history)[-1000:],
            'hold_ratio_history': list(self.hold_ratio_history)[-100:],
            'entropy_history': list(self.action_entropy_history)[-100:],
            'agent_state': agent.state_dict() if hasattr(agent, 'state_dict') else None,
            'optimizer_state': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
            
        # Save checkpoint
        filename = f"checkpoint_step_{self.total_steps}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Also save diversity report
        self.save_diversity_report()
        
        self.last_checkpoint_step = self.total_steps
        
    def save_diversity_report(self):
        """Save detailed diversity analysis report"""
        report = {
            'summary': self.check_diversity(),
            'history': {
                'hold_ratios': list(self.hold_ratio_history)[-200:],
                'entropy': list(self.action_entropy_history)[-200:],
                'episode_returns': list(self.episode_returns)[-200:],
            },
            'statistics': self._calculate_statistics(),
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"diversity_report_step_{self.total_steps}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
    def _calculate_statistics(self) -> Dict[str, float]:
        """Calculate comprehensive statistics"""
        if len(self.action_history) < 100:
            return {}
            
        recent_actions = list(self.action_history)[-1000:]
        action_counts = np.bincount(recent_actions, minlength=self.num_actions)
        
        stats = {
            'mean_hold_ratio': np.mean(self.hold_ratio_history) if self.hold_ratio_history else 0,
            'std_hold_ratio': np.std(self.hold_ratio_history) if self.hold_ratio_history else 0,
            'mean_entropy': np.mean(self.action_entropy_history) if self.action_entropy_history else 0,
            'action_counts': action_counts.tolist(),
            'episodes_completed': self.total_episodes,
            'steps_since_checkpoint': self.total_steps - self.last_checkpoint_step
        }
        
        return stats
        
    def plot_diversity_metrics(self, save_path: Optional[str] = None):
        """Plot diversity metrics over time"""
        if len(self.hold_ratio_history) < 10:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Hold ratio over time
        axes[0, 0].plot(self.hold_ratio_history)
        axes[0, 0].axhline(y=self.conservatism_threshold, color='r', linestyle='--', 
                           label=f'Threshold ({self.conservatism_threshold})')
        axes[0, 0].set_title('Hold Action Ratio Over Time')
        axes[0, 0].set_ylabel('Hold Ratio')
        axes[0, 0].legend()
        
        # Action entropy over time
        axes[0, 1].plot(self.action_entropy_history)
        axes[0, 1].axhline(y=self.diversity_threshold, color='r', linestyle='--',
                           label=f'Threshold ({self.diversity_threshold})')
        axes[0, 1].set_title('Action Entropy Over Time')
        axes[0, 1].set_ylabel('Normalized Entropy')
        axes[0, 1].legend()
        
        # Recent action distribution
        if len(self.action_history) >= 100:
            recent_actions = list(self.action_history)[-500:]
            action_counts = np.bincount(recent_actions, minlength=self.num_actions)
            axes[1, 0].bar(range(self.num_actions), action_counts)
            axes[1, 0].set_title('Recent Action Distribution (Last 500)')
            axes[1, 0].set_xlabel('Action')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(self.num_actions))
            axes[1, 0].set_xticklabels(['Sell', 'Hold', 'Buy'])
            
        # Episode returns if available
        if len(self.episode_returns) > 0:
            axes[1, 1].plot(self.episode_returns)
            axes[1, 1].set_title('Episode Returns')
            axes[1, 1].set_ylabel('Return')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
            
        plt.close()


class TrainingValidator:
    """
    Validates training progress and triggers interventions when needed
    """
    
    def __init__(self,
                 monitor: ActionDiversityMonitor,
                 patience: int = 50,
                 min_improvement: float = 0.01):
        """
        Initialize training validator
        
        Args:
            monitor: Action diversity monitor instance
            patience: Episodes without improvement before intervention
            min_improvement: Minimum improvement to reset patience
        """
        self.monitor = monitor
        self.patience = patience
        self.min_improvement = min_improvement
        
        self.best_performance = -float('inf')
        self.episodes_without_improvement = 0
        self.intervention_count = 0
        
    def validate_episode(self, episode_return: float) -> Dict[str, Any]:
        """
        Validate episode and check if intervention needed
        
        Args:
            episode_return: Return from completed episode
            
        Returns:
            Validation results and recommendations
        """
        # Update monitor
        self.monitor.episode_returns.append(episode_return)
        
        # Check performance improvement
        if episode_return > self.best_performance * (1 + self.min_improvement):
            self.best_performance = episode_return
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
            
        # Get diversity check
        diversity_check = self.monitor.check_diversity()
        
        # Check if intervention needed
        should_intervene, reason = self.monitor.should_intervene()
        
        # Additional checks
        if self.episodes_without_improvement >= self.patience:
            should_intervene = True
            reason = f"No improvement for {self.patience} episodes"
            
        recommendations = []
        
        if should_intervene:
            self.intervention_count += 1
            recommendations.extend([
                "Increase exploration rate significantly",
                "Reset optimizer learning rate",
                "Consider switching to different agent type",
                "Apply action masking to prevent excessive holding"
            ])
            
        return {
            'validation_passed': not should_intervene,
            'reason': reason,
            'diversity_check': diversity_check,
            'recommendations': recommendations,
            'metrics': {
                'episode_return': episode_return,
                'best_performance': self.best_performance,
                'episodes_without_improvement': self.episodes_without_improvement,
                'intervention_count': self.intervention_count
            }
        }


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Training Monitor")
    print("=" * 50)
    
    # Create monitor
    monitor = ActionDiversityMonitor(
        window_size=1000,
        diversity_threshold=0.3,
        conservatism_threshold=0.7
    )
    
    # Simulate conservative training
    print("\n📊 Simulating conservative training:")
    for episode in range(50):
        episode_return = 0
        
        for step in range(100):
            # Conservative action distribution
            if np.random.random() < 0.85:  # 85% hold
                action = 1
            elif np.random.random() < 0.5:  # 7.5% buy
                action = 2
            else:  # 7.5% sell
                action = 0
                
            reward = np.random.randn() * 0.1
            episode_return += reward
            
            monitor.update(action, reward, done=(step == 99))
            
        if episode % 10 == 0:
            diversity_check = monitor.check_diversity()
            print(f"\nEpisode {episode}:")
            print(f"   Status: {diversity_check.get('status', 'N/A')}")
            if 'metrics' in diversity_check:
                metrics = diversity_check['metrics']
                print(f"   Hold ratio: {metrics['hold_ratio']:.1%}")
                print(f"   Entropy: {metrics['entropy']:.3f}")
                
    # Check if intervention needed
    should_intervene, reason = monitor.should_intervene()
    print(f"\n🚨 Intervention needed: {should_intervene}")
    if should_intervene:
        print(f"   Reason: {reason}")
        
    # Save report
    monitor.save_diversity_report()
    print(f"\n📊 Diversity report saved to {monitor.checkpoint_dir}")
    
    # Plot metrics
    monitor.plot_diversity_metrics("diversity_analysis.png")
    print("📊 Diversity plots saved to diversity_analysis.png")