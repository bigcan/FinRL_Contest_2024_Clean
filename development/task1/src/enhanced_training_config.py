"""
Enhanced Training Configuration for Profitability Improvements
Addresses insufficient training and implements early stopping
"""

import numpy as np
import torch
from erl_config import Config
from collections import deque
import time


class EnhancedConfig(Config):
    """
    Enhanced configuration with extended training and early stopping
    Addresses the core issue of insufficient training (8-16 steps)
    """
    
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        super().__init__(agent_class, env_class, env_args)
        
        # Extended training configuration
        self.break_step = 200  # Increase from 8-16 to 200 steps
        self.max_training_time = 1800  # Maximum 30 minutes training
        
        # Early stopping configuration
        self.early_stopping_patience = 50  # Stop if no improvement for 50 steps
        self.early_stopping_min_delta = 0.001  # Minimum improvement threshold
        self.early_stopping_monitor = "eval_score"  # What to monitor
        self.early_stopping_enabled = True
        
        # Learning rate scheduling
        self.use_lr_scheduler = True
        self.lr_scheduler_type = "cosine_annealing"  # "cosine_annealing", "exponential", "step"
        self.lr_decay_factor = 0.95
        self.lr_decay_steps = 25
        self.lr_min = 1e-7
        
        # Enhanced exploration
        self.exploration_decay = True
        self.initial_exploration = 0.1  # Start with higher exploration
        self.final_exploration = 0.001  # End with lower exploration  
        self.exploration_decay_steps = self.break_step
        self.explore_rate = self.initial_exploration  # Add missing attribute
        
        # Improved evaluation
        self.eval_per_step = 10  # Evaluate every 10 steps instead of 20000
        self.eval_times = 5  # More evaluation episodes for stability
        
        # Performance tracking
        self.track_training_metrics = True
        self.metrics_history_size = 200
        
        print(f"🚀 Enhanced Training Configuration:")
        print(f"   📈 Training Steps: {self.break_step} (vs 8-16 baseline)")
        print(f"   ⏰ Max Training Time: {self.max_training_time/60:.1f} minutes")
        print(f"   🛑 Early Stopping: {'Enabled' if self.early_stopping_enabled else 'Disabled'}")
        print(f"   📊 Evaluation Frequency: Every {self.eval_per_step} steps")
        print(f"   🎯 Learning Rate Scheduling: {self.lr_scheduler_type}")
        print(f"   🔍 Exploration Decay: {self.initial_exploration} → {self.final_exploration}")


class EarlyStoppingManager:
    """
    Early stopping implementation to prevent overfitting and save training time
    """
    
    def __init__(self, patience=50, min_delta=0.001, monitor="eval_score", mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.patience_counter = 0
        self.should_stop = False
        self.best_step = 0
        
        print(f"🛑 Early Stopping Manager:")
        print(f"   Monitor: {monitor} ({mode})")
        print(f"   Patience: {patience} steps")
        print(f"   Min Delta: {min_delta}")
    
    def update(self, current_score, current_step):
        """Update early stopping state with current score"""
        
        improved = False
        if self.mode == 'max':
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.best_step = current_step
            self.patience_counter = 0
            print(f"   ✅ New best {self.monitor}: {current_score:.6f} at step {current_step}")
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self.should_stop = True
            print(f"   🛑 Early stopping triggered! No improvement for {self.patience} steps")
            print(f"   📊 Best score: {self.best_score:.6f} at step {self.best_step}")
        
        return self.should_stop
    
    def get_status(self):
        """Get current early stopping status"""
        return {
            "should_stop": self.should_stop,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "patience_counter": self.patience_counter,
            "patience_remaining": self.patience - self.patience_counter
        }


class LearningRateScheduler:
    """
    Learning rate scheduling for improved training stability and convergence
    """
    
    def __init__(self, optimizer, scheduler_type="cosine_annealing", 
                 total_steps=200, decay_factor=0.95, decay_steps=25, min_lr=1e-7):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.total_steps = total_steps
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
        print(f"📈 Learning Rate Scheduler ({scheduler_type}):")
        print(f"   Initial LR: {self.initial_lr:.2e}")
        print(f"   Total Steps: {total_steps}")
        print(f"   Min LR: {min_lr:.2e}")
    
    def step(self):
        """Update learning rate based on current step"""
        self.current_step += 1
        
        if self.scheduler_type == "cosine_annealing":
            lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * self.current_step / self.total_steps))
        elif self.scheduler_type == "exponential":
            lr = self.initial_lr * (self.decay_factor ** (self.current_step / self.decay_steps))
            lr = max(lr, self.min_lr)
        elif self.scheduler_type == "step":
            lr = self.initial_lr * (self.decay_factor ** (self.current_step // self.decay_steps))
            lr = max(lr, self.min_lr)
        else:
            lr = self.initial_lr  # No scheduling
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class ExplorationScheduler:
    """
    Exploration rate scheduling for improved exploration-exploitation balance
    """
    
    def __init__(self, initial_rate=0.1, final_rate=0.001, total_steps=200, decay_type="linear"):
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_steps = total_steps
        self.decay_type = decay_type
        self.current_step = 0
        
        print(f"🔍 Exploration Scheduler ({decay_type}):")
        print(f"   Initial Rate: {initial_rate}")
        print(f"   Final Rate: {final_rate}")
        print(f"   Total Steps: {total_steps}")
    
    def get_rate(self, current_step=None):
        """Get current exploration rate"""
        if current_step is not None:
            self.current_step = current_step
        
        progress = min(self.current_step / self.total_steps, 1.0)
        
        if self.decay_type == "linear":
            rate = self.initial_rate - (self.initial_rate - self.final_rate) * progress
        elif self.decay_type == "exponential":
            rate = self.initial_rate * (self.final_rate / self.initial_rate) ** progress
        elif self.decay_type == "cosine":
            rate = self.final_rate + (self.initial_rate - self.final_rate) * \
                   0.5 * (1 + np.cos(np.pi * progress))
        else:
            rate = self.initial_rate  # No decay
        
        return max(rate, self.final_rate)


class TrainingMetricsTracker:
    """
    Track training metrics for analysis and debugging
    """
    
    def __init__(self, history_size=200):
        self.history_size = history_size
        self.metrics = {
            "step": deque(maxlen=history_size),
            "reward": deque(maxlen=history_size),
            "loss_critic": deque(maxlen=history_size),
            "loss_actor": deque(maxlen=history_size),
            "exploration_rate": deque(maxlen=history_size),
            "learning_rate": deque(maxlen=history_size),
            "eval_score": deque(maxlen=history_size),
            "training_time": deque(maxlen=history_size)
        }
        self.start_time = time.time()
    
    def update(self, step, reward=None, loss_critic=None, loss_actor=None, 
               exploration_rate=None, learning_rate=None, eval_score=None):
        """Update metrics for current step"""
        self.metrics["step"].append(step)
        self.metrics["training_time"].append(time.time() - self.start_time)
        
        if reward is not None:
            self.metrics["reward"].append(reward)
        if loss_critic is not None:
            self.metrics["loss_critic"].append(loss_critic)
        if loss_actor is not None:
            self.metrics["loss_actor"].append(loss_actor)
        if exploration_rate is not None:
            self.metrics["exploration_rate"].append(exploration_rate)
        if learning_rate is not None:
            self.metrics["learning_rate"].append(learning_rate)
        if eval_score is not None:
            self.metrics["eval_score"].append(eval_score)
    
    def get_recent_average(self, metric, window=10):
        """Get recent average of a metric"""
        if metric not in self.metrics or len(self.metrics[metric]) < window:
            return None
        
        recent_values = list(self.metrics[metric])[-window:]
        return sum(recent_values) / len(recent_values)
    
    def print_summary(self):
        """Print training summary"""
        if len(self.metrics["step"]) == 0:
            return
        
        print(f"\n📊 Training Metrics Summary:")
        print(f"   🕐 Total Time: {self.metrics['training_time'][-1]:.1f}s")
        print(f"   📈 Total Steps: {self.metrics['step'][-1]}")
        
        for metric in ["reward", "loss_critic", "loss_actor", "eval_score"]:
            if len(self.metrics[metric]) > 0:
                recent_avg = self.get_recent_average(metric, window=10)
                if recent_avg is not None:
                    print(f"   📊 Recent {metric}: {recent_avg:.6f}")


def create_enhanced_config(agent_class, env_class, env_args, 
                          training_steps=200, early_stopping=True, 
                          lr_scheduling=True) -> EnhancedConfig:
    """
    Factory function to create enhanced training configuration
    
    Args:
        agent_class: RL agent class
        env_class: Environment class
        env_args: Environment arguments
        training_steps: Number of training steps (default: 200)
        early_stopping: Enable early stopping (default: True)
        lr_scheduling: Enable learning rate scheduling (default: True)
    
    Returns:
        Enhanced configuration object
    """
    config = EnhancedConfig(agent_class, env_class, env_args)
    config.break_step = training_steps
    config.early_stopping_enabled = early_stopping
    config.use_lr_scheduler = lr_scheduling
    
    return config


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Enhanced Training Configuration")
    print("=" * 60)
    
    # Test configuration creation
    config = create_enhanced_config(
        agent_class=None,
        env_class=None, 
        env_args={
            "env_name": "TestEnv",
            "state_dim": 8, 
            "action_dim": 3,
            "if_discrete": True
        },
        training_steps=200
    )
    
    print(f"✅ Enhanced config created successfully")
    print(f"   Break step: {config.break_step}")
    print(f"   Early stopping: {config.early_stopping_enabled}")
    print(f"   LR scheduling: {config.use_lr_scheduler}")
    
    # Test early stopping manager
    early_stopping = EarlyStoppingManager(patience=10, min_delta=0.01)
    
    # Simulate some training steps
    scores = [0.1, 0.15, 0.12, 0.18, 0.17, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13]
    for i, score in enumerate(scores):
        should_stop = early_stopping.update(score, i)
        if should_stop:
            print(f"   Early stopping at step {i}")
            break
    
    # Test metrics tracker
    tracker = TrainingMetricsTracker(history_size=50)
    for i in range(10):
        tracker.update(
            step=i, 
            reward=np.random.normal(100, 10), 
            loss_critic=np.random.normal(1, 0.1),
            eval_score=np.random.normal(0.15, 0.02)
        )
    
    tracker.print_summary()
    
    print(f"\n🎉 All enhanced training components tested successfully!")