"""
Ensemble training orchestrator for the FinRL Contest 2024 framework.

This module provides comprehensive training coordination for ensemble learning
with multiple RL agents, including advanced scheduling and evaluation.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import torch
import numpy as np
import time
import json
import os
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from pathlib import Path

from ..ensemble.base_ensemble import BaseEnsemble, EnsembleStrategy, EnsembleMetrics
from ..ensemble.voting_ensemble import VotingEnsemble
from ..ensemble.stacking_ensemble import StackingEnsemble
from ..agents import create_agent, create_ensemble_agents
from ..core.types import StateType, ActionType


class TrainingPhase(Enum):
    """Training phases for ensemble learning."""
    INDIVIDUAL = "individual"
    ENSEMBLE = "ensemble"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"


@dataclass
class TrainingConfig:
    """Configuration for ensemble training."""
    # Basic training parameters
    total_episodes: int = 1000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 100
    save_frequency: int = 200
    
    # Phase-specific parameters
    individual_episodes: int = 300
    ensemble_episodes: int = 500
    fine_tuning_episodes: int = 200
    
    # Ensemble parameters
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_VOTE
    voting_temperature: float = 1.0
    confidence_threshold: float = 0.5
    meta_learning_rate: float = 1e-4
    
    # Evaluation parameters
    eval_episodes: int = 10
    eval_deterministic: bool = True
    
    # Logging and visualization
    log_level: str = "INFO"
    plot_frequency: int = 50
    save_plots: bool = True
    
    # Early stopping
    patience: int = 50
    min_improvement: float = 0.01
    
    # Resource management
    device: str = "auto"
    num_workers: int = 1


@dataclass
class TrainingResults:
    """Results from ensemble training."""
    training_rewards: List[float] = field(default_factory=list)
    evaluation_rewards: List[float] = field(default_factory=list)
    individual_agent_rewards: Dict[str, List[float]] = field(default_factory=dict)
    ensemble_metrics: EnsembleMetrics = field(default_factory=EnsembleMetrics)
    training_times: List[float] = field(default_factory=list)
    phase_transitions: Dict[TrainingPhase, int] = field(default_factory=dict)
    best_episode: int = 0
    best_reward: float = float('-inf')
    total_training_time: float = 0.0
    
    def save_results(self, filepath: str):
        """Save training results to file."""
        results_dict = {
            'training_rewards': self.training_rewards,
            'evaluation_rewards': self.evaluation_rewards,
            'individual_agent_rewards': self.individual_agent_rewards,
            'training_times': self.training_times,
            'phase_transitions': {phase.value: step for phase, step in self.phase_transitions.items()},
            'best_episode': self.best_episode,
            'best_reward': self.best_reward,
            'total_training_time': self.total_training_time,
            'summary_statistics': {
                'mean_training_reward': np.mean(self.training_rewards) if self.training_rewards else 0.0,
                'std_training_reward': np.std(self.training_rewards) if self.training_rewards else 0.0,
                'mean_eval_reward': np.mean(self.evaluation_rewards) if self.evaluation_rewards else 0.0,
                'final_training_reward': self.training_rewards[-1] if self.training_rewards else 0.0,
                'improvement': (self.training_rewards[-1] - self.training_rewards[0] 
                              if len(self.training_rewards) > 1 else 0.0)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Training results saved to {filepath}")


class EnsembleTrainer:
    """
    Comprehensive training orchestrator for ensemble learning.
    
    Manages multi-phase training including individual agent training,
    ensemble coordination, fine-tuning, and evaluation.
    """
    
    def __init__(self,
                 environment: Any,
                 agent_configs: Dict[str, Dict[str, Any]],
                 config: TrainingConfig,
                 save_dir: str = "./ensemble_training"):
        """
        Initialize ensemble trainer.
        
        Args:
            environment: Training environment
            agent_configs: Dictionary of agent configurations
            config: Training configuration
            save_dir: Directory for saving results
        """
        self.environment = environment
        self.agent_configs = agent_configs
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize agents and ensemble
        self.agents = None
        self.ensemble = None
        self.current_phase = TrainingPhase.INDIVIDUAL
        
        # Training state
        self.results = TrainingResults()
        self.episode = 0
        self.best_ensemble_checkpoint = None
        self.early_stopping_counter = 0
        
        # Evaluation callback
        self.evaluation_callback: Optional[Callable] = None
        
        print(f"Initialized EnsembleTrainer:")
        print(f"  Device: {self.device}")
        print(f"  Save directory: {self.save_dir}")
        print(f"  Total episodes: {config.total_episodes}")
        print(f"  Agent types: {list(agent_configs.keys())}")
    
    def train(self) -> TrainingResults:
        """
        Execute complete ensemble training pipeline.
        
        Returns:
            Training results
        """
        start_time = time.time()
        
        try:
            # Phase 1: Individual agent training
            self._train_individual_agents()
            
            # Phase 2: Ensemble coordination training
            self._train_ensemble()
            
            # Phase 3: Fine-tuning
            self._fine_tune_ensemble()
            
            # Phase 4: Final evaluation
            self._final_evaluation()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
        
        finally:
            self.results.total_training_time = time.time() - start_time
            self._save_final_results()
        
        return self.results
    
    def _train_individual_agents(self):
        """Phase 1: Train individual agents independently."""
        print("\n" + "="*60)
        print("PHASE 1: Individual Agent Training")
        print("="*60)
        
        self.current_phase = TrainingPhase.INDIVIDUAL
        self.results.phase_transitions[TrainingPhase.INDIVIDUAL] = self.episode
        
        # Create agents
        state_dim = getattr(self.environment, 'state_dim', 100)
        action_dim = getattr(self.environment, 'action_dim', 3)
        
        self.agents = create_ensemble_agents(
            self.agent_configs,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        print(f"Created {len(self.agents)} agents: {list(self.agents.keys())}")
        
        # Train each agent individually
        for episode in range(self.config.individual_episodes):
            self.episode = episode
            
            # Training step for each agent
            episode_rewards = {}
            for name, agent in self.agents.items():
                reward = self._train_single_episode(agent, name)
                episode_rewards[name] = reward
                
                # Track individual agent performance
                if name not in self.results.individual_agent_rewards:
                    self.results.individual_agent_rewards[name] = []
                self.results.individual_agent_rewards[name].append(reward)
            
            # Calculate average reward
            avg_reward = np.mean(list(episode_rewards.values()))
            self.results.training_rewards.append(avg_reward)
            
            # Logging and evaluation
            if episode % 20 == 0:
                print(f"Episode {episode}: Average reward = {avg_reward:.3f}")
                for name, reward in episode_rewards.items():
                    print(f"  {name}: {reward:.3f}")
            
            if episode % self.config.eval_frequency == 0:
                self._evaluate_agents()
            
            if episode % self.config.plot_frequency == 0:
                self._plot_training_progress()
        
        print(f"Individual training completed after {self.config.individual_episodes} episodes")
    
    def _train_ensemble(self):
        """Phase 2: Train ensemble coordination."""
        print("\n" + "="*60)
        print("PHASE 2: Ensemble Coordination Training")
        print("="*60)
        
        self.current_phase = TrainingPhase.ENSEMBLE
        self.results.phase_transitions[TrainingPhase.ENSEMBLE] = self.episode
        
        # Create ensemble
        self.ensemble = self._create_ensemble()
        
        print(f"Created {type(self.ensemble).__name__} with strategy: {self.config.ensemble_strategy.value}")
        
        # Train ensemble
        for episode in range(self.config.ensemble_episodes):
            self.episode += 1
            
            reward = self._train_ensemble_episode()
            self.results.training_rewards.append(reward)
            
            # Logging
            if episode % 20 == 0:
                print(f"Episode {self.episode}: Ensemble reward = {reward:.3f}")
                if hasattr(self.ensemble, 'get_voting_statistics'):
                    stats = self.ensemble.get_voting_statistics()
                    print(f"  Active agents: {len(self.ensemble.get_active_agents())}/{len(self.agents)}")
                    print(f"  Mean confidence: {np.mean(list(stats['agent_confidences'].values())):.3f}")
            
            # Evaluation and checkpointing
            if episode % self.config.eval_frequency == 0:
                self._evaluate_ensemble()
            
            if episode % self.config.save_frequency == 0:
                self._save_checkpoint()
            
            if episode % self.config.plot_frequency == 0:
                self._plot_training_progress()
            
            # Early stopping check
            if self._check_early_stopping():
                print(f"Early stopping triggered at episode {self.episode}")
                break
        
        print(f"Ensemble training completed after {episode + 1} episodes")
    
    def _fine_tune_ensemble(self):
        """Phase 3: Fine-tune ensemble performance."""
        print("\n" + "="*60)
        print("PHASE 3: Ensemble Fine-tuning")
        print("="*60)
        
        self.current_phase = TrainingPhase.FINE_TUNING
        self.results.phase_transitions[TrainingPhase.FINE_TUNING] = self.episode
        
        # Load best checkpoint if available
        if self.best_ensemble_checkpoint:
            print("Loading best ensemble checkpoint for fine-tuning")
            # Implementation would load the checkpoint here
        
        # Fine-tuning with reduced learning rates
        for name, agent in self.agents.items():
            if hasattr(agent, 'adjust_learning_rate'):
                current_lr = getattr(agent.config, 'learning_rate', 1e-4)
                agent.adjust_learning_rate(current_lr * 0.1)
        
        # Fine-tuning episodes
        for episode in range(self.config.fine_tuning_episodes):
            self.episode += 1
            
            reward = self._train_ensemble_episode()
            self.results.training_rewards.append(reward)
            
            if episode % 20 == 0:
                print(f"Episode {self.episode}: Fine-tuning reward = {reward:.3f}")
            
            if episode % self.config.eval_frequency == 0:
                self._evaluate_ensemble()
        
        print(f"Fine-tuning completed after {self.config.fine_tuning_episodes} episodes")
    
    def _final_evaluation(self):
        """Phase 4: Final comprehensive evaluation."""
        print("\n" + "="*60)
        print("PHASE 4: Final Evaluation")
        print("="*60)
        
        self.current_phase = TrainingPhase.EVALUATION
        self.results.phase_transitions[TrainingPhase.EVALUATION] = self.episode
        
        # Extended evaluation
        eval_rewards = []
        for eval_episode in range(self.config.eval_episodes * 2):  # More thorough evaluation
            reward = self._evaluate_single_episode()
            eval_rewards.append(reward)
        
        # Calculate final statistics
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        print(f"Final evaluation results ({len(eval_rewards)} episodes):")
        print(f"  Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"  Best reward: {max(eval_rewards):.3f}")
        print(f"  Worst reward: {min(eval_rewards):.3f}")
        
        # Save evaluation results
        self.results.evaluation_rewards.extend(eval_rewards)
    
    def _create_ensemble(self) -> BaseEnsemble:
        """Create ensemble based on configuration."""
        if self.config.ensemble_strategy == EnsembleStrategy.STACKING:
            return StackingEnsemble(
                agents=self.agents,
                action_dim=getattr(self.environment, 'action_dim', 3),
                device=self.device,
                meta_learning_rate=self.config.meta_learning_rate
            )
        else:
            return VotingEnsemble(
                agents=self.agents,
                strategy=self.config.ensemble_strategy,
                device=self.device,
                confidence_threshold=self.config.confidence_threshold,
                temperature=self.config.voting_temperature
            )
    
    def _train_single_episode(self, agent: Any, agent_name: str) -> float:
        """Train single agent for one episode."""
        state = self.environment.reset()
        total_reward = 0.0
        
        for step in range(self.config.max_steps_per_episode):
            # Agent action selection
            action = agent.select_action(state, deterministic=False)
            
            # Environment step
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            
            # Agent update (if supported)
            if hasattr(agent, 'update') and hasattr(self.environment, 'get_last_transition'):
                try:
                    transition = self.environment.get_last_transition()
                    agent.update(transition)
                except Exception as e:
                    # Silent fail for unsupported environments
                    pass
            
            state = next_state
            if done:
                break
        
        return total_reward
    
    def _train_ensemble_episode(self) -> float:
        """Train ensemble for one episode."""
        state = self.environment.reset()
        total_reward = 0.0
        
        for step in range(self.config.max_steps_per_episode):
            # Ensemble action selection
            action = self.ensemble.select_action(state, deterministic=False)
            
            # Environment step
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            
            # Ensemble update
            if hasattr(self.environment, 'get_last_transition'):
                try:
                    transition = self.environment.get_last_transition()
                    self.ensemble.update(transition)
                except Exception as e:
                    # Create simple transition for update
                    simple_transition = (state, action, reward, done)
                    self.ensemble.update(simple_transition)
            
            state = next_state
            if done:
                break
        
        return total_reward
    
    def _evaluate_agents(self):
        """Evaluate individual agents."""
        eval_rewards = {}
        for name, agent in self.agents.items():
            rewards = []
            for _ in range(self.config.eval_episodes):
                reward = self._evaluate_single_agent(agent)
                rewards.append(reward)
            eval_rewards[name] = np.mean(rewards)
        
        avg_eval_reward = np.mean(list(eval_rewards.values()))
        self.results.evaluation_rewards.append(avg_eval_reward)
        
        print(f"Evaluation at episode {self.episode}:")
        for name, reward in eval_rewards.items():
            print(f"  {name}: {reward:.3f}")
        print(f"  Average: {avg_eval_reward:.3f}")
    
    def _evaluate_ensemble(self):
        """Evaluate ensemble performance."""
        eval_rewards = []
        for _ in range(self.config.eval_episodes):
            reward = self._evaluate_single_episode()
            eval_rewards.append(reward)
        
        avg_reward = np.mean(eval_rewards)
        self.results.evaluation_rewards.append(avg_reward)
        
        # Check for best performance
        if avg_reward > self.results.best_reward:
            self.results.best_reward = avg_reward
            self.results.best_episode = self.episode
            self.best_ensemble_checkpoint = self._create_checkpoint()
            self.early_stopping_counter = 0
            print(f"New best ensemble performance: {avg_reward:.3f}")
        else:
            self.early_stopping_counter += 1
        
        print(f"Ensemble evaluation at episode {self.episode}: {avg_reward:.3f}")
    
    def _evaluate_single_agent(self, agent: Any) -> float:
        """Evaluate single agent for one episode."""
        state = self.environment.reset()
        total_reward = 0.0
        
        for step in range(self.config.max_steps_per_episode):
            action = agent.select_action(state, deterministic=self.config.eval_deterministic)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        
        return total_reward
    
    def _evaluate_single_episode(self) -> float:
        """Evaluate ensemble for one episode."""
        state = self.environment.reset()
        total_reward = 0.0
        
        for step in range(self.config.max_steps_per_episode):
            action = self.ensemble.select_action(state, deterministic=self.config.eval_deterministic)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        
        return total_reward
    
    def _check_early_stopping(self) -> bool:
        """Check early stopping criteria."""
        if self.early_stopping_counter >= self.config.patience:
            return True
        
        # Check for minimum improvement
        if len(self.results.evaluation_rewards) >= 2:
            recent_improvement = (
                self.results.evaluation_rewards[-1] - 
                self.results.evaluation_rewards[-2]
            )
            if recent_improvement < self.config.min_improvement:
                self.early_stopping_counter += 1
            else:
                self.early_stopping_counter = 0
        
        return False
    
    def _plot_training_progress(self):
        """Plot training progress."""
        if not self.config.save_plots:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training rewards
        if self.results.training_rewards:
            axes[0, 0].plot(self.results.training_rewards)
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
        
        # Evaluation rewards
        if self.results.evaluation_rewards:
            eval_episodes = np.arange(0, len(self.results.evaluation_rewards)) * self.config.eval_frequency
            axes[0, 1].plot(eval_episodes, self.results.evaluation_rewards)
            axes[0, 1].set_title('Evaluation Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
        
        # Individual agent performance
        if self.results.individual_agent_rewards:
            for name, rewards in self.results.individual_agent_rewards.items():
                axes[1, 0].plot(rewards[:len(self.results.training_rewards)], label=name)
            axes[1, 0].set_title('Individual Agent Performance')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].legend()
        
        # Ensemble metrics
        if hasattr(self.ensemble, 'metrics') and self.ensemble.metrics.agreement_scores:
            axes[1, 1].plot(self.ensemble.metrics.agreement_scores, label='Agreement')
            axes[1, 1].plot(self.ensemble.metrics.diversity_scores, label='Diversity')
            axes[1, 1].set_title('Ensemble Metrics')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_progress_episode_{self.episode}.png')
        plt.close()
    
    def _create_checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of current state."""
        return {
            'episode': self.episode,
            'phase': self.current_phase.value,
            'ensemble_info': self.ensemble.get_ensemble_info() if self.ensemble else None,
            'results': self.results
        }
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        if self.ensemble:
            checkpoint_path = self.save_dir / f'ensemble_checkpoint_episode_{self.episode}.pth'
            self.ensemble.save_ensemble(str(checkpoint_path))
    
    def _save_final_results(self):
        """Save final training results."""
        # Save results
        results_path = self.save_dir / 'training_results.json'
        self.results.save_results(str(results_path))
        
        # Save final plot
        if self.config.save_plots:
            self._plot_training_progress()
            
            # Create summary plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.results.training_rewards, label='Training', alpha=0.7)
            if self.results.evaluation_rewards:
                eval_episodes = np.arange(0, len(self.results.evaluation_rewards)) * self.config.eval_frequency
                ax.plot(eval_episodes, self.results.evaluation_rewards, label='Evaluation', linewidth=2)
            
            # Mark phase transitions
            for phase, episode in self.results.phase_transitions.items():
                ax.axvline(x=episode, linestyle='--', alpha=0.5, label=f'{phase.value} start')
            
            ax.set_title('Complete Training Progress')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'final_training_summary.png', dpi=300)
            plt.close()
        
        print(f"\nTraining completed!")
        print(f"Best performance: {self.results.best_reward:.3f} at episode {self.results.best_episode}")
        print(f"Total training time: {self.results.total_training_time:.1f} seconds")
        print(f"Results saved to: {self.save_dir}")
    
    def set_evaluation_callback(self, callback: Callable):
        """Set custom evaluation callback."""
        self.evaluation_callback = callback
    
    def __repr__(self) -> str:
        return (f"EnsembleTrainer("
                f"agents={len(self.agent_configs) if self.agent_configs else 0}, "
                f"strategy={self.config.ensemble_strategy.value}, "
                f"episode={self.episode}, "
                f"phase={self.current_phase.value})")