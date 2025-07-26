"""
Base ensemble framework for the FinRL Contest 2024 refactored architecture.

This module provides the foundational classes and interfaces for ensemble learning
with multiple RL agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from ..core.types import StateType, ActionType, RewardType
from ..core.base_agent import BaseAgent
from ..core.interfaces import TrainingProtocol


class EnsembleStrategy(Enum):
    """Ensemble decision-making strategies."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    STACKING = "stacking"
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    DYNAMIC_WEIGHTED = "dynamic_weighted"


@dataclass
class EnsembleMetrics:
    """Metrics for ensemble performance tracking."""
    individual_rewards: Dict[str, List[float]] = field(default_factory=dict)
    ensemble_rewards: List[float] = field(default_factory=list)
    individual_losses: Dict[str, List[float]] = field(default_factory=dict)
    agreement_scores: List[float] = field(default_factory=list)
    diversity_scores: List[float] = field(default_factory=list)
    weights_history: List[Dict[str, float]] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    
    def add_step_metrics(self, 
                        individual_rewards: Dict[str, float],
                        ensemble_reward: float,
                        individual_losses: Dict[str, float],
                        agreement_score: float,
                        diversity_score: float,
                        weights: Dict[str, float],
                        confidence: float):
        """Add metrics for a single training step."""
        for agent_name, reward in individual_rewards.items():
            if agent_name not in self.individual_rewards:
                self.individual_rewards[agent_name] = []
            self.individual_rewards[agent_name].append(reward)
        
        for agent_name, loss in individual_losses.items():
            if agent_name not in self.individual_losses:
                self.individual_losses[agent_name] = []
            self.individual_losses[agent_name].append(loss)
        
        self.ensemble_rewards.append(ensemble_reward)
        self.agreement_scores.append(agreement_score)
        self.diversity_scores.append(diversity_score)
        self.weights_history.append(weights.copy())
        self.confidence_scores.append(confidence)


class BaseEnsemble(ABC):
    """
    Abstract base class for ensemble methods.
    
    Provides common functionality for managing multiple agents and combining
    their decisions through various ensemble strategies.
    """
    
    def __init__(self,
                 agents: Dict[str, BaseAgent],
                 strategy: EnsembleStrategy = EnsembleStrategy.MAJORITY_VOTE,
                 device: Optional[torch.device] = None):
        """
        Initialize base ensemble.
        
        Args:
            agents: Dictionary mapping agent names to agent instances
            strategy: Ensemble decision-making strategy
            device: Computing device
        """
        self.agents = agents
        self.strategy = strategy
        self.device = device or torch.device('cpu')
        self.agent_names = list(agents.keys())
        
        # Initialize tracking variables
        self.metrics = EnsembleMetrics()
        self.weights = {name: 1.0 / len(agents) for name in self.agent_names}
        self.performance_history = {name: [] for name in self.agent_names}
        self.training_step = 0
        
        # Validation
        if not agents:
            raise ValueError("At least one agent must be provided")
        
        print(f"Initialized {self.__class__.__name__} with {len(agents)} agents")
        print(f"Strategy: {strategy.value}")
        print(f"Agents: {list(agents.keys())}")
    
    @abstractmethod
    def select_action(self, state: StateType, deterministic: bool = False) -> ActionType:
        """
        Select action using ensemble strategy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Ensemble action
        """
        pass
    
    @abstractmethod
    def update(self, batch_data: Any) -> Dict[str, Any]:
        """
        Update all agents in the ensemble.
        
        Args:
            batch_data: Training batch data
            
        Returns:
            Dictionary of training statistics
        """
        pass
    
    def get_individual_actions(self, state: StateType, deterministic: bool = False) -> Dict[str, ActionType]:
        """
        Get actions from all individual agents.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Dictionary mapping agent names to their actions
        """
        actions = {}
        for name, agent in self.agents.items():
            try:
                action = agent.select_action(state, deterministic=deterministic)
                actions[name] = action
            except Exception as e:
                print(f"Warning: Agent {name} failed to select action: {e}")
                # Use a default action (assuming discrete action space)
                actions[name] = 0
        
        return actions
    
    def get_individual_q_values(self, state: StateType) -> Dict[str, torch.Tensor]:
        """
        Get Q-values from all agents (if available).
        
        Args:
            state: Current state
            
        Returns:
            Dictionary mapping agent names to their Q-values
        """
        q_values = {}
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'get_q_values'):
                    q_vals = agent.get_q_values(state)
                    q_values[name] = q_vals
                elif hasattr(agent, 'online_network'):
                    # Try to get Q-values directly from network
                    if isinstance(state, (list, tuple)):
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                    elif not isinstance(state, torch.Tensor):
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                    else:
                        state_tensor = state.to(self.device)
                    
                    if state_tensor.dim() == 1:
                        state_tensor = state_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        if hasattr(agent.online_network, 'get_q1_q2'):
                            q1, q2 = agent.online_network.get_q1_q2(state_tensor)
                            q_vals = torch.min(q1, q2)
                        else:
                            q_vals = agent.online_network(state_tensor)
                        q_values[name] = q_vals
            except Exception as e:
                print(f"Warning: Could not get Q-values from agent {name}: {e}")
        
        return q_values
    
    def calculate_agreement_score(self, actions: Dict[str, ActionType]) -> float:
        """
        Calculate agreement score between agents.
        
        Args:
            actions: Dictionary of agent actions
            
        Returns:
            Agreement score (0.0 to 1.0)
        """
        if len(actions) < 2:
            return 1.0
        
        action_list = list(actions.values())
        
        # For discrete actions, calculate mode agreement
        if isinstance(action_list[0], (int, np.integer)):
            from collections import Counter
            counter = Counter(action_list)
            most_common_count = counter.most_common(1)[0][1]
            return most_common_count / len(action_list)
        
        # For continuous actions, use variance-based agreement
        action_array = np.array(action_list)
        if action_array.ndim == 1:
            variance = np.var(action_array)
            # Convert variance to agreement score (lower variance = higher agreement)
            agreement = 1.0 / (1.0 + variance)
        else:
            # Multi-dimensional actions
            variances = np.var(action_array, axis=0)
            mean_variance = np.mean(variances)
            agreement = 1.0 / (1.0 + mean_variance)
        
        return float(np.clip(agreement, 0.0, 1.0))
    
    def calculate_diversity_score(self, q_values: Dict[str, torch.Tensor]) -> float:
        """
        Calculate diversity score between agent Q-values.
        
        Args:
            q_values: Dictionary of agent Q-values
            
        Returns:
            Diversity score (0.0 to 1.0)
        """
        if len(q_values) < 2:
            return 0.0
        
        try:
            # Stack Q-values and calculate pairwise distances
            q_stack = torch.stack(list(q_values.values()))
            
            # Calculate pairwise cosine distances
            q_normalized = torch.nn.functional.normalize(q_stack, dim=-1)
            similarities = torch.mm(q_normalized, q_normalized.t())
            
            # Extract upper triangular part (excluding diagonal)
            mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()
            pairwise_similarities = similarities[mask]
            
            # Convert similarity to diversity (1 - mean similarity)
            diversity = 1.0 - pairwise_similarities.mean().item()
            
            return float(np.clip(diversity, 0.0, 1.0))
        
        except Exception as e:
            print(f"Warning: Could not calculate diversity score: {e}")
            return 0.0
    
    def update_weights(self, performance_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Update agent weights based on performance.
        
        Args:
            performance_scores: Dictionary mapping agent names to performance scores
            
        Returns:
            Updated weights dictionary
        """
        if self.strategy == EnsembleStrategy.PERFORMANCE_WEIGHTED:
            # Weight by recent performance
            total_performance = sum(performance_scores.values())
            if total_performance > 0:
                self.weights = {
                    name: score / total_performance 
                    for name, score in performance_scores.items()
                }
            else:
                # Fallback to uniform weights
                self.weights = {name: 1.0 / len(self.agents) for name in self.agent_names}
        
        elif self.strategy == EnsembleStrategy.DYNAMIC_WEIGHTED:
            # Exponential moving average of performance
            alpha = 0.1  # Learning rate for weight updates
            for name, score in performance_scores.items():
                if name in self.weights:
                    self.weights[name] = (1 - alpha) * self.weights[name] + alpha * score
            
            # Normalize weights
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {name: w / total_weight for name, w in self.weights.items()}
        
        return self.weights.copy()
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get comprehensive ensemble information."""
        return {
            'num_agents': len(self.agents),
            'agent_names': self.agent_names,
            'strategy': self.strategy.value,
            'current_weights': self.weights.copy(),
            'training_step': self.training_step,
            'metrics_summary': self._get_metrics_summary()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble metrics."""
        if not self.metrics.ensemble_rewards:
            return {}
        
        return {
            'mean_ensemble_reward': np.mean(self.metrics.ensemble_rewards[-100:]),
            'mean_agreement_score': np.mean(self.metrics.agreement_scores[-100:]),
            'mean_diversity_score': np.mean(self.metrics.diversity_scores[-100:]),
            'mean_confidence': np.mean(self.metrics.confidence_scores[-100:]),
            'total_steps': len(self.metrics.ensemble_rewards)
        }
    
    def save_ensemble(self, filepath: str):
        """
        Save ensemble state including all agents.
        
        Args:
            filepath: Path to save ensemble checkpoint
        """
        import os
        
        # Create directory for ensemble
        ensemble_dir = filepath.replace('.pth', '_ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Save individual agents
        for name, agent in self.agents.items():
            agent_path = os.path.join(ensemble_dir, f"{name}.pth")
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(agent_path)
        
        # Save ensemble state
        ensemble_state = {
            'strategy': self.strategy.value,
            'weights': self.weights,
            'performance_history': self.performance_history,
            'training_step': self.training_step,
            'agent_names': self.agent_names,
            'metrics': {
                'individual_rewards': self.metrics.individual_rewards,
                'ensemble_rewards': self.metrics.ensemble_rewards,
                'individual_losses': self.metrics.individual_losses,
                'agreement_scores': self.metrics.agreement_scores,
                'diversity_scores': self.metrics.diversity_scores,
                'weights_history': self.metrics.weights_history,
                'confidence_scores': self.metrics.confidence_scores,
            }
        }
        
        torch.save(ensemble_state, filepath)
        print(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """
        Load ensemble state including all agents.
        
        Args:
            filepath: Path to ensemble checkpoint
        """
        import os
        
        # Load ensemble state
        ensemble_state = torch.load(filepath, map_location=self.device)
        
        self.strategy = EnsembleStrategy(ensemble_state['strategy'])
        self.weights = ensemble_state['weights']
        self.performance_history = ensemble_state['performance_history']
        self.training_step = ensemble_state.get('training_step', 0)
        
        # Load metrics
        metrics_data = ensemble_state.get('metrics', {})
        self.metrics = EnsembleMetrics(
            individual_rewards=metrics_data.get('individual_rewards', {}),
            ensemble_rewards=metrics_data.get('ensemble_rewards', []),
            individual_losses=metrics_data.get('individual_losses', {}),
            agreement_scores=metrics_data.get('agreement_scores', []),
            diversity_scores=metrics_data.get('diversity_scores', []),
            weights_history=metrics_data.get('weights_history', []),
            confidence_scores=metrics_data.get('confidence_scores', []),
        )
        
        # Load individual agents
        ensemble_dir = filepath.replace('.pth', '_ensemble')
        for name, agent in self.agents.items():
            agent_path = os.path.join(ensemble_dir, f"{name}.pth")
            if os.path.exists(agent_path) and hasattr(agent, 'load_checkpoint'):
                agent.load_checkpoint(agent_path)
        
        print(f"Ensemble loaded from {filepath}")
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"agents={list(self.agents.keys())}, "
                f"strategy={self.strategy.value}, "
                f"training_step={self.training_step})")