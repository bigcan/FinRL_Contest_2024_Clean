"""
Voting-based ensemble strategies for the FinRL Contest 2024 framework.

This module implements various voting mechanisms for combining decisions
from multiple RL agents.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from collections import Counter

from .base_ensemble import BaseEnsemble, EnsembleStrategy
from ..core.types import StateType, ActionType, TrainingStats


class VotingEnsemble(BaseEnsemble):
    """
    Ensemble that combines agent decisions through voting mechanisms.
    
    Supports multiple voting strategies:
    - Majority vote: Most common action wins
    - Weighted vote: Weighted by agent performance or confidence
    - Uncertainty-weighted: Weight by prediction uncertainty
    """
    
    def __init__(self,
                 agents: Dict[str, Any],
                 strategy: EnsembleStrategy = EnsembleStrategy.MAJORITY_VOTE,
                 device: Optional[torch.device] = None,
                 confidence_threshold: float = 0.5,
                 temperature: float = 1.0):
        """
        Initialize voting ensemble.
        
        Args:
            agents: Dictionary of agents
            strategy: Voting strategy
            device: Computing device
            confidence_threshold: Minimum confidence for agent participation
            temperature: Temperature for softmax voting (higher = more diverse)
        """
        super().__init__(agents, strategy, device)
        
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        
        # Initialize confidence tracking
        self.agent_confidences = {name: 1.0 for name in self.agent_names}
        self.confidence_history = {name: [] for name in self.agent_names}
        
        print(f"Initialized VotingEnsemble with confidence_threshold={confidence_threshold}, "
              f"temperature={temperature}")
    
    def select_action(self, state: StateType, deterministic: bool = False) -> ActionType:
        """
        Select action using voting strategy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic selection
            
        Returns:
            Ensemble action
        """
        # Get individual actions
        individual_actions = self.get_individual_actions(state, deterministic)
        
        if self.strategy == EnsembleStrategy.MAJORITY_VOTE:
            return self._majority_vote(individual_actions)
        
        elif self.strategy == EnsembleStrategy.WEIGHTED_VOTE:
            return self._weighted_vote(individual_actions, state)
        
        elif self.strategy == EnsembleStrategy.UNCERTAINTY_WEIGHTED:
            return self._uncertainty_weighted_vote(individual_actions, state)
        
        else:
            # Fallback to majority vote
            return self._majority_vote(individual_actions)
    
    def _majority_vote(self, actions: Dict[str, ActionType]) -> ActionType:
        """
        Select action by majority vote.
        
        Args:
            actions: Dictionary of agent actions
            
        Returns:
            Most popular action
        """
        action_list = list(actions.values())
        
        if isinstance(action_list[0], (int, np.integer)):
            # Discrete actions - use mode
            counter = Counter(action_list)
            return counter.most_common(1)[0][0]
        else:
            # Continuous actions - use mean
            return np.mean(action_list, axis=0)
    
    def _weighted_vote(self, actions: Dict[str, ActionType], state: StateType) -> ActionType:
        """
        Select action by weighted vote using current agent weights.
        
        Args:
            actions: Dictionary of agent actions
            state: Current state
            
        Returns:
            Weighted action
        """
        action_list = list(actions.values())
        weights_list = [self.weights[name] for name in actions.keys()]
        
        if isinstance(action_list[0], (int, np.integer)):
            # Discrete actions - use weighted probability
            action_probs = {}
            for action, weight in zip(action_list, weights_list):
                action_probs[action] = action_probs.get(action, 0) + weight
            
            # Select action with highest weighted probability
            return max(action_probs, key=action_probs.get)
        else:
            # Continuous actions - weighted average
            weighted_actions = [action * weight for action, weight in zip(action_list, weights_list)]
            return np.sum(weighted_actions, axis=0)
    
    def _uncertainty_weighted_vote(self, actions: Dict[str, ActionType], state: StateType) -> ActionType:
        """
        Select action weighted by agent uncertainty (confidence).
        
        Args:
            actions: Dictionary of agent actions
            state: Current state
            
        Returns:
            Uncertainty-weighted action
        """
        # Get Q-values to calculate uncertainty
        q_values = self.get_individual_q_values(state)
        
        # Calculate uncertainty-based weights
        uncertainty_weights = {}
        for name in actions.keys():
            if name in q_values:
                q_vals = q_values[name]
                # Uncertainty as entropy of softmax distribution
                if q_vals.numel() > 1:
                    probs = torch.softmax(q_vals / self.temperature, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    # Convert entropy to confidence (lower entropy = higher confidence)
                    confidence = 1.0 / (1.0 + entropy.item())
                else:
                    confidence = self.agent_confidences.get(name, 1.0)
            else:
                confidence = self.agent_confidences.get(name, 1.0)
            
            # Only include agents above confidence threshold
            if confidence >= self.confidence_threshold:
                uncertainty_weights[name] = confidence
        
        # Normalize weights
        total_weight = sum(uncertainty_weights.values())
        if total_weight > 0:
            uncertainty_weights = {name: w / total_weight for name, w in uncertainty_weights.items()}
        else:
            # Fallback to uniform weights if all below threshold
            uncertainty_weights = {name: 1.0 / len(actions) for name in actions.keys()}
        
        # Apply weighted voting
        action_list = [actions[name] for name in uncertainty_weights.keys()]
        weights_list = list(uncertainty_weights.values())
        
        if isinstance(action_list[0], (int, np.integer)):
            # Discrete actions
            action_probs = {}
            for action, weight in zip(action_list, weights_list):
                action_probs[action] = action_probs.get(action, 0) + weight
            return max(action_probs, key=action_probs.get)
        else:
            # Continuous actions
            weighted_actions = [action * weight for action, weight in zip(action_list, weights_list)]
            return np.sum(weighted_actions, axis=0)
    
    def update(self, batch_data: Any) -> Dict[str, Any]:
        """
        Update all agents and ensemble statistics.
        
        Args:
            batch_data: Training batch data
            
        Returns:
            Comprehensive training statistics
        """
        self.training_step += 1
        
        # Update individual agents
        individual_stats = {}
        individual_rewards = {}
        individual_losses = {}
        
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'update'):
                    stats = agent.update(batch_data)
                    individual_stats[name] = stats
                    
                    # Extract key metrics
                    if isinstance(stats, dict):
                        individual_losses[name] = stats.get('critic_loss', 0.0)
                        individual_rewards[name] = stats.get('reward', 0.0)
                    else:
                        # Handle TrainingStats object
                        individual_losses[name] = getattr(stats, 'critic_loss', 0.0)
                        individual_rewards[name] = getattr(stats, 'reward', 0.0)
                else:
                    # Agent doesn't support update
                    individual_losses[name] = 0.0
                    individual_rewards[name] = 0.0
                    
            except Exception as e:
                print(f"Warning: Failed to update agent {name}: {e}")
                individual_losses[name] = float('inf')
                individual_rewards[name] = 0.0
        
        # Calculate ensemble metrics
        ensemble_reward = np.mean(list(individual_rewards.values()))
        
        # Get sample state for agreement/diversity calculation
        if hasattr(batch_data, '__len__') and len(batch_data) > 0:
            sample_state = batch_data[0] if isinstance(batch_data, (list, tuple)) else None
            if sample_state is not None:
                # Calculate agreement and diversity
                individual_actions = self.get_individual_actions(sample_state)
                agreement_score = self.calculate_agreement_score(individual_actions)
                
                q_values = self.get_individual_q_values(sample_state)
                diversity_score = self.calculate_diversity_score(q_values)
            else:
                agreement_score = 0.0
                diversity_score = 0.0
        else:
            agreement_score = 0.0
            diversity_score = 0.0
        
        # Update agent weights based on performance
        performance_scores = {
            name: max(0.0, 1.0 - loss) for name, loss in individual_losses.items()
            if not np.isnan(loss) and not np.isinf(loss)
        }
        self.update_weights(performance_scores)
        
        # Update confidence tracking
        self._update_confidence_tracking(individual_losses)
        
        # Calculate overall confidence
        confidence = np.mean(list(self.agent_confidences.values()))
        
        # Update metrics
        self.metrics.add_step_metrics(
            individual_rewards=individual_rewards,
            ensemble_reward=ensemble_reward,
            individual_losses=individual_losses,
            agreement_score=agreement_score,
            diversity_score=diversity_score,
            weights=self.weights.copy(),
            confidence=confidence
        )
        
        # Compile comprehensive statistics
        ensemble_stats = {
            'ensemble_reward': ensemble_reward,
            'individual_rewards': individual_rewards,
            'individual_losses': individual_losses,
            'agreement_score': agreement_score,
            'diversity_score': diversity_score,
            'current_weights': self.weights.copy(),
            'agent_confidences': self.agent_confidences.copy(),
            'overall_confidence': confidence,
            'training_step': self.training_step,
            'individual_stats': individual_stats
        }
        
        return ensemble_stats
    
    def _update_confidence_tracking(self, losses: Dict[str, float]):
        """
        Update agent confidence based on recent performance.
        
        Args:
            losses: Dictionary of agent losses
        """
        for name, loss in losses.items():
            if not np.isnan(loss) and not np.isinf(loss):
                # Convert loss to confidence (lower loss = higher confidence)
                confidence = 1.0 / (1.0 + loss)
                
                # Exponential moving average
                alpha = 0.1
                if name in self.agent_confidences:
                    self.agent_confidences[name] = (
                        (1 - alpha) * self.agent_confidences[name] + alpha * confidence
                    )
                else:
                    self.agent_confidences[name] = confidence
                
                # Track history
                self.confidence_history[name].append(confidence)
                if len(self.confidence_history[name]) > 1000:
                    self.confidence_history[name].pop(0)
    
    def get_voting_statistics(self) -> Dict[str, Any]:
        """Get detailed voting statistics."""
        return {
            'voting_strategy': self.strategy.value,
            'confidence_threshold': self.confidence_threshold,
            'temperature': self.temperature,
            'agent_confidences': self.agent_confidences.copy(),
            'confidence_history_length': {
                name: len(history) for name, history in self.confidence_history.items()
            },
            'mean_confidence_per_agent': {
                name: np.mean(history[-100:]) if history else 0.0
                for name, history in self.confidence_history.items()
            },
            'weights': self.weights.copy(),
            'ensemble_info': self.get_ensemble_info()
        }
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold for agent participation.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def set_temperature(self, temperature: float):
        """
        Update temperature for uncertainty calculations.
        
        Args:
            temperature: New temperature value (> 0.0)
        """
        self.temperature = max(0.1, temperature)
        print(f"Updated temperature to {self.temperature}")
    
    def get_active_agents(self) -> List[str]:
        """
        Get list of agents currently above confidence threshold.
        
        Returns:
            List of active agent names
        """
        return [
            name for name, confidence in self.agent_confidences.items()
            if confidence >= self.confidence_threshold
        ]
    
    def __repr__(self) -> str:
        active_agents = len(self.get_active_agents())
        return (f"VotingEnsemble("
                f"total_agents={len(self.agents)}, "
                f"active_agents={active_agents}, "
                f"strategy={self.strategy.value}, "
                f"confidence_threshold={self.confidence_threshold})")