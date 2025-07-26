import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter, deque
import pickle
import os


class WeightedVotingEnsemble:
    """
    Advanced ensemble using weighted voting based on individual agent performance
    """
    
    def __init__(self, agents: List[Any], performance_window: int = 1000):
        """
        Initialize weighted voting ensemble
        
        Args:
            agents: List of trained agents
            performance_window: Window size for performance tracking
        """
        self.agents = agents
        self.performance_window = performance_window
        
        # Performance tracking for each agent
        self.agent_performances = [deque(maxlen=performance_window) for _ in agents]
        self.agent_weights = [1.0 / len(agents)] * len(agents)  # Initial equal weights
        
        # Ensemble statistics
        self.prediction_count = 0
        self.weight_update_frequency = 100
        
    def predict(self, state: torch.Tensor, update_weights: bool = True) -> int:
        """
        Make ensemble prediction using weighted voting
        
        Args:
            state: Input state tensor
            update_weights: Whether to update weights based on recent performance
            
        Returns:
            Ensemble action prediction
        """
        # Get predictions from all agents
        agent_actions = []
        agent_confidences = []
        
        with torch.no_grad():
            for agent in self.agents:
                # Get Q-values and action
                if hasattr(agent.act, 'get_q1_q2'):
                    q1, q2 = agent.act.get_q1_q2(state)
                    q_values = torch.min(q1, q2)
                else:
                    q_values = agent.act(state)
                
                action = q_values.argmax(dim=1).item()
                confidence = torch.softmax(q_values, dim=1).max().item()
                
                agent_actions.append(action)
                agent_confidences.append(confidence)
        
        # Update weights periodically
        if update_weights and self.prediction_count % self.weight_update_frequency == 0:
            self._update_weights()
        
        # Weighted voting
        ensemble_action = self._weighted_vote(agent_actions, agent_confidences)
        
        self.prediction_count += 1
        return ensemble_action
    
    def _weighted_vote(self, actions: List[int], confidences: List[float]) -> int:
        """
        Perform weighted voting combining agent weights and prediction confidence
        
        Args:
            actions: List of agent actions
            confidences: List of prediction confidences
            
        Returns:
            Ensemble action
        """
        # Combine agent weights with prediction confidence
        combined_weights = [
            self.agent_weights[i] * confidences[i] 
            for i in range(len(actions))
        ]
        
        # Weight votes by combined score
        action_scores = {}
        for action, weight in zip(actions, combined_weights):
            action_scores[action] = action_scores.get(action, 0) + weight
        
        # Return action with highest weighted score
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def update_performance(self, agent_rewards: List[float]):
        """
        Update individual agent performance metrics
        
        Args:
            agent_rewards: List of rewards for each agent
        """
        for i, reward in enumerate(agent_rewards):
            self.agent_performances[i].append(reward)
    
    def _update_weights(self):
        """Update agent weights based on recent performance"""
        if not self.agent_performances[0]:  # No performance data yet
            return
            
        # Calculate performance metrics for each agent
        performance_scores = []
        for perf_history in self.agent_performances:
            if len(perf_history) > 10:
                # Use Sharpe ratio-like metric (mean/std of returns)
                mean_perf = np.mean(perf_history)
                std_perf = np.std(perf_history) + 1e-8  # Avoid division by zero
                score = mean_perf / std_perf
            else:
                score = np.mean(perf_history) if perf_history else 0.0
            performance_scores.append(score)
        
        # Convert to weights using softmax
        scores_tensor = torch.tensor(performance_scores, dtype=torch.float32)
        self.agent_weights = torch.softmax(scores_tensor * 2.0, dim=0).tolist()  # Temperature scaling
    
    def get_agent_weights(self) -> List[float]:
        """Get current agent weights"""
        return self.agent_weights.copy()


class StackingEnsemble:
    """
    Stacking ensemble that uses a meta-learner to combine agent predictions
    """
    
    def __init__(self, agents: List[Any], state_dim: int, action_dim: int, device: torch.device):
        """
        Initialize stacking ensemble
        
        Args:
            agents: List of base agents
            state_dim: State dimensionality
            action_dim: Action dimensionality
            device: PyTorch device
        """
        self.agents = agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Meta-learner network
        self.meta_learner = self._build_meta_learner()
        self.meta_optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        
        # Training data collection
        self.meta_training_data = []
        self.max_meta_data = 10000
        
    def _build_meta_learner(self) -> nn.Module:
        """Build meta-learner network"""
        # Input: state + agent predictions
        input_dim = self.state_dim + len(self.agents) * self.action_dim
        
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        ).to(self.device)
    
    def predict(self, state: torch.Tensor) -> int:
        """
        Make ensemble prediction using meta-learner
        
        Args:
            state: Input state tensor
            
        Returns:
            Ensemble action prediction
        """
        with torch.no_grad():
            # Get base agent predictions
            agent_predictions = []
            for agent in self.agents:
                if hasattr(agent.act, 'get_q1_q2'):
                    q1, q2 = agent.act.get_q1_q2(state)
                    q_values = torch.min(q1, q2)
                else:
                    q_values = agent.act(state)
                agent_predictions.append(q_values)
            
            # Concatenate state and agent predictions
            agent_preds_flat = torch.cat(agent_predictions, dim=1)
            meta_input = torch.cat([state, agent_preds_flat], dim=1)
            
            # Meta-learner prediction
            meta_output = self.meta_learner(meta_input)
            action = meta_output.argmax(dim=1).item()
            
        return action
    
    def add_training_data(self, state: torch.Tensor, agent_predictions: List[torch.Tensor], 
                         target_q_values: torch.Tensor):
        """
        Add training data for meta-learner
        
        Args:
            state: State tensor
            agent_predictions: List of agent Q-value predictions
            target_q_values: Target Q-values (from actual experience)
        """
        # Prepare meta-learner input
        agent_preds_flat = torch.cat(agent_predictions, dim=1)
        meta_input = torch.cat([state, agent_preds_flat], dim=1)
        
        # Store training sample
        self.meta_training_data.append((meta_input.cpu(), target_q_values.cpu()))
        
        # Limit training data size
        if len(self.meta_training_data) > self.max_meta_data:
            self.meta_training_data.pop(0)
    
    def train_meta_learner(self, batch_size: int = 32, num_epochs: int = 10):
        """
        Train the meta-learner on collected data
        
        Args:
            batch_size: Training batch size
            num_epochs: Number of training epochs
        """
        if len(self.meta_training_data) < batch_size:
            return
        
        self.meta_learner.train()
        
        for epoch in range(num_epochs):
            # Sample batch
            indices = np.random.choice(len(self.meta_training_data), batch_size, replace=False)
            batch_inputs = torch.stack([self.meta_training_data[i][0] for i in indices]).to(self.device)
            batch_targets = torch.stack([self.meta_training_data[i][1] for i in indices]).to(self.device)
            
            # Forward pass
            predictions = self.meta_learner(batch_inputs)
            loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            self.meta_optimizer.zero_grad()
            loss.backward()
            self.meta_optimizer.step()
        
        self.meta_learner.eval()


class UncertaintyBasedEnsemble:
    """
    Ensemble that uses prediction uncertainty for weighting and exploration
    """
    
    def __init__(self, agents: List[Any], uncertainty_threshold: float = 0.1):
        """
        Initialize uncertainty-based ensemble
        
        Args:
            agents: List of agents
            uncertainty_threshold: Threshold for high uncertainty decisions
        """
        self.agents = agents
        self.uncertainty_threshold = uncertainty_threshold
        
        # Track uncertainty statistics
        self.uncertainty_history = deque(maxlen=1000)
        
    def predict(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Make prediction with uncertainty estimation
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, uncertainty)
        """
        agent_predictions = []
        agent_entropies = []
        
        with torch.no_grad():
            for agent in self.agents:
                if hasattr(agent.act, 'get_q1_q2'):
                    q1, q2 = agent.act.get_q1_q2(state)
                    q_values = torch.min(q1, q2)
                else:
                    q_values = agent.act(state)
                
                # Convert to action probabilities
                probs = torch.softmax(q_values, dim=1)
                
                # Calculate entropy (uncertainty measure)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).item()
                
                agent_predictions.append(q_values)
                agent_entropies.append(entropy)
        
        # Calculate ensemble prediction and uncertainty
        ensemble_q = torch.stack(agent_predictions).mean(dim=0)
        ensemble_action = ensemble_q.argmax(dim=1).item()
        
        # Uncertainty is based on disagreement between agents and individual entropies
        q_std = torch.stack(agent_predictions).std(dim=0).mean().item()
        mean_entropy = np.mean(agent_entropies)
        uncertainty = q_std + mean_entropy
        
        self.uncertainty_history.append(uncertainty)
        
        return ensemble_action, uncertainty
    
    def should_explore(self, uncertainty: float) -> bool:
        """
        Decide whether to explore based on uncertainty
        
        Args:
            uncertainty: Current prediction uncertainty
            
        Returns:
            Whether to take exploratory action
        """
        return uncertainty > self.uncertainty_threshold
    
    def get_uncertainty_stats(self) -> Dict[str, float]:
        """Get uncertainty statistics"""
        if not self.uncertainty_history:
            return {}
        
        return {
            'mean_uncertainty': np.mean(self.uncertainty_history),
            'std_uncertainty': np.std(self.uncertainty_history),
            'max_uncertainty': np.max(self.uncertainty_history),
            'min_uncertainty': np.min(self.uncertainty_history)
        }


class AdaptiveEnsembleSelector:
    """
    Dynamically selects the best ensemble strategy based on market conditions
    """
    
    def __init__(self, ensemble_strategies: Dict[str, Any], adaptation_window: int = 500):
        """
        Initialize adaptive ensemble selector
        
        Args:
            ensemble_strategies: Dictionary of ensemble strategies
            adaptation_window: Window for performance evaluation
        """
        self.strategies = ensemble_strategies
        self.adaptation_window = adaptation_window
        
        # Performance tracking
        self.strategy_performances = {name: deque(maxlen=adaptation_window) 
                                    for name in ensemble_strategies.keys()}
        self.current_strategy = list(ensemble_strategies.keys())[0]
        
        # Adaptation parameters
        self.evaluation_frequency = 100
        self.prediction_count = 0
        
    def predict(self, state: torch.Tensor) -> int:
        """
        Make prediction using current best strategy
        
        Args:
            state: Input state tensor
            
        Returns:
            Ensemble action prediction
        """
        # Periodically evaluate and switch strategies
        if self.prediction_count % self.evaluation_frequency == 0:
            self._update_strategy_selection()
        
        # Make prediction with current strategy
        current_ensemble = self.strategies[self.current_strategy]
        
        if hasattr(current_ensemble, 'predict'):
            action = current_ensemble.predict(state)
        else:
            # Fallback for simpler ensembles
            action = self._simple_majority_vote(state)
        
        self.prediction_count += 1
        return action
    
    def update_performance(self, strategy_rewards: Dict[str, float]):
        """
        Update performance for all strategies
        
        Args:
            strategy_rewards: Dictionary of rewards for each strategy
        """
        for strategy_name, reward in strategy_rewards.items():
            if strategy_name in self.strategy_performances:
                self.strategy_performances[strategy_name].append(reward)
    
    def _update_strategy_selection(self):
        """Update current strategy based on recent performance"""
        if not any(self.strategy_performances.values()):
            return
        
        # Calculate average performance for each strategy
        strategy_scores = {}
        for name, performances in self.strategy_performances.items():
            if performances:
                # Use Sharpe-like ratio
                mean_perf = np.mean(performances)
                std_perf = np.std(performances) + 1e-8
                strategy_scores[name] = mean_perf / std_perf
            else:
                strategy_scores[name] = 0.0
        
        # Select best performing strategy
        self.current_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def _simple_majority_vote(self, state: torch.Tensor) -> int:
        """Fallback majority voting"""
        # This would need to be implemented based on available agents
        # For now, return a placeholder
        return 1  # Hold action
    
    def get_current_strategy(self) -> str:
        """Get name of current strategy"""
        return self.current_strategy
    
    def get_strategy_performances(self) -> Dict[str, float]:
        """Get recent performance of all strategies"""
        performances = {}
        for name, perf_history in self.strategy_performances.items():
            if perf_history:
                performances[name] = np.mean(perf_history)
            else:
                performances[name] = 0.0
        return performances