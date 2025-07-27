"""
Stacking ensemble implementation for the FinRL Contest 2024 framework.

This module implements stacking (stacked generalization) where a meta-learner
is trained to combine predictions from multiple base agents.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from .base_ensemble import BaseEnsemble, EnsembleStrategy
from ..core.types import StateType, ActionType, TrainingStats
from ..networks.base_networks import QNetBase


class MetaLearnerNetwork(QNetBase):
    """
    Meta-learner network for stacking ensemble.
    
    Takes as input the concatenated predictions/features from base agents
    and outputs the final ensemble decision.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 activation: str = "relu",
                 dropout_rate: float = 0.1):
        """
        Initialize meta-learner network.
        
        Args:
            input_dim: Input dimension (concatenated agent features)
            output_dim: Output dimension (action space)
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "elu":
            return nn.ELU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through meta-learner.
        
        Args:
            x: Concatenated agent features/predictions
            
        Returns:
            Meta-learner output
        """
        return self.network(x)


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble that uses a meta-learner to combine base agent decisions.
    
    The meta-learner is trained to predict the optimal action or Q-values
    based on the outputs of the base agents.
    """
    
    def __init__(self,
                 agents: Dict[str, Any],
                 action_dim: int,
                 device: Optional[torch.device] = None,
                 meta_learning_rate: float = 1e-4,
                 meta_hidden_dims: List[int] = [128, 64],
                 meta_update_frequency: int = 10,
                 buffer_size: int = 10000):
        """
        Initialize stacking ensemble.
        
        Args:
            agents: Dictionary of base agents
            action_dim: Action space dimensionality
            device: Computing device
            meta_learning_rate: Learning rate for meta-learner
            meta_hidden_dims: Hidden dimensions for meta-learner
            meta_update_frequency: Steps between meta-learner updates
            buffer_size: Size of meta-learning buffer
        """
        super().__init__(agents, EnsembleStrategy.STACKING, device)
        
        self.action_dim = action_dim
        self.meta_learning_rate = meta_learning_rate
        self.meta_update_frequency = meta_update_frequency
        self.buffer_size = buffer_size
        
        # Calculate meta-learner input dimension
        # Each agent contributes Q-values (action_dim) + confidence (1)
        self.meta_input_dim = len(agents) * (action_dim + 1)
        
        # Initialize meta-learner network
        self.meta_learner = MetaLearnerNetwork(
            input_dim=self.meta_input_dim,
            output_dim=action_dim,
            hidden_dims=meta_hidden_dims
        ).to(self.device)
        
        # Initialize meta-learner optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.meta_learner.parameters(), 
            lr=meta_learning_rate
        )
        
        # Meta-learning buffer for training data
        self.meta_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.meta_losses = []
        self.meta_update_count = 0
        
        print(f"Initialized StackingEnsemble with meta-learner:")
        print(f"  Input dim: {self.meta_input_dim}")
        print(f"  Output dim: {action_dim}")
        print(f"  Hidden dims: {meta_hidden_dims}")
        print(f"  Learning rate: {meta_learning_rate}")
    
    def _extract_agent_features(self, state: StateType) -> torch.Tensor:
        """
        Extract features from all base agents for meta-learner input.
        
        Args:
            state: Current state
            
        Returns:
            Concatenated agent features
        """
        features = []
        
        # Get Q-values and confidence from each agent
        q_values = self.get_individual_q_values(state)
        
        for name in self.agent_names:
            if name in q_values:
                # Use actual Q-values
                q_vals = q_values[name]
                if q_vals.dim() > 1:
                    q_vals = q_vals.squeeze(0)  # Remove batch dimension
                
                # Calculate confidence as max Q-value difference
                if q_vals.numel() > 1:
                    confidence = (q_vals.max() - q_vals.mean()).unsqueeze(0)
                else:
                    confidence = torch.tensor([1.0], device=self.device)
                
                agent_features = torch.cat([q_vals, confidence])
            else:
                # Fallback: use zero features
                agent_features = torch.zeros(self.action_dim + 1, device=self.device)
            
            features.append(agent_features)
        
        return torch.cat(features)
    
    def select_action(self, state: StateType, deterministic: bool = False) -> ActionType:
        """
        Select action using meta-learner.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic selection
            
        Returns:
            Meta-learner selected action
        """
        # Extract features from base agents
        agent_features = self._extract_agent_features(state)
        
        # Get meta-learner prediction
        with torch.no_grad():
            meta_q_values = self.meta_learner(agent_features.unsqueeze(0))
            
            if deterministic:
                action = meta_q_values.argmax(dim=1).item()
            else:
                # Epsilon-greedy or softmax sampling could be added here
                action = meta_q_values.argmax(dim=1).item()
        
        return action
    
    def update(self, batch_data: Any) -> Dict[str, Any]:
        """
        Update base agents and meta-learner.
        
        Args:
            batch_data: Training batch data
            
        Returns:
            Comprehensive training statistics
        """
        self.training_step += 1
        
        # Update base agents first
        base_stats = self._update_base_agents(batch_data)
        
        # Collect meta-learning data
        self._collect_meta_data(batch_data, base_stats)
        
        # Update meta-learner if enough data and time
        meta_stats = {}
        if (len(self.meta_buffer) >= 64 and 
            self.training_step % self.meta_update_frequency == 0):
            meta_stats = self._update_meta_learner()
        
        # Calculate ensemble metrics
        ensemble_reward = np.mean([stats.get('reward', 0.0) for stats in base_stats.values()])
        individual_rewards = {name: stats.get('reward', 0.0) for name, stats in base_stats.items()}
        individual_losses = {name: stats.get('critic_loss', 0.0) for name, stats in base_stats.items()}
        
        # Calculate agreement and diversity if possible
        if hasattr(batch_data, '__len__') and len(batch_data) > 0:
            sample_state = batch_data[0] if isinstance(batch_data, (list, tuple)) else None
            if sample_state is not None:
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
        
        # Update ensemble metrics
        confidence = 1.0 - meta_stats.get('meta_loss', 0.0) if meta_stats else 1.0
        
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
            'meta_loss': meta_stats.get('meta_loss', 0.0),
            'meta_accuracy': meta_stats.get('meta_accuracy', 0.0),
            'meta_buffer_size': len(self.meta_buffer),
            'meta_update_count': self.meta_update_count,
            'training_step': self.training_step,
            'base_agent_stats': base_stats
        }
        
        return ensemble_stats
    
    def _update_base_agents(self, batch_data: Any) -> Dict[str, Dict[str, Any]]:
        """Update all base agents."""
        base_stats = {}
        
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'update'):
                    stats = agent.update(batch_data)
                    base_stats[name] = stats if isinstance(stats, dict) else {
                        'critic_loss': getattr(stats, 'critic_loss', 0.0),
                        'reward': getattr(stats, 'reward', 0.0)
                    }
                else:
                    base_stats[name] = {'critic_loss': 0.0, 'reward': 0.0}
            except Exception as e:
                print(f"Warning: Failed to update base agent {name}: {e}")
                base_stats[name] = {'critic_loss': float('inf'), 'reward': 0.0}
        
        return base_stats
    
    def _collect_meta_data(self, batch_data: Any, base_stats: Dict[str, Dict[str, Any]]):
        """
        Collect training data for meta-learner.
        
        Args:
            batch_data: Original training batch
            base_stats: Statistics from base agent updates
        """
        try:
            # Extract state and target from batch_data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 4:
                states, actions, rewards, dones = batch_data[:4]
                
                # Sample a few states for meta-learning
                if hasattr(states, '__len__') and len(states) > 0:
                    # Take first state as sample
                    sample_state = states[0] if hasattr(states, '__getitem__') else states
                    target_action = actions[0] if hasattr(actions, '__getitem__') else actions
                    
                    # Extract agent features
                    agent_features = self._extract_agent_features(sample_state)
                    
                    # Store for meta-learning
                    self.meta_buffer.append({
                        'features': agent_features.detach().cpu(),
                        'target_action': target_action,
                        'reward': rewards[0] if hasattr(rewards, '__getitem__') else rewards
                    })
                    
        except Exception as e:
            print(f"Warning: Failed to collect meta-learning data: {e}")
    
    def _update_meta_learner(self) -> Dict[str, float]:
        """
        Update meta-learner using collected data.
        
        Returns:
            Meta-learner training statistics
        """
        if len(self.meta_buffer) < 32:
            return {}
        
        # Sample batch from meta buffer
        batch_size = min(64, len(self.meta_buffer))
        indices = np.random.choice(len(self.meta_buffer), batch_size, replace=False)
        
        # Prepare batch
        features_batch = []
        targets_batch = []
        
        for idx in indices:
            sample = self.meta_buffer[idx]
            features_batch.append(sample['features'])
            targets_batch.append(sample['target_action'])
        
        features_tensor = torch.stack(features_batch).to(self.device)
        targets_tensor = torch.tensor(targets_batch, dtype=torch.long, device=self.device)
        
        # Update meta-learner
        self.meta_optimizer.zero_grad()
        
        # Forward pass
        meta_outputs = self.meta_learner(features_tensor)
        
        # Calculate loss (cross-entropy for discrete actions)
        loss = F.cross_entropy(meta_outputs, targets_tensor)
        
        # Backward pass
        loss.backward()
        self.meta_optimizer.step()
        
        # Calculate accuracy
        predictions = meta_outputs.argmax(dim=1)
        accuracy = (predictions == targets_tensor).float().mean().item()
        
        # Update statistics
        self.meta_losses.append(loss.item())
        if len(self.meta_losses) > 1000:
            self.meta_losses.pop(0)
        
        self.meta_update_count += 1
        
        return {
            'meta_loss': loss.item(),
            'meta_accuracy': accuracy,
            'meta_lr': self.meta_optimizer.param_groups[0]['lr']
        }
    
    def get_meta_statistics(self) -> Dict[str, Any]:
        """Get detailed meta-learner statistics."""
        return {
            'meta_learner_params': sum(p.numel() for p in self.meta_learner.parameters()),
            'meta_buffer_size': len(self.meta_buffer),
            'meta_update_count': self.meta_update_count,
            'meta_update_frequency': self.meta_update_frequency,
            'recent_meta_loss': np.mean(self.meta_losses[-100:]) if self.meta_losses else 0.0,
            'meta_input_dim': self.meta_input_dim,
            'meta_output_dim': self.action_dim,
            'meta_learning_rate': self.meta_learning_rate
        }
    
    def save_ensemble(self, filepath: str):
        """Save ensemble including meta-learner state."""
        # Save base ensemble
        super().save_ensemble(filepath)
        
        # Save meta-learner state
        meta_state = {
            'meta_learner': self.meta_learner.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'meta_losses': self.meta_losses,
            'meta_update_count': self.meta_update_count,
            'meta_buffer': list(self.meta_buffer),
            'action_dim': self.action_dim,
            'meta_input_dim': self.meta_input_dim
        }
        
        meta_filepath = filepath.replace('.pth', '_meta.pth')
        torch.save(meta_state, meta_filepath)
        print(f"Meta-learner saved to {meta_filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble including meta-learner state."""
        # Load base ensemble
        super().load_ensemble(filepath)
        
        # Load meta-learner state
        meta_filepath = filepath.replace('.pth', '_meta.pth')
        try:
            meta_state = torch.load(meta_filepath, map_location=self.device)
            
            self.meta_learner.load_state_dict(meta_state['meta_learner'])
            self.meta_optimizer.load_state_dict(meta_state['meta_optimizer'])
            self.meta_losses = meta_state.get('meta_losses', [])
            self.meta_update_count = meta_state.get('meta_update_count', 0)
            
            # Restore meta buffer
            buffer_data = meta_state.get('meta_buffer', [])
            self.meta_buffer = deque(buffer_data, maxlen=self.buffer_size)
            
            print(f"Meta-learner loaded from {meta_filepath}")
            
        except FileNotFoundError:
            print(f"Warning: Meta-learner checkpoint not found at {meta_filepath}")
        except Exception as e:
            print(f"Warning: Failed to load meta-learner: {e}")
    
    def __repr__(self) -> str:
        return (f"StackingEnsemble("
                f"agents={len(self.agents)}, "
                f"meta_input_dim={self.meta_input_dim}, "
                f"meta_output_dim={self.action_dim}, "
                f"meta_updates={self.meta_update_count}, "
                f"buffer_size={len(self.meta_buffer)})")