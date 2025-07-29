"""
Ensemble strategies and meta-learning approaches for the FinRL Contest 2024 framework.

This module provides comprehensive ensemble learning capabilities including:
- Base ensemble framework with multiple decision strategies
- Voting ensembles (majority, weighted, uncertainty-based)
- Stacking ensembles with meta-learning
- Advanced metrics and evaluation systems
"""

from typing import Dict, List, Any, Optional
import torch
from enum import Enum

# Try importing core ensemble components with fallbacks
try:
    from .base_ensemble import (
        BaseEnsemble, 
        EnsembleStrategy, 
        EnsembleMetrics
    )
    from .voting_ensemble import VotingEnsemble
    from .stacking_ensemble import StackingEnsemble, MetaLearnerNetwork
except ImportError as e:
    print(f"Warning: Could not import ensemble implementations: {e}")
    
    # Fallback minimal implementations
    class EnsembleStrategy(Enum):
        MAJORITY_VOTE = "majority_vote"
        WEIGHTED_VOTE = "weighted_vote"
        UNCERTAINTY_WEIGHTED = "uncertainty_weighted"
        STACKING = "stacking"
    
    class EnsembleMetrics:
        def __init__(self):
            self.accuracy = 0.0
            self.diversity = 0.0
    
    class BaseEnsemble:
        def __init__(self, agents, device=None, **kwargs):
            self.agents = agents
            self.device = device
        
        def select_action(self, state, deterministic=False):
            # Simple fallback: use first agent
            if self.agents:
                first_agent = next(iter(self.agents.values()))
                return first_agent.select_action(state, deterministic)
            return 0
        
        def select_action_with_confidence(self, state):
            action = self.select_action(state, deterministic=True)
            return action, 1.0
    
    VotingEnsemble = StackingEnsemble = BaseEnsemble
    
    class MetaLearnerNetwork:
        def __init__(self, *args, **kwargs):
            pass

# Factory functions
def create_ensemble(ensemble_type: str,
                   agents: Dict[str, Any],
                   strategy: EnsembleStrategy = EnsembleStrategy.MAJORITY_VOTE,
                   device: Optional[torch.device] = None,
                   **kwargs) -> BaseEnsemble:
    """
    Factory function to create ensemble instances.
    
    Args:
        ensemble_type: Type of ensemble ("voting" or "stacking")
        agents: Dictionary of agents
        strategy: Ensemble strategy
        device: Computing device
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ensemble instance
        
    Raises:
        ValueError: If ensemble_type is not supported
    """
    if ensemble_type.lower() == "voting":
        return VotingEnsemble(
            agents=agents,
            strategy=strategy,
            device=device,
            **kwargs
        )
    elif ensemble_type.lower() == "stacking":
        action_dim = kwargs.get('action_dim', 3)
        return StackingEnsemble(
            agents=agents,
            action_dim=action_dim,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported ensemble type: {ensemble_type}")


def create_voting_ensemble(agents: Dict[str, Any],
                          strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_VOTE,
                          confidence_threshold: float = 0.5,
                          temperature: float = 1.0,
                          device: Optional[torch.device] = None) -> VotingEnsemble:
    """
    Create a voting ensemble with common configurations.
    
    Args:
        agents: Dictionary of agents
        strategy: Voting strategy
        confidence_threshold: Minimum confidence for participation
        temperature: Temperature for uncertainty weighting
        device: Computing device
        
    Returns:
        Configured voting ensemble
    """
    return VotingEnsemble(
        agents=agents,
        strategy=strategy,
        device=device,
        confidence_threshold=confidence_threshold,
        temperature=temperature
    )


def create_stacking_ensemble(agents: Dict[str, Any],
                           action_dim: int,
                           meta_learning_rate: float = 1e-4,
                           meta_hidden_dims: List[int] = [128, 64],
                           device: Optional[torch.device] = None) -> StackingEnsemble:
    """
    Create a stacking ensemble with common configurations.
    
    Args:
        agents: Dictionary of agents
        action_dim: Action space dimensionality
        meta_learning_rate: Learning rate for meta-learner
        meta_hidden_dims: Hidden dimensions for meta-learner
        device: Computing device
        
    Returns:
        Configured stacking ensemble
    """
    return StackingEnsemble(
        agents=agents,
        action_dim=action_dim,
        device=device,
        meta_learning_rate=meta_learning_rate,
        meta_hidden_dims=meta_hidden_dims
    )


def evaluate_ensemble_diversity(ensemble: BaseEnsemble, 
                               test_states: List[Any],
                               num_samples: int = 100) -> Dict[str, float]:
    """
    Evaluate ensemble diversity across multiple test states.
    
    Args:
        ensemble: Ensemble to evaluate
        test_states: List of test states
        num_samples: Number of samples to use
        
    Returns:
        Dictionary of diversity metrics
    """
    agreement_scores = []
    diversity_scores = []
    
    sample_states = test_states[:num_samples] if len(test_states) > num_samples else test_states
    
    for state in sample_states:
        # Get individual actions
        actions = ensemble.get_individual_actions(state, deterministic=True)
        agreement = ensemble.calculate_agreement_score(actions)
        agreement_scores.append(agreement)
        
        # Get Q-values for diversity
        q_values = ensemble.get_individual_q_values(state)
        if q_values:
            diversity = ensemble.calculate_diversity_score(q_values)
            diversity_scores.append(diversity)
    
    return {
        'mean_agreement': float(torch.tensor(agreement_scores).mean()) if agreement_scores else 0.0,
        'std_agreement': float(torch.tensor(agreement_scores).std()) if agreement_scores else 0.0,
        'mean_diversity': float(torch.tensor(diversity_scores).mean()) if diversity_scores else 0.0,
        'std_diversity': float(torch.tensor(diversity_scores).std()) if diversity_scores else 0.0,
        'num_samples': len(sample_states)
    }


def compare_ensemble_strategies(agents: Dict[str, Any],
                              test_states: List[Any],
                              strategies: List[EnsembleStrategy] = None,
                              action_dim: int = 3,
                              device: Optional[torch.device] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare different ensemble strategies on the same agents.
    
    Args:
        agents: Dictionary of agents
        test_states: List of test states for evaluation
        strategies: List of strategies to compare
        action_dim: Action space dimensionality
        device: Computing device
        
    Returns:
        Dictionary mapping strategy names to their metrics
    """
    if strategies is None:
        strategies = [
            EnsembleStrategy.MAJORITY_VOTE,
            EnsembleStrategy.WEIGHTED_VOTE,
            EnsembleStrategy.UNCERTAINTY_WEIGHTED
        ]
    
    results = {}
    
    for strategy in strategies:
        print(f"Evaluating strategy: {strategy.value}")
        
        if strategy == EnsembleStrategy.STACKING:
            ensemble = create_stacking_ensemble(agents, action_dim, device=device)
        else:
            ensemble = create_voting_ensemble(agents, strategy=strategy, device=device)
        
        # Evaluate diversity
        diversity_metrics = evaluate_ensemble_diversity(ensemble, test_states)
        
        results[strategy.value] = diversity_metrics
    
    return results


# Export all public interfaces
__all__ = [
    # Base classes
    'BaseEnsemble',
    'EnsembleStrategy', 
    'EnsembleMetrics',
    
    # Ensemble implementations
    'VotingEnsemble',
    'StackingEnsemble',
    'MetaLearnerNetwork',
    
    # Factory functions
    'create_ensemble',
    'create_voting_ensemble', 
    'create_stacking_ensemble',
    
    # Evaluation utilities
    'evaluate_ensemble_diversity',
    'compare_ensemble_strategies',
]


# Module information
def get_ensemble_info() -> Dict[str, Any]:
    """Get information about the ensemble module."""
    return {
        'module': 'src_refactored.ensemble',
        'description': 'Comprehensive ensemble learning framework',
        'strategies': [strategy.value for strategy in EnsembleStrategy],
        'ensemble_types': ['voting', 'stacking'],
        'features': [
            'Multiple voting strategies',
            'Meta-learning with stacking',
            'Uncertainty-based weighting',
            'Performance tracking and metrics',
            'Dynamic weight adaptation',
            'Comprehensive evaluation tools'
        ],
        'version': '1.0.0'
    }


if __name__ == "__main__":
    # Demo usage
    print("FinRL Contest 2024 - Ensemble Framework Demo")
    print("=" * 50)
    
    # Show available strategies
    print("\nAvailable Ensemble Strategies:")
    for strategy in EnsembleStrategy:
        print(f"  - {strategy.value}")
    
    # Show module info
    info = get_ensemble_info()
    print(f"\nModule: {info['description']}")
    print(f"Features: {info['features'][:3]}...")  # Show first 3 features