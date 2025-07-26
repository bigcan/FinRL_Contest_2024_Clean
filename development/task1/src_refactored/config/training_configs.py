"""
Training-specific configuration definitions for the FinRL Contest 2024 framework.
Provides structured configuration classes for training parameters, environment settings,
and ensemble configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum


class EnsembleStrategy(Enum):
    """Ensemble strategy types"""
    MAJORITY_VOTING = "majority_voting"
    WEIGHTED_VOTING = "weighted_voting"
    STACKING = "stacking"
    UNCERTAINTY_BASED = "uncertainty_based"
    ADAPTIVE = "adaptive"


class BufferType(Enum):
    """Replay buffer types"""
    UNIFORM = "uniform"
    PRIORITIZED = "prioritized"


@dataclass
class EnvironmentConfig:
    """Configuration for trading environment"""
    gpu_id: int = 0
    num_sims: int = 64               # Number of parallel simulations
    num_ignore_step: int = 60        # Steps to ignore at start
    max_position: int = 1            # Maximum position size
    step_gap: int = 2                # Gap between steps
    slippage: float = 7e-7           # Trading slippage
    starting_cash: float = 1000000   # Initial cash amount
    data_length: int = 4800          # Length of data for training
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'gpu_id': self.gpu_id,
            'num_sims': self.num_sims,
            'num_ignore_step': self.num_ignore_step,
            'max_position': self.max_position,
            'step_gap': self.step_gap,
            'slippage': self.slippage,
            'starting_cash': self.starting_cash,
            'data_length': self.data_length,
        }


@dataclass
class DataConfig:
    """Configuration for data paths and sources"""
    csv_path: str = "data/BTC_1sec.csv"
    predict_path: str = "data/BTC_1sec_predict.npy"
    enhanced_features: Dict[str, str] = field(default_factory=lambda: {
        "enhanced_v3": "data/BTC_1sec_predict_enhanced_v3.npy",
        "optimized": "data/BTC_1sec_predict_optimized.npy", 
        "enhanced": "data/BTC_1sec_predict_enhanced.npy"
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'csv_path': self.csv_path,
            'predict_path': self.predict_path,
            'enhanced_features': self.enhanced_features,
        }


@dataclass
class ReplayBufferConfig:
    """Configuration for replay buffer"""
    buffer_type: BufferType = BufferType.UNIFORM
    buffer_size_multiplier: int = 8   # Multiplier for buffer size
    
    # Prioritized Experience Replay specific
    per_alpha: float = 0.6            # Prioritization exponent
    per_beta: float = 0.4             # Importance sampling initial value
    per_beta_annealing_steps: int = 100000  # Steps to anneal beta to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'buffer_type': self.buffer_type.value,
            'buffer_size_multiplier': self.buffer_size_multiplier,
            'per_alpha': self.per_alpha,
            'per_beta': self.per_beta,
            'per_beta_annealing_steps': self.per_beta_annealing_steps,
        }


@dataclass
class TrainingConfig:
    """Main training configuration"""
    break_step: int = 100             # Steps before breaking training
    horizon_len_multiplier: int = 2   # Horizon length multiplier
    eval_per_step_multiplier: int = 1 # Evaluation frequency multiplier
    num_workers: int = 1              # Number of training workers
    save_gap: int = 8                 # Gap between model saves
    
    # Training monitoring
    log_frequency: int = 100          # Steps between logging
    eval_frequency: int = 1000        # Steps between evaluations
    checkpoint_frequency: int = 5000  # Steps between checkpoints
    
    # Early stopping
    patience: int = 10                # Early stopping patience
    min_improvement: float = 0.001    # Minimum improvement threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'break_step': self.break_step,
            'horizon_len_multiplier': self.horizon_len_multiplier,
            'eval_per_step_multiplier': self.eval_per_step_multiplier,
            'num_workers': self.num_workers,
            'save_gap': self.save_gap,
            'log_frequency': self.log_frequency,
            'eval_frequency': self.eval_frequency,
            'checkpoint_frequency': self.checkpoint_frequency,
            'patience': self.patience,
            'min_improvement': self.min_improvement,
        }


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training and strategy"""
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_VOTING
    
    # Weighted voting parameters
    performance_window: int = 1000    # Window for performance tracking
    weight_update_frequency: int = 100  # How often to update weights
    
    # Stacking ensemble parameters
    meta_learner_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    meta_learner_lr: float = 1e-4
    meta_training_frequency: int = 1000
    max_meta_data: int = 10000
    
    # Uncertainty-based parameters
    uncertainty_threshold: float = 0.1
    
    # Adaptive ensemble parameters
    adaptation_window: int = 500
    evaluation_frequency: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'strategy': self.strategy.value,
            'performance_window': self.performance_window,
            'weight_update_frequency': self.weight_update_frequency,
            'meta_learner_hidden_dims': self.meta_learner_hidden_dims,
            'meta_learner_lr': self.meta_learner_lr,
            'meta_training_frequency': self.meta_training_frequency,
            'max_meta_data': self.max_meta_data,
            'uncertainty_threshold': self.uncertainty_threshold,
            'adaptation_window': self.adaptation_window,
            'evaluation_frequency': self.evaluation_frequency,
        }


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all components"""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # Experiment metadata
    experiment_name: str = "finrl_contest_experiment"
    experiment_description: str = "FinRL Contest 2024 Trading Experiment"
    random_seed: Optional[int] = None
    
    # Output paths
    save_dir: str = "ensemble_teamname/ensemble_models"
    log_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'environment': self.environment.to_dict(),
            'data': self.data.to_dict(),
            'training': self.training.to_dict(),
            'replay_buffer': self.replay_buffer.to_dict(),
            'ensemble': self.ensemble.to_dict(),
            'experiment_name': self.experiment_name,
            'experiment_description': self.experiment_description,
            'random_seed': self.random_seed,
            'save_dir': self.save_dir,
            'log_dir': self.log_dir,
        }
    
    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        return cls(
            environment=EnvironmentConfig(**config_dict.get('environment', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            replay_buffer=ReplayBufferConfig(
                buffer_type=BufferType(config_dict.get('replay_buffer', {}).get('buffer_type', 'uniform')),
                **{k: v for k, v in config_dict.get('replay_buffer', {}).items() if k != 'buffer_type'}
            ),
            ensemble=EnsembleConfig(
                strategy=EnsembleStrategy(config_dict.get('ensemble', {}).get('strategy', 'weighted_voting')),
                **{k: v for k, v in config_dict.get('ensemble', {}).items() if k != 'strategy'}
            ),
            experiment_name=config_dict.get('experiment_name', 'finrl_contest_experiment'),
            experiment_description=config_dict.get('experiment_description', 'FinRL Contest 2024 Trading Experiment'),
            random_seed=config_dict.get('random_seed'),
            save_dir=config_dict.get('save_dir', 'ensemble_teamname/ensemble_models'),
            log_dir=config_dict.get('log_dir', 'logs'),
        )


def load_experiment_config_from_ensemble_json(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load experiment configuration from existing ensemble_config.json
    
    Args:
        config_path: Path to ensemble configuration JSON file
        
    Returns:
        Complete experiment configuration
    """
    import json
    
    with open(config_path, 'r') as f:
        ensemble_config = json.load(f)
    
    # Map ensemble config to new structure
    environment_config = EnvironmentConfig(**ensemble_config.get('ensemble', {}))
    data_config = DataConfig(**ensemble_config.get('data_paths', {}))
    training_config = TrainingConfig(**ensemble_config.get('training', {}))
    
    # Determine buffer type based on agents
    agents = ensemble_config.get('agents', [])
    buffer_type = BufferType.PRIORITIZED if any(
        agent in ["AgentPrioritizedDQN", "AgentRainbowDQN"] for agent in agents
    ) else BufferType.UNIFORM
    
    replay_buffer_config = ReplayBufferConfig(
        buffer_type=buffer_type,
        **ensemble_config.get('per_config', {})
    )
    
    # Map ensemble strategy
    strategy_mapping = {
        'majority_voting': EnsembleStrategy.MAJORITY_VOTING,
        'weighted_voting': EnsembleStrategy.WEIGHTED_VOTING,
        'stacking': EnsembleStrategy.STACKING,
        'uncertainty_based': EnsembleStrategy.UNCERTAINTY_BASED,
        'adaptive': EnsembleStrategy.ADAPTIVE,
    }
    
    ensemble_strategy = strategy_mapping.get(
        ensemble_config.get('ensemble_strategy', 'weighted_voting'),
        EnsembleStrategy.WEIGHTED_VOTING
    )
    
    ensemble_config_obj = EnsembleConfig(strategy=ensemble_strategy)
    
    return ExperimentConfig(
        environment=environment_config,
        data=data_config,
        training=training_config,
        replay_buffer=replay_buffer_config,
        ensemble=ensemble_config_obj,
    )


# Predefined experiment configurations
DEFAULT_EXPERIMENT_CONFIGS = {
    "quick_test": ExperimentConfig(
        environment=EnvironmentConfig(num_sims=16, data_length=1200),
        training=TrainingConfig(break_step=50, save_gap=4),
        experiment_name="quick_test_experiment",
    ),
    "full_training": ExperimentConfig(
        environment=EnvironmentConfig(num_sims=64, data_length=4800),
        training=TrainingConfig(break_step=100, save_gap=8),
        experiment_name="full_training_experiment",
    ),
    "gpu_optimized": ExperimentConfig(
        environment=EnvironmentConfig(num_sims=128, data_length=4800),
        training=TrainingConfig(break_step=200, save_gap=10, num_workers=2),
        experiment_name="gpu_optimized_experiment",
    ),
}


def get_default_experiment_config(config_type: str = "full_training") -> ExperimentConfig:
    """
    Get predefined experiment configuration
    
    Args:
        config_type: Type of experiment configuration
        
    Returns:
        Experiment configuration
        
    Raises:
        ValueError: If config_type is not supported
    """
    if config_type not in DEFAULT_EXPERIMENT_CONFIGS:
        raise ValueError(f"Unsupported config type: {config_type}")
    
    # Return a copy to avoid modifications to the default
    import copy
    return copy.deepcopy(DEFAULT_EXPERIMENT_CONFIGS[config_type])