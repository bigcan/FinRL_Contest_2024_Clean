"""
Meta-Learning Hyperparameter Optimizer
Advanced hyperparameter optimization for meta-learning components
"""

import os
import time
import numpy as np
import torch
import json
import itertools
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import meta-learning components
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from meta_learning_framework import MetaLearningEnsembleManager
from meta_learning_agent_wrapper import MetaLearningAgentWrapper  
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN

@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization"""
    
    # Meta-learning parameters
    meta_lookback: int = 500
    regime_features: int = 50
    training_batch_size: int = 32
    training_epochs: int = 10
    
    # Regime detection parameters
    regime_stability_threshold: float = 0.7
    regime_confidence_threshold: float = 0.6
    regime_update_frequency: int = 10
    
    # Performance prediction parameters
    performance_window: int = 50
    prediction_horizon: int = 20
    prediction_confidence_threshold: float = 0.5
    
    # Algorithm selection parameters
    max_weight: float = 0.6
    min_diversification: int = 2
    weight_smoothing_factor: float = 0.3
    
    # Training parameters
    meta_training_frequency: int = 100
    performance_evaluation_window: int = 30
    adaptation_learning_rate: float = 0.001
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'meta_lookback': self.meta_lookback,
            'regime_features': self.regime_features,
            'training_batch_size': self.training_batch_size,
            'training_epochs': self.training_epochs,
            'regime_stability_threshold': self.regime_stability_threshold,
            'regime_confidence_threshold': self.regime_confidence_threshold,
            'regime_update_frequency': self.regime_update_frequency,
            'performance_window': self.performance_window,
            'prediction_horizon': self.prediction_horizon,
            'prediction_confidence_threshold': self.prediction_confidence_threshold,
            'max_weight': self.max_weight,
            'min_diversification': self.min_diversification,
            'weight_smoothing_factor': self.weight_smoothing_factor,
            'meta_training_frequency': self.meta_training_frequency,
            'performance_evaluation_window': self.performance_evaluation_window,
            'adaptation_learning_rate': self.adaptation_learning_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create from dictionary"""
        return cls(**config_dict)

class MetaLearningOptimizer:
    """Hyperparameter optimizer for meta-learning components"""
    
    def __init__(self, base_agents_path: str, optimization_budget: int = 50):
        self.base_agents_path = base_agents_path
        self.optimization_budget = optimization_budget
        self.base_agents = {}
        self.optimization_history = []
        self.best_config = None
        self.best_score = -np.inf
        
        # Load base agents
        self._load_base_agents()
        
        # Define search spaces
        self._define_search_spaces()
    
    def _load_base_agents(self):
        """Load pre-trained base agents"""
        agent_classes = {
            'AgentD3QN': AgentD3QN,
            'AgentDoubleDQN': AgentDoubleDQN,
            'AgentTwinD3QN': AgentTwinD3QN
        }
        
        ensemble_models_path = os.path.join(self.base_agents_path, 'ensemble_models')
        
        if not os.path.exists(ensemble_models_path):
            raise FileNotFoundError(f"Base agents not found at: {ensemble_models_path}")
        
        # Get state dimension
        temp_sim = TradeSimulator(num_sims=1)
        state_dim = temp_sim.state_dim
        action_dim = 3
        
        for agent_name in os.listdir(ensemble_models_path):
            agent_path = os.path.join(ensemble_models_path, agent_name)
            
            if os.path.isdir(agent_path) and agent_name in agent_classes:
                try:
                    agent_class = agent_classes[agent_name]
                    agent = agent_class(
                        net_dims=(128, 64, 32),
                        state_dim=state_dim,
                        action_dim=action_dim,
                        gpu_id=0
                    )
                    
                    agent.save_or_load_agent(agent_path, if_save=False)
                    self.base_agents[agent_name] = agent
                    
                    print(f"✅ Loaded base agent: {agent_name}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {agent_name}: {e}")
        
        print(f"📊 Loaded {len(self.base_agents)} base agents for optimization")
    
    def _define_search_spaces(self):
        """Define hyperparameter search spaces"""
        self.search_spaces = {
            'meta_lookback': [250, 500, 750, 1000],
            'regime_features': [30, 40, 50, 60],
            'training_batch_size': [16, 32, 64, 128],
            'training_epochs': [5, 10, 15, 20],
            'regime_stability_threshold': [0.5, 0.6, 0.7, 0.8],
            'regime_confidence_threshold': [0.4, 0.5, 0.6, 0.7],
            'performance_window': [30, 50, 70, 100],
            'max_weight': [0.4, 0.5, 0.6, 0.7],
            'min_diversification': [2, 3, 4],
            'weight_smoothing_factor': [0.1, 0.2, 0.3, 0.4],
            'meta_training_frequency': [50, 100, 150, 200],
            'adaptation_learning_rate': [0.0001, 0.0005, 0.001, 0.002]
        }
        
        print(f"🔧 Defined search spaces for {len(self.search_spaces)} hyperparameters")
    
    def optimize_hyperparameters(self, method='random_search') -> HyperparameterConfig:
        """Optimize meta-learning hyperparameters"""
        
        print(f"🚀 Starting hyperparameter optimization")
        print(f"   Method: {method}")
        print(f"   Budget: {self.optimization_budget} evaluations")
        
        start_time = time.time()
        
        if method == 'random_search':
            best_config = self._random_search()
        elif method == 'grid_search':
            best_config = self._grid_search()
        elif method == 'bayesian_optimization':
            best_config = self._bayesian_optimization()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimization_time = time.time() - start_time
        
        print(f"🎉 Optimization completed in {optimization_time:.1f}s")
        print(f"   Best score: {self.best_score:.4f}")
        
        return best_config
    
    def _random_search(self) -> HyperparameterConfig:
        """Random search optimization"""
        
        print(f"🎲 Random search optimization")
        
        for iteration in range(self.optimization_budget):
            # Sample random configuration
            config = self._sample_random_config()
            
            # Evaluate configuration
            score = self._evaluate_config(config, iteration)
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"   🌟 New best config at iteration {iteration}: score={score:.4f}")
            
            # Progress update
            if (iteration + 1) % 10 == 0:
                print(f"   Progress: {iteration + 1}/{self.optimization_budget} "
                      f"(best={self.best_score:.4f})")
        
        return self.best_config
    
    def _grid_search(self) -> HyperparameterConfig:
        """Grid search optimization (limited scope)"""
        
        print(f"🔍 Grid search optimization (limited)")
        
        # Define limited grid for feasible computation
        limited_spaces = {
            'meta_lookback': [500, 750],
            'regime_stability_threshold': [0.6, 0.7],
            'max_weight': [0.5, 0.6],
            'weight_smoothing_factor': [0.2, 0.3],
            'adaptation_learning_rate': [0.001, 0.002]
        }
        
        # Generate all combinations
        keys = list(limited_spaces.keys())
        values = list(limited_spaces.values())
        combinations = list(itertools.product(*values))
        
        # Limit to budget
        combinations = combinations[:self.optimization_budget]
        
        print(f"   Evaluating {len(combinations)} configurations")
        
        for iteration, combination in enumerate(combinations):
            # Create configuration
            config_dict = dict(zip(keys, combination))
            config = HyperparameterConfig()
            for key, value in config_dict.items():
                setattr(config, key, value)
            
            # Evaluate configuration
            score = self._evaluate_config(config, iteration)
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"   🌟 New best config at iteration {iteration}: score={score:.4f}")
        
        return self.best_config
    
    def _bayesian_optimization(self) -> HyperparameterConfig:
        """Bayesian optimization (simplified implementation)"""
        
        print(f"🧠 Bayesian optimization (simplified)")
        
        # Initialize with random samples
        initial_samples = min(10, self.optimization_budget // 2)
        
        for iteration in range(initial_samples):
            config = self._sample_random_config()
            score = self._evaluate_config(config, iteration)
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
        
        # Continue with informed sampling
        for iteration in range(initial_samples, self.optimization_budget):
            # Sample based on previous results (simplified heuristic)
            config = self._sample_informed_config()
            score = self._evaluate_config(config, iteration)
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"   🌟 New best config at iteration {iteration}: score={score:.4f}")
        
        return self.best_config
    
    def _sample_random_config(self) -> HyperparameterConfig:
        """Sample random configuration"""
        config = HyperparameterConfig()
        
        for param, values in self.search_spaces.items():
            if hasattr(config, param):
                setattr(config, param, np.random.choice(values))
        
        return config
    
    def _sample_informed_config(self) -> HyperparameterConfig:
        """Sample configuration based on optimization history"""
        # Simplified: focus on better-performing regions
        
        if not self.optimization_history:
            return self._sample_random_config()
        
        # Get top 25% configurations
        sorted_history = sorted(self.optimization_history, key=lambda x: x['score'], reverse=True)
        top_configs = sorted_history[:max(1, len(sorted_history) // 4)]
        
        # Sample around top configurations with some randomness
        base_config = np.random.choice(top_configs)['config']
        new_config = HyperparameterConfig()
        
        for param, values in self.search_spaces.items():
            if hasattr(new_config, param) and hasattr(base_config, param):
                base_value = getattr(base_config, param)
                
                # 70% chance to stay close to base value, 30% random
                if np.random.random() < 0.7 and base_value in values:
                    # Stay close to base value
                    base_idx = values.index(base_value)
                    nearby_indices = [max(0, base_idx-1), base_idx, min(len(values)-1, base_idx+1)]
                    selected_idx = np.random.choice(nearby_indices)
                    setattr(new_config, param, values[selected_idx])
                else:
                    # Random selection
                    setattr(new_config, param, np.random.choice(values))
        
        return new_config
    
    def _evaluate_config(self, config: HyperparameterConfig, iteration: int) -> float:
        """Evaluate a hyperparameter configuration"""
        
        try:
            print(f"   Evaluating config {iteration + 1}/{self.optimization_budget}")
            
            # Create meta-learning manager with this configuration
            meta_manager = self._create_meta_manager(config)
            
            # Run short evaluation
            score = self._run_meta_learning_evaluation(meta_manager, config)
            
            # Store in history
            self.optimization_history.append({
                'iteration': iteration,
                'config': config,
                'score': score,
                'timestamp': time.time()
            })
            
            print(f"     Score: {score:.4f}")
            return score
            
        except Exception as e:
            print(f"     ⚠️  Evaluation failed: {e}")
            return -1.0
    
    def _create_meta_manager(self, config: HyperparameterConfig):
        """Create meta-learning manager with given configuration"""
        
        # Convert base agents to list
        agent_list = list(self.base_agents.values())
        
        # Create meta manager with configuration
        meta_manager = MetaLearningEnsembleManager(
            agents=agent_list,
            meta_lookback=config.meta_lookback,
            regime_features=config.regime_features,
            training_batch_size=config.training_batch_size,
            training_epochs=config.training_epochs
        )
        
        # Configure additional parameters
        meta_manager.regime_stability_threshold = config.regime_stability_threshold
        meta_manager.regime_confidence_threshold = config.regime_confidence_threshold
        meta_manager.performance_window = config.performance_window
        meta_manager.max_weight = config.max_weight
        meta_manager.min_diversification = config.min_diversification
        meta_manager.weight_smoothing_factor = config.weight_smoothing_factor
        meta_manager.meta_training_frequency = config.meta_training_frequency
        meta_manager.adaptation_learning_rate = config.adaptation_learning_rate
        
        return meta_manager
    
    def _run_meta_learning_evaluation(self, meta_manager, config: HyperparameterConfig) -> float:
        """Run meta-learning evaluation and return performance score"""
        
        # Create evaluation environment
        eval_env = EvalTradeSimulator(num_sims=1)
        state = eval_env.reset()
        
        total_reward = 0.0
        regime_detection_accuracy = 0.0
        adaptation_speed = 0.0
        steps = 0
        max_steps = 200  # Limited evaluation for speed
        
        regime_changes = 0
        correct_regime_predictions = 0
        
        try:
            for step in range(max_steps):
                # Get current market features (simplified)
                market_features = self._extract_market_features(state)
                
                # Detect regime
                predicted_regime = meta_manager.detect_market_regime(market_features)
                
                # Get algorithm weights
                weights = meta_manager.get_algorithm_weights()
                
                # Generate ensemble action (simplified)
                ensemble_action = self._get_ensemble_action(weights, state)
                
                # Execute action
                next_state, reward, done, _ = eval_env.step([ensemble_action])
                
                total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                steps += 1
                
                # Update meta-learning system (simplified)
                performance_metrics = {'reward': reward[0] if isinstance(reward, (list, np.ndarray)) else reward}
                meta_manager.update_performance(performance_metrics)
                
                state = next_state
                
                if done[0] if isinstance(done, (list, np.ndarray)) else done:
                    break
            
            # Calculate performance score
            avg_reward = total_reward / max(steps, 1)
            
            # Additional scoring based on meta-learning effectiveness
            regime_score = meta_manager.get_regime_detection_accuracy() if hasattr(meta_manager, 'get_regime_detection_accuracy') else 0.5
            adaptation_score = meta_manager.get_adaptation_effectiveness() if hasattr(meta_manager, 'get_adaptation_effectiveness') else 0.5
            
            # Combined score
            final_score = 0.6 * avg_reward + 0.2 * regime_score + 0.2 * adaptation_score
            
            return final_score
            
        except Exception as e:
            print(f"       Evaluation error: {e}")
            return -1.0
    
    def _extract_market_features(self, state) -> np.ndarray:
        """Extract market features for regime detection (simplified)"""
        
        # Convert state to numpy if needed
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy().flatten()
        elif isinstance(state, (list, np.ndarray)):
            state_np = np.array(state).flatten()
        else:
            state_np = np.array([state]).flatten()
        
        # Create expanded feature vector (pad or truncate to 50 features)
        target_features = 50
        if len(state_np) >= target_features:
            features = state_np[:target_features]
        else:
            features = np.pad(state_np, (0, target_features - len(state_np)), 'constant')
        
        # Add some derived features (simplified)
        features = np.concatenate([features, [
            np.std(features),  # Volatility proxy
            np.mean(features), # Level proxy
            np.max(features) - np.min(features),  # Range proxy
        ]])
        
        return features[:target_features]  # Ensure exact size
    
    def _get_ensemble_action(self, weights: Dict, state) -> int:
        """Get ensemble action based on agent weights (simplified)"""
        
        # Get actions from all agents
        agent_actions = {}
        for agent_name, agent in self.base_agents.items():
            try:
                if isinstance(state, np.ndarray):
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                else:
                    state_tensor = state
                
                with torch.no_grad():
                    action = agent.select_action(state_tensor)
                    agent_actions[agent_name] = action[0] if isinstance(action, (list, np.ndarray)) else action
            except:
                agent_actions[agent_name] = 0  # Default action
        
        # Weighted voting
        action_votes = {}
        for agent_name, action in agent_actions.items():
            weight = weights.get(agent_name, 1.0 / len(agent_actions))
            action_votes[action] = action_votes.get(action, 0) + weight
        
        # Return action with highest weighted vote
        return max(action_votes.items(), key=lambda x: x[1])[0] if action_votes else 0
    
    def save_optimization_results(self, output_path: str):
        """Save optimization results"""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save best configuration
        if self.best_config:
            config_path = os.path.join(output_path, 'best_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.best_config.to_dict(), f, indent=2)
        
        # Save optimization history
        history_path = os.path.join(output_path, 'optimization_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.optimization_history, f)
        
        # Save summary report
        self._create_optimization_report(output_path)
        
        print(f"📊 Optimization results saved to: {output_path}")
    
    def _create_optimization_report(self, output_path: str):
        """Create optimization summary report"""
        
        report_path = os.path.join(output_path, 'optimization_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Meta-Learning Hyperparameter Optimization Report\n\n")
            
            # Summary
            f.write("## Optimization Summary\n\n")
            f.write(f"- **Evaluations**: {len(self.optimization_history)}\n")
            f.write(f"- **Best Score**: {self.best_score:.4f}\n")
            if self.optimization_history:
                scores = [h['score'] for h in self.optimization_history]
                f.write(f"- **Mean Score**: {np.mean(scores):.4f}\n")
                f.write(f"- **Score Std**: {np.std(scores):.4f}\n")
            f.write("\n")
            
            # Best configuration
            if self.best_config:
                f.write("## Best Configuration\n\n")
                config_dict = self.best_config.to_dict()
                for param, value in config_dict.items():
                    f.write(f"- **{param}**: {value}\n")
                f.write("\n")
            
            # Top configurations
            if len(self.optimization_history) >= 5:
                f.write("## Top 5 Configurations\n\n")
                sorted_history = sorted(self.optimization_history, key=lambda x: x['score'], reverse=True)
                
                for i, entry in enumerate(sorted_history[:5]):
                    f.write(f"### Rank {i+1} (Score: {entry['score']:.4f})\n")
                    config_dict = entry['config'].to_dict()
                    for param, value in config_dict.items():
                        f.write(f"- {param}: {value}\n")
                    f.write("\n")

def run_meta_learning_optimization(base_agents_path, output_path="meta_learning_optimization", 
                                 method='random_search', budget=50):
    """Run meta-learning hyperparameter optimization"""
    
    print(f"🚀 Meta-Learning Hyperparameter Optimization")
    print(f"📂 Base Agents: {base_agents_path}")
    print(f"📂 Output: {output_path}")
    print(f"🔧 Method: {method}")
    print(f"💰 Budget: {budget} evaluations")
    
    try:
        # Initialize optimizer
        optimizer = MetaLearningOptimizer(base_agents_path, budget)
        
        # Run optimization
        best_config = optimizer.optimize_hyperparameters(method)
        
        # Save results
        optimizer.save_optimization_results(output_path)
        
        print(f"🎉 Optimization completed successfully!")
        print(f"📊 Best score: {optimizer.best_score:.4f}")
        
        return best_config
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    # Default parameters
    base_agents_path = "ensemble_optimized_phase2"
    output_path = "meta_learning_optimization"
    method = 'random_search'
    budget = 50
    
    if len(sys.argv) > 1:
        base_agents_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        method = sys.argv[3]
    if len(sys.argv) > 4:
        budget = int(sys.argv[4])
    
    best_config = run_meta_learning_optimization(base_agents_path, output_path, method, budget)