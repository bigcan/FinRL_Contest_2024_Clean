import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class MarketRegimeClassifier(nn.Module):
    """
    Neural network to classify market regimes based on multiple features
    """
    
    def __init__(self, input_dim: int = 50, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.input_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        # Output layer for 7 market regimes
        layers.append(nn.Linear(prev_dim, 7))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
        
        # Regime labels
        self.regime_labels = [
            'trending_bull', 'trending_bear', 'high_vol_range', 
            'low_vol_range', 'breakout', 'reversal', 'crisis'
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.network.eval()  # Set to evaluation mode to handle single samples
        return self.network(x)
    
    def predict_regime(self, features: torch.Tensor) -> str:
        """Predict single regime from features"""
        self.network.eval()
        with torch.no_grad():
            probs = self.forward(features.unsqueeze(0))
            regime_idx = torch.argmax(probs, dim=1).item()
            return self.regime_labels[regime_idx]


class AlgorithmPerformancePredictor(nn.Module):
    """
    Predicts algorithm performance based on market conditions and historical data
    """
    
    def __init__(self, market_features_dim: int = 50, 
                 agent_history_dim: int = 20,
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        input_dim = market_features_dim + agent_history_dim
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        # Output: predicted Sharpe ratio (can be negative)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, market_features: torch.Tensor, 
                agent_history: torch.Tensor) -> torch.Tensor:
        self.network.eval()  # Set to evaluation mode to handle single samples
        combined_input = torch.cat([market_features, agent_history], dim=-1)
        return self.network(combined_input)


class MarketFeatureExtractor:
    """
    Extracts comprehensive market features for regime classification
    """
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.price_history = deque(maxlen=lookback_window)
        self.volume_history = deque(maxlen=lookback_window)
        self.feature_scaler = StandardScaler()
        self.fitted = False
        
    def update_data(self, price: float, volume: float):
        """Update with new market data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def extract_features(self) -> torch.Tensor:
        """Extract comprehensive market features"""
        if len(self.price_history) < 20:
            return torch.zeros(50)
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        returns = np.diff(np.log(prices))
        
        features = []
        
        # Price-based features
        features.extend(self._price_features(prices, returns))
        
        # Volatility features
        features.extend(self._volatility_features(returns))
        
        # Momentum features
        features.extend(self._momentum_features(prices, returns))
        
        # Volume features
        features.extend(self._volume_features(volumes, returns))
        
        # Microstructure features (simplified)
        features.extend(self._microstructure_features(prices, volumes))
        
        feature_array = np.array(features)
        
        # Normalize features
        if not self.fitted and len(feature_array) > 0:
            try:
                self.feature_scaler.fit(feature_array.reshape(1, -1))
                self.fitted = True
            except:
                pass
        
        if self.fitted:
            try:
                feature_array = self.feature_scaler.transform(feature_array.reshape(1, -1)).flatten()
            except:
                pass
        
        # Ensure exactly 50 features
        if len(feature_array) < 50:
            feature_array = np.pad(feature_array, (0, 50 - len(feature_array)))
        elif len(feature_array) > 50:
            feature_array = feature_array[:50]
            
        return torch.tensor(feature_array, dtype=torch.float32)
    
    def _price_features(self, prices: np.ndarray, returns: np.ndarray) -> List[float]:
        """Extract price-based features"""
        features = []
        
        # Moving averages
        for window in [5, 10, 20]:
            if len(prices) >= window:
                ma = np.mean(prices[-window:])
                features.append((prices[-1] - ma) / ma)  # Price relative to MA
            else:
                features.append(0.0)
        
        # Price momentum
        for window in [5, 10, 20]:
            if len(prices) >= window + 1:
                momentum = (prices[-1] - prices[-window-1]) / prices[-window-1]
                features.append(momentum)
            else:
                features.append(0.0)
        
        return features
    
    def _volatility_features(self, returns: np.ndarray) -> List[float]:
        """Extract volatility features"""
        features = []
        
        if len(returns) > 5:
            # Rolling volatility
            for window in [5, 10, 20]:
                if len(returns) >= window:
                    vol = np.std(returns[-window:])
                    features.append(vol)
                else:
                    features.append(0.0)
            
            # Volatility of volatility
            if len(returns) >= 20:
                vol_series = [np.std(returns[i:i+5]) for i in range(len(returns)-5)]
                features.append(np.std(vol_series))
            else:
                features.append(0.0)
                
        else:
            features.extend([0.0] * 4)
        
        return features
    
    def _momentum_features(self, prices: np.ndarray, returns: np.ndarray) -> List[float]:
        """Extract momentum features"""
        features = []
        
        if len(returns) > 10:
            # RSI approximation
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            
            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    features.append(rsi / 100.0)  # Normalize to [0,1]
                else:
                    features.append(1.0)
            else:
                features.append(0.5)
            
            # MACD approximation
            if len(prices) >= 26:
                ema12 = np.mean(prices[-12:])
                ema26 = np.mean(prices[-26:])
                macd = (ema12 - ema26) / ema26
                features.append(macd)
            else:
                features.append(0.0)
                
        else:
            features.extend([0.5, 0.0])
        
        return features
    
    def _volume_features(self, volumes: np.ndarray, returns: np.ndarray) -> List[float]:
        """Extract volume features"""
        features = []
        
        if len(volumes) > 5:
            # Volume trend
            recent_vol = np.mean(volumes[-5:])
            older_vol = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else recent_vol
            if older_vol > 0:
                vol_trend = (recent_vol - older_vol) / older_vol
                features.append(vol_trend)
            else:
                features.append(0.0)
            
            # Volume-price correlation
            if len(returns) >= len(volumes) - 1:
                vol_returns_corr = np.corrcoef(volumes[1:], returns[:len(volumes)-1])[0, 1]
                if np.isnan(vol_returns_corr):
                    vol_returns_corr = 0.0
                features.append(vol_returns_corr)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _microstructure_features(self, prices: np.ndarray, volumes: np.ndarray) -> List[float]:
        """Extract microstructure features (simplified)"""
        features = []
        
        if len(prices) > 10:
            # Price impact approximation
            price_changes = np.diff(prices)
            volume_changes = np.diff(volumes) if len(volumes) > 1 else np.array([0])
            
            if len(volume_changes) > 0 and len(price_changes) > 0:
                # Correlation between price change and volume
                if len(price_changes) == len(volume_changes):
                    impact_corr = np.corrcoef(price_changes, volume_changes)[0, 1]
                    if np.isnan(impact_corr):
                        impact_corr = 0.0
                    features.append(impact_corr)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
                
            # Liquidity approximation (inverse of price volatility)
            if len(price_changes) > 0:
                liquidity = 1.0 / (1.0 + np.std(price_changes))
                features.append(liquidity)
            else:
                features.append(0.5)
        else:
            features.extend([0.0, 0.5])
        
        return features


class MetaLearningEnsembleManager:
    """
    Main meta-learning framework for adaptive algorithm selection
    """
    
    def __init__(self, agents: Dict, meta_lookback: int = 500):
        self.agents = agents
        self.agent_names = list(agents.keys())
        self.meta_lookback = meta_lookback
        
        # Core components
        self.market_feature_extractor = MarketFeatureExtractor()
        self.market_regime_classifier = MarketRegimeClassifier()
        self.performance_predictors = {}
        
        # Initialize performance predictors for each agent
        for agent_name in self.agent_names:
            self.performance_predictors[agent_name] = AlgorithmPerformancePredictor()
        
        # Historical data storage
        self.market_state_history = deque(maxlen=meta_lookback)
        self.agent_performance_history = defaultdict(lambda: deque(maxlen=meta_lookback))
        self.regime_history = deque(maxlen=meta_lookback)
        
        # Training data
        self.training_data = {
            'market_features': [],
            'agent_histories': defaultdict(list),
            'performance_labels': defaultdict(list)
        }
        
        # Current state
        self.current_regime = 'low_vol_range'
        self.training_step = 0
        
    def update_market_data(self, price: float, volume: float):
        """Update with new market data"""
        self.market_feature_extractor.update_data(price, volume)
        
        # Extract current features and detect regime
        current_features = self.market_feature_extractor.extract_features()
        self.market_state_history.append(current_features)
        
        # Update current regime
        self.current_regime = self.market_regime_classifier.predict_regime(current_features)
        self.regime_history.append(self.current_regime)
    
    def update_agent_performance(self, agent_name: str, performance_metrics: Dict):
        """Update agent performance history"""
        if agent_name in self.agent_names:
            self.agent_performance_history[agent_name].append(performance_metrics)
    
    def get_agent_history_features(self, agent_name: str) -> torch.Tensor:
        """Extract recent performance features for an agent"""
        if agent_name not in self.agent_performance_history:
            return torch.zeros(20)
        
        history = list(self.agent_performance_history[agent_name])
        if len(history) == 0:
            return torch.zeros(20)
        
        # Extract last 20 performance metrics
        features = []
        lookback = min(20, len(history))
        
        for i in range(lookback):
            perf = history[-(i+1)]
            features.extend([
                perf.get('sharpe_ratio', 0.0),
                perf.get('win_rate', 0.5),
                perf.get('avg_return', 0.0),
                perf.get('volatility', 0.1)
            ])
        
        # Pad if necessary
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features[:20], dtype=torch.float32)
    
    def predict_agent_performance(self, agent_name: str) -> float:
        """Predict agent performance for current market conditions"""
        if len(self.market_state_history) == 0:
            return 0.0
        
        current_market_features = self.market_state_history[-1]
        agent_history_features = self.get_agent_history_features(agent_name)
        
        predictor = self.performance_predictors[agent_name]
        
        with torch.no_grad():
            predicted_sharpe = predictor(
                current_market_features.unsqueeze(0),
                agent_history_features.unsqueeze(0)
            ).item()
        
        return predicted_sharpe
    
    def get_adaptive_algorithm_weights(self, risk_constraints: Optional[Dict] = None) -> Dict[str, float]:
        """Get optimal algorithm weights based on meta-learning predictions"""
        
        # Get performance predictions for all agents
        predictions = {}
        for agent_name in self.agent_names:
            predictions[agent_name] = self.predict_agent_performance(agent_name)
        
        # Convert to softmax weights (favor better predicted performance)
        pred_values = list(predictions.values())
        if len(pred_values) == 0:
            return {name: 1.0/len(self.agent_names) for name in self.agent_names}
        
        # Apply temperature scaling to sharpen/smooth distribution
        temperature = 0.5
        exp_values = np.exp(np.array(pred_values) / temperature)
        weights = exp_values / np.sum(exp_values)
        
        # Apply risk constraints if provided
        if risk_constraints:
            weights = self._apply_risk_constraints(weights, risk_constraints)
        
        # Ensure minimum diversification (no single agent > 70%)
        max_weight = 0.7
        if np.max(weights) > max_weight:
            weights = np.clip(weights, 0, max_weight)
            weights = weights / np.sum(weights)
        
        return {name: float(weight) for name, weight in zip(self.agent_names, weights)}
    
    def _apply_risk_constraints(self, weights: np.ndarray, constraints: Dict) -> np.ndarray:
        """Apply risk constraints to algorithm weights"""
        
        # Max weight per algorithm
        if 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            weights = np.clip(weights, 0, max_weight)
            weights = weights / np.sum(weights)
        
        # Minimum diversification
        if 'min_diversification' in constraints:
            min_agents = constraints['min_diversification']
            top_weights = np.sort(weights)[-min_agents:]
            min_total = 0.5  # Top N agents should have at least 50% weight
            if np.sum(top_weights) < min_total:
                # Boost top N agents
                boost_factor = min_total / np.sum(top_weights)
                for i in range(len(weights)):
                    if weights[i] >= top_weights[0]:
                        weights[i] *= boost_factor
                weights = weights / np.sum(weights)
        
        return weights
    
    def collect_training_data(self):
        """Collect data for training meta-learning models"""
        if len(self.market_state_history) < 2:
            return
        
        # Use previous market state and current performance as training pair
        prev_market_state = self.market_state_history[-2]
        
        for agent_name in self.agent_names:
            if len(self.agent_performance_history[agent_name]) > 0:
                current_perf = self.agent_performance_history[agent_name][-1]
                prev_agent_history = self.get_agent_history_features(agent_name)
                
                # Store training data
                self.training_data['market_features'].append(prev_market_state.numpy())
                self.training_data['agent_histories'][agent_name].append(prev_agent_history.numpy())
                self.training_data['performance_labels'][agent_name].append(
                    current_perf.get('sharpe_ratio', 0.0)
                )
    
    def train_meta_models(self, batch_size: int = 32, epochs: int = 10):
        """Train meta-learning models on collected data"""
        
        if len(self.training_data['market_features']) < batch_size:
            return
        
        print(f"Training meta-learning models with {len(self.training_data['market_features'])} samples...")
        
        # Prepare training data
        market_features = torch.tensor(np.array(self.training_data['market_features']), dtype=torch.float32)
        
        # Train each agent's performance predictor
        for agent_name in self.agent_names:
            if len(self.training_data['performance_labels'][agent_name]) < batch_size:
                continue
                
            predictor = self.performance_predictors[agent_name]
            optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            agent_histories = torch.tensor(
                np.array(self.training_data['agent_histories'][agent_name]), 
                dtype=torch.float32
            )
            labels = torch.tensor(
                self.training_data['performance_labels'][agent_name], 
                dtype=torch.float32
            ).unsqueeze(1)
            
            # Training loop
            predictor.train()
            for epoch in range(epochs):
                # Shuffle data
                indices = torch.randperm(len(market_features))
                
                total_loss = 0
                num_batches = 0
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    if len(batch_indices) < 2:  # Skip small batches
                        continue
                    
                    # Ensure indices are within bounds
                    valid_indices = batch_indices[batch_indices < len(agent_histories)]
                    if len(valid_indices) < 2:
                        continue
                    
                    batch_market = market_features[valid_indices]
                    batch_history = agent_histories[valid_indices]
                    batch_labels = labels[valid_indices]
                    
                    optimizer.zero_grad()
                    predictions = predictor(batch_market, batch_history)
                    loss = criterion(predictions, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    if epoch % 5 == 0:
                        print(f"Agent {agent_name}, Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.training_step += 1
        print("Meta-learning model training completed.")
    
    def get_regime_info(self) -> Dict:
        """Get current market regime information"""
        return {
            'current_regime': self.current_regime,
            'regime_confidence': 1.0,  # Could be extended with actual confidence
            'regime_history': list(self.regime_history)[-10:],  # Last 10 regimes
            'regime_stability': self._calculate_regime_stability()
        }
    
    def _calculate_regime_stability(self) -> float:
        """Calculate how stable the current regime is"""
        if len(self.regime_history) < 10:
            return 0.5
        
        recent_regimes = list(self.regime_history)[-10:]
        regime_changes = sum(1 for i in range(1, len(recent_regimes)) 
                           if recent_regimes[i] != recent_regimes[i-1])
        
        stability = 1.0 - (regime_changes / 9.0)  # 9 possible changes in 10 samples
        return max(0.0, min(1.0, stability))
    
    def save_meta_models(self, save_path: str):
        """Save trained meta-learning models"""
        save_dict = {
            'regime_classifier': self.market_regime_classifier.state_dict(),
            'performance_predictors': {
                name: predictor.state_dict() 
                for name, predictor in self.performance_predictors.items()
            },
            'training_step': self.training_step,
            'current_regime': self.current_regime
        }
        
        torch.save(save_dict, f"{save_path}/meta_learning_models.pth")
        print(f"Meta-learning models saved to {save_path}")
    
    def load_meta_models(self, load_path: str):
        """Load trained meta-learning models"""
        try:
            save_dict = torch.load(f"{load_path}/meta_learning_models.pth")
            
            self.market_regime_classifier.load_state_dict(save_dict['regime_classifier'])
            
            for name, state_dict in save_dict['performance_predictors'].items():
                if name in self.performance_predictors:
                    self.performance_predictors[name].load_state_dict(state_dict)
            
            self.training_step = save_dict.get('training_step', 0)
            self.current_regime = save_dict.get('current_regime', 'low_vol_range')
            
            print(f"Meta-learning models loaded from {load_path}")
            
        except Exception as e:
            print(f"Failed to load meta-learning models: {e}")


class MetaLearningRiskManagedEnsemble:
    """
    Integration of meta-learning with existing risk management
    """
    
    def __init__(self, agents: Dict, meta_learning_manager: MetaLearningEnsembleManager,
                 risk_manager=None):
        self.agents = agents
        self.meta_learning_manager = meta_learning_manager
        self.risk_manager = risk_manager
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.decision_history = deque(maxlen=1000)
        
    def get_trading_action(self, state: torch.Tensor, current_price: float, 
                          current_volume: float = 1000.0) -> Tuple[int, Dict]:
        """
        Get trading action with meta-learning and risk management
        """
        
        # Update meta-learning with current market data
        self.meta_learning_manager.update_market_data(current_price, current_volume)
        
        # Get adaptive algorithm weights
        risk_constraints = {
            'max_weight': 0.6,
            'min_diversification': 3
        }
        
        algorithm_weights = self.meta_learning_manager.get_adaptive_algorithm_weights(
            risk_constraints
        )
        
        # Get individual agent actions
        agent_actions = {}
        agent_q_values = {}
        
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'act'):
                    action_result = agent.act(state)
                    if isinstance(action_result, tuple):
                        action, q_values = action_result
                    else:
                        action = action_result
                        q_values = None
                    
                    agent_actions[agent_name] = action
                    agent_q_values[agent_name] = q_values
                    
            except Exception as e:
                print(f"Error getting action from agent {agent_name}: {e}")
                agent_actions[agent_name] = 1  # Hold action as fallback
                agent_q_values[agent_name] = None
        
        # Weighted ensemble decision
        ensemble_action = self._compute_weighted_ensemble_action(
            agent_actions, algorithm_weights
        )
        
        # Apply risk management if available
        if self.risk_manager:
            ensemble_action = self._apply_risk_management(
                ensemble_action, current_price, algorithm_weights
            )
        
        # Prepare decision info
        decision_info = {
            'ensemble_action': ensemble_action,
            'algorithm_weights': algorithm_weights,
            'agent_actions': agent_actions,
            'current_regime': self.meta_learning_manager.current_regime,
            'regime_info': self.meta_learning_manager.get_regime_info()
        }
        
        # Store decision for analysis
        self.decision_history.append({
            'timestamp': time.time(),
            'state': state.numpy() if isinstance(state, torch.Tensor) else state,
            'price': current_price,
            'action': ensemble_action,
            'weights': algorithm_weights,
            'regime': self.meta_learning_manager.current_regime
        })
        
        return ensemble_action, decision_info
    
    def _compute_weighted_ensemble_action(self, agent_actions: Dict[str, int], 
                                        weights: Dict[str, float]) -> int:
        """Compute weighted ensemble action"""
        
        if not agent_actions:
            return 1  # Hold
        
        # Weighted voting
        action_votes = defaultdict(float)
        
        for agent_name, action in agent_actions.items():
            weight = weights.get(agent_name, 0.0)
            action_votes[action] += weight
        
        # Return action with highest weighted vote
        if action_votes:
            best_action = max(action_votes.items(), key=lambda x: x[1])[0]
            return best_action
        
        return 1  # Hold as fallback
    
    def _apply_risk_management(self, action: int, current_price: float, 
                             weights: Dict[str, float]) -> int:
        """Apply risk management constraints"""
        
        # Basic risk management - can be extended
        if self.risk_manager and hasattr(self.risk_manager, 'check_risk_constraints'):
            risk_check = self.risk_manager.check_risk_constraints(
                action, current_price, weights
            )
            if not risk_check['allowed']:
                return risk_check.get('alternative_action', 1)  # Force hold
        
        return action
    
    def update_performance(self, returns: float, sharpe_ratio: float, 
                          additional_metrics: Dict = None):
        """Update performance metrics for meta-learning"""
        
        perf_metrics = {
            'timestamp': time.time(),
            'returns': returns,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': additional_metrics.get('win_rate', 0.5) if additional_metrics else 0.5,
            'avg_return': additional_metrics.get('avg_return', returns) if additional_metrics else returns,
            'volatility': additional_metrics.get('volatility', 0.1) if additional_metrics else 0.1
        }
        
        self.performance_history.append(perf_metrics)
        
        # Update meta-learning manager with individual agent performance
        # This would typically be done with actual agent performance tracking
        for agent_name in self.agents.keys():
            self.meta_learning_manager.update_agent_performance(agent_name, perf_metrics)
        
        # Collect training data for meta-learning
        self.meta_learning_manager.collect_training_data()
        
        # Periodically train meta-models
        if len(self.performance_history) % 100 == 0:
            self.meta_learning_manager.train_meta_models(batch_size=16, epochs=5)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        
        if len(self.performance_history) == 0:
            return {}
        
        recent_performance = list(self.performance_history)[-50:]  # Last 50 trades
        
        summary = {
            'total_trades': len(self.performance_history),
            'recent_sharpe': np.mean([p['sharpe_ratio'] for p in recent_performance]),
            'recent_returns': np.mean([p['returns'] for p in recent_performance]),
            'regime_info': self.meta_learning_manager.get_regime_info(),
            'meta_learning_stats': {
                'training_steps': self.meta_learning_manager.training_step,
                'data_samples': len(self.meta_learning_manager.training_data['market_features'])
            }
        }
        
        return summary