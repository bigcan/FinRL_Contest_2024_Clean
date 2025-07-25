"""
Meta-Learning Agent Wrapper System
Provides unified interface between existing agents and meta-learning framework
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import time
from collections import deque

# Import existing agents
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN

try:
    from erl_agent_ppo import AgentPPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False

try:
    from erl_agent_rainbow import AgentRainbow
    RAINBOW_AVAILABLE = True
except ImportError:
    RAINBOW_AVAILABLE = False


class BaseAgentWrapper(ABC):
    """
    Abstract base class for agent wrappers
    Provides standardized interface for meta-learning integration
    """
    
    def __init__(self, agent, agent_name: str):
        self.agent = agent
        self.agent_name = agent_name
        self.agent_type = type(agent).__name__
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.decision_history = deque(maxlen=1000)
        self.training_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'last_update_time': 0,
            'average_update_time': 0
        }
        
        # Meta-learning compatibility attributes
        self.confidence_scores = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        self.action_distribution = {0: 0, 1: 0, 2: 0}  # sell, hold, buy counts
        
        print(f"🤖 Agent wrapper created for {self.agent_name} ({self.agent_type})")
    
    @abstractmethod
    def get_action_with_confidence(self, state: torch.Tensor) -> Tuple[int, float, Dict]:
        """
        Get action with confidence score and additional information
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action, confidence_score, additional_info)
        """
        pass
    
    @abstractmethod
    def update_with_feedback(self, buffer, learning_info: Optional[Dict] = None) -> Dict:
        """
        Update agent with experience buffer and optional learning info
        
        Args:
            buffer: Experience replay buffer
            learning_info: Additional learning information from meta-learning
            
        Returns:
            Dictionary with update statistics
        """
        pass
    
    def get_performance_metrics(self, window: int = 50) -> Dict[str, float]:
        """Get recent performance metrics for meta-learning"""
        if len(self.performance_history) < window:
            window = len(self.performance_history)
        
        if window == 0:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.5,
                'avg_return': 0.0,
                'volatility': 0.1,
                'confidence': 0.5,
                'activity_rate': 0.0
            }
        
        recent_performance = list(self.performance_history)[-window:]
        
        # Calculate basic metrics
        returns = [p.get('return', 0.0) for p in recent_performance]
        avg_return = np.mean(returns) if returns else 0.0
        volatility = np.std(returns) if len(returns) > 1 else 0.1
        sharpe_ratio = avg_return / (volatility + 1e-8)
        
        # Calculate win rate
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns) if returns else 0.5
        
        # Calculate average confidence
        confidences = [p.get('confidence', 0.5) for p in recent_performance]
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        # Calculate activity rate (non-hold actions)
        actions = [p.get('action', 1) for p in recent_performance]
        non_hold_actions = sum(1 for a in actions if a != 1)
        activity_rate = non_hold_actions / len(actions) if actions else 0.0
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'volatility': float(volatility),
            'confidence': float(avg_confidence),
            'activity_rate': float(activity_rate)
        }
    
    def update_performance(self, action: int, reward: float, confidence: float = 0.5):
        """Update performance history with latest action and reward"""
        performance_entry = {
            'timestamp': time.time(),
            'action': action,
            'return': reward,
            'confidence': confidence
        }
        
        self.performance_history.append(performance_entry)
        self.decision_history.append(performance_entry)
        
        # Update action distribution
        self.action_distribution[action] = self.action_distribution.get(action, 0) + 1
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        total_decisions = len(self.decision_history)
        
        stats = {
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'total_decisions': total_decisions,
            'training_stats': self.training_stats.copy(),
            'performance_metrics': self.get_performance_metrics(),
            'action_distribution': self.action_distribution.copy()
        }
        
        if total_decisions > 0:
            # Calculate action percentages
            total_actions = sum(self.action_distribution.values())
            if total_actions > 0:
                stats['action_percentages'] = {
                    'sell': self.action_distribution[0] / total_actions,
                    'hold': self.action_distribution[1] / total_actions,
                    'buy': self.action_distribution[2] / total_actions
                }
        
        return stats


class DQNAgentWrapper(BaseAgentWrapper):
    """
    Wrapper for DQN-based agents (D3QN, DoubleDQN, TwinD3QN)
    """
    
    def __init__(self, agent, agent_name: str):
        super().__init__(agent, agent_name)
        self.q_value_threshold = 0.1  # Threshold for confidence calculation
    
    def get_action_with_confidence(self, state: torch.Tensor) -> Tuple[int, float, Dict]:
        """Get action with confidence based on Q-value distribution"""
        try:
            # Get Q-values from agent
            q_values = self.agent.act(state)
            
            if isinstance(q_values, tuple):
                action, q_vals = q_values
            else:
                # If agent returns only action, try to get Q-values separately
                action = q_values
                q_vals = None
                
                # Try to get Q-values from the agent's network
                if hasattr(self.agent, 'act_target'):
                    with torch.no_grad():
                        q_vals = self.agent.act_target(state.unsqueeze(0)).squeeze(0)
                elif hasattr(self.agent, 'act'):
                    # Fallback: assume uniform confidence
                    q_vals = torch.ones(3) * 0.33
            
            # Calculate confidence from Q-value distribution
            if q_vals is not None:
                q_vals_np = q_vals.detach().cpu().numpy() if isinstance(q_vals, torch.Tensor) else np.array(q_vals)
                
                # Softmax to get probabilities
                exp_q = np.exp(q_vals_np - np.max(q_vals_np))
                probs = exp_q / np.sum(exp_q)
                
                # Confidence is the maximum probability
                confidence = float(np.max(probs))
                
                # Alternative confidence: entropy-based (lower entropy = higher confidence)
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                confidence_entropy = 1.0 - (entropy / np.log(len(probs)))
                
                # Use maximum of both measures
                final_confidence = max(confidence, confidence_entropy)
            else:
                final_confidence = 0.5  # Default confidence
            
            # Store Q-values for analysis
            self.q_value_history.append(q_vals_np if q_vals is not None else [0.33, 0.33, 0.34])
            self.confidence_scores.append(final_confidence)
            
            additional_info = {
                'q_values': q_vals_np if q_vals is not None else None,
                'action_probs': probs if q_vals is not None else [0.33, 0.33, 0.34],
                'entropy': entropy if q_vals is not None else np.log(3)
            }
            
            return int(action), final_confidence, additional_info
            
        except Exception as e:
            print(f"⚠️ Error in DQN wrapper action: {e}")
            return 1, 0.5, {'error': str(e)}  # Return hold action with low confidence
    
    def update_with_feedback(self, buffer, learning_info: Optional[Dict] = None) -> Dict:
        """Update DQN agent with experience buffer"""
        start_time = time.time()
        
        try:
            # Standard DQN update
            if hasattr(self.agent, 'update_net'):
                obj_critic, obj_actor = self.agent.update_net(buffer)
            else:
                obj_critic, obj_actor = 0.0, 0.0
            
            # Apply meta-learning feedback if available
            if learning_info and 'learning_rate_adjustment' in learning_info:
                lr_factor = learning_info['learning_rate_adjustment']
                self._adjust_learning_rate(lr_factor)
            
            # Update training statistics
            update_time = time.time() - start_time
            self.training_stats['total_updates'] += 1
            self.training_stats['successful_updates'] += 1
            self.training_stats['last_update_time'] = update_time
            self.training_stats['average_update_time'] = (
                (self.training_stats['average_update_time'] * (self.training_stats['total_updates'] - 1) + update_time) /
                self.training_stats['total_updates']
            )
            
            return {
                'obj_critic': float(obj_critic),
                'obj_actor': float(obj_actor),
                'update_time': update_time,
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️ Error in DQN wrapper update: {e}")
            self.training_stats['total_updates'] += 1
            return {
                'obj_critic': 0.0,
                'obj_actor': 0.0,
                'update_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _adjust_learning_rate(self, factor: float):
        """Adjust learning rate based on meta-learning feedback"""
        if hasattr(self.agent, 'act_optimizer'):
            for param_group in self.agent.act_optimizer.param_groups:
                param_group['lr'] *= factor
        
        if hasattr(self.agent, 'cri_optimizer'):
            for param_group in self.agent.cri_optimizer.param_groups:
                param_group['lr'] *= factor


class PPOAgentWrapper(BaseAgentWrapper):
    """
    Wrapper for PPO agent
    """
    
    def __init__(self, agent, agent_name: str):
        super().__init__(agent, agent_name)
        self.entropy_threshold = 0.5  # Threshold for confidence calculation
    
    def get_action_with_confidence(self, state: torch.Tensor) -> Tuple[int, float, Dict]:
        """Get action with confidence based on action probability distribution"""
        try:
            # Get action and probabilities from PPO agent
            if hasattr(self.agent, 'act'):
                result = self.agent.act(state)
                
                if isinstance(result, tuple):
                    action, action_probs = result
                else:
                    action = result
                    action_probs = None
            else:
                action = 1  # Default hold
                action_probs = None
            
            # Calculate confidence from action probabilities  
            if action_probs is not None:
                if isinstance(action_probs, torch.Tensor):
                    probs = action_probs.detach().cpu().numpy()
                else:
                    probs = np.array(action_probs)
                
                # Confidence is the maximum probability
                confidence = float(np.max(probs))
                
                # Calculate entropy for additional confidence measure
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                confidence_entropy = 1.0 - (entropy / np.log(len(probs)))
                
                final_confidence = max(confidence, confidence_entropy)
            else:
                probs = [0.33, 0.33, 0.34]
                entropy = np.log(3)
                final_confidence = 0.5
            
            self.confidence_scores.append(final_confidence)
            
            additional_info = {
                'action_probs': probs,
                'entropy': entropy,
                'policy_type': 'stochastic'
            }
            
            return int(action), final_confidence, additional_info
            
        except Exception as e:
            print(f"⚠️ Error in PPO wrapper action: {e}")
            return 1, 0.5, {'error': str(e)}
    
    def update_with_feedback(self, buffer, learning_info: Optional[Dict] = None) -> Dict:
        """Update PPO agent with experience buffer"""
        start_time = time.time()
        
        try:
            # PPO-specific update
            if hasattr(self.agent, 'update_net'):
                obj_critic, obj_actor = self.agent.update_net(buffer)
            else:
                obj_critic, obj_actor = 0.0, 0.0
            
            # Apply meta-learning feedback
            if learning_info:
                if 'clip_ratio_adjustment' in learning_info:
                    self._adjust_clip_ratio(learning_info['clip_ratio_adjustment'])
                if 'entropy_coefficient_adjustment' in learning_info:
                    self._adjust_entropy_coefficient(learning_info['entropy_coefficient_adjustment'])
            
            # Update training statistics
            update_time = time.time() - start_time
            self.training_stats['total_updates'] += 1
            self.training_stats['successful_updates'] += 1
            self.training_stats['last_update_time'] = update_time
            self.training_stats['average_update_time'] = (
                (self.training_stats['average_update_time'] * (self.training_stats['total_updates'] - 1) + update_time) /
                self.training_stats['total_updates']
            )
            
            return {
                'obj_critic': float(obj_critic),
                'obj_actor': float(obj_actor),
                'update_time': update_time,
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️ Error in PPO wrapper update: {e}")
            self.training_stats['total_updates'] += 1
            return {
                'obj_critic': 0.0,
                'obj_actor': 0.0,
                'update_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _adjust_clip_ratio(self, factor: float):
        """Adjust PPO clip ratio based on meta-learning feedback"""
        if hasattr(self.agent, 'clip_ratio'):
            self.agent.clip_ratio *= factor
    
    def _adjust_entropy_coefficient(self, factor: float):
        """Adjust entropy coefficient based on meta-learning feedback"""
        if hasattr(self.agent, 'entropy_coef'):
            self.agent.entropy_coef *= factor


class RainbowAgentWrapper(BaseAgentWrapper):
    """
    Wrapper for Rainbow DQN agent
    """
    
    def __init__(self, agent, agent_name: str):
        super().__init__(agent, agent_name)
        self.distributional_confidence = True  # Use distributional info for confidence
    
    def get_action_with_confidence(self, state: torch.Tensor) -> Tuple[int, float, Dict]:
        """Get action with confidence based on distributional Q-values"""
        try:
            # Get action from Rainbow agent
            result = self.agent.act(state)
            
            if isinstance(result, tuple):
                action, q_dist = result
            else:
                action = result
                q_dist = None
            
            # Calculate confidence from distributional Q-values
            if q_dist is not None and self.distributional_confidence:
                if isinstance(q_dist, torch.Tensor):
                    q_vals = q_dist.detach().cpu().numpy()
                else:
                    q_vals = np.array(q_dist)
                
                # For distributional RL, confidence comes from distribution sharpness
                if len(q_vals.shape) > 1:  # Distributional Q-values
                    # Calculate variance of each action's distribution
                    variances = np.var(q_vals, axis=1) if len(q_vals.shape) > 1 else [0.1]
                    # Lower variance = higher confidence
                    confidence = 1.0 / (1.0 + np.min(variances))
                else:
                    # Standard Q-values
                    exp_q = np.exp(q_vals - np.max(q_vals))
                    probs = exp_q / np.sum(exp_q)
                    confidence = float(np.max(probs))
            else:
                confidence = 0.5
            
            self.confidence_scores.append(confidence)
            
            additional_info = {
                'q_distribution': q_dist if q_dist is not None else None,
                'distributional': self.distributional_confidence,
                'agent_type': 'rainbow'
            }
            
            return int(action), confidence, additional_info
            
        except Exception as e:
            print(f"⚠️ Error in Rainbow wrapper action: {e}")
            return 1, 0.5, {'error': str(e)}
    
    def update_with_feedback(self, buffer, learning_info: Optional[Dict] = None) -> Dict:
        """Update Rainbow agent with experience buffer"""
        start_time = time.time()
        
        try:
            # Rainbow-specific update
            if hasattr(self.agent, 'update_net'):
                obj_critic, obj_actor = self.agent.update_net(buffer)
            else:
                obj_critic, obj_actor = 0.0, 0.0
            
            # Apply meta-learning feedback
            if learning_info:
                if 'noisy_net_adjustment' in learning_info:
                    self._adjust_noise_parameters(learning_info['noisy_net_adjustment'])
                if 'priority_adjustment' in learning_info:
                    self._adjust_priority_parameters(learning_info['priority_adjustment'])
            
            # Update training statistics
            update_time = time.time() - start_time
            self.training_stats['total_updates'] += 1
            self.training_stats['successful_updates'] += 1
            self.training_stats['last_update_time'] = update_time
            self.training_stats['average_update_time'] = (
                (self.training_stats['average_update_time'] * (self.training_stats['total_updates'] - 1) + update_time) /
                self.training_stats['total_updates']
            )
            
            return {
                'obj_critic': float(obj_critic),
                'obj_actor': float(obj_actor),
                'update_time': update_time,
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️ Error in Rainbow wrapper update: {e}")
            self.training_stats['total_updates'] += 1
            return {
                'obj_critic': 0.0,
                'obj_actor': 0.0,
                'update_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _adjust_noise_parameters(self, factor: float):
        """Adjust noisy network parameters"""
        if hasattr(self.agent, 'noise_std'):
            self.agent.noise_std *= factor
    
    def _adjust_priority_parameters(self, factor: float):
        """Adjust prioritized replay parameters"""
        if hasattr(self.agent, 'priority_alpha'):
            self.agent.priority_alpha *= factor


class AgentWrapperFactory:
    """
    Factory for creating appropriate agent wrappers
    """
    
    @staticmethod
    def create_wrapper(agent, agent_name: str) -> BaseAgentWrapper:
        """
        Create appropriate wrapper for given agent
        
        Args:
            agent: Agent instance
            agent_name: Name for the agent
            
        Returns:
            Appropriate agent wrapper
        """
        agent_type = type(agent).__name__
        
        if agent_type in ['AgentD3QN', 'AgentDoubleDQN', 'AgentTwinD3QN']:
            return DQNAgentWrapper(agent, agent_name)
        elif agent_type == 'AgentPPO' and PPO_AVAILABLE:
            return PPOAgentWrapper(agent, agent_name)
        elif agent_type == 'AgentRainbow' and RAINBOW_AVAILABLE:
            return RainbowAgentWrapper(agent, agent_name)
        else:
            print(f"⚠️ Unknown agent type {agent_type}, using DQN wrapper")
            return DQNAgentWrapper(agent, agent_name)
    
    @staticmethod
    def create_multiple_wrappers(agents_dict: Dict) -> Dict[str, BaseAgentWrapper]:
        """
        Create wrappers for multiple agents
        
        Args:
            agents_dict: Dictionary of {agent_name: agent_instance}
            
        Returns:
            Dictionary of {agent_name: agent_wrapper}
        """
        wrappers = {}
        
        for agent_name, agent in agents_dict.items():
            wrapper = AgentWrapperFactory.create_wrapper(agent, agent_name)
            wrappers[agent_name] = wrapper
            
        print(f"🏭 Created {len(wrappers)} agent wrappers")
        return wrappers


class AgentEnsembleWrapper:
    """
    Wrapper for managing ensemble of agent wrappers
    """
    
    def __init__(self, agent_wrappers: Dict[str, BaseAgentWrapper]):
        self.agent_wrappers = agent_wrappers
        self.agent_names = list(agent_wrappers.keys())
        self.ensemble_statistics = {
            'total_decisions': 0,
            'consensus_decisions': 0,
            'disagreement_rate': 0.0
        }
        
        print(f"🎭 Ensemble wrapper created with {len(self.agent_wrappers)} agents:")
        for name, wrapper in self.agent_wrappers.items():
            print(f"   - {name} ({wrapper.agent_type})")
    
    def get_all_actions_with_confidence(self, state: torch.Tensor) -> Dict[str, Tuple[int, float, Dict]]:
        """Get actions and confidence from all agents"""
        results = {}
        
        for agent_name, wrapper in self.agent_wrappers.items():
            try:
                action, confidence, info = wrapper.get_action_with_confidence(state)
                results[agent_name] = (action, confidence, info)
            except Exception as e:
                print(f"⚠️ Error getting action from {agent_name}: {e}")
                results[agent_name] = (1, 0.5, {'error': str(e)})  # Default hold
        
        # Update ensemble statistics
        self.ensemble_statistics['total_decisions'] += 1
        actions = [result[0] for result in results.values()]
        
        # Check for consensus (all agents agree)
        if len(set(actions)) == 1:
            self.ensemble_statistics['consensus_decisions'] += 1
        
        # Update disagreement rate
        self.ensemble_statistics['disagreement_rate'] = (
            1.0 - self.ensemble_statistics['consensus_decisions'] / 
            self.ensemble_statistics['total_decisions']
        )
        
        return results
    
    def update_all_agents(self, buffers: Dict, learning_info: Optional[Dict] = None) -> Dict[str, Dict]:
        """Update all agents with their respective buffers"""
        update_results = {}
        
        for agent_name, wrapper in self.agent_wrappers.items():
            if agent_name in buffers:
                try:
                    agent_learning_info = learning_info.get(agent_name) if learning_info else None
                    result = wrapper.update_with_feedback(buffers[agent_name], agent_learning_info)
                    update_results[agent_name] = result
                except Exception as e:
                    print(f"⚠️ Error updating {agent_name}: {e}")
                    update_results[agent_name] = {'success': False, 'error': str(e)}
        
        return update_results
    
    def get_ensemble_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all agents"""
        summary = {
            'ensemble_stats': self.ensemble_statistics.copy(),
            'agent_performances': {},
            'ensemble_metrics': {}
        }
        
        # Get individual agent performances
        all_performances = []
        for agent_name, wrapper in self.agent_wrappers.items():
            agent_perf = wrapper.get_performance_metrics()
            summary['agent_performances'][agent_name] = agent_perf
            all_performances.append(agent_perf)
        
        # Calculate ensemble-level metrics
        if all_performances:
            summary['ensemble_metrics'] = {
                'avg_sharpe_ratio': np.mean([p['sharpe_ratio'] for p in all_performances]),
                'avg_win_rate': np.mean([p['win_rate'] for p in all_performances]),
                'avg_confidence': np.mean([p['confidence'] for p in all_performances]),
                'avg_activity_rate': np.mean([p['activity_rate'] for p in all_performances]),
                'performance_variance': np.var([p['sharpe_ratio'] for p in all_performances])
            }
        
        return summary


# Utility functions for agent wrapper management
def create_agent_wrappers_from_config(agents_config: Dict, 
                                     net_dims: List[int],
                                     state_dim: int,
                                     action_dim: int,
                                     gpu_id: int = 0) -> Dict[str, BaseAgentWrapper]:
    """
    Create agent wrappers from configuration
    
    Args:
        agents_config: Configuration dictionary for agents
        net_dims: Network dimensions
        state_dim: State dimension
        action_dim: Action dimension
        gpu_id: GPU device ID
        
    Returns:
        Dictionary of agent wrappers
    """
    agents = {}
    
    # Create DQN agents
    if 'd3qn' in agents_config:
        agents['d3qn'] = AgentD3QN(net_dims, state_dim, action_dim, gpu_id=gpu_id)
    
    if 'double_dqn' in agents_config:
        agents['double_dqn'] = AgentDoubleDQN(net_dims, state_dim, action_dim, gpu_id=gpu_id)
    
    if 'twin_d3qn' in agents_config:
        agents['twin_d3qn'] = AgentTwinD3QN(net_dims, state_dim, action_dim, gpu_id=gpu_id)
    
    # Create PPO agent if available
    if 'ppo' in agents_config and PPO_AVAILABLE:
        try:
            agents['ppo'] = AgentPPO(net_dims, state_dim, action_dim, gpu_id=gpu_id)
        except Exception as e:
            print(f"⚠️ Failed to create PPO agent: {e}")
    
    # Create Rainbow agent if available
    if 'rainbow' in agents_config and RAINBOW_AVAILABLE:
        try:
            agents['rainbow'] = AgentRainbow(net_dims, state_dim, action_dim, gpu_id=gpu_id)
        except Exception as e:
            print(f"⚠️ Failed to create Rainbow agent: {e}")
    
    # Create wrappers
    wrappers = AgentWrapperFactory.create_multiple_wrappers(agents)
    
    return wrappers


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Agent Wrapper System")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        'd3qn': True,
        'double_dqn': True,
        'twin_d3qn': True,
        'ppo': PPO_AVAILABLE,
        'rainbow': RAINBOW_AVAILABLE
    }
    
    # Create test wrappers
    wrappers = create_agent_wrappers_from_config(
        agents_config=test_config,
        net_dims=[256, 256],
        state_dim=50,
        action_dim=3,
        gpu_id=0
    )
    
    print(f"✅ Created {len(wrappers)} agent wrappers")
    
    # Test ensemble wrapper
    ensemble = AgentEnsembleWrapper(wrappers)
    
    # Test with dummy state
    test_state = torch.randn(50)
    
    print(f"\n🎯 Testing ensemble action generation:")
    results = ensemble.get_all_actions_with_confidence(test_state)
    
    for agent_name, (action, confidence, info) in results.items():
        print(f"   {agent_name}: Action={action}, Confidence={confidence:.3f}")
    
    # Test performance tracking
    print(f"\n📊 Testing performance tracking:")
    for wrapper in wrappers.values():
        # Simulate some performance data
        for i in range(10):
            wrapper.update_performance(
                action=np.random.choice([0, 1, 2]),
                reward=np.random.normal(0.01, 0.02),
                confidence=np.random.uniform(0.4, 0.9)
            )
    
    # Get performance summary
    summary = ensemble.get_ensemble_performance_summary()
    
    print(f"   Ensemble disagreement rate: {summary['ensemble_stats']['disagreement_rate']:.3f}")
    print(f"   Average Sharpe ratio: {summary['ensemble_metrics']['avg_sharpe_ratio']:.3f}")
    print(f"   Average confidence: {summary['ensemble_metrics']['avg_confidence']:.3f}")
    
    print(f"\n🎉 Agent wrapper system tested successfully!")