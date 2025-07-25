import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meta_learning_framework import (
    MetaLearningEnsembleManager,
    MetaLearningRiskManagedEnsemble
)
from enhanced_ensemble_manager import EnhancedEnsembleManager
from dynamic_risk_manager import DynamicRiskManager
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN

try:
    from erl_agent_ppo import AgentPPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("PPO agent not available, using DQN agents only")

try:
    from erl_agent_rainbow import AgentRainbow
    RAINBOW_AVAILABLE = True
except ImportError:
    RAINBOW_AVAILABLE = False
    print("Rainbow agent not available, using standard DQN agents")


class MetaLearningEnsembleTrainer:
    """
    Training framework that integrates meta-learning with ensemble management
    """
    
    def __init__(self, env, state_dim: int, action_dim: int, 
                 net_dims: List[int] = [512, 512],
                 save_dir: str = "./meta_learning_models"):
        
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_dims = net_dims
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize meta-learning components
        self.meta_learning_manager = MetaLearningEnsembleManager(
            agents=self.agents,
            meta_lookback=1000
        )
        
        # Initialize risk management
        self.risk_manager = DynamicRiskManager(
            max_position_size=0.95,
            stop_loss_threshold=0.05,
            max_drawdown_threshold=0.15
        )
        
        # Initialize integrated ensemble
        self.meta_ensemble = MetaLearningRiskManagedEnsemble(
            agents=self.agents,
            meta_learning_manager=self.meta_learning_manager,
            risk_manager=self.risk_manager
        )
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'meta_learning_updates': 0,
            'performance_history': []
        }
        
        print(f"Meta-learning ensemble initialized with {len(self.agents)} agents")
        print(f"Agents: {list(self.agents.keys())}")
    
    def _initialize_agents(self) -> Dict:
        """Initialize all available agents"""
        agents = {}
        
        # Standard DQN agents
        agents['d3qn'] = AgentD3QN(
            net_dims=self.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gpu_id=0
        )
        
        agents['double_dqn'] = AgentDoubleDQN(
            net_dims=self.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gpu_id=0
        )
        
        agents['twin_d3qn'] = AgentTwinD3QN(
            net_dims=self.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gpu_id=0
        )
        
        # PPO agent if available
        if PPO_AVAILABLE:
            try:
                agents['ppo'] = AgentPPO(
                    net_dims=self.net_dims,
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    gpu_id=0
                )
                print("PPO agent successfully initialized")
            except Exception as e:
                print(f"Failed to initialize PPO agent: {e}")
        
        # Rainbow agent if available
        if RAINBOW_AVAILABLE:
            try:
                agents['rainbow'] = AgentRainbow(
                    net_dims=self.net_dims,
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    gpu_id=0
                )
                print("Rainbow agent successfully initialized")
            except Exception as e:
                print(f"Failed to initialize Rainbow agent: {e}")
        
        return agents
    
    def train_episode(self, episode_num: int, max_steps: int = 1000) -> Dict:
        """Train single episode with meta-learning"""
        
        episode_stats = {
            'episode': episode_num,
            'total_reward': 0.0,
            'steps': 0,
            'actions_taken': {'buy': 0, 'hold': 0, 'sell': 0},
            'regime_changes': 0,
            'meta_learning_active': False
        }
        
        # Reset environment
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        done = False
        step = 0
        episode_reward = 0.0
        previous_regime = None
        
        # Episode variables for tracking
        episode_returns = []
        episode_actions = []
        episode_prices = []
        
        while not done and step < max_steps:
            # Get current market price (assuming it's available in the environment)
            current_price = getattr(self.env, 'current_price', 100.0 + np.random.randn() * 0.1)
            current_volume = getattr(self.env, 'current_volume', 1000.0)
            
            # Get meta-learning enhanced action
            action, decision_info = self.meta_ensemble.get_trading_action(
                state_tensor, current_price, current_volume
            )
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            # Update statistics
            episode_reward += reward
            episode_returns.append(reward)
            episode_actions.append(action)
            episode_prices.append(current_price)
            
            action_names = ['sell', 'hold', 'buy']
            if 0 <= action < len(action_names):
                episode_stats['actions_taken'][action_names[action]] += 1
            
            # Track regime changes
            current_regime = decision_info.get('current_regime', 'unknown')
            if previous_regime and previous_regime != current_regime:
                episode_stats['regime_changes'] += 1
            previous_regime = current_regime
            
            # Update individual agents (simplified - would need proper buffer management)
            self._update_agents_simplified(state_tensor, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            state_tensor = torch.tensor(state, dtype=torch.float32)
            step += 1
        
        # Calculate episode performance metrics
        episode_stats.update({
            'total_reward': episode_reward,
            'steps': step,
            'avg_reward_per_step': episode_reward / max(step, 1),
            'final_regime': current_regime,
            'meta_learning_active': len(self.meta_learning_manager.training_data['market_features']) > 50
        })
        
        # Calculate additional metrics
        if len(episode_returns) > 1:
            returns_array = np.array(episode_returns)
            episode_stats.update({
                'returns_std': np.std(returns_array),
                'sharpe_ratio': np.mean(returns_array) / (np.std(returns_array) + 1e-8),
                'max_single_return': np.max(returns_array),
                'min_single_return': np.min(returns_array)
            })
        
        # Update meta-learning with episode performance
        self.meta_ensemble.update_performance(
            returns=episode_reward,
            sharpe_ratio=episode_stats.get('sharpe_ratio', 0.0),
            additional_metrics={
                'win_rate': np.mean(np.array(episode_returns) > 0),
                'avg_return': np.mean(episode_returns),
                'volatility': np.std(episode_returns)
            }
        )
        
        # Update training statistics
        self.training_stats['episodes'] += 1
        self.training_stats['total_steps'] += step
        self.training_stats['performance_history'].append(episode_stats)
        
        return episode_stats
    
    def _update_agents_simplified(self, state: torch.Tensor, action: int, 
                                reward: float, next_state: np.ndarray, done: bool):
        """Simplified agent update - would need proper implementation with buffers"""
        
        # This is a placeholder - actual implementation would require:
        # 1. Proper replay buffer management for each agent
        # 2. Batch updates with collected experience
        # 3. Different update mechanisms for DQN vs PPO agents
        
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        
        # For now, just track that agents need updating
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'update_needed'):
                agent.update_needed = True
    
    def train_full_session(self, num_episodes: int = 100, 
                          save_interval: int = 20) -> Dict:
        """Train complete session with multiple episodes"""
        
        print(f"Starting meta-learning training session: {num_episodes} episodes")
        print(f"Save interval: {save_interval} episodes")
        
        session_stats = {
            'total_episodes': num_episodes,
            'episode_rewards': [],
            'regime_stability': [],
            'meta_learning_progress': [],
            'best_episode_reward': float('-inf'),
            'final_performance': {}
        }
        
        for episode in range(num_episodes):
            # Train single episode
            episode_stats = self.train_episode(episode, max_steps=1000)
            
            # Track session statistics
            session_stats['episode_rewards'].append(episode_stats['total_reward'])
            
            # Track meta-learning progress
            meta_progress = {
                'episode': episode,
                'training_samples': len(self.meta_learning_manager.training_data['market_features']),
                'training_steps': self.meta_learning_manager.training_step,
                'current_regime': episode_stats.get('final_regime', 'unknown')
            }
            session_stats['meta_learning_progress'].append(meta_progress)
            
            # Update best performance
            if episode_stats['total_reward'] > session_stats['best_episode_reward']:
                session_stats['best_episode_reward'] = episode_stats['total_reward']
            
            # Print progress
            if episode % 10 == 0 or episode == num_episodes - 1:
                avg_reward = np.mean(session_stats['episode_rewards'][-10:])
                regime_info = self.meta_learning_manager.get_regime_info()
                
                print(f"Episode {episode:3d}: "
                      f"Reward={episode_stats['total_reward']:.3f}, "
                      f"Avg10={avg_reward:.3f}, "
                      f"Regime={regime_info['current_regime']}, "
                      f"Stability={regime_info['regime_stability']:.2f}")
            
            # Save models periodically
            if episode % save_interval == 0 and episode > 0:
                self.save_models(f"checkpoint_episode_{episode}")
                print(f"Models saved at episode {episode}")
        
        # Final performance summary
        session_stats['final_performance'] = self.meta_ensemble.get_performance_summary()
        
        # Save final models
        self.save_models("final_models")
        
        print(f"\nTraining session completed!")
        print(f"Best episode reward: {session_stats['best_episode_reward']:.3f}")
        print(f"Final 10-episode average: {np.mean(session_stats['episode_rewards'][-10:]):.3f}")
        
        return session_stats
    
    def save_models(self, checkpoint_name: str):
        """Save all models and meta-learning state"""
        
        checkpoint_dir = os.path.join(self.save_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save individual agents
        agents_dir = os.path.join(checkpoint_dir, "agents")
        os.makedirs(agents_dir, exist_ok=True)
        
        for agent_name, agent in self.agents.items():
            try:
                agent_path = os.path.join(agents_dir, agent_name)
                os.makedirs(agent_path, exist_ok=True)
                agent.save_or_load_agent(agent_path, if_save=True)
            except Exception as e:
                print(f"Failed to save agent {agent_name}: {e}")
        
        # Save meta-learning models
        self.meta_learning_manager.save_meta_models(checkpoint_dir)
        
        # Save training statistics
        stats_path = os.path.join(checkpoint_dir, "training_stats.json")
        import json
        with open(stats_path, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            serializable_stats = self._make_json_serializable(self.training_stats)
            json.dump(serializable_stats, f, indent=2)
        
        print(f"All models saved to {checkpoint_dir}")
    
    def load_models(self, checkpoint_name: str):
        """Load all models and meta-learning state"""
        
        checkpoint_dir = os.path.join(self.save_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory {checkpoint_dir} does not exist")
            return False
        
        # Load individual agents
        agents_dir = os.path.join(checkpoint_dir, "agents")
        
        for agent_name, agent in self.agents.items():
            try:
                agent_path = os.path.join(agents_dir, agent_name)
                if os.path.exists(agent_path):
                    agent.save_or_load_agent(agent_path, if_save=False)
                    print(f"Loaded agent {agent_name}")
            except Exception as e:
                print(f"Failed to load agent {agent_name}: {e}")
        
        # Load meta-learning models
        try:
            self.meta_learning_manager.load_meta_models(checkpoint_dir)
        except Exception as e:
            print(f"Failed to load meta-learning models: {e}")
        
        # Load training statistics
        stats_path = os.path.join(checkpoint_dir, "training_stats.json")
        if os.path.exists(stats_path):
            try:
                import json
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
                print("Training statistics loaded")
            except Exception as e:
                print(f"Failed to load training statistics: {e}")
        
        return True
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj
    
    def evaluate_performance(self, num_episodes: int = 10) -> Dict:
        """Evaluate trained meta-learning ensemble"""
        
        print(f"Evaluating meta-learning ensemble over {num_episodes} episodes...")
        
        evaluation_stats = {
            'episode_rewards': [],
            'regime_distributions': {},
            'algorithm_usage': {},
            'performance_metrics': {}
        }
        
        for episode in range(num_episodes):
            episode_stats = self.train_episode(episode, max_steps=1000)
            evaluation_stats['episode_rewards'].append(episode_stats['total_reward'])
            
            # Track regime distribution
            regime = episode_stats.get('final_regime', 'unknown')
            evaluation_stats['regime_distributions'][regime] = \
                evaluation_stats['regime_distributions'].get(regime, 0) + 1
        
        # Calculate performance metrics
        rewards = evaluation_stats['episode_rewards']
        evaluation_stats['performance_metrics'] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'sharpe_ratio': np.mean(rewards) / (np.std(rewards) + 1e-8),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'success_rate': np.mean(np.array(rewards) > 0)
        }
        
        print("Evaluation completed!")
        print(f"Mean reward: {evaluation_stats['performance_metrics']['mean_reward']:.3f}")
        print(f"Sharpe ratio: {evaluation_stats['performance_metrics']['sharpe_ratio']:.3f}")
        print(f"Success rate: {evaluation_stats['performance_metrics']['success_rate']:.2%}")
        
        return evaluation_stats


def demo_meta_learning_training():
    """
    Demonstration of meta-learning framework training
    """
    
    print("=== Meta-Learning Framework Demo ===")
    
    # Mock environment for demonstration
    class MockTradingEnv:
        def __init__(self):
            self.state_dim = 50
            self.action_dim = 3
            self.current_step = 0
            self.max_steps = 1000
            self.current_price = 100.0
            self.current_volume = 1000.0
        
        def reset(self):
            self.current_step = 0
            self.current_price = 100.0 + np.random.randn() * 10
            return np.random.randn(self.state_dim)
        
        def step(self, action):
            self.current_step += 1
            
            # Simulate price movement
            price_change = np.random.randn() * 0.02
            if action == 0:  # Sell
                reward = -price_change
            elif action == 2:  # Buy
                reward = price_change
            else:  # Hold
                reward = 0.0
            
            self.current_price *= (1 + price_change)
            self.current_volume = 1000.0 + np.random.randn() * 100
            
            next_state = np.random.randn(self.state_dim)
            done = self.current_step >= self.max_steps
            info = {'price': self.current_price, 'volume': self.current_volume}
            
            return next_state, reward, done, info
    
    # Initialize mock environment
    env = MockTradingEnv()
    
    # Initialize meta-learning trainer
    trainer = MetaLearningEnsembleTrainer(
        env=env,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        net_dims=[256, 256],
        save_dir="./demo_meta_learning_models"
    )
    
    # Train for a few episodes
    print("\nStarting demo training...")
    session_stats = trainer.train_full_session(num_episodes=20, save_interval=10)
    
    # Evaluate performance
    print("\nEvaluating performance...")
    eval_stats = trainer.evaluate_performance(num_episodes=5)
    
    print("\n=== Demo Complete ===")
    print(f"Training completed with {len(session_stats['episode_rewards'])} episodes")
    print(f"Best episode reward: {session_stats['best_episode_reward']:.3f}")
    print(f"Evaluation Sharpe ratio: {eval_stats['performance_metrics']['sharpe_ratio']:.3f}")
    
    return trainer, session_stats, eval_stats


if __name__ == "__main__":
    # Run demonstration
    trainer, session_stats, eval_stats = demo_meta_learning_training()