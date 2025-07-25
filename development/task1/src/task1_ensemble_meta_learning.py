#!/usr/bin/env python3
"""
Meta-Learning Ensemble Training System
Advanced ensemble that adapts algorithm selection based on market conditions
"""

import os
import sys
import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
import argparse
from collections import deque

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import core components
from trade_simulator import TradeSimulator
from erl_config import build_env
from erl_replay_buffer import ReplayBuffer

# Import meta-learning framework
from meta_learning_framework import (
    MetaLearningEnsembleManager,
    MetaLearningRiskManagedEnsemble
)
from meta_learning_config import (
    MetaLearningConfig,
    MetaLearningTracker,
    create_meta_learning_config
)
from meta_learning_agent_wrapper import (
    AgentWrapperFactory,
    AgentEnsembleWrapper,
    create_agent_wrappers_from_config
)

# Import existing components
from enhanced_ensemble_manager import EnhancedEnsembleManager
from dynamic_risk_manager import DynamicRiskManager
from optimized_hyperparameters import get_optimized_hyperparameters
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown

# Import different buffer types
try:
    from erl_replay_buffer_ppo import PPOReplayBuffer
    PPO_BUFFER_AVAILABLE = True
except ImportError:
    PPO_BUFFER_AVAILABLE = False

try:
    from erl_agent_rainbow import PrioritizedReplayBuffer
    PRIORITIZED_BUFFER_AVAILABLE = True
except ImportError:
    PRIORITIZED_BUFFER_AVAILABLE = False


class MetaLearningEnsembleTrainer:
    """
    Advanced ensemble trainer with meta-learning capabilities
    Combines multiple RL agents with intelligent algorithm selection
    """
    
    def __init__(self, 
                 config: MetaLearningConfig,
                 team_name: str = "meta_learning_ensemble",
                 save_dir: str = "./meta_learning_models"):
        
        self.config = config
        self.team_name = team_name
        self.save_dir = save_dir
        self.gpu_id = getattr(config, 'gpu_id', 0)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize environment
        try:
            self.env = build_env(config.env_class, config.env_args, gpu_id=self.gpu_id)
        except Exception as e:
            print(f"⚠️ Failed to create environment with build_env: {e}")
            # Fallback: create test environment
            from test_environment import create_test_environment
            self.env = create_test_environment(
                state_dim=getattr(config, 'state_dim', 50),
                action_dim=getattr(config, 'action_dim', 3),
                max_steps=1000
            )
            print(f"✅ Using test environment as fallback")
        
        # Get state dimensions from environment or TradeSimulator
        if hasattr(self.env, 'state_dim'):
            self.state_dim = self.env.state_dim
        else:
            temp_sim = TradeSimulator(num_sims=1)
            self.state_dim = temp_sim.state_dim
        
        self.action_dim = getattr(config, 'action_dim', 3)
        self.net_dims = getattr(config, 'net_dims', [512, 512])
        
        print(f"🧠 Meta-Learning Ensemble Trainer Initialized:")
        print(f"   📊 State Dimension: {self.state_dim}")
        print(f"   🎯 Action Dimension: {self.action_dim}")
        print(f"   🏗️ Network Dimensions: {self.net_dims}")
        print(f"   💾 Save Directory: {save_dir}")
        
        # Initialize agents and wrappers
        self.agent_wrappers = self._initialize_agent_wrappers()
        self.ensemble_wrapper = AgentEnsembleWrapper(self.agent_wrappers)
        
        # Initialize replay buffers
        self.replay_buffers = self._initialize_replay_buffers()
        
        # Initialize meta-learning components
        raw_agents = {name: wrapper.agent for name, wrapper in self.agent_wrappers.items()}
        self.meta_learning_manager = MetaLearningEnsembleManager(
            agents=raw_agents,
            meta_lookback=config.meta_lookback
        )
        
        # Initialize risk management
        self.risk_manager = DynamicRiskManager(
            max_position_size=getattr(config, 'max_position_risk', 0.95),
            stop_loss_threshold=0.05,
            max_drawdown_threshold=0.15
        )
        
        # Initialize integrated ensemble
        self.meta_ensemble = MetaLearningRiskManagedEnsemble(
            agents=raw_agents,
            meta_learning_manager=self.meta_learning_manager,
            risk_manager=self.risk_manager
        )
        
        # Initialize tracking and logging
        self.tracker = MetaLearningTracker(history_size=config.decision_history_size)
        
        # Training state
        self.training_stats = {
            'episodes_completed': 0,
            'total_steps': 0,
            'best_performance': float('-inf'),
            'meta_learning_updates': 0,
            'start_time': time.time()
        }
        
        # Performance history
        self.episode_rewards = deque(maxlen=100)
        self.episode_sharpe_ratios = deque(maxlen=100)
        self.episode_max_drawdowns = deque(maxlen=100)
        
        print(f"🎭 Initialized with {len(self.agent_wrappers)} agents:")
        for name, wrapper in self.agent_wrappers.items():
            print(f"   - {name} ({wrapper.agent_type})")
    
    def _initialize_agent_wrappers(self) -> Dict:
        """Initialize agent wrappers based on configuration"""
        
        # Define which agents to use
        agents_config = {
            'd3qn': True,
            'double_dqn': True,
            'twin_d3qn': True,
            'ppo': getattr(self.config, 'use_ppo', True),
            'rainbow': getattr(self.config, 'use_rainbow', True)
        }
        
        # Apply optimized hyperparameters if available
        try:
            optimized_params = get_optimized_hyperparameters()
            if optimized_params:
                print("📈 Applying optimized hyperparameters")
        except Exception as e:
            print(f"⚠️ Could not load optimized hyperparameters: {e}")
            optimized_params = None
        
        # Create agent wrappers
        wrappers = create_agent_wrappers_from_config(
            agents_config=agents_config,
            net_dims=self.net_dims,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gpu_id=self.gpu_id
        )
        
        return wrappers
    
    def _initialize_replay_buffers(self) -> Dict:
        """Initialize replay buffers for each agent type"""
        buffers = {}
        buffer_size = getattr(self.config, 'replay_buffer_size', 100000)
        
        for agent_name, wrapper in self.agent_wrappers.items():
            agent_type = wrapper.agent_type
            
            if agent_type == 'AgentPPO' and PPO_BUFFER_AVAILABLE:
                # PPO uses different buffer
                buffers[agent_name] = PPOReplayBuffer(
                    max_size=buffer_size,
                    state_dim=self.state_dim,
                    action_dim=1,  # Discrete action
                    device=torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
                )
            elif agent_type == 'AgentRainbow' and PRIORITIZED_BUFFER_AVAILABLE:
                # Rainbow uses prioritized buffer
                try:
                    buffers[agent_name] = PrioritizedReplayBuffer(
                        capacity=buffer_size,
                        state_dim=self.state_dim,
                        action_dim=1,
                        device=torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
                    )
                except Exception:
                    # Fallback to standard buffer
                    buffers[agent_name] = ReplayBuffer(
                        max_size=buffer_size,
                        state_dim=self.state_dim,
                        action_dim=1,
                        gpu_id=self.gpu_id
                    )
            else:
                # Standard DQN buffer
                buffers[agent_name] = ReplayBuffer(
                    max_size=buffer_size,
                    state_dim=self.state_dim,
                    action_dim=1,
                    gpu_id=self.gpu_id
                )
        
        print(f"💾 Initialized {len(buffers)} replay buffers")
        return buffers
    
    def train_episode(self, episode_num: int, max_steps: int = 1000) -> Dict:
        """Train single episode with meta-learning"""
        
        episode_start_time = time.time()
        episode_stats = {
            'episode': episode_num,
            'total_reward': 0.0,
            'steps': 0,
            'actions_taken': {'sell': 0, 'hold': 0, 'buy': 0},
            'regime_changes': 0,
            'meta_learning_updates': 0,
            'ensemble_decisions': [],
            'agent_agreements': 0,
            'confidence_scores': []
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
        
        # Episode tracking
        episode_returns = []
        episode_actions = []
        episode_prices = []
        episode_volumes = []
        
        while not done and step < max_steps:
            
            # Get current market data (mock if not available from environment)
            current_price = getattr(self.env, 'current_price', 100.0 + np.random.randn() * 0.5)
            current_volume = getattr(self.env, 'current_volume', 1000.0 + np.random.randn() * 50)
            
            episode_prices.append(current_price)
            episode_volumes.append(current_volume)
            
            # Get meta-learning enhanced ensemble action
            ensemble_action, decision_info = self.meta_ensemble.get_trading_action(
                state_tensor, current_price, current_volume
            )
            
            # Store decision information
            episode_stats['ensemble_decisions'].append(ensemble_action)
            
            # Track regime changes
            current_regime = decision_info.get('current_regime', 'unknown')
            if previous_regime and previous_regime != current_regime:
                episode_stats['regime_changes'] += 1
            previous_regime = current_regime
            
            # Get individual agent actions for buffer storage
            agent_results = self.ensemble_wrapper.get_all_actions_with_confidence(state_tensor)
            
            # Track ensemble agreement
            individual_actions = [result[0] for result in agent_results.values()]
            if len(set(individual_actions)) == 1:  # All agents agree
                episode_stats['agent_agreements'] += 1
            
            # Track confidence scores
            confidences = [result[1] for result in agent_results.values()]
            episode_stats['confidence_scores'].append(np.mean(confidences))
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(ensemble_action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            
            # Store experience in replay buffers
            for agent_name, (agent_action, confidence, agent_info) in agent_results.items():
                if agent_name in self.replay_buffers:
                    try:
                        self.replay_buffers[agent_name].add(
                            state, agent_action, reward, next_state, done
                        )
                    except Exception as e:
                        print(f"⚠️ Error storing experience for {agent_name}: {e}")
            
            # Update episode statistics
            episode_reward += reward
            episode_returns.append(reward)
            episode_actions.append(ensemble_action)
            
            action_names = ['sell', 'hold', 'buy']
            if 0 <= ensemble_action < len(action_names):
                episode_stats['actions_taken'][action_names[ensemble_action]] += 1
            
            # Update agent performance tracking
            for agent_name, wrapper in self.agent_wrappers.items():
                if agent_name in agent_results:
                    agent_action, confidence, _ = agent_results[agent_name]
                    wrapper.update_performance(agent_action, reward, confidence)
            
            # Update agents periodically
            if step > 0 and step % 10 == 0:
                self._update_agents(step)
            
            # Meta-learning updates
            if (step > 0 and 
                step % self.config.meta_training_frequency == 0 and
                len(self.meta_learning_manager.training_data['market_features']) > self.config.meta_batch_size):
                
                self._perform_meta_learning_update()
                episode_stats['meta_learning_updates'] += 1
            
            # Update tracking
            self.tracker.update_meta_metrics(
                step=step,
                regime=current_regime,
                regime_confidence=decision_info.get('regime_info', {}).get('regime_stability', 0.5),
                agent_weights=decision_info.get('algorithm_weights', {}),
                ensemble_decision=ensemble_action
            )
            
            # Move to next state
            state = next_state
            state_tensor = next_state_tensor
            step += 1
        
        # Calculate episode performance metrics
        episode_stats.update({
            'total_reward': episode_reward,
            'steps': step,
            'avg_reward_per_step': episode_reward / max(step, 1),
            'final_regime': current_regime,
            'training_time': time.time() - episode_start_time,
            'agreement_rate': episode_stats['agent_agreements'] / max(step, 1),
            'avg_confidence': np.mean(episode_stats['confidence_scores']) if episode_stats['confidence_scores'] else 0.5
        })
        
        # Calculate additional performance metrics
        if len(episode_returns) > 1:
            returns_array = np.array(episode_returns)
            episode_stats.update({
                'returns_std': np.std(returns_array),
                'sharpe_ratio': sharpe_ratio(returns_array),
                'max_drawdown': max_drawdown(np.cumsum(returns_array)),
                'romd': return_over_max_drawdown(returns_array),
                'win_rate': np.mean(returns_array > 0),
                'profit_factor': np.sum(returns_array[returns_array > 0]) / max(abs(np.sum(returns_array[returns_array < 0])), 1e-8)
            })
        
        # Update meta-learning with episode performance
        self.meta_ensemble.update_performance(
            returns=episode_reward,
            sharpe_ratio=episode_stats.get('sharpe_ratio', 0.0),
            additional_metrics={
                'win_rate': episode_stats.get('win_rate', 0.5),
                'avg_return': np.mean(episode_returns) if episode_returns else 0.0,
                'volatility': np.std(episode_returns) if len(episode_returns) > 1 else 0.1,
                'max_drawdown': episode_stats.get('max_drawdown', 0.0),
                'agreement_rate': episode_stats['agreement_rate'],
                'regime_stability': decision_info.get('regime_info', {}).get('regime_stability', 0.5)
            }
        )
        
        # Update training statistics
        self.training_stats['episodes_completed'] += 1
        self.training_stats['total_steps'] += step
        
        # Track episode performance
        self.episode_rewards.append(episode_reward)
        if 'sharpe_ratio' in episode_stats:
            self.episode_sharpe_ratios.append(episode_stats['sharpe_ratio'])
        if 'max_drawdown' in episode_stats:
            self.episode_max_drawdowns.append(episode_stats['max_drawdown'])
        
        # Update best performance
        if episode_reward > self.training_stats['best_performance']:
            self.training_stats['best_performance'] = episode_reward
            self._save_best_models(episode_num)
        
        return episode_stats
    
    def _update_agents(self, current_step: int):
        """Update all agents with their replay buffers"""
        
        # Check if buffers have enough data
        min_buffer_size = 1000
        ready_buffers = {
            name: buffer for name, buffer in self.replay_buffers.items()
            if len(buffer) >= min_buffer_size
        }
        
        if not ready_buffers:
            return
        
        # Get meta-learning feedback for each agent
        learning_info = {}
        for agent_name in ready_buffers.keys():
            performance_metrics = self.agent_wrappers[agent_name].get_performance_metrics()
            
            # Adjust learning based on performance
            if performance_metrics['sharpe_ratio'] < 0:
                learning_info[agent_name] = {'learning_rate_adjustment': 0.95}  # Reduce LR for poor performance
            elif performance_metrics['sharpe_ratio'] > 1.0:
                learning_info[agent_name] = {'learning_rate_adjustment': 1.05}  # Increase LR for good performance
        
        # Update agents
        update_results = self.ensemble_wrapper.update_all_agents(ready_buffers, learning_info)
        
        # Log update results
        successful_updates = sum(1 for result in update_results.values() if result.get('success', False))
        if successful_updates > 0:
            print(f"   🔄 Updated {successful_updates}/{len(ready_buffers)} agents at step {current_step}")
    
    def _perform_meta_learning_update(self):
        """Perform meta-learning model updates"""
        try:
            self.meta_learning_manager.train_meta_models(
                batch_size=self.config.meta_batch_size,
                epochs=self.config.meta_epochs
            )
            self.training_stats['meta_learning_updates'] += 1
        except Exception as e:
            print(f"⚠️ Meta-learning update failed: {e}")
    
    def train_full_session(self, num_episodes: int = 100, 
                          save_interval: int = 20,
                          evaluation_interval: int = 10) -> Dict:
        """Train complete session with multiple episodes"""
        
        print(f"🚀 Starting Meta-Learning Training Session:")
        print(f"   📚 Episodes: {num_episodes}")
        print(f"   💾 Save Interval: {save_interval}")
        print(f"   📊 Evaluation Interval: {evaluation_interval}")
        print(f"   🧠 Meta-Learning: {'Enabled' if self.config.meta_learning_enabled else 'Disabled'}")
        
        session_stats = {
            'total_episodes': num_episodes,
            'episode_rewards': [],
            'episode_sharpe_ratios': [],
            'regime_distributions': {},
            'meta_learning_progress': [],
            'best_episode_reward': float('-inf'),
            'best_episode_sharpe': float('-inf'),
            'training_duration': 0,
            'final_performance': {}
        }
        
        session_start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # Train single episode
            episode_stats = self.train_episode(episode, max_steps=1000)
            
            # Track session statistics
            session_stats['episode_rewards'].append(episode_stats['total_reward'])
            if 'sharpe_ratio' in episode_stats:
                session_stats['episode_sharpe_ratios'].append(episode_stats['sharpe_ratio'])
            
            # Track regime distribution
            regime = episode_stats.get('final_regime', 'unknown')
            session_stats['regime_distributions'][regime] = \
                session_stats['regime_distributions'].get(regime, 0) + 1
            
            # Track meta-learning progress
            meta_progress = {
                'episode': episode,
                'training_samples': len(self.meta_learning_manager.training_data['market_features']),
                'training_steps': self.meta_learning_manager.training_step,
                'current_regime': regime,
                'meta_updates_this_episode': episode_stats['meta_learning_updates']
            }
            session_stats['meta_learning_progress'].append(meta_progress)
            
            # Update best performance
            if episode_stats['total_reward'] > session_stats['best_episode_reward']:
                session_stats['best_episode_reward'] = episode_stats['total_reward']
            
            if 'sharpe_ratio' in episode_stats and episode_stats['sharpe_ratio'] > session_stats['best_episode_sharpe']:
                session_stats['best_episode_sharpe'] = episode_stats['sharpe_ratio']
            
            # Print progress
            if episode % 5 == 0 or episode == num_episodes - 1:
                self._print_training_progress(episode, episode_stats, session_stats)
            
            # Evaluation
            if episode % evaluation_interval == 0 and episode > 0:
                eval_results = self._evaluate_current_performance()
                print(f"   📊 Evaluation at episode {episode}: Sharpe={eval_results.get('sharpe_ratio', 0):.3f}")
            
            # Save models periodically
            if episode % save_interval == 0 and episode > 0:
                self._save_checkpoint(f"episode_{episode}")
                print(f"   💾 Checkpoint saved at episode {episode}")
        
        # Final session statistics
        session_stats['training_duration'] = time.time() - session_start_time
        session_stats['final_performance'] = self._get_final_performance_summary()
        
        # Save final models
        self._save_checkpoint("final_session")
        
        # Print final summary
        self._print_session_summary(session_stats, num_episodes)
        
        return session_stats
    
    def _print_training_progress(self, episode: int, episode_stats: Dict, session_stats: Dict):
        """Print training progress information"""
        
        # Recent performance averages
        recent_rewards = session_stats['episode_rewards'][-10:] if len(session_stats['episode_rewards']) >= 10 else session_stats['episode_rewards']
        recent_sharpe = session_stats['episode_sharpe_ratios'][-10:] if len(session_stats['episode_sharpe_ratios']) >= 10 else session_stats['episode_sharpe_ratios']
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_sharpe = np.mean(recent_sharpe) if recent_sharpe else 0
        
        # Meta-learning info
        meta_samples = len(self.meta_learning_manager.training_data['market_features'])
        meta_updates = self.training_stats['meta_learning_updates']
        
        print(f"Episode {episode:3d}: "
              f"Reward={episode_stats['total_reward']:.3f}, "
              f"Avg10={avg_reward:.3f}, "
              f"Sharpe={episode_stats.get('sharpe_ratio', 0):.3f}, "
              f"Regime={episode_stats.get('final_regime', 'unknown')}, "
              f"Agree={episode_stats.get('agreement_rate', 0):.2f}, "
              f"Meta={meta_samples}/{meta_updates}")
    
    def _evaluate_current_performance(self, num_eval_episodes: int = 5) -> Dict:
        """Evaluate current ensemble performance"""
        
        eval_rewards = []
        eval_sharpe_ratios = []
        
        for eval_ep in range(num_eval_episodes):
            # Run evaluation episode (similar to training but without updates)
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            episode_reward = 0.0
            episode_returns = []
            done = False
            step = 0
            max_eval_steps = 500
            
            while not done and step < max_eval_steps:
                current_price = getattr(self.env, 'current_price', 100.0 + np.random.randn() * 0.5)
                current_volume = getattr(self.env, 'current_volume', 1000.0 + np.random.randn() * 50)
                
                # Get action without exploration/training
                ensemble_action, _ = self.meta_ensemble.get_trading_action(
                    state_tensor, current_price, current_volume
                )
                
                next_state, reward, done, info = self.env.step(ensemble_action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                
                episode_reward += reward
                episode_returns.append(reward)
                
                state = next_state
                state_tensor = torch.tensor(state, dtype=torch.float32)
                step += 1
            
            eval_rewards.append(episode_reward)
            if len(episode_returns) > 1:
                eval_sharpe_ratios.append(sharpe_ratio(np.array(episode_returns)))
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'sharpe_ratio': np.mean(eval_sharpe_ratios) if eval_sharpe_ratios else 0.0,
            'evaluation_episodes': num_eval_episodes
        }
    
    def _get_final_performance_summary(self) -> Dict:
        """Get comprehensive final performance summary"""
        
        # Basic performance metrics
        summary = {
            'total_episodes': self.training_stats['episodes_completed'],
            'total_steps': self.training_stats['total_steps'],
            'best_episode_reward': self.training_stats['best_performance'],
            'meta_learning_updates': self.training_stats['meta_learning_updates']
        }
        
        # Recent performance (last 20 episodes)
        if len(self.episode_rewards) > 0:
            recent_rewards = list(self.episode_rewards)[-20:]
            summary.update({
                'recent_mean_reward': np.mean(recent_rewards),
                'recent_std_reward': np.std(recent_rewards),
                'recent_best_reward': np.max(recent_rewards),
                'recent_worst_reward': np.min(recent_rewards)
            })
        
        if len(self.episode_sharpe_ratios) > 0:
            recent_sharpe = list(self.episode_sharpe_ratios)[-20:]
            summary.update({
                'recent_mean_sharpe': np.mean(recent_sharpe),
                'recent_best_sharpe': np.max(recent_sharpe)
            })
        
        # Meta-learning specific metrics
        meta_summary = self.meta_ensemble.get_performance_summary()
        summary.update({
            'meta_learning_summary': meta_summary,
            'regime_info': self.meta_learning_manager.get_regime_info()
        })
        
        # Agent-specific performance
        ensemble_summary = self.ensemble_wrapper.get_ensemble_performance_summary()
        summary['agent_performances'] = ensemble_summary['agent_performances']
        summary['ensemble_metrics'] = ensemble_summary['ensemble_metrics']
        
        return summary
    
    def _print_session_summary(self, session_stats: Dict, num_episodes: int):
        """Print comprehensive session summary"""
        
        print(f"\n" + "="*80)
        print(f"🎉 META-LEARNING TRAINING SESSION COMPLETED")
        print(f"="*80)
        
        # Training overview
        print(f"📊 Training Overview:")
        print(f"   Episodes Completed: {num_episodes}")
        print(f"   Training Duration: {session_stats['training_duration']:.1f}s ({session_stats['training_duration']/60:.1f}m)")
        print(f"   Total Steps: {self.training_stats['total_steps']}")
        print(f"   Meta-Learning Updates: {self.training_stats['meta_learning_updates']}")
        
        # Performance metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   Best Episode Reward: {session_stats['best_episode_reward']:.4f}")
        print(f"   Best Episode Sharpe: {session_stats['best_episode_sharpe']:.4f}")
        
        if session_stats['episode_rewards']:
            recent_performance = session_stats['episode_rewards'][-10:]
            print(f"   Final 10 Episodes Avg: {np.mean(recent_performance):.4f}")
            print(f"   Performance Std: {np.std(session_stats['episode_rewards']):.4f}")
        
        # Meta-learning insights
        print(f"\n🧠 Meta-Learning Insights:")
        final_perf = session_stats['final_performance']
        if 'regime_info' in final_perf:
            regime_info = final_perf['regime_info']
            print(f"   Final Regime: {regime_info.get('current_regime', 'unknown')}")
            print(f"   Regime Stability: {regime_info.get('regime_stability', 0):.3f}")
        
        if 'meta_learning_summary' in final_perf:
            meta_summary = final_perf['meta_learning_summary']
            print(f"   Total Decisions: {meta_summary.get('total_trades', 0)}")
            print(f"   Recent Sharpe: {meta_summary.get('recent_sharpe', 0):.3f}")
        
        # Regime distribution
        print(f"\n🌍 Market Regime Distribution:")
        total_episodes = sum(session_stats['regime_distributions'].values())
        for regime, count in session_stats['regime_distributions'].items():
            percentage = (count / total_episodes) * 100 if total_episodes > 0 else 0
            print(f"   {regime}: {count} episodes ({percentage:.1f}%)")
        
        # Agent performance comparison
        if 'agent_performances' in final_perf:
            print(f"\n🤖 Agent Performance Comparison:")
            for agent_name, perf in final_perf['agent_performances'].items():
                print(f"   {agent_name}: Sharpe={perf['sharpe_ratio']:.3f}, "
                      f"Win={perf['win_rate']:.2%}, Conf={perf['confidence']:.3f}")
        
        print(f"\n💾 Models saved to: {self.save_dir}")
        print(f"="*80)
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint"""
        
        checkpoint_dir = os.path.join(self.save_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save individual agents
        agents_dir = os.path.join(checkpoint_dir, "agents")
        os.makedirs(agents_dir, exist_ok=True)
        
        for agent_name, wrapper in self.agent_wrappers.items():
            try:
                agent_path = os.path.join(agents_dir, agent_name)
                os.makedirs(agent_path, exist_ok=True)
                wrapper.agent.save_or_load_agent(agent_path, if_save=True)
            except Exception as e:
                print(f"⚠️ Failed to save agent {agent_name}: {e}")
        
        # Save meta-learning models
        try:
            self.meta_learning_manager.save_meta_models(checkpoint_dir)
        except Exception as e:
            print(f"⚠️ Failed to save meta-learning models: {e}")
        
        # Save training statistics
        stats_path = os.path.join(checkpoint_dir, "training_stats.json")
        try:
            with open(stats_path, 'w') as f:
                # Make stats JSON serializable
                serializable_stats = self._make_json_serializable({
                    'training_stats': self.training_stats,
                    'config_summary': {
                        'meta_learning_enabled': self.config.meta_learning_enabled,
                        'meta_lookback': self.config.meta_lookback,
                        'max_agent_weight': self.config.max_agent_weight,
                        'regime_features_dim': self.config.regime_features_dim
                    }
                })
                json.dump(serializable_stats, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save training statistics: {e}")
    
    def _save_best_models(self, episode_num: int):
        """Save best performing models"""
        best_dir = os.path.join(self.save_dir, "best_models")
        self._save_checkpoint("best_models")
        
        # Save additional info about the best performance
        best_info_path = os.path.join(best_dir, "best_performance_info.json")
        best_info = {
            'episode': episode_num,
            'reward': self.training_stats['best_performance'],
            'timestamp': time.time(),
            'total_steps': self.training_stats['total_steps']
        }
        
        try:
            with open(best_info_path, 'w') as f:
                json.dump(best_info, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save best performance info: {e}")
    
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
        elif isinstance(obj, deque):
            return list(obj)
        else:
            return obj


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Meta-Learning Ensemble Training")
    
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes (default: 100)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--team_name', type=str, default='meta_learning_ensemble',
                        help='Team name for saving models (default: meta_learning_ensemble)')
    parser.add_argument('--config_preset', type=str, default='balanced',
                        choices=['conservative', 'aggressive', 'balanced', 'research'],
                        help='Configuration preset (default: balanced)')
    parser.add_argument('--save_dir', type=str, default='./meta_learning_models',
                        help='Directory for saving models (default: ./meta_learning_models)')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Episode interval for saving checkpoints (default: 20)')
    parser.add_argument('--evaluation_interval', type=int, default=10,
                        help='Episode interval for evaluation (default: 10)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    
    print("🧠 Meta-Learning Ensemble Training System")
    print("="*50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create meta-learning configuration
    env_args = {
        'env_name': 'TradeSimulator-v0',
        'state_dim': 50,  # Will be updated from TradeSimulator
        'action_dim': 3,
        'if_discrete': True
    }
    
    config = create_meta_learning_config(
        preset=args.config_preset,
        env_args=env_args,
        custom_params={
            'gpu_id': args.gpu_id,
            'net_dims': [512, 512]
        }
    )
    
    # Create trainer
    trainer = MetaLearningEnsembleTrainer(
        config=config,
        team_name=args.team_name,
        save_dir=args.save_dir
    )
    
    # Train the ensemble
    try:
        session_stats = trainer.train_full_session(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            evaluation_interval=args.evaluation_interval
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Models saved in: {args.save_dir}")
        
        return session_stats
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        trainer._save_checkpoint("interrupted")
        print(f"💾 Emergency checkpoint saved")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save emergency checkpoint
        try:
            trainer._save_checkpoint("error_checkpoint")
            print(f"💾 Emergency checkpoint saved")
        except Exception as save_error:
            print(f"❌ Failed to save emergency checkpoint: {save_error}")


if __name__ == "__main__":
    main()