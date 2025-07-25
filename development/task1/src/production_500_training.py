#!/usr/bin/env python3
"""
Production-Level 500-Episode Training Session
State-of-the-art implementation with comprehensive monitoring and checkpointing
"""

import os
import sys
import time
import torch
import numpy as np
import json
import pickle
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from enhanced_training_config import EnhancedConfig, EarlyStoppingManager, TrainingMetricsTracker
from optimized_hyperparameters import get_optimized_hyperparameters, apply_optimized_hyperparameters
from erl_config import build_env
from erl_replay_buffer import ReplayBuffer


class Production500TrainingManager:
    """
    Production-level training manager for 500-episode sessions
    Features comprehensive monitoring, checkpointing, and robust error handling
    """
    
    def __init__(self, 
                 reward_type: str = "simple",
                 gpu_id: int = 0,
                 team_name: str = "production_500_training",
                 max_episodes: int = 500):
        """
        Initialize production training manager
        
        Args:
            reward_type: Reward function to use
            gpu_id: GPU device ID
            team_name: Team name for saving models
            max_episodes: Maximum number of episodes to train
        """
        self.reward_type = reward_type
        self.gpu_id = gpu_id
        self.team_name = f"{team_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_episodes = max_episodes
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")
        
        # Get state dimension
        temp_sim = TradeSimulator(num_sims=1)
        self.state_dim = temp_sim.state_dim
        temp_sim.set_reward_type(reward_type)
        
        # Create directories
        self.base_dir = Path(f"production_training_results/{self.team_name}")
        self.models_dir = self.base_dir / "models"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.logs_dir = self.base_dir / "logs"
        self.metrics_dir = self.base_dir / "metrics"
        
        for dir_path in [self.models_dir, self.checkpoints_dir, self.logs_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Agent configurations
        self.agent_configs = self._setup_production_configs()
        
        # Training state
        self.training_state = {
            'session_start_time': time.time(),
            'current_agent': None,
            'current_episode': 0,
            'agents_completed': 0,
            'total_agents': len(self.agent_configs),
            'best_performance': -np.inf,
            'training_interrupted': False
        }
        
        self.logger.info(f"🚀 PRODUCTION 500-EPISODE TRAINING INITIALIZED")
        self.logger.info(f"   Reward type: {reward_type}")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   State dimension: {self.state_dim}")
        self.logger.info(f"   Maximum episodes: {max_episodes}")
        self.logger.info(f"   Agent configurations: {len(self.agent_configs)}")
        self.logger.info(f"   Results directory: {self.base_dir}")
        
        print(f"🌟 PRODUCTION 500-EPISODE TRAINING MANAGER")
        print("=" * 80)
        print(f"🎯 Reward type: {reward_type}")
        print(f"🖥️  Device: {self.device}")
        print(f"📊 State dimension: {self.state_dim}")
        print(f"🎲 Maximum episodes: {max_episodes}")
        print(f"🤖 Agent types: {len(self.agent_configs)}")
        print(f"📁 Results: {self.base_dir}")
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Main logger
        self.logger = logging.getLogger('Production500Training')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.logs_dir / "training_session.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Performance logger
        self.perf_logger = logging.getLogger('Performance')
        self.perf_logger.setLevel(logging.INFO)
        perf_file = self.logs_dir / "performance_metrics.log"
        perf_handler = logging.FileHandler(perf_file)
        perf_handler.setFormatter(formatter)
        self.perf_logger.addHandler(perf_handler)
        
        print(f"📝 Logging configured: {log_file}")
    
    def _setup_production_configs(self) -> Dict:
        """Setup production-level configurations for all agents"""
        
        # Environment arguments
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": 16,
            "max_step": 2370,
            "state_dim": self.state_dim,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "num_sims": 16,
            "step_gap": 2,
        }
        
        # Get optimized hyperparameters
        optimized_params = get_optimized_hyperparameters(self.reward_type)
        
        configs = {}
        
        # Agent specifications with production parameters
        agent_specs = [
            ("AgentD3QN", AgentD3QN, {
                'learning_rate': 6e-6,
                'gamma': 0.997,
                'explore_rate': 0.008,
                'batch_size': 512,
                'buffer_size': 200000,
                'net_dims': (256, 128, 64)
            }),
            ("AgentDoubleDQN", AgentDoubleDQN, {
                'learning_rate': 5e-6,
                'gamma': 0.996,
                'explore_rate': 0.010,
                'batch_size': 512,
                'buffer_size': 200000,
                'net_dims': (256, 128, 64)
            }),
            ("AgentTwinD3QN", AgentTwinD3QN, {
                'learning_rate': 7e-6,
                'gamma': 0.998,
                'explore_rate': 0.006,
                'batch_size': 512,
                'buffer_size': 200000,
                'net_dims': (256, 128, 64)
            })
        ]
        
        for agent_name, agent_class, agent_params in agent_specs:
            config = EnhancedConfig(agent_class=agent_class, env_class=TradeSimulator, env_args=env_args)
            config.gpu_id = self.gpu_id
            config.random_seed = 42
            config.state_dim = self.state_dim
            
            # Production-level training parameters
            config.break_step = self.max_episodes
            config.max_training_time = 14400  # 4 hours max per agent
            
            # Enhanced early stopping for 500 episodes
            config.early_stopping_patience = 100  # More patience for longer training
            config.early_stopping_min_delta = 0.0005  # Finer improvement threshold
            config.early_stopping_enabled = True
            
            # Evaluation settings
            config.eval_per_step = 25  # Evaluate every 25 episodes
            config.eval_times = 8  # More thorough evaluation
            
            # Apply agent-specific parameters
            for param, value in agent_params.items():
                setattr(config, param, value)
            
            # Apply base optimized hyperparameters
            config = apply_optimized_hyperparameters(config, optimized_params, env_args)
            
            configs[agent_name] = {
                'config': config,
                'agent_class': agent_class,
                'agent_params': agent_params,
                'env_args': env_args
            }
            
            self.logger.info(f"   {agent_name} configured: LR={agent_params['learning_rate']:.2e}, "
                           f"Gamma={agent_params['gamma']}, Explore={agent_params['explore_rate']}")
        
        return configs
    
    def train_agent_production(self, 
                             agent_name: str, 
                             agent_config: Dict) -> Dict:
        """
        Train single agent with production-level monitoring and checkpointing
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration dictionary
            
        Returns:
            Comprehensive training results
        """
        self.logger.info(f"🚀 Starting production training: {agent_name}")
        self.training_state['current_agent'] = agent_name
        self.training_state['current_episode'] = 0
        
        training_start = time.time()
        
        try:
            config = agent_config['config']
            agent_class = agent_config['agent_class']
            env_args = agent_config['env_args']
            
            # Setup agent-specific directories
            agent_dir = self.models_dir / agent_name.lower()
            agent_checkpoints = self.checkpoints_dir / agent_name.lower()
            agent_metrics = self.metrics_dir / agent_name.lower()
            
            for dir_path in [agent_dir, agent_checkpoints, agent_metrics]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Build environment
            env = build_env(config.env_class, env_args, config.gpu_id)
            env.set_reward_type(self.reward_type)
            
            # Create agent
            agent = agent_class(
                config.net_dims,
                config.state_dim,
                config.action_dim,
                gpu_id=config.gpu_id,
                args=config,
            )
            
            # Initialize state
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state = state.to(agent.device)
            agent.last_state = state.detach()
            
            # Create replay buffer
            buffer = ReplayBuffer(
                gpu_id=config.gpu_id,
                num_seqs=config.num_envs,
                max_size=config.buffer_size,
                state_dim=config.state_dim,
                action_dim=1,
            )
            
            # Initialize systems
            early_stopping = EarlyStoppingManager(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                monitor="validation_score",
                mode="max"
            )
            
            metrics_tracker = TrainingMetricsTracker(history_size=self.max_episodes)
            
            # Warm up buffer
            self.logger.info(f"   🔄 Warming up replay buffer...")
            buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
            buffer.update(buffer_items)
            
            # Training metrics
            episode_rewards = []
            validation_scores = []
            training_losses = []
            best_validation = -np.inf
            best_episode = 0
            checkpoint_interval = 50
            
            self.logger.info(f"   🏋️  Starting {self.max_episodes}-episode training...")
            print(f"\n🎯 Training {agent_name} for up to {self.max_episodes} episodes")
            
            # Main training loop
            for episode in range(self.max_episodes):
                episode_start = time.time()
                self.training_state['current_episode'] = episode
                
                # Check for interruption
                if self._check_interruption():
                    self.logger.warning(f"   ⚠️ Training interrupted at episode {episode}")
                    break
                
                # Collect experience
                buffer_items = agent.explore_env(env, config.horizon_len)
                episode_reward = buffer_items[2].mean().item()
                episode_rewards.append(episode_reward)
                
                # Update buffer
                buffer.update(buffer_items)
                
                # Update networks
                torch.set_grad_enabled(True)
                logging_tuple = agent.update_net(buffer)
                torch.set_grad_enabled(False)
                
                # Extract loss
                training_loss = logging_tuple[0] if (logging_tuple and len(logging_tuple) > 0) else 0.0
                training_losses.append(training_loss)
                
                # Periodic validation
                validation_score = None
                if episode % config.eval_per_step == 0:
                    validation_score = self._comprehensive_validation(agent, env)
                    validation_scores.append((episode, validation_score))
                    
                    # Update metrics tracker
                    metrics_tracker.update(
                        step=episode,
                        reward=episode_reward,
                        loss_critic=training_loss,
                        eval_score=validation_score,
                        exploration_rate=config.explore_rate,
                        learning_rate=config.learning_rate
                    )
                    
                    # Check for best performance
                    if validation_score > best_validation:
                        best_validation = validation_score
                        best_episode = episode
                        self._save_best_model(agent, agent_dir / "best_model")
                        self.logger.info(f"   ✨ New best validation: {validation_score:.6f} at episode {episode}")
                    
                    # Early stopping check
                    if early_stopping.update(validation_score, episode):
                        self.logger.info(f"   🛑 Early stopping triggered at episode {episode}")
                        break
                    
                    # Performance logging
                    self.perf_logger.info(f"{agent_name} Episode {episode}: "
                                        f"Reward={episode_reward:.4f}, "
                                        f"Validation={validation_score:.4f}, "
                                        f"Loss={training_loss:.6f}")
                
                # Progress reporting
                episode_time = time.time() - episode_start
                if episode % 10 == 0:
                    recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
                    total_time = time.time() - training_start
                    
                    progress_msg = (f"   Episode {episode:3d}/{self.max_episodes}: "
                                  f"Reward={episode_reward:.4f}, "
                                  f"Recent={recent_reward:.4f}, "
                                  f"Time={episode_time:.1f}s, "
                                  f"Total={total_time/60:.1f}min")
                    
                    if validation_score is not None:
                        progress_msg += f", Val={validation_score:.4f}"
                    
                    print(progress_msg)
                    self.logger.info(progress_msg)
                
                # Checkpointing
                if (episode + 1) % checkpoint_interval == 0:
                    checkpoint_path = agent_checkpoints / f"episode_{episode+1:03d}"
                    self._save_checkpoint(agent, checkpoint_path, episode, {
                        'episode_rewards': episode_rewards,
                        'validation_scores': validation_scores,
                        'training_losses': training_losses,
                        'best_validation': best_validation,
                        'best_episode': best_episode
                    })
                    self.logger.info(f"   💾 Checkpoint saved: episode {episode+1}")
                
                # Check maximum training time
                if (time.time() - training_start) > config.max_training_time:
                    self.logger.warning(f"   ⏰ Maximum training time reached at episode {episode}")
                    break
            
            # Training completed
            final_episode = episode
            training_time = time.time() - training_start
            
            # Load best model
            best_model_path = agent_dir / "best_model"
            if best_model_path.exists():
                agent.save_or_load_agent(str(best_model_path), if_save=False)
                self.logger.info(f"   📥 Loaded best model from episode {best_episode}")
            
            # Save final model
            agent.save_or_load_agent(str(agent_dir), if_save=True)
            
            # Comprehensive final evaluation
            final_validation = self._comprehensive_validation(agent, env, num_runs=20)
            
            # Cleanup
            env.close() if hasattr(env, "close") else None
            
            # Compile results
            results = {
                'agent_name': agent_name,
                'success': True,
                'episodes_completed': final_episode + 1,
                'training_time': training_time,
                'best_validation_score': best_validation,
                'best_episode': best_episode,
                'final_validation_score': final_validation,
                'average_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
                'final_episode_reward': episode_rewards[-1] if episode_rewards else 0.0,
                'improvement': (episode_rewards[-1] - episode_rewards[0]) if len(episode_rewards) > 1 else 0.0,
                'training_stability': np.std(episode_rewards[-50:]) if len(episode_rewards) >= 50 else 0.0,
                'convergence_episode': best_episode,
                'early_stopping_triggered': early_stopping.should_stop,
                'total_validation_runs': len(validation_scores),
                'metrics_history': {
                    'episode_rewards': episode_rewards,
                    'validation_scores': validation_scores,
                    'training_losses': training_losses
                }
            }
            
            # Save detailed results
            results_file = agent_metrics / "training_results.json"
            with open(results_file, 'w') as f:
                # Prepare serializable data
                serializable_results = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[key] = value.tolist()
                    elif isinstance(value, dict):
                        serializable_results[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (list, tuple)) and len(subvalue) > 0:
                                if isinstance(subvalue[0], tuple):
                                    serializable_results[key][subkey] = [list(item) for item in subvalue]
                                else:
                                    serializable_results[key][subkey] = list(subvalue)
                            else:
                                serializable_results[key][subkey] = subvalue
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"✅ {agent_name} training completed successfully")
            self.logger.info(f"   Episodes: {final_episode + 1}, Time: {training_time/60:.1f}min")
            self.logger.info(f"   Best validation: {best_validation:.6f} at episode {best_episode}")
            self.logger.info(f"   Final validation: {final_validation:.6f}")
            
            print(f"✅ {agent_name} completed: {final_episode + 1} episodes in {training_time/60:.1f}min")
            print(f"   🏆 Best validation: {best_validation:.6f} (episode {best_episode})")
            print(f"   🎯 Final validation: {final_validation:.6f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ {agent_name} training failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                'agent_name': agent_name,
                'success': False,
                'error': str(e),
                'episodes_completed': self.training_state.get('current_episode', 0),
                'training_time': time.time() - training_start,
                'best_validation_score': -1000.0,
                'final_validation_score': -1000.0
            }
    
    def _comprehensive_validation(self, agent, env, num_runs: int = 10) -> float:
        """Run comprehensive validation evaluation"""
        try:
            total_scores = []
            
            for run in range(num_runs):
                state = env.reset()
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                state = state.to(agent.device)
                
                episode_reward = 0.0
                steps = 0
                max_steps = 200
                
                for step in range(max_steps):
                    with torch.no_grad():
                        q_values = agent.act(state)
                        action = q_values.argmax(dim=1, keepdim=True)
                    
                    next_state, reward, done, _ = env.step(action)
                    
                    if not isinstance(next_state, torch.Tensor):
                        next_state = torch.tensor(next_state, dtype=torch.float32)
                    next_state = next_state.to(agent.device)
                    
                    reward_val = reward.mean().item() if hasattr(reward, 'mean') else reward
                    episode_reward += reward_val
                    steps += 1
                    
                    if done.any() if hasattr(done, 'any') else done:
                        break
                    
                    state = next_state
                
                total_scores.append(episode_reward / max(steps, 1))
            
            return np.mean(total_scores)
            
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            return 0.0
    
    def _save_best_model(self, agent, model_path: Path):
        """Save the best performing model"""
        model_path.mkdir(parents=True, exist_ok=True)
        agent.save_or_load_agent(str(model_path), if_save=True)
    
    def _save_checkpoint(self, agent, checkpoint_path: Path, episode: int, metrics: Dict):
        """Save training checkpoint with comprehensive state"""
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        agent.save_or_load_agent(str(checkpoint_path), if_save=True)
        
        # Save metrics
        metrics_file = checkpoint_path / "checkpoint_metrics.json"
        checkpoint_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'training_state': self.training_state.copy(),
            'metrics': metrics
        }
        
        # Make serializable
        serializable_data = {}
        for key, value in checkpoint_data.items():
            if isinstance(value, dict):
                serializable_data[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (list, tuple)) and len(subvalue) > 0:
                        if isinstance(subvalue[0], tuple):
                            serializable_data[key][subkey] = [list(item) for item in subvalue]
                        else:
                            serializable_data[key][subkey] = list(subvalue)
                    else:
                        serializable_data[key][subkey] = subvalue
            else:
                serializable_data[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def _check_interruption(self) -> bool:
        """Check for training interruption signals"""
        stop_file = self.base_dir / "STOP_TRAINING"
        if stop_file.exists():
            self.training_state['training_interrupted'] = True
            return True
        return False
    
    def run_production_training(self) -> Dict:
        """Execute complete production-level 500-episode training"""
        
        self.logger.info(f"🚀 STARTING PRODUCTION 500-EPISODE TRAINING SESSION")
        self.logger.info(f"   Session ID: {self.team_name}")
        self.logger.info(f"   Maximum episodes per agent: {self.max_episodes}")
        self.logger.info(f"   Total agents: {len(self.agent_configs)}")
        
        print(f"\n🌟 PRODUCTION 500-EPISODE TRAINING SESSION")
        print("=" * 80)
        print(f"📅 Session: {self.team_name}")
        print(f"🎯 Max episodes: {self.max_episodes}")
        print(f"🤖 Agents: {list(self.agent_configs.keys())}")
        print(f"📊 Expected duration: 8-12 hours")
        
        session_start = time.time()
        all_results = {}
        
        # Train each agent
        for i, (agent_name, agent_config) in enumerate(self.agent_configs.items(), 1):
            print(f"\n{'='*60}")
            print(f"🤖 Training Agent {i}/{len(self.agent_configs)}: {agent_name}")
            print(f"{'='*60}")
            
            self.logger.info(f"Starting agent {i}/{len(self.agent_configs)}: {agent_name}")
            
            # Train agent
            result = self.train_agent_production(agent_name, agent_config)
            all_results[agent_name] = result
            
            self.training_state['agents_completed'] += 1
            
            # Update best performance
            if result.get('success', False):
                validation_score = result.get('best_validation_score', -np.inf)
                if validation_score > self.training_state['best_performance']:
                    self.training_state['best_performance'] = validation_score
            
            # Save session state
            self._save_session_state(all_results)
            
            # Check for interruption
            if self._check_interruption():
                self.logger.warning("Training session interrupted")
                break
            
            print(f"✅ Agent {i} completed. Progress: {i}/{len(self.agent_configs)}")
        
        # Session completed
        session_time = time.time() - session_start
        
        # Generate comprehensive report
        self._generate_session_report(all_results, session_time)
        
        self.logger.info(f"🎉 PRODUCTION TRAINING SESSION COMPLETED")
        self.logger.info(f"   Total time: {session_time/3600:.2f} hours")
        self.logger.info(f"   Agents completed: {self.training_state['agents_completed']}")
        self.logger.info(f"   Best performance: {self.training_state['best_performance']:.6f}")
        
        print(f"\n🎉 PRODUCTION TRAINING SESSION COMPLETED!")
        print(f"⏱️  Total time: {session_time/3600:.2f} hours")
        print(f"✅ Agents completed: {self.training_state['agents_completed']}/{len(self.agent_configs)}")
        print(f"🏆 Best performance: {self.training_state['best_performance']:.6f}")
        print(f"📁 Results saved to: {self.base_dir}")
        
        return all_results
    
    def _save_session_state(self, results: Dict):
        """Save current session state"""
        session_file = self.base_dir / "session_state.json"
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'training_state': self.training_state.copy(),
            'completed_agents': list(results.keys()),
            'session_config': {
                'reward_type': self.reward_type,
                'gpu_id': self.gpu_id,
                'max_episodes': self.max_episodes,
                'total_agents': len(self.agent_configs)
            }
        }
        
        # Add basic result summaries
        session_data['agent_summaries'] = {}
        for agent_name, result in results.items():
            session_data['agent_summaries'][agent_name] = {
                'success': result.get('success', False),
                'episodes_completed': result.get('episodes_completed', 0),
                'best_validation_score': result.get('best_validation_score', -1000.0),
                'training_time': result.get('training_time', 0.0)
            }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def _generate_session_report(self, results: Dict, session_time: float):
        """Generate comprehensive session report"""
        report_file = self.base_dir / "PRODUCTION_TRAINING_REPORT.md"
        
        # Calculate statistics
        successful_agents = [r for r in results.values() if r.get('success', False)]
        total_episodes = sum(r.get('episodes_completed', 0) for r in results.values())
        avg_episodes = total_episodes / len(results) if results else 0
        best_agent = max(successful_agents, key=lambda x: x.get('best_validation_score', -np.inf)) if successful_agents else None
        
        report_content = f"""# Production 500-Episode Training Report

## Session Overview
- **Session ID**: {self.team_name}
- **Start Time**: {datetime.fromtimestamp(self.training_state['session_start_time']).isoformat()}
- **Duration**: {session_time/3600:.2f} hours
- **Reward Type**: {self.reward_type}
- **Device**: {self.device}

## Training Results
- **Total Agents**: {len(results)}
- **Successful Agents**: {len(successful_agents)}/{len(results)}
- **Total Episodes**: {total_episodes:,}
- **Average Episodes per Agent**: {avg_episodes:.1f}
- **Success Rate**: {len(successful_agents)/len(results)*100:.1f}%

## Performance Summary
"""
        
        if best_agent:
            report_content += f"""
### Best Performing Agent
- **Agent**: {best_agent['agent_name']}
- **Best Validation Score**: {best_agent['best_validation_score']:.6f}
- **Episodes Completed**: {best_agent['episodes_completed']}
- **Training Time**: {best_agent['training_time']/3600:.2f} hours
- **Convergence Episode**: {best_agent.get('best_episode', 'N/A')}
"""
        
        report_content += f"""
## Individual Agent Results

| Agent | Success | Episodes | Best Score | Training Time | Final Score |
|-------|---------|----------|------------|---------------|-------------|
"""
        
        for agent_name, result in results.items():
            success_icon = "✅" if result.get('success', False) else "❌"
            episodes = result.get('episodes_completed', 0)
            best_score = result.get('best_validation_score', -1000.0)
            training_time = result.get('training_time', 0.0) / 3600
            final_score = result.get('final_validation_score', -1000.0)
            
            report_content += f"| {agent_name} | {success_icon} | {episodes} | {best_score:.4f} | {training_time:.2f}h | {final_score:.4f} |\n"
        
        report_content += f"""
## Technical Details
- **State Dimension**: {self.state_dim}
- **Maximum Episodes per Agent**: {self.max_episodes}
- **Early Stopping**: Enabled (patience=100, min_delta=0.0005)
- **Validation Frequency**: Every 25 episodes
- **Checkpoint Frequency**: Every 50 episodes

## Files Generated
- **Models**: `models/` directory
- **Checkpoints**: `checkpoints/` directory
- **Logs**: `logs/` directory
- **Metrics**: `metrics/` directory

## Recommendations
"""
        
        if successful_agents:
            report_content += f"""
1. **Deploy Best Model**: Use {best_agent['agent_name']} with validation score {best_agent['best_validation_score']:.6f}
2. **Ensemble Strategy**: Combine top {min(3, len(successful_agents))} performing agents
3. **Production Deployment**: Models are ready for live trading evaluation
"""
        else:
            report_content += f"""
1. **Debug Training**: Investigate why no agents completed successfully
2. **Hyperparameter Tuning**: Consider adjusting learning rates and exploration
3. **Environment Validation**: Verify trading environment setup
"""
        
        report_content += f"""
## Session Statistics
- **GPU Utilization**: CUDA device {self.gpu_id}
- **Memory Usage**: Optimized with checkpointing
- **Monitoring**: Comprehensive logging enabled
- **Error Handling**: Robust exception management

---
*Report generated on {datetime.now().isoformat()}*
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📋 Comprehensive report generated: {report_file}")
        print(f"📋 Report generated: {report_file}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production 500-Episode Training")
    parser.add_argument("gpu_id", type=int, help="GPU ID (0, 1, etc. or -1 for CPU)")
    parser.add_argument("--reward", type=str, default="simple", 
                        choices=["simple", "transaction_cost_adjusted", "multi_objective"],
                        help="Reward function type")
    parser.add_argument("--episodes", type=int, default=500, help="Maximum episodes per agent")
    parser.add_argument("--team", type=str, default="production_500", help="Team name")
    
    args = parser.parse_args()
    
    print(f"🚀 PRODUCTION 500-EPISODE TRAINING LAUNCH")
    print("=" * 80)
    print(f"Configuration:")
    print(f"   🖥️  GPU ID: {args.gpu_id}")
    print(f"   🎯 Reward Type: {args.reward}")
    print(f"   🎲 Max Episodes: {args.episodes}")
    print(f"   🏷️  Team Name: {args.team}")
    print(f"   ⏱️  Expected Duration: 8-12 hours")
    print(f"   🧠 Advanced Features: Early stopping, checkpointing, comprehensive monitoring")
    
    # Create production trainer
    trainer = Production500TrainingManager(
        reward_type=args.reward,
        gpu_id=args.gpu_id,
        team_name=args.team,
        max_episodes=args.episodes
    )
    
    # Execute training
    results = trainer.run_production_training()
    
    print(f"\n🎉 Production training completed!")
    print(f"📊 Check results in: {trainer.base_dir}")
    
    return results


if __name__ == "__main__":
    results = main()