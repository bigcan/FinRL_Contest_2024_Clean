"""
Full 500-Episode Training with Advanced Early Stopping
Complete implementation of long training sessions with sophisticated convergence detection
"""

import os
import time
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
import json
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedEarlyStoppingTrainer:
    """Advanced training with multiple early stopping criteria"""
    
    def __init__(self, save_path, agent_classes, args: Config):
        self.save_path = save_path
        self.agent_classes = agent_classes
        self.args = args
        self.agents = []
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Advanced early stopping parameters
        self.early_stopping_config = {
            'patience': 75,                    # Episodes to wait for improvement
            'min_delta': 0.001,               # Minimum change to qualify as improvement
            'monitor_metric': 'validation',    # 'validation', 'training', or 'combined'
            'plateau_patience': 40,           # Episodes to detect performance plateau
            'convergence_window': 30,         # Window for convergence detection
            'min_episodes': 150,              # Minimum episodes before early stopping
            'max_episodes': 500,              # Maximum episodes
            'validation_frequency': 5,        # Evaluate validation every N episodes
        }
        
        # Performance tracking
        self.training_metrics = {
            'training_scores': [],
            'validation_scores': [],
            'training_losses': [],
            'action_diversities': [],
            'learning_rates': [],
            'episode_durations': [],
            'convergence_indicators': [],
            'plateau_detections': [],
            'early_stop_reasons': {}
        }
        
        print(f"🎯 Advanced Early Stopping Configuration:")
        for key, value in self.early_stopping_config.items():
            print(f"   {key}: {value}")
    
    def train_full_ensemble(self):
        """Train complete ensemble with 500-episode capability"""
        print(f"\n🚀 Starting Full 500-Episode Ensemble Training")
        print(f"   Agents: {[cls.__name__ for cls in self.agent_classes]}")
        
        ensemble_start_time = time.time()
        
        for i, agent_class in enumerate(self.agent_classes):
            print(f"\n{'='*60}")
            print(f"📊 Training Agent {i+1}/{len(self.agent_classes)}: {agent_class.__name__}")
            print(f"{'='*60}")
            
            agent_start_time = time.time()
            
            # Train individual agent with advanced early stopping
            agent, training_info = self._train_agent_with_advanced_stopping(agent_class, i)
            
            agent_duration = time.time() - agent_start_time
            
            self.agents.append(agent)
            
            # Store comprehensive training information
            self.training_metrics['early_stop_reasons'][agent_class.__name__] = training_info
            
            print(f"✅ {agent_class.__name__} completed in {agent_duration/60:.1f} minutes")
            print(f"   Episodes trained: {training_info['episodes_completed']}")
            print(f"   Stop reason: {training_info['stop_reason']}")
            print(f"   Best validation score: {training_info['best_validation_score']:.4f}")
        
        ensemble_duration = time.time() - ensemble_start_time
        
        # Save comprehensive results
        self._save_comprehensive_results(ensemble_duration)
        
        # Final evaluation
        self._comprehensive_ensemble_evaluation()
        
        print(f"\n🎉 Full Ensemble Training Completed!")
        print(f"   Total time: {ensemble_duration/60:.1f} minutes")
        print(f"   Models saved to: {self.save_path}")
        
        return self.agents
    
    def _train_agent_with_advanced_stopping(self, agent_class, agent_idx):
        """Train single agent with sophisticated early stopping"""
        
        # Setup agent-specific configuration
        args = self._setup_agent_config(agent_class)
        args.init_before_training()
        torch.set_grad_enabled(False)
        
        # Build environments
        env = build_env(args.env_class, args.env_args, args.gpu_id)
        eval_env = build_env(args.eval_env_class, args.eval_env_args, args.gpu_id)
        
        # Initialize agent
        agent = args.agent_class(
            args.net_dims,
            args.state_dim,
            args.action_dim,
            gpu_id=args.gpu_id,
            args=args,
        )
        
        # Initialize state
        state = env.reset()
        if args.num_envs == 1:
            state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            state = state.to(agent.device)
        agent.last_state = state.detach()
        
        # Initialize replay buffer
        if args.if_off_policy:
            buffer = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs,
                max_size=args.buffer_size,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
            )
            # Pre-fill buffer
            buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
            buffer.update(buffer_items)
        else:
            buffer = []
        
        # Initialize evaluator
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)
        
        # Advanced tracking variables
        episode_count = 0
        best_validation_score = -np.inf
        best_training_score = -np.inf
        patience_counter = 0
        plateau_counter = 0
        
        # Performance tracking
        training_scores = []
        validation_scores = []
        losses = []
        action_diversities = []
        learning_rates = []
        episode_durations = []
        
        # Convergence tracking
        convergence_window = deque(maxlen=self.early_stopping_config['convergence_window'])
        validation_window = deque(maxlen=self.early_stopping_config['plateau_patience'])
        
        print(f"   🔧 Starting training loop with advanced early stopping...")
        
        while episode_count < self.early_stopping_config['max_episodes']:
            episode_start_time = time.time()
            
            # Training step
            buffer_items = agent.explore_env(env, args.horizon_len)
            
            # Action diversity analysis
            actions = buffer_items[1].flatten()
            action_counts = torch.bincount(actions).cpu().numpy()
            if len(action_counts) > 1:
                action_diversity = 1.0 - np.max(action_counts) / np.sum(action_counts)
            else:
                action_diversity = 0.0
            action_diversities.append(action_diversity)
            
            exp_r = buffer_items[2].mean().item()
            if args.if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items
            
            # Update network
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
            
            # Extract loss information
            if logging_tuple and len(logging_tuple) > 0:
                current_loss = logging_tuple[0] if isinstance(logging_tuple[0], (int, float)) else 0.0
            else:
                current_loss = 0.0
            losses.append(current_loss)
            
            # Get current learning rate
            if hasattr(agent, 'optimizer') and hasattr(agent.optimizer, 'param_groups'):
                current_lr = agent.optimizer.param_groups[0]['lr']
            else:
                current_lr = args.learning_rate
            learning_rates.append(current_lr)
            
            # Training evaluation
            eval_result = evaluator.evaluate_and_save(
                actor=agent.act,
                steps=args.horizon_len,
                exp_r=exp_r,
                logging_tuple=logging_tuple,
            )
            
            current_training_score = evaluator.recorder[0][-1] if evaluator.recorder[0] else exp_r
            training_scores.append(current_training_score)
            convergence_window.append(current_training_score)
            
            episode_duration = time.time() - episode_start_time
            episode_durations.append(episode_duration)
            
            # Validation evaluation
            validation_score = None
            if episode_count % self.early_stopping_config['validation_frequency'] == 0:
                validation_score = self._evaluate_validation(agent, eval_env)
                validation_scores.append(validation_score)
                validation_window.append(validation_score)
                
                # Track best scores
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    patience_counter = 0
                    self._save_best_model(agent, args.cwd, f"best_validation_{agent_class.__name__}")
                else:
                    patience_counter += 1
                
                if current_training_score > best_training_score:
                    best_training_score = current_training_score
                    self._save_best_model(agent, args.cwd, f"best_training_{agent_class.__name__}")
            
            # Progress reporting
            if validation_score is not None:
                print(f"   Episode {episode_count:3d}: Train={current_training_score:.4f}, "
                      f"Val={validation_score:.4f}, Loss={current_loss:.4f}, "
                      f"ActionDiv={action_diversity:.3f}, Time={episode_duration:.1f}s")
            else:
                print(f"   Episode {episode_count:3d}: Train={current_training_score:.4f}, "
                      f"Loss={current_loss:.4f}, ActionDiv={action_diversity:.3f}, "
                      f"Time={episode_duration:.1f}s")
            
            # Advanced early stopping checks
            stop_reason = self._check_early_stopping_conditions(
                episode_count, patience_counter, plateau_counter,
                convergence_window, validation_window, training_scores
            )
            
            if stop_reason:
                print(f"   🛑 Early stopping triggered: {stop_reason}")
                break
            
            episode_count += 1
            
            # Manual stop check
            if os.path.exists(f"{args.cwd}/stop"):
                stop_reason = "Manual stop requested"
                print(f"   🛑 {stop_reason}")
                break
            
            # Checkpoint saving
            if episode_count % 50 == 0:
                checkpoint_path = os.path.join(args.cwd, f"checkpoint_ep{episode_count}")
                self._save_checkpoint(agent, checkpoint_path, episode_count, {
                    'training_scores': training_scores,
                    'validation_scores': validation_scores,
                    'losses': losses,
                    'best_validation_score': best_validation_score
                })
                print(f"   💾 Checkpoint saved at episode {episode_count}")
        
        # Training completed
        final_duration = time.time() - evaluator.start_time
        print(f"   ✅ Training completed: {episode_count} episodes in {final_duration/60:.1f} minutes")
        
        # Load best model
        best_model_path = os.path.join(args.cwd, f"best_validation_{agent_class.__name__}")
        if os.path.exists(best_model_path):
            agent.save_or_load_agent(best_model_path, if_save=False)
            print(f"   📥 Loaded best validation model")
        
        # Save final model
        agent.save_or_load_agent(args.cwd, if_save=True)
        evaluator.save_training_curve_jpg()
        
        # Cleanup
        env.close() if hasattr(env, "close") else None
        eval_env.close() if hasattr(eval_env, "close") else None
        
        # Compile training information
        training_info = {
            'episodes_completed': episode_count,
            'stop_reason': stop_reason or 'Max episodes reached',
            'best_validation_score': best_validation_score,
            'best_training_score': best_training_score,
            'final_training_score': training_scores[-1] if training_scores else 0.0,
            'final_validation_score': validation_scores[-1] if validation_scores else 0.0,
            'training_duration': final_duration,
            'average_episode_duration': np.mean(episode_durations) if episode_durations else 0.0,
            'final_action_diversity': action_diversities[-1] if action_diversities else 0.0,
            'training_scores': training_scores,
            'validation_scores': validation_scores,
            'losses': losses,
            'action_diversities': action_diversities,
            'learning_rates': learning_rates,
            'episode_durations': episode_durations
        }
        
        return agent, training_info
    
    def _check_early_stopping_conditions(self, episode_count, patience_counter, plateau_counter,
                                        convergence_window, validation_window, training_scores):
        """Check multiple early stopping conditions"""
        
        # Must meet minimum episodes requirement
        if episode_count < self.early_stopping_config['min_episodes']:
            return None
        
        # Patience-based early stopping
        if patience_counter >= self.early_stopping_config['patience']:
            return f"Patience exceeded ({patience_counter} episodes without improvement)"
        
        # Convergence detection
        if len(convergence_window) >= self.early_stopping_config['convergence_window']:
            convergence_check = self._detect_convergence(list(convergence_window))
            if convergence_check:
                return f"Training converged ({convergence_check})"
        
        # Performance plateau detection
        if len(validation_window) >= self.early_stopping_config['plateau_patience']:
            plateau_check = self._detect_performance_plateau(list(validation_window))
            if plateau_check:
                return f"Performance plateau detected ({plateau_check})"
        
        # Catastrophic performance degradation
        if len(training_scores) >= 20:
            recent_scores = training_scores[-20:]
            early_scores = training_scores[:20]
            if np.mean(recent_scores) < np.mean(early_scores) * 0.5:
                return "Catastrophic performance degradation detected"
        
        return None
    
    def _detect_convergence(self, scores):
        """Detect if training has converged using statistical tests"""
        if len(scores) < 10:
            return False
        
        # Check for minimal variance (training has stabilized)
        recent_std = np.std(scores[-10:])
        if recent_std < self.early_stopping_config['min_delta']:
            return f"Low variance (std={recent_std:.4f})"
        
        # Trend analysis - check if slope is near zero
        x = np.arange(len(scores))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
            if abs(slope) < self.early_stopping_config['min_delta'] / len(scores):
                return f"Flat trend (slope={slope:.6f})"
        except:
            pass
        
        return False
    
    def _detect_performance_plateau(self, scores):
        """Detect performance plateau using trend analysis"""
        if len(scores) < 10:
            return False
        
        # Check if recent performance is not improving
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        if np.mean(second_half) <= np.mean(first_half) + self.early_stopping_config['min_delta']:
            return f"No improvement (recent={np.mean(second_half):.4f} vs early={np.mean(first_half):.4f})"
        
        # Check for sustained decline
        if len(scores) >= 20:
            recent_trend = np.polyfit(range(10), scores[-10:], 1)[0]
            if recent_trend < -self.early_stopping_config['min_delta']:
                return f"Declining trend (slope={recent_trend:.4f})"
        
        return False
    
    def _evaluate_validation(self, agent, eval_env):
        """Comprehensive validation evaluation"""
        try:
            state = eval_env.reset()
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            total_reward = 0.0
            steps = 0
            max_eval_steps = 200  # More comprehensive evaluation
            
            for _ in range(max_eval_steps):
                with torch.no_grad():
                    action = agent.select_action(state)
                
                state, reward, done, _ = eval_env.step(action)
                
                if isinstance(state, np.ndarray):
                    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                
                reward_val = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                total_reward += reward_val
                steps += 1
                
                if done[0] if isinstance(done, (list, np.ndarray)) else done:
                    break
            
            return total_reward / max(steps, 1)
            
        except Exception as e:
            print(f"   ⚠️ Validation evaluation failed: {e}")
            return 0.0
    
    def _save_best_model(self, agent, base_path, model_name):
        """Save best performing model"""
        best_path = os.path.join(base_path, model_name)
        os.makedirs(best_path, exist_ok=True)
        agent.save_or_load_agent(best_path, if_save=True)
    
    def _save_checkpoint(self, agent, checkpoint_path, episode, metrics):
        """Save training checkpoint"""
        os.makedirs(checkpoint_path, exist_ok=True)
        agent.save_or_load_agent(checkpoint_path, if_save=True)
        
        # Save metrics
        metrics_path = os.path.join(checkpoint_path, "metrics.json")
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_metrics[key] = [v.tolist() for v in value]
                else:
                    serializable_metrics[key] = value
            json.dump(serializable_metrics, f, indent=2)
    
    def _setup_agent_config(self, agent_class):
        """Setup optimized configuration for specific agent"""
        args = self.args
        args.agent_class = agent_class
        
        # Agent-specific optimizations
        if agent_class.__name__ == "AgentD3QN":
            args.learning_rate = 8e-6
            args.gamma = 0.996
            args.explore_rate = 0.012
        elif agent_class.__name__ == "AgentDoubleDQN":
            args.learning_rate = 6e-6
            args.gamma = 0.995
            args.explore_rate = 0.015
        elif agent_class.__name__ == "AgentTwinD3QN":
            args.learning_rate = 1e-5
            args.gamma = 0.997
            args.explore_rate = 0.010
        
        return args
    
    def _save_comprehensive_results(self, ensemble_duration):
        """Save comprehensive training results and visualizations"""
        os.makedirs(self.save_path, exist_ok=True)
        
        # Save training metrics
        metrics_path = os.path.join(self.save_path, "comprehensive_metrics.json")
        with open(metrics_path, 'w') as f:
            # Prepare serializable data
            serializable_metrics = {}
            for key, value in self.training_metrics.items():
                if isinstance(value, dict):
                    serializable_metrics[key] = {}
                    for subkey, subvalue in value.items():
                        if hasattr(subvalue, 'tolist'):
                            serializable_metrics[key][subkey] = subvalue.tolist()
                        else:
                            serializable_metrics[key][subkey] = subvalue
                elif hasattr(value, 'tolist'):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value
            
            serializable_metrics['ensemble_duration'] = ensemble_duration
            json.dump(serializable_metrics, f, indent=2)
        
        # Save detailed agent information
        detailed_path = os.path.join(self.save_path, "detailed_training_info.pkl")
        with open(detailed_path, 'wb') as f:
            pickle.dump(self.training_metrics, f)
        
        # Generate training visualizations
        self._generate_training_visualizations()
        
        print(f"📊 Comprehensive results saved to: {self.save_path}")
        
    def _generate_training_visualizations(self):
        """Generate comprehensive training visualizations"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Full 500-Episode Training Analysis', fontsize=16)
            
            # Extract data for visualization
            agent_names = list(self.training_metrics['early_stop_reasons'].keys())
            
            for i, agent_name in enumerate(agent_names):
                agent_data = self.training_metrics['early_stop_reasons'][agent_name]
                
                # Training and validation scores
                if 'training_scores' in agent_data and agent_data['training_scores']:
                    axes[0, 0].plot(agent_data['training_scores'], label=f"{agent_name} Training", alpha=0.7)
                if 'validation_scores' in agent_data and agent_data['validation_scores']:
                    val_episodes = range(0, len(agent_data['training_scores']), 5)[:len(agent_data['validation_scores'])]
                    axes[0, 0].plot(val_episodes, agent_data['validation_scores'], 
                                   label=f"{agent_name} Validation", linestyle='--', alpha=0.7)
            
            axes[0, 0].set_title('Training & Validation Scores')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Loss curves
            for i, agent_name in enumerate(agent_names):
                agent_data = self.training_metrics['early_stop_reasons'][agent_name]
                if 'losses' in agent_data and agent_data['losses']:
                    axes[0, 1].plot(agent_data['losses'], label=f"{agent_name}", alpha=0.7)
            
            axes[0, 1].set_title('Training Losses')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Action diversity
            for i, agent_name in enumerate(agent_names):
                agent_data = self.training_metrics['early_stop_reasons'][agent_name]
                if 'action_diversities' in agent_data and agent_data['action_diversities']:
                    axes[0, 2].plot(agent_data['action_diversities'], label=f"{agent_name}", alpha=0.7)
            
            axes[0, 2].set_title('Action Diversity')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Diversity')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # Episode durations
            for i, agent_name in enumerate(agent_names):
                agent_data = self.training_metrics['early_stop_reasons'][agent_name]
                if 'episode_durations' in agent_data and agent_data['episode_durations']:
                    axes[1, 0].plot(agent_data['episode_durations'], label=f"{agent_name}", alpha=0.7)
            
            axes[1, 0].set_title('Episode Durations')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Duration (s)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Learning rates
            for i, agent_name in enumerate(agent_names):
                agent_data = self.training_metrics['early_stop_reasons'][agent_name]
                if 'learning_rates' in agent_data and agent_data['learning_rates']:
                    axes[1, 1].plot(agent_data['learning_rates'], label=f"{agent_name}", alpha=0.7)
            
            axes[1, 1].set_title('Learning Rates')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Performance summary
            episodes_completed = [self.training_metrics['early_stop_reasons'][name]['episodes_completed'] 
                                for name in agent_names]
            best_scores = [self.training_metrics['early_stop_reasons'][name]['best_validation_score'] 
                          for name in agent_names]
            
            bars = axes[1, 2].bar(range(len(agent_names)), episodes_completed, alpha=0.7)
            axes[1, 2].set_title('Episodes Completed')
            axes[1, 2].set_xlabel('Agent')
            axes[1, 2].set_ylabel('Episodes')
            axes[1, 2].set_xticks(range(len(agent_names)))
            axes[1, 2].set_xticklabels([name.replace('Agent', '') for name in agent_names], rotation=45)
            
            # Add best scores as text on bars
            for i, (bar, score) in enumerate(zip(bars, best_scores)):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 5,
                               f'Best: {score:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            viz_path = os.path.join(self.save_path, "training_analysis.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Training visualizations saved to: {viz_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to generate visualizations: {e}")
    
    def _comprehensive_ensemble_evaluation(self):
        """Comprehensive evaluation of the trained ensemble"""
        print(f"\n📊 Comprehensive Ensemble Evaluation")
        print(f"{'='*60}")
        
        total_episodes = sum(info['episodes_completed'] 
                           for info in self.training_metrics['early_stop_reasons'].values())
        total_time = sum(info['training_duration'] 
                        for info in self.training_metrics['early_stop_reasons'].values())
        
        print(f"📈 Overall Statistics:")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Total Training Time: {total_time/60:.1f} minutes")
        print(f"   Average Time per Episode: {total_time/max(total_episodes,1):.2f} seconds")
        
        print(f"\n🤖 Individual Agent Performance:")
        for agent_name, info in self.training_metrics['early_stop_reasons'].items():
            print(f"   {agent_name}:")
            print(f"     Episodes: {info['episodes_completed']}")
            print(f"     Stop Reason: {info['stop_reason']}")
            print(f"     Best Validation: {info['best_validation_score']:.4f}")
            print(f"     Final Training: {info['final_training_score']:.4f}")
            print(f"     Training Time: {info['training_duration']/60:.1f} min")
            print(f"     Avg Episode Time: {info['average_episode_duration']:.1f}s")
        
        # Find best performing agent
        best_agent = max(self.training_metrics['early_stop_reasons'].items(),
                        key=lambda x: x[1]['best_validation_score'])
        
        print(f"\n🏆 Best Performing Agent: {best_agent[0]}")
        print(f"   Best Validation Score: {best_agent[1]['best_validation_score']:.4f}")
        print(f"   Episodes to Best: {best_agent[1]['episodes_completed']}")

def run_full_500_episode_training():
    """Run complete 500-episode training with advanced early stopping"""
    import sys
    
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    print(f"🚀 Full 500-Episode Training with Advanced Early Stopping")
    print(f"🖥️  Device: {'GPU ' + str(gpu_id) if gpu_id >= 0 else 'CPU'}")
    
    # Enhanced configuration for long training
    num_sims = 12  # Optimized for stability
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7
    
    max_step = (4800 - num_ignore_step) // step_gap
    
    # Get actual state dimension
    temp_sim = TradeSimulator(num_sims=1)
    actual_state_dim = temp_sim.state_dim
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": actual_state_dim,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
    }
    
    print(f"📊 Training Configuration:")
    print(f"   State Dimension: {actual_state_dim}")
    print(f"   Parallel Environments: {num_sims}")
    print(f"   Max Steps per Episode: {max_step}")
    print(f"   Maximum Episodes: 500")
    
    # Create configuration
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    
    # Optimized network architecture
    if actual_state_dim <= 8:
        args.net_dims = (128, 64, 32)
    else:
        args.net_dims = (256, 128, 64)
    
    # Long training hyperparameters
    args.gamma = 0.996
    args.explore_rate = 0.01
    args.state_value_tau = 0.005
    args.soft_update_tau = 5e-6
    args.learning_rate = 8e-6  # Conservative for stability
    args.batch_size = 256
    args.buffer_size = int(max_step * 15)
    args.repeat_times = 1.0
    args.horizon_len = int(max_step * 2)
    args.eval_per_step = int(max_step // 2)
    args.num_workers = 1
    args.save_gap = 32
    
    # Evaluation environment
    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = env_args.copy()
    
    print(f"🔧 Hyperparameters:")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Buffer Size: {args.buffer_size}")
    print(f"   Gamma: {args.gamma}")
    
    # Agent list for ensemble
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    
    # Create trainer
    trainer = AdvancedEarlyStoppingTrainer(
        save_path="ensemble_full_500_episode_training",
        agent_classes=agent_list,
        args=args
    )
    
    # Run training
    agents = trainer.train_full_ensemble()
    
    return agents

if __name__ == "__main__":
    run_full_500_episode_training()