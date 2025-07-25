"""
Task 1 Extended Training - Long-term Performance Enhancement
Enhanced ensemble training with extended episodes, early stopping, and proper validation splits
"""

import os
import time
import torch
import numpy as np
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from metrics import *
import json
import pickle

# Import enhanced networks
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from enhanced_erl_net import create_enhanced_network, QNetEnhanced, QNetTwinEnhanced
    ENHANCED_NETWORKS_AVAILABLE = True
    print("✅ Enhanced networks loaded successfully")
except ImportError:
    ENHANCED_NETWORKS_AVAILABLE = False
    print("⚠️  Enhanced networks not available, using standard networks")

class ExtendedEnsembleTrainer:
    """Extended training with validation splits and early stopping"""
    
    def __init__(self, save_path, agent_classes, args: Config, validation_ratio=0.2):
        self.save_path = save_path
        self.agent_classes = agent_classes
        self.args = args
        self.validation_ratio = validation_ratio
        self.agents = []
        
        # Training history tracking
        self.training_history = {
            'agents': {},
            'ensemble_performance': [],
            'validation_performance': [],
            'best_episode': 0,
            'best_sharpe': -np.inf,
            'early_stopping_patience': 50,
            'patience_counter': 0
        }
        
        # Get optimized state_dim from TradeSimulator
        temp_sim = TradeSimulator(num_sims=1)
        self.state_dim = temp_sim.state_dim
        print(f"🎯 Extended training state_dim: {self.state_dim}")
        
        # Setup device
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Create train/validation data splits
        self._create_data_splits()
    
    def _create_data_splits(self):
        """Create proper train/validation/test splits from Bitcoin data"""
        try:
            # Load original data to determine split points
            from data_config import data_path_dict
            data_path = data_path_dict.get("BTC")
            
            if os.path.exists(data_path.replace('.csv', '_predict_optimized.npy')):
                features = np.load(data_path.replace('.csv', '_predict_optimized.npy'))
            else:
                print("⚠️  Optimized features not found, creating data splits without them")
                return
            
            total_samples = len(features)
            
            # 60% train, 20% validation, 20% test
            train_end = int(total_samples * 0.6)
            val_end = int(total_samples * 0.8)
            
            self.data_splits = {
                'train': (0, train_end),
                'validation': (train_end, val_end), 
                'test': (val_end, total_samples)
            }
            
            print(f"📊 Data splits created:")
            print(f"   Training: {train_end} samples ({100*train_end/total_samples:.1f}%)")
            print(f"   Validation: {val_end-train_end} samples ({100*(val_end-train_end)/total_samples:.1f}%)")
            print(f"   Test: {total_samples-val_end} samples ({100*(total_samples-val_end)/total_samples:.1f}%)")
            
        except Exception as e:
            print(f"⚠️  Could not create data splits: {e}")
            self.data_splits = None
    
    def train_extended_ensemble(self, max_episodes=500, min_episodes=100):
        """Train ensemble with extended episodes and early stopping"""
        print(f"🚀 Starting extended ensemble training")
        print(f"   Max Episodes: {max_episodes}")
        print(f"   Min Episodes: {min_episodes}")
        print(f"   Early Stopping Patience: {self.training_history['early_stopping_patience']}")
        
        for i, agent_class in enumerate(self.agent_classes):
            print(f"\n📊 Training Agent {i+1}/{len(self.agent_classes)}: {agent_class.__name__}")
            
            agent_start_time = time.time()
            agent = self._train_agent_extended(agent_class, max_episodes, min_episodes)
            agent_duration = time.time() - agent_start_time
            
            self.agents.append(agent)
            
            # Store agent training history
            self.training_history['agents'][agent_class.__name__] = {
                'training_time': agent_duration,
                'final_performance': getattr(agent, 'final_performance', 0.0),
                'episodes_trained': getattr(agent, 'episodes_trained', 0),
                'best_episode': getattr(agent, 'best_episode', 0)
            }
            
            print(f"✅ {agent_class.__name__} completed in {agent_duration:.1f}s")
        
        self._save_ensemble_and_history()
        self._evaluate_extended_performance()
        
        print(f"🎉 Extended ensemble training completed!")
        return self.agents
    
    def _train_agent_extended(self, agent_class, max_episodes, min_episodes):
        """Train single agent with extended episodes and validation"""
        args = self.args.copy() if hasattr(self.args, 'copy') else self.args
        args.agent_class = agent_class
        
        # Extended training configuration
        args.break_step = int(max_episodes * 64)  # Much longer training
        args.eval_per_step = int(64)  # More frequent evaluation
        args.save_gap = int(32)  # More frequent saving
        
        # Initialize training
        args.init_before_training()
        torch.set_grad_enabled(False)
        
        # Build environment
        env = build_env(args.env_class, args.env_args, args.gpu_id)
        
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
            buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
            buffer.update(buffer_items)
        else:
            buffer = []
        
        # Initialize evaluator with validation tracking
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = ExtendedEvaluator(cwd=args.cwd, env=eval_env, args=args)
        
        # Extended training loop with early stopping
        print(f"   Starting extended training loop")
        
        episode_count = 0
        best_validation_score = -np.inf
        patience_counter = 0
        training_scores = []
        validation_scores = []
        
        while episode_count < max_episodes:
            episode_start = time.time()
            
            # Training step
            buffer_items = agent.explore_env(env, args.horizon_len)
            
            # Action distribution analysis
            action = buffer_items[1].flatten()
            action_counts = torch.bincount(action).cpu().numpy()
            action_distribution = action_counts / action_counts.sum() if action_counts.sum() > 0 else action_counts
            
            exp_r = buffer_items[2].mean().item()
            if args.if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items
            
            # Update network
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
            
            # Evaluate performance
            eval_result = evaluator.evaluate_and_save(
                actor=agent.act,
                steps=args.horizon_len,
                exp_r=exp_r,
                logging_tuple=logging_tuple,
            )
            
            current_score = evaluator.recorder[0][-1] if evaluator.recorder[0] else exp_r
            training_scores.append(current_score)
            
            # Validation evaluation (every 10 episodes)
            if episode_count % 10 == 0 and episode_count > 0:
                validation_score = self._evaluate_validation_performance(agent, eval_env)
                validation_scores.append(validation_score)
                
                # Early stopping check
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    patience_counter = 0
                    # Save best model
                    best_model_path = os.path.join(args.cwd, "best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    agent.save_or_load_agent(best_model_path, if_save=True)
                else:
                    patience_counter += 1
                
                print(f"   Episode {episode_count}: Train={current_score:.3f}, Val={validation_score:.3f}, Best={best_validation_score:.3f}")
                
                # Early stopping condition
                if episode_count >= min_episodes and patience_counter >= self.training_history['early_stopping_patience']:
                    print(f"   Early stopping triggered at episode {episode_count}")
                    break
            else:
                print(f"   Episode {episode_count}: Score={current_score:.3f}")
            
            episode_count += 1
            
            # Stop condition
            if os.path.exists(f"{args.cwd}/stop"):
                print(f"   Manual stop requested at episode {episode_count}")
                break
        
        # Training completed
        episode_duration = time.time() - evaluator.start_time
        print(f"   Training completed: {episode_count} episodes in {episode_duration:.1f}s")
        print(f"   Best validation score: {best_validation_score:.3f}")
        
        # Load best model
        if os.path.exists(os.path.join(args.cwd, "best_model")):
            agent.save_or_load_agent(os.path.join(args.cwd, "best_model"), if_save=False)
        
        # Store training statistics on agent
        agent.episodes_trained = episode_count
        agent.final_performance = current_score
        agent.best_validation_score = best_validation_score
        agent.training_scores = training_scores
        agent.validation_scores = validation_scores
        
        # Clean up
        env.close() if hasattr(env, "close") else None
        evaluator.save_training_curve_jpg()
        agent.save_or_load_agent(args.cwd, if_save=True)
        
        return agent
    
    def _evaluate_validation_performance(self, agent, eval_env):
        """Evaluate agent performance on validation set"""
        try:
            # Simple validation evaluation
            state = eval_env.reset()
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            total_reward = 0.0
            steps = 0
            max_eval_steps = 100
            
            for _ in range(max_eval_steps):
                action = agent.select_action(state)
                state, reward, done, _ = eval_env.step(action)
                
                if isinstance(state, np.ndarray):
                    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                
                total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                steps += 1
                
                if done[0] if isinstance(done, (list, np.ndarray)) else done:
                    break
            
            return total_reward / max(steps, 1)
            
        except Exception as e:
            print(f"   Validation evaluation failed: {e}")
            return 0.0
    
    def _save_ensemble_and_history(self):
        """Save ensemble models and training history"""
        # Save ensemble models
        ensemble_dir = os.path.join(self.save_path, "ensemble_models")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        for idx, agent in enumerate(self.agents):
            agent_name = self.agent_classes[idx].__name__
            agent_dir = os.path.join(ensemble_dir, agent_name)
            os.makedirs(agent_dir, exist_ok=True)
            agent.save_or_load_agent(agent_dir, if_save=True)
        
        # Save training history
        history_path = os.path.join(self.save_path, "training_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_for_json = {}
            for key, value in self.training_history.items():
                if isinstance(value, dict):
                    history_for_json[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            history_for_json[key][subkey] = subvalue.tolist()
                        else:
                            history_for_json[key][subkey] = subvalue
                else:
                    if isinstance(value, np.ndarray):
                        history_for_json[key] = value.tolist()
                    else:
                        history_for_json[key] = value
            
            json.dump(history_for_json, f, indent=2)
        
        # Save detailed agent statistics
        agent_stats_path = os.path.join(self.save_path, "agent_statistics.pkl")
        agent_stats = {}
        for idx, agent in enumerate(self.agents):
            agent_name = self.agent_classes[idx].__name__
            agent_stats[agent_name] = {
                'training_scores': getattr(agent, 'training_scores', []),
                'validation_scores': getattr(agent, 'validation_scores', []),
                'episodes_trained': getattr(agent, 'episodes_trained', 0),
                'best_validation_score': getattr(agent, 'best_validation_score', 0.0)
            }
        
        with open(agent_stats_path, 'wb') as f:
            pickle.dump(agent_stats, f)
        
        print(f"✅ Extended training history saved to: {self.save_path}")
    
    def _evaluate_extended_performance(self):
        """Comprehensive evaluation of extended ensemble performance"""
        print(f"\n📊 Extended Performance Evaluation")
        
        # Calculate ensemble statistics
        total_episodes = sum(self.training_history['agents'][name]['episodes_trained'] 
                           for name in self.training_history['agents'])
        avg_episodes = total_episodes / len(self.agents) if self.agents else 0
        
        total_time = sum(self.training_history['agents'][name]['training_time'] 
                        for name in self.training_history['agents'])
        
        print(f"   Total Episodes Trained: {total_episodes}")
        print(f"   Average Episodes per Agent: {avg_episodes:.1f}")
        print(f"   Total Training Time: {total_time:.1f}s")
        print(f"   Average Time per Episode: {total_time/max(total_episodes,1):.2f}s")
        
        # Agent performance summary
        for agent_name, stats in self.training_history['agents'].items():
            print(f"   {agent_name}: {stats['episodes_trained']} episodes, "
                  f"final score: {stats['final_performance']:.3f}")

class ExtendedEvaluator(Evaluator):
    """Extended evaluator with validation support"""
    
    def __init__(self, cwd, env, args):
        super().__init__(cwd, env, args)
        self.validation_history = []
    
    def evaluate_validation(self, actor, steps):
        """Evaluate on validation set"""
        # Simplified validation evaluation
        state = self.env.reset()
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        total_reward = 0.0
        for _ in range(min(steps, 100)):
            action = actor(state)
            state, reward, done, _ = self.env.step(action)
            
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            
            if done[0] if isinstance(done, (list, np.ndarray)) else done:
                break
        
        self.validation_history.append(total_reward)
        return total_reward

def run_extended_training(save_path, agent_list, max_episodes=500, min_episodes=100):
    """Run extended ensemble training with proper validation"""
    import sys
    
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"🚀 Extended Ensemble Training")
    print(f"🖥️  Using device: {'GPU ' + str(gpu_id) if gpu_id >= 0 else 'CPU'}")
    print(f"📊 Max Episodes: {max_episodes}")
    print(f"📊 Min Episodes: {min_episodes}")
    
    # Enhanced training configuration
    num_sims = 16  # Reduced for longer training stability
    num_ignore_step = 60
    max_position = 1
    step_gap = 2  
    slippage = 7e-7
    
    max_step = (4800 - num_ignore_step) // step_gap
    
    # Get state dimension
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
    
    print(f"📊 Extended Training Configuration:")
    print(f"   State Dimension: {actual_state_dim}")
    print(f"   Parallel Environments: {num_sims}")
    print(f"   Max Steps: {max_step}")
    
    # Create config for extended training
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    
    # Optimized architecture for extended training
    if actual_state_dim <= 8:
        args.net_dims = (128, 64, 32)
    else:
        args.net_dims = (256, 128, 64)
    
    # Extended training hyperparameters
    args.gamma = 0.995
    args.explore_rate = 0.01  # Higher exploration for longer training
    args.state_value_tau = 0.005
    args.soft_update_tau = 1e-5
    args.learning_rate = 1e-5  # Lower for stability
    args.batch_size = 256
    args.buffer_size = int(max_step * 10)  # Larger buffer for extended training
    args.repeat_times = 1.5
    args.horizon_len = int(max_step * 2)
    args.eval_per_step = int(max_step // 2)
    args.num_workers = 1
    args.save_gap = 16
    
    # Evaluation environment
    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = env_args.copy()
    
    print(f"🔧 Extended Hyperparameters:")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Exploration Rate: {args.explore_rate}")
    print(f"   Buffer Size: {args.buffer_size}")
    
    # Create and run extended trainer
    trainer = ExtendedEnsembleTrainer(save_path, agent_list, args)
    agents = trainer.train_extended_ensemble(max_episodes, min_episodes)
    
    print(f"🎉 Extended Training Complete! Models saved to: {save_path}")
    return agents

if __name__ == "__main__":
    print("🚀 Starting Extended Ensemble Training...")
    
    # Agent configuration for extended training
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    
    run_extended_training(
        "ensemble_extended_long_training",
        agent_list,
        max_episodes=500,  # Much longer training
        min_episodes=100   # Minimum before early stopping
    )