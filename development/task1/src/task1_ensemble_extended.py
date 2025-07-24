#!/usr/bin/env python3
"""
Extended Training Ensemble for Profitability Improvements
Addresses insufficient training with 200 steps, early stopping, and enhanced configuration
"""

import os
import sys
import torch
import numpy as np
import time
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from enhanced_training_config import EnhancedConfig, EarlyStoppingManager, LearningRateScheduler, ExplorationScheduler, TrainingMetricsTracker
from optimized_hyperparameters import get_optimized_hyperparameters, apply_optimized_hyperparameters
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator


class ExtendedEnsembleTrainer:
    """
    Enhanced ensemble trainer with extended training and profitability improvements
    """
    
    def __init__(self, save_path=None, reward_type="multi_objective"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = save_path or f"ensemble_extended_{self.timestamp}"
        self.reward_type = reward_type
        
        # Agent configuration  
        self.agent_classes = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
        self.trained_agents = []
        
        print(f"🚀 Extended Ensemble Training")
        print(f"📁 Save path: {self.save_path}")
        print(f"🎯 Reward type: {reward_type}")
        print(f"🤖 Agents: {[cls.__name__ for cls in self.agent_classes]}")
    
    def setup_enhanced_config(self):
        """Setup enhanced training configuration with extended training"""
        
        # Get optimized state dimension
        temp_sim = TradeSimulator(num_sims=1)
        state_dim = temp_sim.state_dim
        temp_sim.set_reward_type(self.reward_type)  # Set reward type
        
        print(f"📊 Configuration:")
        print(f"   State dimension: {state_dim}")
        print(f"   Reward type: {self.reward_type}")
        print(f"   Features: {getattr(temp_sim, 'feature_names', 'optimized')}")
        
        # Enhanced training configuration
        num_sims = 32  # Balanced for performance
        max_step = 2370  # Full data sequence
        
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": num_sims,
            "max_step": max_step,
            "state_dim": state_dim,
            "action_dim": 3,
            "if_discrete": True,
            "max_position": 1,
            "slippage": 7e-7,
            "num_sims": num_sims,
            "step_gap": 2,
        }
        
        # Create enhanced configuration with extended training
        config = EnhancedConfig(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
        config.gpu_id = 0 if torch.cuda.is_available() else -1
        config.random_seed = 42
        config.state_dim = state_dim
        
        # Apply optimized hyperparameters - MAJOR IMPROVEMENT
        optimized_params = get_optimized_hyperparameters(self.reward_type)
        config = apply_optimized_hyperparameters(config, optimized_params, env_args)
        
        # Additional configuration
        config.repeat_times = 2
        
        print(f"⚙️  Optimized Training Configuration:")
        print(f"   📈 Training Steps: {config.break_step} (vs 8-16 baseline = {config.break_step/16:.0f}x)")
        print(f"   🧠 Network: {config.net_dims} (optimized for 8 features)")
        print(f"   📚 Learning Rate: {config.learning_rate:.2e} (vs 2e-6 baseline = {config.learning_rate/2e-6:.0f}x)")
        print(f"   🔍 Exploration: {config.initial_exploration:.3f} → {config.final_exploration:.3f} (vs 0.005 baseline)")
        print(f"   💾 Buffer Size: {config.buffer_size} ({config.buffer_size//max_step}x max_step)")
        print(f"   ⏰ Max Training Time: {config.max_training_time/60:.1f} minutes")
        print(f"   🎯 Reward Type: {self.reward_type}")
        print(f"   🛑 Early Stopping: {config.early_stopping_enabled} (patience: {config.early_stopping_patience})")
        
        return config
    
    def train_single_agent_extended(self, agent_class, config, agent_idx):
        """Train a single agent with extended training and improvements"""
        
        agent_name = agent_class.__name__
        agent_start_time = time.time()
        
        print(f"\n🤖 Training Agent {agent_idx+1}/3: {agent_name}")
        print("=" * 60)
        
        # Setup agent-specific directory
        agent_dir = os.path.join(self.save_path, agent_name)
        os.makedirs(agent_dir, exist_ok=True)
        config.cwd = agent_dir
        
        # Initialize training
        config.agent_class = agent_class
        config.init_before_training()
        torch.set_grad_enabled(False)
        
        # Build environment with enhanced reward system
        env = self.build_enhanced_env(config)
        print(f"🌍 Environment created with {self.reward_type} rewards")
        
        # Create agent with enhanced architecture
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
        
        print(f"🎮 Initial state shape: {state.shape}")
        
        # Initialize buffer
        buffer = ReplayBuffer(
            gpu_id=config.gpu_id,
            num_seqs=config.num_envs,
            max_size=config.buffer_size,
            state_dim=config.state_dim,
            action_dim=1,
        )
        
        # Enhanced training components
        early_stopping = EarlyStoppingManager(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            monitor="eval_score"
        )
        
        lr_scheduler = LearningRateScheduler(
            agent.act_optimizer,
            scheduler_type=config.lr_scheduler_type,
            total_steps=config.break_step,
            min_lr=config.lr_min
        )
        
        exploration_scheduler = ExplorationScheduler(
            initial_rate=config.initial_exploration,
            final_rate=config.final_exploration,
            total_steps=config.break_step
        )
        
        metrics_tracker = TrainingMetricsTracker(
            history_size=config.metrics_history_size
        )
        
        # Warm up buffer
        print(f"🔄 Warming up replay buffer...")
        buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
        buffer.update(buffer_items)
        print(f"   Buffer size: {len(buffer) if hasattr(buffer, '__len__') else 'N/A'}")
        
        # Extended training loop
        print(f"🏋️  Starting Extended Training ({config.break_step} steps)...")
        best_eval_score = float('-inf')
        
        for step in range(config.break_step):
            step_start = time.time()
            
            # Check time limit
            elapsed_time = time.time() - agent_start_time
            if elapsed_time > config.max_training_time:
                print(f"   ⏰ Time limit reached ({elapsed_time:.1f}s), stopping training")
                break
            
            # Update exploration rate
            current_exploration = exploration_scheduler.get_rate(step)
            
            # Collect experience with current exploration
            buffer_items = agent.explore_env(env, config.horizon_len)
            exp_r = buffer_items[2].mean().item()
            
            # Update buffer
            buffer.update(buffer_items)
            
            # Update network
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
            
            # Update learning rate
            current_lr = lr_scheduler.step()
            
            # Extract losses
            obj_critic = obj_actor = 0.0
            if logging_tuple:
                obj_critic, obj_actor = logging_tuple[:2]
            
            step_time = time.time() - step_start
            
            # Update metrics
            metrics_tracker.update(
                step=step,
                reward=exp_r,
                loss_critic=obj_critic,
                loss_actor=obj_actor,
                exploration_rate=current_exploration,
                learning_rate=current_lr
            )
            
            # Periodic evaluation
            if step % config.eval_per_step == 0:
                # Quick evaluation
                eval_score = self.quick_evaluate_agent(agent, env)
                metrics_tracker.update(step=step, eval_score=eval_score)
                
                # Track best performance
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    # Save best model
                    agent.save_or_load_agent(config.cwd, if_save=True)
                
                # Check early stopping
                if config.early_stopping_enabled:
                    should_stop = early_stopping.update(eval_score, step)
                    if should_stop:
                        print(f"   🛑 Early stopping triggered at step {step}")
                        break
                
                print(f"   Step {step+1}/{config.break_step}: Reward={exp_r:.4f}, Eval={eval_score:.4f}, LR={current_lr:.2e}, Time={step_time:.1f}s")
            else:
                print(f"   Step {step+1}/{config.break_step}: Reward={exp_r:.4f}, LR={current_lr:.2e}, Time={step_time:.1f}s")
        
        # Final save
        agent.save_or_load_agent(config.cwd, if_save=True)
        
        agent_time = time.time() - agent_start_time
        print(f"✅ {agent_name} extended training completed in {agent_time:.1f}s")
        print(f"   📊 Best evaluation score: {best_eval_score:.4f}")
        
        # Print training summary
        metrics_tracker.print_summary()
        
        # Final evaluation
        final_eval_score = self.quick_evaluate_agent(agent, env)
        print(f"   🎯 Final evaluation score: {final_eval_score:.4f}")
        
        env.close() if hasattr(env, "close") else None
        self.trained_agents.append((agent_name, agent_dir, best_eval_score))
        
        return agent, best_eval_score
    
    def build_enhanced_env(self, config):
        """Build environment with enhanced reward system"""
        from erl_config import build_env
        
        env = build_env(config.env_class, config.env_args, config.gpu_id)
        
        # Set reward type if environment supports it
        if hasattr(env, 'set_reward_type'):
            env.set_reward_type(self.reward_type)
        elif hasattr(env, 'envs') and len(env.envs) > 0:
            # For vectorized environments
            for single_env in env.envs:
                if hasattr(single_env, 'set_reward_type'):
                    single_env.set_reward_type(self.reward_type)
        
        return env
    
    def quick_evaluate_agent(self, agent, env):
        """Quick evaluation of agent performance"""
        
        action_counts = {0: 0, 1: 0, 2: 0}
        rewards = []
        
        eval_episodes = 3
        for _ in range(eval_episodes):
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state = state.to(agent.device)
            
            total_reward = 0
            for _ in range(20):  # Short evaluation episodes
                with torch.no_grad():
                    q_values = agent.act(state)
                    action = q_values.argmax(dim=1, keepdim=True)
                    action_counts[action[0].item()] += 1
                
                next_state, reward, done, _ = env.step(action)
                total_reward += reward.mean().item()
                
                if done.any():
                    break
                    
                state = next_state
            
            rewards.append(total_reward)
        
        avg_reward = np.mean(rewards)
        action_variety = len([v for v in action_counts.values() if v > 0])
        
        # Reward with bonus for trading diversity (addresses conservative trading)
        diversity_bonus = action_variety / 3.0 * 0.1  # 10% bonus for full diversity
        final_score = avg_reward + diversity_bonus
        
        return final_score
    
    def run_extended_ensemble_training(self):
        """Run complete extended ensemble training"""
        start_time = time.time()
        
        # Setup enhanced configuration
        config = self.setup_enhanced_config()
        
        print(f"\n🚀 Starting Extended Ensemble Training")
        print("=" * 80)
        
        # Train each agent with extended training
        all_scores = []
        for i, agent_class in enumerate(self.agent_classes):
            try:
                agent, best_score = self.train_single_agent_extended(config, i)
                all_scores.append(best_score)
                print(f"✅ Agent {i+1}/3 completed with score {best_score:.4f}")
            except Exception as e:
                print(f"❌ Agent {i+1}/3 failed: {e}")
                import traceback
                traceback.print_exc()
                all_scores.append(0.0)
        
        total_time = time.time() - start_time
        
        # Final summary
        print(f"\n🎉 EXTENDED ENSEMBLE TRAINING COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"🤖 Agents Trained: {len(self.trained_agents)}/3")
        print(f"📁 Models Saved: {self.save_path}")
        
        print(f"\n📊 Agent Performance Summary:")
        for i, (agent_name, agent_dir, score) in enumerate(self.trained_agents):
            print(f"   {i+1}. {agent_name}: {score:.4f}")
        
        if all_scores:
            print(f"\n📈 Ensemble Statistics:")
            print(f"   📊 Average Score: {np.mean(all_scores):.4f}")
            print(f"   🎯 Best Score: {max(all_scores):.4f}")
            print(f"   📉 Worst Score: {min(all_scores):.4f}")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Run evaluation: python3 task1_eval.py 0")
        print(f"   2. Compare with baseline performance")
        print(f"   3. Analyze improved trading behavior")
        print(f"   4. Test different reward functions")
        
        return len(self.trained_agents) == 3, all_scores


def main():
    """Main execution"""
    print("🚀 Extended Training for Profitability Improvements")
    print("=" * 80)
    
    # Get command line arguments
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    reward_type = sys.argv[2] if len(sys.argv) > 2 else "multi_objective"
    
    print(f"🖥️  GPU ID: {gpu_id}")
    print(f"🎯 Reward Type: {reward_type}")
    
    # Create extended trainer
    save_path = f"ensemble_extended_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = ExtendedEnsembleTrainer(save_path=save_path, reward_type=reward_type)
    
    # Run extended training
    success, scores = trainer.run_extended_ensemble_training()
    
    if success:
        print(f"\n🎉 ALL AGENTS TRAINED SUCCESSFULLY!")
        print(f"📈 Expected improvements:")
        print(f"   - Better win rate (target: 55-60% vs 45% baseline)")
        print(f"   - Positive returns (target: +0.5-2% vs -0.19% baseline)")
        print(f"   - Improved Sharpe ratio (target: +0.2-0.5 vs -0.036 baseline)")
        return True
    else:
        print(f"\n⚠️  Some agents failed to train properly")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)