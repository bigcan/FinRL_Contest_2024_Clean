"""
Run Optimized Ensemble Training - Phase 3
Simplified ensemble training with progress monitoring
"""

import os
import sys
import torch
import numpy as np
import time
from datetime import datetime

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator

class OptimizedEnsembleRunner:
    """Simplified ensemble runner with monitoring"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = f"../../../results/task1_results/trained_models/ensemble_optimized_{self.timestamp}"
        os.makedirs(self.save_path, exist_ok=True)
        
        # Agent configuration
        self.agent_classes = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
        self.trained_agents = []
        
        print(f"🚀 Phase 3: Optimized Ensemble Training")
        print(f"📁 Save path: {self.save_path}")
        print(f"🤖 Agents to train: {[cls.__name__ for cls in self.agent_classes]}")
    
    def setup_config(self):
        """Setup training configuration"""
        # Get optimized state dimension
        temp_sim = TradeSimulator(num_sims=1)
        state_dim = temp_sim.state_dim
        
        print(f"📊 Optimized state dimension: {state_dim}")
        print(f"🎯 Features: {temp_sim.feature_names}")
        
        # Training configuration
        num_sims = 16  # Balanced for performance
        max_step = 500  # Manageable training length
        
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
        
        # Create base config
        config = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
        config.gpu_id = 0 if torch.cuda.is_available() else -1
        config.random_seed = 42
        
        # Optimized hyperparameters
        config.net_dims = (128, 64, 32)
        config.learning_rate = 2e-6
        config.batch_size = 256
        config.gamma = 0.995
        config.explore_rate = 0.005
        config.break_step = 8  # Reasonable training length
        config.horizon_len = max_step
        config.buffer_size = max_step * 4
        config.repeat_times = 2
        config.eval_per_step = max_step
        config.eval_times = 1
        
        print(f"⚙️  Training Configuration:")
        print(f"   Parallel Environments: {num_sims}")
        print(f"   Max Steps per Episode: {max_step}")
        print(f"   Network Architecture: {config.net_dims}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Training Steps: {config.break_step}")
        
        return config
    
    def train_single_agent(self, agent_class, config, agent_idx):
        """Train a single agent"""
        agent_name = agent_class.__name__
        agent_start_time = time.time()
        
        print(f"\n🤖 Training Agent {agent_idx+1}/3: {agent_name}")
        print("=" * 50)
        
        # Setup agent-specific directory
        agent_dir = os.path.join(self.save_path, agent_name)
        os.makedirs(agent_dir, exist_ok=True)
        config.cwd = agent_dir
        
        # Initialize training
        config.agent_class = agent_class
        config.init_before_training()
        torch.set_grad_enabled(False)
        
        # Build environment
        env = build_env(config.env_class, config.env_args, config.gpu_id)
        print(f"🌍 Environment created for {agent_name}")
        
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
        
        print(f"🎮 Initial state shape: {state.shape}")
        
        # Initialize buffer
        buffer = ReplayBuffer(
            gpu_id=config.gpu_id,
            num_seqs=config.num_envs,
            max_size=config.buffer_size,
            state_dim=config.state_dim,
            action_dim=1,
        )
        
        # Warm up buffer
        print(f"🔄 Warming up replay buffer...")
        buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
        buffer.update(buffer_items)
        
        # Training loop
        print(f"🏋️  Starting {agent_name} training...")
        
        for step in range(config.break_step):
            step_start = time.time()
            
            # Collect experience
            buffer_items = agent.explore_env(env, config.horizon_len)
            exp_r = buffer_items[2].mean().item()
            
            # Update buffer
            buffer.update(buffer_items)
            
            # Update network
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
            
            step_time = time.time() - step_start
            
            print(f"   Step {step+1}/{config.break_step}: Reward={exp_r:.4f}, Time={step_time:.1f}s")
            
            if logging_tuple:
                obj_critic, obj_actor = logging_tuple[:2]
                print(f"     Losses - Critic: {obj_critic:.4f}, Actor: {obj_actor:.4f}")
        
        # Save agent
        agent.save_or_load_agent(config.cwd, if_save=True)
        
        agent_time = time.time() - agent_start_time
        print(f"✅ {agent_name} training completed in {agent_time:.1f}s")
        
        # Quick evaluation
        self.quick_evaluate_agent(agent, env, agent_name)
        
        env.close() if hasattr(env, "close") else None
        self.trained_agents.append((agent_name, agent_dir))
        
        return agent
    
    def quick_evaluate_agent(self, agent, env, agent_name):
        """Quick evaluation of trained agent"""
        print(f"🧪 Quick evaluation of {agent_name}...")
        
        action_counts = {0: 0, 1: 0, 2: 0}
        rewards = []
        
        for _ in range(50):
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state = state.to(agent.device)
            
            total_reward = 0
            for _ in range(10):  # Short episodes
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
        
        print(f"   Average Reward: {avg_reward:.4f}")
        print(f"   Action Distribution: {action_counts}")
        print(f"   Action Variety: {action_variety}/3 actions used")
        
        if action_variety >= 2:
            print(f"   ✅ Good trading diversity")
        else:
            print(f"   ⚠️  Limited trading diversity")
    
    def run_ensemble_training(self):
        """Run complete ensemble training"""
        start_time = time.time()
        
        # Setup configuration
        config = self.setup_config()
        
        print(f"\n🚀 Starting Ensemble Training")
        print("=" * 60)
        
        # Train each agent
        for i, agent_class in enumerate(self.agent_classes):
            try:
                agent = self.train_single_agent(agent_class, config, i)
                print(f"✅ Agent {i+1}/3 completed successfully")
            except Exception as e:
                print(f"❌ Agent {i+1}/3 failed: {e}")
                import traceback
                traceback.print_exc()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 ENSEMBLE TRAINING COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"🤖 Agents Trained: {len(self.trained_agents)}/3")
        print(f"📁 Models Saved: {self.save_path}")
        
        for agent_name, agent_dir in self.trained_agents:
            print(f"   ✅ {agent_name}: {agent_dir}")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Run evaluation: python3 task1_eval.py 0")
        print(f"   2. Compare with baseline performance")
        print(f"   3. Analyze trading behavior")
        
        return len(self.trained_agents) == 3

def main():
    """Main execution"""
    print("🚀 Phase 3: Optimized Ensemble Training")
    print("=" * 60)
    
    runner = OptimizedEnsembleRunner()
    success = runner.run_ensemble_training()
    
    if success:
        print("\n🎉 ALL AGENTS TRAINED SUCCESSFULLY!")
        return True
    else:
        print("\n⚠️  Some agents failed to train")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)