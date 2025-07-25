#!/usr/bin/env python3
"""
Ultimate Ensemble Training: DQN + PPO + Rainbow
State-of-the-art ensemble combining all advanced algorithms
"""

import os
import sys
import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import argparse

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trade_simulator import TradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_agent_ppo import AgentPPO
from erl_agent_rainbow import AgentRainbow, PrioritizedReplayBuffer
from enhanced_training_config import EnhancedConfig
from optimized_hyperparameters import get_optimized_hyperparameters, apply_optimized_hyperparameters
from enhanced_ensemble_manager import EnhancedEnsembleManager
from erl_config import build_env
from erl_replay_buffer import ReplayBuffer
from erl_replay_buffer_ppo import PPOReplayBuffer


class UltimateEnsembleTrainer:
    """
    Ultimate ensemble trainer with DQN, PPO, and Rainbow agents
    Represents the pinnacle of deep reinforcement learning for trading
    """
    
    def __init__(self, 
                 reward_type: str = "simple",
                 gpu_id: int = 0,
                 team_name: str = "ultimate_ensemble"):
        """
        Initialize ultimate ensemble trainer
        
        Args:
            reward_type: Reward function to use
            gpu_id: GPU device ID
            team_name: Team name for saving models
        """
        self.reward_type = reward_type
        self.gpu_id = gpu_id
        self.team_name = team_name
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")
        
        # Get state dimension
        temp_sim = TradeSimulator(num_sims=1)
        self.state_dim = temp_sim.state_dim
        temp_sim.set_reward_type(reward_type)
        
        # Agent configurations
        self.agent_configs = self._setup_agent_configs()
        
        # Training metrics
        self.training_results = {}
        
        print(f"🌟 ULTIMATE ENSEMBLE TRAINER")
        print("=" * 80)
        print(f"🎯 Reward type: {reward_type}")
        print(f"🖥️  Device: {self.device}")
        print(f"📊 State dimension: {self.state_dim}")
        print(f"🤖 Agent types: {len(self.agent_configs)}")
        print(f"🏆 Algorithms: DQN variants + PPO + Rainbow DQN")
    
    def _setup_agent_configs(self) -> Dict:
        """Setup configurations for all agent types"""
        
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
        
        # Standard DQN agents
        dqn_agents = [
            ("AgentD3QN", AgentD3QN),
            ("AgentDoubleDQN", AgentDoubleDQN), 
            ("AgentTwinD3QN", AgentTwinD3QN)
        ]
        
        for agent_name, agent_class in dqn_agents:
            config = EnhancedConfig(agent_class=agent_class, env_class=TradeSimulator, env_args=env_args)
            config.gpu_id = self.gpu_id
            config.random_seed = 42
            config.state_dim = self.state_dim
            
            # Apply optimized hyperparameters
            config = apply_optimized_hyperparameters(config, optimized_params, env_args)
            
            configs[agent_name] = {
                'config': config,
                'agent_class': agent_class,
                'buffer_type': 'dqn',
                'env_args': env_args,
                'category': 'dqn'
            }
        
        # PPO agent
        ppo_config = EnhancedConfig(agent_class=AgentPPO, env_class=TradeSimulator, env_args=env_args)
        ppo_config.gpu_id = self.gpu_id
        ppo_config.random_seed = 42
        ppo_config.state_dim = self.state_dim
        
        # PPO-specific parameters
        ppo_config.ppo_clip_ratio = 0.2
        ppo_config.ppo_policy_epochs = 4
        ppo_config.ppo_value_epochs = 4
        ppo_config.ppo_gae_lambda = 0.95
        ppo_config.ppo_entropy_coeff = 0.01
        ppo_config.ppo_max_grad_norm = 0.5
        
        # Apply base hyperparameters
        ppo_config = apply_optimized_hyperparameters(ppo_config, optimized_params, env_args)
        
        configs["AgentPPO"] = {
            'config': ppo_config,
            'agent_class': AgentPPO,
            'buffer_type': 'ppo',
            'env_args': env_args,
            'category': 'policy_gradient'
        }
        
        # Rainbow DQN agent
        rainbow_config = EnhancedConfig(agent_class=AgentRainbow, env_class=TradeSimulator, env_args=env_args)
        rainbow_config.gpu_id = self.gpu_id
        rainbow_config.random_seed = 42
        rainbow_config.state_dim = self.state_dim
        
        # Rainbow-specific parameters
        rainbow_config.rainbow_n_step = 3
        rainbow_config.rainbow_n_atoms = 51
        rainbow_config.rainbow_v_min = -10.0
        rainbow_config.rainbow_v_max = 10.0
        rainbow_config.rainbow_use_noisy = True
        rainbow_config.rainbow_use_prioritized = True
        rainbow_config.target_update_freq = 4
        
        # Apply base hyperparameters
        rainbow_config = apply_optimized_hyperparameters(rainbow_config, optimized_params, env_args)
        
        configs["AgentRainbow"] = {
            'config': rainbow_config,
            'agent_class': AgentRainbow,
            'buffer_type': 'prioritized',
            'env_args': env_args,
            'category': 'advanced_dqn'
        }
        
        print(f"\n🔧 AGENT CONFIGURATION SUMMARY:")
        for name, config in configs.items():
            category = config['category']
            buffer_type = config['buffer_type']
            print(f"   {name:15}: {category:15} + {buffer_type} buffer")
        
        return configs
    
    def train_single_agent(self, 
                          agent_name: str, 
                          agent_config: Dict,
                          save_dir: str) -> Dict:
        """
        Train a single agent (DQN, PPO, or Rainbow)
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration dictionary
            save_dir: Directory to save model
            
        Returns:
            Training results dictionary
        """
        print(f"\n🤖 Training {agent_name} ({agent_config['category']})...")
        training_start = time.time()
        
        try:
            config = agent_config['config']
            agent_class = agent_config['agent_class']
            buffer_type = agent_config['buffer_type']
            env_args = agent_config['env_args']
            
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
            
            # Create appropriate buffer
            if buffer_type == 'ppo':
                # PPO buffer
                ppo_buffer_size = config.horizon_len * config.num_envs
                buffer = PPOReplayBuffer(
                    max_size=ppo_buffer_size,
                    state_dim=config.state_dim,
                    action_dim=1,
                    gpu_id=config.gpu_id,
                    num_seqs=config.num_envs
                )
            elif buffer_type == 'prioritized':
                # Prioritized replay for Rainbow
                buffer = PrioritizedReplayBuffer(
                    max_size=config.buffer_size,
                    state_dim=config.state_dim,
                    action_dim=1,
                    gpu_id=config.gpu_id,
                    alpha=0.6,
                    beta_start=0.4,
                    beta_frames=config.break_step * config.horizon_len
                )
            else:
                # Standard DQN buffer
                buffer = ReplayBuffer(
                    gpu_id=config.gpu_id,
                    num_seqs=config.num_envs,
                    max_size=config.buffer_size,
                    state_dim=config.state_dim,
                    action_dim=1,
                )
            
            # Warm up buffer
            print(f"   🔄 Warming up {buffer_type} buffer...")
            buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
            buffer.update(buffer_items)
            
            # Training loop
            rewards_history = []
            best_reward = -np.inf
            best_step = 0
            patience_counter = 0
            
            print(f"   🏋️  Starting {config.break_step} training steps...")
            
            for step in range(config.break_step):
                step_start = time.time()
                
                # Collect experience
                buffer_items = agent.explore_env(env, config.horizon_len)
                exp_r = buffer_items[2].mean().item()
                rewards_history.append(exp_r)
                
                # Update buffer
                buffer.update(buffer_items)
                
                # Special handling for PPO
                if buffer_type == 'ppo':
                    buffer.compute_gae_advantages(gamma=config.gamma, gae_lambda=config.ppo_gae_lambda)
                
                # Update networks
                torch.set_grad_enabled(True)
                logging_tuple = agent.update_net(buffer)
                torch.set_grad_enabled(False)
                
                # Clear buffer for on-policy algorithms
                if buffer_type == 'ppo':
                    buffer.clear()
                
                # Early stopping check
                if hasattr(config, 'early_stopping_enabled') and config.early_stopping_enabled:
                    if exp_r > best_reward + config.early_stopping_min_delta:
                        best_reward = exp_r
                        best_step = step
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                        if patience_counter >= config.early_stopping_patience:
                            print(f"   ⏹️  Early stopping at step {step} (best: {best_reward:.4f} at step {best_step})")
                            break
                
                # Progress updates
                if step % config.eval_per_step == 0:
                    step_time = time.time() - step_start
                    recent_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else exp_r
                    
                    print(f"     Step {step:3d}: Reward={exp_r:.4f}, Recent={recent_reward:.4f}, Time={step_time:.2f}s")
                    
                    # Print agent-specific stats
                    if hasattr(agent, 'print_training_stats'):
                        agent.print_training_stats()
                    elif hasattr(agent, 'get_training_stats'):
                        stats = agent.get_training_stats()
                        if agent_name == "AgentRainbow":
                            print(f"        📊 Rainbow: Loss={stats['distributional_loss']:.4f}, Steps={stats['training_steps']}")
            
            # Save model
            os.makedirs(save_dir, exist_ok=True)
            agent.save_or_load_agent(save_dir, if_save=True)
            
            # Final evaluation
            print(f"   🎮 Running final evaluation...")
            eval_rewards = []
            eval_actions = []
            
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state = state.to(agent.device)
            
            for _ in range(100):
                with torch.no_grad():
                    if buffer_type == 'ppo':
                        action_probs = agent.act(state)
                        action = torch.multinomial(action_probs, 1)
                    else:
                        if agent_name == "AgentRainbow":
                            q_values = agent.act.get_q_values(state)
                        else:
                            q_values = agent.act(state)
                        action = q_values.argmax(dim=1, keepdim=True)
                    
                    eval_actions.extend(action.cpu().numpy().flatten())
                
                next_state, reward, done, _ = env.step(action)
                eval_rewards.append(reward.mean().item())
                
                if done.any():
                    state = env.reset()
                    if not isinstance(state, torch.Tensor):
                        state = torch.tensor(state, dtype=torch.float32)
                    state = state.to(agent.device)
                else:
                    state = next_state
            
            # Calculate metrics
            total_return = sum(eval_rewards)
            avg_reward = np.mean(rewards_history)
            final_reward = rewards_history[-1] if rewards_history else 0
            
            # Trading activity analysis
            action_counts = {0: 0, 1: 0, 2: 0}
            for action in eval_actions:
                action_counts[int(action)] += 1
            
            total_actions = sum(action_counts.values())
            trading_pct = ((action_counts[1] + action_counts[2]) / max(1, total_actions)) * 100
            
            training_time = time.time() - training_start
            
            result = {
                'agent_name': agent_name,
                'category': agent_config['category'],
                'total_return': total_return,
                'avg_training_reward': avg_reward,
                'final_training_reward': final_reward,
                'improvement': final_reward - rewards_history[0] if rewards_history else 0,
                'training_time': training_time,
                'trading_activity_pct': trading_pct,
                'action_distribution': action_counts,
                'convergence_step': best_step,
                'rewards_history': rewards_history,
                'buffer_type': buffer_type,
                'success': True
            }
            
            print(f"   ✅ {agent_name} training completed:")
            print(f"      📈 Total return: {total_return:.4f}")
            print(f"      🎯 Trading activity: {trading_pct:.1f}%")
            print(f"      ⏱️  Training time: {training_time:.1f}s")
            print(f"      🧠 Category: {agent_config['category']}")
            
            env.close() if hasattr(env, "close") else None
            return result
            
        except Exception as e:
            print(f"   ❌ {agent_name} training failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'agent_name': agent_name,
                'category': agent_config.get('category', 'unknown'),
                'total_return': -1000.0,
                'avg_training_reward': -1000.0,
                'final_training_reward': -1000.0,
                'improvement': -1000.0,
                'training_time': time.time() - training_start,
                'trading_activity_pct': 0.0,
                'action_distribution': {0: 0, 1: 0, 2: 0},
                'convergence_step': -1,
                'rewards_history': [],
                'buffer_type': agent_config.get('buffer_type', 'unknown'),
                'success': False,
                'error': str(e)
            }
    
    def train_ensemble(self) -> Dict:
        """Train the complete ultimate ensemble"""
        
        print(f"\n🚀 STARTING ULTIMATE ENSEMBLE TRAINING")
        print("=" * 80)
        print(f"🎯 Reward function: {self.reward_type}")
        print(f"🏷️  Team name: {self.team_name}")
        print(f"🤖 Agents to train: {list(self.agent_configs.keys())}")
        print(f"📊 Training represents state-of-the-art deep RL")
        
        ensemble_start = time.time()
        
        # Create save directory
        save_root = f"{self.team_name}/ensemble_models"
        os.makedirs(save_root, exist_ok=True)
        
        # Train each agent
        for agent_name, agent_config in self.agent_configs.items():
            agent_save_dir = f"{save_root}/{agent_name.lower()}"
            result = self.train_single_agent(agent_name, agent_config, agent_save_dir)
            self.training_results[agent_name] = result
        
        total_time = time.time() - ensemble_start
        
        # Analyze results
        self._analyze_ensemble_results(total_time)
        
        return self.training_results
    
    def _analyze_ensemble_results(self, total_time: float):
        """Analyze and display ultimate ensemble training results"""
        
        print(f"\n🌟 ULTIMATE ENSEMBLE TRAINING RESULTS")
        print("=" * 80)
        
        successful_agents = [r for r in self.training_results.values() if r['success']]
        failed_agents = [r for r in self.training_results.values() if not r['success']]
        
        print(f"⏱️  Training completed in {total_time:.1f}s")
        print(f"✅ Successful agents: {len(successful_agents)}/{len(self.training_results)}")
        
        if successful_agents:
            # Performance comparison by category
            categories = {}
            for agent in successful_agents:
                cat = agent['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(agent)
            
            print(f"\n🏆 PERFORMANCE BY ALGORITHM CATEGORY:")
            for category, agents in categories.items():
                avg_return = np.mean([a['total_return'] for a in agents])
                best_agent = max(agents, key=lambda x: x['total_return'])
                print(f"   {category:15}: Avg={avg_return:6.3f}, Best={best_agent['agent_name']} ({best_agent['total_return']:.3f})")
            
            # Overall ranking
            print(f"\n🥇 OVERALL AGENT PERFORMANCE RANKING:")
            ranked_agents = sorted(successful_agents, key=lambda x: x['total_return'], reverse=True)
            
            for i, agent in enumerate(ranked_agents, 1):
                category_emoji = {
                    'dqn': '🔵',
                    'policy_gradient': '🟢', 
                    'advanced_dqn': '🌈'
                }.get(agent['category'], '⚪')
                
                print(f"   {i}. {category_emoji} {agent['agent_name']:15}: Return={agent['total_return']:6.3f}, "
                      f"Trading={agent['trading_activity_pct']:4.1f}%, {agent['category']}")
            
            # Champion analysis
            champion = ranked_agents[0]
            print(f"\n🏆 ULTIMATE CHAMPION: {champion['agent_name']}")
            print(f"   📈 Total return: {champion['total_return']:.4f}")
            print(f"   🎯 Category: {champion['category']}")
            print(f"   📊 Trading improvement: {champion['improvement']:.4f}")
            print(f"   🎪 Trading activity: {champion['trading_activity_pct']:.1f}%")
            print(f"   ⚡ Convergence step: {champion['convergence_step']}")
            print(f"   🔧 Buffer type: {champion['buffer_type']}")
            
            # Algorithm diversity analysis
            print(f"\n🎭 ALGORITHM DIVERSITY ANALYSIS:")
            dqn_agents = [a for a in successful_agents if a['category'] == 'dqn']
            ppo_agents = [a for a in successful_agents if a['category'] == 'policy_gradient']
            rainbow_agents = [a for a in successful_agents if a['category'] == 'advanced_dqn']
            
            print(f"   🔵 Standard DQN: {len(dqn_agents)} agents")
            print(f"   🟢 Policy Gradient: {len(ppo_agents)} agents")
            print(f"   🌈 Advanced DQN: {len(rainbow_agents)} agents")
            
            # Ensemble diversity score
            category_diversity = len(set([a['category'] for a in successful_agents]))
            buffer_diversity = len(set([a['buffer_type'] for a in successful_agents]))
            
            print(f"\n🎯 ENSEMBLE DIVERSITY METRICS:")
            print(f"   Algorithm categories: {category_diversity}/3")
            print(f"   Buffer types: {buffer_diversity}/4")
            print(f"   Total successful agents: {len(successful_agents)}")
            
            if category_diversity >= 2 and len(successful_agents) >= 4:
                print(f"   ✅ EXCELLENT diversity for robust ensemble!")
            elif category_diversity >= 2:
                print(f"   ✅ GOOD diversity across algorithm types")
            else:
                print(f"   ⚠️  Limited diversity - consider debugging failed agents")
            
            # Performance insights
            avg_return = np.mean([a['total_return'] for a in successful_agents])
            std_return = np.std([a['total_return'] for a in successful_agents])
            avg_trading = np.mean([a['trading_activity_pct'] for a in successful_agents])
            
            print(f"\n📊 ENSEMBLE STATISTICS:")
            print(f"   Average return: {avg_return:.4f} ± {std_return:.4f}")
            print(f"   Average trading activity: {avg_trading:.1f}%")
            print(f"   Success rate: {len(successful_agents)/len(self.training_results)*100:.1f}%")
            
            # Performance assessment
            if avg_return > 5.0:
                print(f"   🚀 OUTSTANDING: Ensemble shows exceptional returns!")
            elif avg_return > 1.0:
                print(f"   ✅ EXCELLENT: Strong positive returns across agents")
            elif avg_return > 0.0:
                print(f"   ✅ GOOD: Positive returns achieved")
            else:
                print(f"   ⚠️  NEEDS IMPROVEMENT: Returns below expectations")
        
        if failed_agents:
            print(f"\n❌ FAILED AGENTS:")
            for agent in failed_agents:
                print(f"   {agent['agent_name']} ({agent['category']}): {agent.get('error', 'Unknown error')}")
        
        print(f"\n📋 NEXT STEPS:")
        if successful_agents:
            print(f"   1. 🎯 Deploy ultimate ensemble with top {min(3, len(successful_agents))} agents")
            print(f"   2. 🚀 Run full evaluation: python3 task1_eval_ultimate.py")
            print(f"   3. 📊 Compare against baseline and previous ensembles")
            print(f"   4. 🏆 Submit best performing model configuration")
        else:
            print(f"   1. 🔧 Debug training issues across all algorithms")
            print(f"   2. 🎛️  Adjust hyperparameters and retry")
            print(f"   3. 🧪 Test individual agents before ensemble training")


def main():
    """Main training execution"""
    
    parser = argparse.ArgumentParser(description="Ultimate Ensemble Training")
    parser.add_argument("gpu_id", type=int, help="GPU ID (0, 1, etc. or -1 for CPU)")
    parser.add_argument("--reward", type=str, default="simple", 
                        choices=["simple", "transaction_cost_adjusted", "multi_objective"],
                        help="Reward function type")
    parser.add_argument("--team", type=str, default="ultimate_ensemble", help="Team name")
    
    args = parser.parse_args()
    
    print(f"🌟 ULTIMATE ENSEMBLE TRAINING - STATE OF THE ART")
    print("=" * 80)
    print(f"Configuration:")
    print(f"   🖥️  GPU ID: {args.gpu_id}")
    print(f"   🎯 Reward Type: {args.reward}")
    print(f"   🏷️  Team Name: {args.team}")
    print(f"   🤖 Algorithms: DQN + Double DQN + Dueling DQN + PPO + Rainbow DQN")
    print(f"   🧠 Techniques: Prioritized Replay + Noisy Networks + N-Step + GAE")
    
    # Create trainer
    trainer = UltimateEnsembleTrainer(
        reward_type=args.reward,
        gpu_id=args.gpu_id,
        team_name=args.team
    )
    
    # Train ensemble
    results = trainer.train_ensemble()
    
    print(f"\n🎉 Ultimate ensemble training completed!")
    return results


if __name__ == "__main__":
    results = main()