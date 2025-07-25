#!/usr/bin/env python3
"""
Enhanced Ensemble Training with PPO Agent
Combines DQN variants (D3QN, DoubleDQN, TwinD3QN) with PPO for comprehensive ensemble
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
from enhanced_training_config import EnhancedConfig
from optimized_hyperparameters import get_optimized_hyperparameters, apply_optimized_hyperparameters
from enhanced_ensemble_manager import EnhancedEnsembleManager
from erl_config import build_env
from erl_replay_buffer import ReplayBuffer
from erl_replay_buffer_ppo import PPOReplayBuffer


class HybridEnsembleTrainer:
    """
    Hybrid ensemble trainer supporting both DQN and PPO agents
    """
    
    def __init__(self, 
                 reward_type: str = "simple",
                 gpu_id: int = 0,
                 team_name: str = "hybrid_ensemble"):
        """
        Initialize hybrid ensemble trainer
        
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
        
        print(f"🚀 Hybrid Ensemble Trainer initialized:")
        print(f"   Reward type: {reward_type}")
        print(f"   Device: {self.device}")
        print(f"   State dimension: {self.state_dim}")
        print(f"   Agent types: {len(self.agent_configs)}")
    
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
        
        # DQN agent configs
        dqn_agents = [
            ("AgentD3QN", AgentD3QN),
            ("AgentDoubleDQN", AgentDoubleDQN), 
            ("AgentTwinD3QN", AgentTwinD3QN)
        ]
        
        configs = {}
        
        # Configure DQN agents
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
                'env_args': env_args
            }
        
        # Configure PPO agent
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
            'env_args': env_args
        }
        
        return configs
    
    def train_single_agent(self, 
                          agent_name: str, 
                          agent_config: Dict,
                          save_dir: str) -> Dict:
        """
        Train a single agent (DQN or PPO)
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration dictionary
            save_dir: Directory to save model
            
        Returns:
            Training results dictionary
        """
        print(f"\n🤖 Training {agent_name}...")
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
                # PPO buffer needs to handle flattened data: horizon_len * num_envs
                ppo_buffer_size = config.horizon_len * config.num_envs
                buffer = PPOReplayBuffer(
                    max_size=ppo_buffer_size,
                    state_dim=config.state_dim,
                    action_dim=1,
                    gpu_id=config.gpu_id,
                    num_seqs=config.num_envs
                )
            else:
                buffer = ReplayBuffer(
                    gpu_id=config.gpu_id,
                    num_seqs=config.num_envs,
                    max_size=config.buffer_size,
                    state_dim=config.state_dim,
                    action_dim=1,
                )
            
            # Warm up buffer
            print(f"   Warming up buffer...")
            buffer_items = agent.explore_env(env, config.horizon_len, if_random=True)
            buffer.update(buffer_items)
            
            # Training loop
            rewards_history = []
            best_reward = -np.inf
            best_step = 0
            patience_counter = 0
            
            print(f"   Starting {config.break_step} training steps...")
            
            for step in range(config.break_step):
                step_start = time.time()
                
                # Collect experience
                buffer_items = agent.explore_env(env, config.horizon_len)
                exp_r = buffer_items[2].mean().item()
                rewards_history.append(exp_r)
                
                # Update buffer
                buffer.update(buffer_items)
                
                # Compute advantages for PPO
                if buffer_type == 'ppo':
                    buffer.compute_gae_advantages(gamma=config.gamma, gae_lambda=config.ppo_gae_lambda)
                
                # Update networks
                torch.set_grad_enabled(True)
                if buffer_type == 'ppo':
                    logging_tuple = agent.update_net(buffer)
                    # Clear buffer after update (on-policy)
                    buffer.clear()
                else:
                    logging_tuple = agent.update_net(buffer)
                torch.set_grad_enabled(False)
                
                # Early stopping check
                if hasattr(config, 'early_stopping_enabled') and config.early_stopping_enabled:
                    if exp_r > best_reward + config.early_stopping_min_delta:
                        best_reward = exp_r
                        best_step = step
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                        if patience_counter >= config.early_stopping_patience:
                            print(f"   Early stopping at step {step} (best: {best_reward:.4f} at step {best_step})")
                            break
                
                # Progress updates
                if step % config.eval_per_step == 0:
                    step_time = time.time() - step_start
                    recent_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else exp_r
                    
                    print(f"     Step {step:3d}: Reward={exp_r:.4f}, Recent={recent_reward:.4f}, Time={step_time:.2f}s")
                    
                    # Print agent-specific stats
                    if hasattr(agent, 'print_training_stats'):
                        agent.print_training_stats()
            
            # Save model
            os.makedirs(save_dir, exist_ok=True)
            agent.save_or_load_agent(save_dir, if_save=True)
            
            # Final evaluation
            print(f"   Running final evaluation...")
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
                'total_return': total_return,
                'avg_training_reward': avg_reward,
                'final_training_reward': final_reward,
                'improvement': final_reward - rewards_history[0] if rewards_history else 0,
                'training_time': training_time,
                'trading_activity_pct': trading_pct,
                'action_distribution': action_counts,
                'convergence_step': best_step,
                'rewards_history': rewards_history,
                'success': True
            }
            
            print(f"   ✅ {agent_name} training completed:")
            print(f"      Total return: {total_return:.4f}")
            print(f"      Trading activity: {trading_pct:.1f}%")
            print(f"      Training time: {training_time:.1f}s")
            
            env.close() if hasattr(env, "close") else None
            return result
            
        except Exception as e:
            print(f"   ❌ {agent_name} training failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'agent_name': agent_name,
                'total_return': -1000.0,
                'avg_training_reward': -1000.0,
                'final_training_reward': -1000.0,
                'improvement': -1000.0,
                'training_time': time.time() - training_start,
                'trading_activity_pct': 0.0,
                'action_distribution': {0: 0, 1: 0, 2: 0},
                'convergence_step': -1,
                'rewards_history': [],
                'success': False,
                'error': str(e)
            }
    
    def train_ensemble(self) -> Dict:
        """Train the complete hybrid ensemble"""
        
        print(f"\n🚀 STARTING HYBRID ENSEMBLE TRAINING")
        print("=" * 80)
        print(f"Reward function: {self.reward_type}")
        print(f"Team name: {self.team_name}")
        print(f"Agents to train: {list(self.agent_configs.keys())}")
        
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
        """Analyze and display ensemble training results"""
        
        print(f"\n📊 HYBRID ENSEMBLE TRAINING RESULTS")
        print("=" * 80)
        
        successful_agents = [r for r in self.training_results.values() if r['success']]
        failed_agents = [r for r in self.training_results.values() if not r['success']]
        
        print(f"Training completed in {total_time:.1f}s")
        print(f"Successful agents: {len(successful_agents)}/{len(self.training_results)}")
        
        if successful_agents:
            # Performance comparison
            print(f"\n🏆 AGENT PERFORMANCE RANKING:")
            ranked_agents = sorted(successful_agents, key=lambda x: x['total_return'], reverse=True)
            
            for i, agent in enumerate(ranked_agents, 1):
                print(f"   {i}. {agent['agent_name']:15}: Return={agent['total_return']:6.3f}, "
                      f"Trading={agent['trading_activity_pct']:4.1f}%, Time={agent['training_time']:5.1f}s")
            
            # Best agent details
            best_agent = ranked_agents[0]
            print(f"\n🥇 BEST PERFORMER: {best_agent['agent_name']}")
            print(f"   Total return: {best_agent['total_return']:.4f}")
            print(f"   Training improvement: {best_agent['improvement']:.4f}")
            print(f"   Trading activity: {best_agent['trading_activity_pct']:.1f}%")
            print(f"   Convergence step: {best_agent['convergence_step']}")
            
            # Action distribution analysis
            print(f"\n🎯 ACTION DISTRIBUTION ANALYSIS:")
            for agent in ranked_agents:
                actions = agent['action_distribution']
                total = sum(actions.values())
                if total > 0:
                    hold_pct = actions[0] / total * 100
                    buy_pct = actions[1] / total * 100
                    sell_pct = actions[2] / total * 100
                    print(f"   {agent['agent_name']:15}: Hold={hold_pct:4.1f}%, Buy={buy_pct:4.1f}%, Sell={sell_pct:4.1f}%")
            
            # Overall assessment
            avg_return = np.mean([a['total_return'] for a in successful_agents])
            avg_trading = np.mean([a['trading_activity_pct'] for a in successful_agents])
            
            print(f"\n📈 ENSEMBLE SUMMARY:")
            print(f"   Average return: {avg_return:.4f}")
            print(f"   Average trading activity: {avg_trading:.1f}%")
            print(f"   Success rate: {len(successful_agents)/len(self.training_results)*100:.1f}%")
            
            if avg_return > 0.5:
                print(f"   ✅ EXCELLENT: Ensemble shows strong positive returns!")
            elif avg_return > 0.0:
                print(f"   ✅ GOOD: Ensemble shows positive returns")
            else:
                print(f"   ⚠️  NEEDS IMPROVEMENT: Returns below expectations")
        
        if failed_agents:
            print(f"\n❌ FAILED AGENTS:")
            for agent in failed_agents:
                print(f"   {agent['agent_name']}: {agent.get('error', 'Unknown error')}")
        
        print(f"\n📋 NEXT STEPS:")
        if successful_agents:
            print(f"   1. Use hybrid ensemble for evaluation with best {len(successful_agents)} agents")
            print(f"   2. Run: python3 task1_eval_hybrid.py")
            print(f"   3. Compare with baseline ensemble performance")
        else:
            print(f"   1. Debug training issues")
            print(f"   2. Adjust hyperparameters")
            print(f"   3. Consider single-agent training first")


def main():
    """Main training execution"""
    
    parser = argparse.ArgumentParser(description="Hybrid Ensemble Training with PPO")
    parser.add_argument("gpu_id", type=int, help="GPU ID (0, 1, etc. or -1 for CPU)")
    parser.add_argument("--reward", type=str, default="simple", 
                        choices=["simple", "transaction_cost_adjusted", "multi_objective"],
                        help="Reward function type")
    parser.add_argument("--team", type=str, default="hybrid_ensemble", help="Team name")
    
    args = parser.parse_args()
    
    print(f"🚀 HYBRID ENSEMBLE TRAINING - PHASE 2")
    print("=" * 80)
    print(f"Configuration:")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Reward Type: {args.reward}")
    print(f"   Team Name: {args.team}")
    
    # Create trainer
    trainer = HybridEnsembleTrainer(
        reward_type=args.reward,
        gpu_id=args.gpu_id,
        team_name=args.team
    )
    
    # Train ensemble
    results = trainer.train_ensemble()
    
    print(f"\n🎉 Hybrid ensemble training completed!")
    return results


if __name__ == "__main__":
    results = main()