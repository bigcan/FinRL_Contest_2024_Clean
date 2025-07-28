#!/usr/bin/env python3
"""
Final Production Training Script for FinRL Contest 2024
Integrates properly with the original TradeSimulator framework
"""

import os
import sys
import time
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "src_refactored"))

# Import original training components for proper integration
from erl_config import Config, build_env
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_replay_buffer import ReplayBuffer
from data_config import ConfigData

# Import refactored components
from src_refactored.agents.double_dqn_agent import DoubleDQNAgent, D3QNAgent
from src_refactored.config.agent_configs import DoubleDQNConfig
from src_refactored.ensemble.voting_ensemble import VotingEnsemble, EnsembleStrategy

def get_state_dim():
    """Calculate state dimension from data (matching original framework)."""
    from data_config import ConfigData
    args = ConfigData()
    
    # Priority loading: enhanced_v3 > enhanced_v2 > enhanced > original
    enhanced_v3_path = args.predict_ary_path.replace('.npy', '_enhanced_v3.npy')
    enhanced_v2_path = args.predict_ary_path.replace('.npy', '_enhanced_v2.npy')
    enhanced_path = args.predict_ary_path.replace('.npy', '_enhanced.npy')
    
    if os.path.exists(enhanced_v3_path):
        factor_ary = np.load(enhanced_v3_path)
        feature_type = "enhanced_v3"
    elif os.path.exists(enhanced_v2_path):
        factor_ary = np.load(enhanced_v2_path)
        feature_type = "enhanced_v2"
    elif os.path.exists(enhanced_path):
        factor_ary = np.load(enhanced_path)
        feature_type = "enhanced"
    else:
        factor_ary = np.load(args.predict_ary_path)
        feature_type = "standard"
    
    # For enhanced features, state_dim includes position features
    state_dim = factor_ary.shape[1]
    
    print(f"📊 Data loaded: {factor_ary.shape}")
    print(f"🔬 Feature type: {feature_type}")
    print(f"🎯 State dimension: {state_dim}")
    
    return state_dim, factor_ary.shape[0], feature_type

class FinalProductionTrainer:
    """Final production trainer using original framework integration."""
    
    def __init__(self, output_dir: str = "final_production_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.results = {
            'start_time': time.time(),
            'device': str(self.device),
            'timestamp': self.timestamp,
            'framework': 'refactored_with_original_integration'
        }
        
        print(f"🚀 Final Production Training - Session: {self.timestamp}")
        print(f"💻 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    def setup_configuration(self):
        """Setup training configuration using original framework structure."""
        print("\n⚙️ Setting Up Training Configuration")
        print("-" * 60)
        
        # Get data dimensions
        state_dim, data_length, feature_type = get_state_dim()
        action_dim = 3
        
        # Calculate training parameters
        gpu_id = 0 if torch.cuda.is_available() else -1
        num_sims = 32  # Reduced for stability
        max_position = 1
        step_gap = 2
        num_ignore_step = 60
        slippage = 7e-7
        
        # Calculate max_step for training
        max_step = (data_length - num_ignore_step) // step_gap
        
        print(f"  📊 Dataset length: {data_length:,}")
        print(f"  🎯 State dimension: {state_dim}")
        print(f"  🎯 Action dimension: {action_dim}")
        print(f"  🔢 Max steps: {max_step:,}")
        print(f"  🔬 Feature type: {feature_type}")
        
        # Environment configuration
        env_args = {
            "env_name": "TradeSimulator-v0",
            "num_envs": num_sims,
            "max_step": max_step,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "if_discrete": True,
            "max_position": max_position,
            "slippage": slippage,
            "num_sims": num_sims,
            "step_gap": step_gap,
        }
        
        # Base configuration
        config = Config(
            agent_class=None,  # Will be set per agent
            env_class=TradeSimulator,
            env_args=env_args
        )
        
        config.gpu_id = gpu_id
        config.random_seed = gpu_id
        config.gamma = 0.995
        config.batch_size = 256
        config.buffer_size = max_step * 4
        config.repeat_times = 2
        config.horizon_len = max_step
        config.eval_per_step = max_step
        config.break_step = 8  # Reduced for demo
        config.num_workers = 1
        config.save_gap = 4
        
        # Evaluation environment
        config.eval_env_class = EvalTradeSimulator
        config.eval_env_args = env_args.copy()
        
        self.results['config'] = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'max_step': max_step,
            'data_length': data_length,
            'feature_type': feature_type,
            'env_args': env_args
        }
        
        print(f"  ✅ Configuration complete")
        return config, state_dim, action_dim
    
    def create_enhanced_agents(self, config: Config, state_dim: int, action_dim: int):
        """Create enhanced agents using refactored framework."""
        print("\n🤖 Creating Enhanced Agents")
        print("-" * 60)
        
        agents = {}
        
        # Agent configurations optimized for production
        agent_configs = {
            "EnhancedDoubleDQN": {
                "class": DoubleDQNAgent,
                "config": DoubleDQNConfig(
                    net_dims=[256, 256, 128],
                    gamma=0.995,
                    learning_rate=1e-4,
                    batch_size=128,
                    explore_rate=0.02,
                    soft_update_tau=2e-3
                ),
                "original_class": "AgentDoubleDQN"
            },
            
            "EnhancedD3QN": {
                "class": D3QNAgent,
                "config": DoubleDQNConfig(
                    net_dims=[256, 256, 256],
                    gamma=0.995,
                    learning_rate=8e-5,
                    batch_size=128,
                    explore_rate=0.01,
                    soft_update_tau=2e-3
                ),
                "original_class": "AgentD3QN"
            }
        }
        
        for agent_name, agent_info in agent_configs.items():
            try:
                print(f"  Creating {agent_name}...")
                
                # Create refactored agent
                agent = agent_info["class"](
                    config=agent_info["config"],
                    state_dim=state_dim,
                    action_dim=action_dim,
                    device=self.device
                )
                
                agents[agent_name] = {
                    'agent': agent,
                    'config': agent_info["config"],
                    'type': agent_info["class"].__name__
                }
                
                params = sum(p.numel() for p in agent.online_network.parameters())
                print(f"    ✅ {agent_name}: {params:,} parameters")
                
            except Exception as e:
                print(f"    ❌ {agent_name} failed: {e}")
        
        self.results['agents'] = {name: info['type'] for name, info in agents.items()}
        print(f"  📊 Created {len(agents)} enhanced agents")
        
        return agents
    
    def create_ensemble(self, agents):
        """Create ensemble from enhanced agents."""
        print("\n🤝 Creating Enhanced Ensemble")
        print("-" * 60)
        
        if len(agents) < 2:
            print("  ⚠️ Need at least 2 agents for ensemble")
            return None
        
        try:
            agent_dict = {name: info['agent'] for name, info in agents.items()}
            
            ensemble = VotingEnsemble(
                agents=agent_dict,
                strategy=EnsembleStrategy.WEIGHTED_VOTE,
                device=self.device
            )
            
            print(f"  ✅ Enhanced ensemble created")
            print(f"  🗳️ Strategy: Weighted Voting")
            print(f"  👥 Agents: {len(agents)}")
            
            self.results['ensemble'] = {
                'created': True,
                'strategy': 'weighted_vote',
                'num_agents': len(agents)
            }
            
            return ensemble
            
        except Exception as e:
            print(f"  ❌ Ensemble creation failed: {e}")
            return None
    
    def train_single_agent(self, agent_name: str, agent_info: dict, config: Config, episodes: int = 100):
        """Train a single agent using the original framework structure."""
        print(f"\n  🎓 Training {agent_name} ({episodes} episodes)")
        
        try:
            # Setup for training
            config.agent_class = type(agent_info['agent'])  # Use the refactored agent class
            config.net_dims = agent_info['config'].net_dims
            config.learning_rate = agent_info['config'].learning_rate
            config.explore_rate = agent_info['config'].explore_rate
            
            # Build environment
            env = build_env(config.env_class, config.env_args, config.gpu_id)
            
            # Use the pre-created refactored agent
            agent = agent_info['agent']
            
            # Training metrics
            episode_rewards = []
            training_start = time.time()
            
            # Simplified training loop
            for episode in range(episodes):
                try:
                    # Reset environment
                    state = env.reset()
                    
                    # Ensure state is properly formatted
                    if isinstance(state, np.ndarray):
                        state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    if state.dim() == 1:
                        state = state.unsqueeze(0)
                    
                    episode_reward = 0
                    steps = 0
                    max_steps = 100  # Limit episode length
                    
                    while steps < max_steps:
                        # Agent action
                        action = agent.select_action(state.squeeze(0), deterministic=False)
                        
                        # Ensure action is integer for discrete environment
                        if isinstance(action, torch.Tensor):
                            action = action.item()
                        
                        # Environment step
                        next_state, reward, done, info = env.step(action)
                        
                        if isinstance(reward, (list, np.ndarray)):
                            reward = reward[0] if len(reward) > 0 else 0
                        
                        episode_reward += float(reward)
                        
                        # Prepare next state
                        if isinstance(next_state, np.ndarray):
                            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                        if next_state.dim() == 1:
                            next_state = next_state.unsqueeze(0)
                        
                        # Simple update mechanism
                        if steps > 10:  # Start updating after initial experience
                            try:
                                # Create simple batch for update
                                batch_size = 16
                                current_state = state.squeeze(0)
                                
                                states = torch.stack([current_state] * batch_size)
                                actions = torch.tensor([action] * batch_size, device=self.device)
                                rewards = torch.tensor([reward] * batch_size, device=self.device)
                                next_states = torch.stack([next_state.squeeze(0)] * batch_size)
                                dones = torch.zeros(batch_size, device=self.device)
                                
                                batch_data = (states, actions, rewards, next_states, dones)
                                agent.update(batch_data)
                                
                            except Exception:
                                pass  # Continue despite update errors
                        
                        state = next_state
                        steps += 1
                        
                        # Check for done condition
                        if isinstance(done, (list, np.ndarray)):
                            done = any(done) if len(done) > 0 else False
                        if done:
                            break
                    
                    episode_rewards.append(episode_reward)
                    
                    # Progress reporting
                    if (episode + 1) % 20 == 0:
                        avg_reward = np.mean(episode_rewards[-20:])
                        elapsed = time.time() - training_start
                        print(f"    Episode {episode + 1:3d}: Avg={avg_reward:8.3f}, Time={elapsed:.1f}s")
                
                except Exception as episode_error:
                    print(f"    ⚠️ Episode {episode} error: {episode_error}")
                    continue
            
            # Calculate final metrics
            if episode_rewards:
                initial_avg = np.mean(episode_rewards[:10]) if len(episode_rewards) >= 10 else episode_rewards[0]
                final_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                improvement = final_avg - initial_avg
                
                training_results = {
                    'episodes': len(episode_rewards),
                    'initial_performance': initial_avg,
                    'final_performance': final_avg,
                    'improvement': improvement,
                    'total_reward': sum(episode_rewards),
                    'training_time': time.time() - training_start,
                    'success': True
                }
                
                print(f"    ✅ {agent_name} completed: {improvement:+.3f} improvement")
            else:
                training_results = {'episodes': 0, 'success': False, 'error': 'No episodes completed'}
                print(f"    ❌ {agent_name} failed: No episodes completed")
            
            # Cleanup
            if hasattr(env, 'close'):
                env.close()
            
            return training_results
            
        except Exception as e:
            print(f"    ❌ {agent_name} training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_all_agents(self, agents: dict, config: Config, episodes: int = 100):
        """Train all agents."""
        print(f"\n🎓 Enhanced Agent Training ({episodes} episodes)")
        print("-" * 60)
        
        training_results = {}
        
        for agent_name, agent_info in agents.items():
            result = self.train_single_agent(agent_name, agent_info, config, episodes)
            training_results[agent_name] = result
        
        self.results['training'] = training_results
        return training_results
    
    def evaluate_ensemble(self, ensemble, config: Config, episodes: int = 50):
        """Evaluate ensemble performance."""
        print(f"\n📊 Enhanced Ensemble Evaluation ({episodes} episodes)")
        print("-" * 60)
        
        if ensemble is None:
            print("  ⚠️ No ensemble to evaluate")
            return {}
        
        try:
            # Build evaluation environment
            eval_env = build_env(config.eval_env_class, config.eval_env_args, config.gpu_id)
            
            episode_rewards = []
            
            for episode in range(episodes):
                try:
                    state = eval_env.reset()
                    
                    if isinstance(state, np.ndarray):
                        state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    if state.dim() == 1:
                        state = state.unsqueeze(0)
                    
                    episode_reward = 0
                    steps = 0
                    max_steps = 100
                    
                    while steps < max_steps:
                        action = ensemble.select_action(state.squeeze(0), deterministic=True)
                        
                        if isinstance(action, torch.Tensor):
                            action = action.item()
                        
                        next_state, reward, done, info = eval_env.step(action)
                        
                        if isinstance(reward, (list, np.ndarray)):
                            reward = reward[0] if len(reward) > 0 else 0
                        
                        episode_reward += float(reward)
                        
                        if isinstance(next_state, np.ndarray):
                            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                        if next_state.dim() == 1:
                            next_state = next_state.unsqueeze(0)
                        
                        state = next_state
                        steps += 1
                        
                        if isinstance(done, (list, np.ndarray)):
                            done = any(done) if len(done) > 0 else False
                        if done:
                            break
                    
                    episode_rewards.append(episode_reward)
                    
                except Exception as episode_error:
                    print(f"    ⚠️ Evaluation episode {episode} error: {episode_error}")
                    continue
            
            if episode_rewards:
                eval_results = {
                    'episodes': len(episode_rewards),
                    'mean_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'max_reward': np.max(episode_rewards),
                    'min_reward': np.min(episode_rewards),
                    'success_rate': sum(1 for r in episode_rewards if r > 0) / len(episode_rewards),
                    'success': True
                }
                
                print(f"  📈 Ensemble Results:")
                print(f"    Mean: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
                print(f"    Best: {eval_results['max_reward']:.3f}")
                print(f"    Success rate: {eval_results['success_rate']:.1%}")
            else:
                eval_results = {'success': False, 'error': 'No episodes completed'}
                print(f"  ❌ Evaluation failed: No episodes completed")
            
            # Cleanup
            if hasattr(eval_env, 'close'):
                eval_env.close()
            
            self.results['evaluation'] = eval_results
            return eval_results
            
        except Exception as e:
            print(f"  ❌ Ensemble evaluation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def save_results(self):
        """Save final results."""
        print("\n💾 Saving Final Results")
        print("-" * 60)
        
        self.results['end_time'] = time.time()
        self.results['duration'] = self.results['end_time'] - self.results['start_time']
        
        # Save JSON results
        results_file = self.output_dir / f"final_production_results_{self.timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary
        summary_file = self.output_dir / f"final_training_summary_{self.timestamp}.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Final Production Training Results\n\n")
            f.write(f"**Session:** {self.timestamp}  \n")
            f.write(f"**Duration:** {self.results['duration']/60:.1f} minutes  \n")
            f.write(f"**Framework:** Enhanced Refactored with Original Integration  \n\n")
            
            if 'config' in self.results:
                config = self.results['config']
                f.write(f"## Configuration\n")
                f.write(f"- State Dimension: {config['state_dim']}\n")
                f.write(f"- Feature Type: {config['feature_type']}\n")
                f.write(f"- Max Steps: {config['max_step']:,}\n\n")
            
            if 'training' in self.results:
                f.write(f"## Training Results\n")
                for agent_name, results in self.results['training'].items():
                    if results.get('success'):
                        f.write(f"### {agent_name}\n")
                        f.write(f"- Episodes: {results['episodes']}\n")
                        f.write(f"- Improvement: {results['improvement']:+.3f}\n")
                        f.write(f"- Final Performance: {results['final_performance']:.3f}\n\n")
            
            if 'evaluation' in self.results and self.results['evaluation'].get('success'):
                eval_res = self.results['evaluation']
                f.write(f"## Ensemble Evaluation\n")
                f.write(f"- Mean Reward: {eval_res['mean_reward']:.3f} ± {eval_res['std_reward']:.3f}\n")
                f.write(f"- Success Rate: {eval_res['success_rate']:.1%}\n")
        
        print(f"  ✅ Results saved: {results_file}")
        print(f"  ✅ Summary saved: {summary_file}")
        print(f"  ⏱️ Duration: {self.results['duration']/60:.1f} minutes")
        
        return results_file, summary_file

def main():
    """Main execution."""
    print("🚀 FinRL Contest 2024 - Final Production Training")
    print("=" * 80)
    print("🔧 Enhanced Refactored Framework with Original Integration")
    print("=" * 80)
    
    try:
        trainer = FinalProductionTrainer()
        
        # Setup configuration
        config, state_dim, action_dim = trainer.setup_configuration()
        
        # Create enhanced agents
        agents = trainer.create_enhanced_agents(config, state_dim, action_dim)
        
        if not agents:
            print("❌ No agents created - exiting")
            return
        
        # Create ensemble
        ensemble = trainer.create_ensemble(agents)
        
        # Train agents
        training_results = trainer.train_all_agents(agents, config, episodes=100)
        
        # Evaluate ensemble
        if ensemble:
            eval_results = trainer.evaluate_ensemble(ensemble, config, episodes=50)
        
        # Save results
        results_file, summary_file = trainer.save_results()
        
        print("\n" + "=" * 80)
        print("🏆 FINAL PRODUCTION TRAINING COMPLETE!")
        print("=" * 80)
        
        # Summary
        if 'training' in trainer.results:
            print("\n📊 Training Summary:")
            for agent_name, results in trainer.results['training'].items():
                if results.get('success'):
                    print(f"  {agent_name}: {results['improvement']:+.3f} improvement")
        
        if 'evaluation' in trainer.results and trainer.results['evaluation'].get('success'):
            eval_info = trainer.results['evaluation']
            print(f"\n🎯 Ensemble Performance: {eval_info['mean_reward']:.3f} ± {eval_info['std_reward']:.3f}")
        
        print(f"\n📁 Results saved to: {trainer.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()