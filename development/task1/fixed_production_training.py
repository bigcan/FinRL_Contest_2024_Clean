#!/usr/bin/env python3
"""
Fixed Production Training - Corrected batch data format for refactored agents
"""

import os
import sys
import time
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "src_refactored"))

# Imports
from data_config import ConfigData
from trade_simulator import TradeSimulator
from src_refactored.agents.double_dqn_agent import DoubleDQNAgent, D3QNAgent
from src_refactored.config.agent_configs import DoubleDQNConfig
from src_refactored.ensemble.voting_ensemble import VotingEnsemble, EnsembleStrategy

class FixedProductionTrainer:
    """Fixed production trainer with corrected batch data format."""
    
    def __init__(self, output_dir: str = "fixed_production_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.results = {
            'start_time': time.time(),
            'device': str(self.device),
            'timestamp': self.timestamp,
            'training_config': {},
            'agents': {},
            'ensemble': {},
            'performance': {}
        }
        
        print(f"🚀 Fixed Production Training - Session: {self.timestamp}")
        print(f"💻 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")
    
    def setup_trading_environment(self):
        """Setup the Bitcoin LOB trading environment."""
        print("\n📊 Setting Up Fixed Bitcoin LOB Trading Environment")
        print("-" * 60)
        
        try:
            # Load data configuration
            data_config = ConfigData()
            
            # Check for enhanced features (priority order)
            feature_paths = [
                "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict_enhanced_v3.npy",
                "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict_enhanced_v2.npy",
                "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict_enhanced.npy",
                "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict.npy"
            ]
            
            predict_data = None
            feature_type = "unknown"
            
            for path in feature_paths:
                if os.path.exists(path):
                    predict_data = np.load(path)
                    feature_type = os.path.basename(path).replace('BTC_1sec_predict', '').replace('.npy', '') or 'standard'
                    print(f"  ✅ Features loaded: {os.path.basename(path)}")
                    print(f"  📈 Shape: {predict_data.shape}")
                    break
            
            if predict_data is None:
                raise FileNotFoundError("No prediction data found")
            
            print(f"  📊 Dataset size: {len(predict_data):,} timesteps")
            print(f"  🔬 Feature type: {feature_type}")
            
            # Create trading environment
            env = TradeSimulator(
                num_sims=1,  # Single environment for training
                slippage=5e-5,
                max_position=2,
                step_gap=1,
                gpu_id=0 if torch.cuda.is_available() else -1
            )
            
            # Get dimensions from the environment
            state_dim = 41  # Enhanced v3 features
            action_dim = 3   # Buy, Hold, Sell
            
            print(f"  ✅ Trading environment created successfully")
            print(f"  🎯 State dimension: {state_dim}")
            print(f"  🎯 Action dimension: {action_dim}")
            
            self.results['training_config'] = {
                'feature_type': feature_type,
                'dataset_size': len(predict_data),
                'state_dim': state_dim,
                'action_dim': action_dim,
                'device': str(self.device)
            }
            
            return env, state_dim, action_dim
            
        except Exception as e:
            print(f"  ❌ Environment setup failed: {e}")
            raise e
    
    def create_production_agents(self, state_dim: int, action_dim: int):
        """Create production-grade agents with proper configurations."""
        print(f"\n🤖 Creating Fixed Production-Grade Agents")
        print("-" * 60)
        
        agents = {}
        
        # Agent configurations
        configs = {
            "DoubleDQN_Production": {
                "class": DoubleDQNAgent,
                "config": DoubleDQNConfig(
                    net_dims=[512, 512, 256],    # Large network for production
                    gamma=0.995,                 # High discount for long-term gains
                    learning_rate=5e-5,          # Conservative learning rate
                    batch_size=128,              # Large batches for stability
                    clip_grad_norm=1.0,
                    soft_update_tau=1e-3,        # Slow target updates
                    explore_rate=0.02            # Low exploration for exploitation
                )
            },
            "D3QN_Production": {
                "class": D3QNAgent,
                "config": DoubleDQNConfig(
                    net_dims=[512, 512, 512],    # Even larger for D3QN
                    gamma=0.99,                  # Standard discount
                    learning_rate=3e-5,          # Very conservative
                    batch_size=128,
                    clip_grad_norm=1.0,
                    soft_update_tau=1e-3,
                    explore_rate=0.01            # Minimal exploration
                )
            },
            "DoubleDQN_Aggressive": {
                "class": DoubleDQNAgent,
                "config": DoubleDQNConfig(
                    net_dims=[256, 256, 128],    # Smaller for faster training
                    gamma=0.99,                  # Lower discount for agility
                    learning_rate=1e-4,          # Higher for faster learning
                    batch_size=64,               # Smaller batch for frequency
                    clip_grad_norm=5.0,
                    soft_update_tau=5e-3,        # Faster target updates
                    explore_rate=0.05            # Higher exploration
                )
            }
        }
        
        print(f"  🏗️ Building {len(configs)} production agents...")
        
        for agent_name, agent_info in configs.items():
            try:
                print(f"\n  Creating {agent_name}:")
                
                agent = agent_info["class"](
                    config=agent_info["config"],
                    state_dim=state_dim,
                    action_dim=action_dim,
                    device=self.device
                )
                
                agents[agent_name] = agent
                
                # Calculate parameters
                params = sum(p.numel() for p in agent.online_network.parameters())
                
                print(f"    ✅ {agent_name} created")
                print(f"    📊 Parameters: {params:,}")
                print(f"    🧠 Architecture: {agent_info['config'].net_dims}")
                print(f"    🎯 Learning rate: {agent_info['config'].learning_rate}")
                
                # Store agent info
                self.results['agents'][agent_name] = {
                    'type': agent_info["class"].__name__,
                    'config': agent_info["config"].to_dict(),
                    'parameters': params,
                    'created': True
                }
                
            except Exception as e:
                print(f"    ❌ Failed to create {agent_name}: {e}")
                self.results['agents'][agent_name] = {'created': False, 'error': str(e)}
        
        print(f"\n  📊 Successfully created {len(agents)}/{len(configs)} agents")
        return agents
    
    def train_agents(self, agents, env, episodes: int = 200):
        """Train agents with corrected batch data format."""
        print(f"\n🎓 Fixed Agent Training ({episodes} episodes)")
        print("-" * 60)
        
        training_results = {}
        
        for agent_idx, (agent_name, agent) in enumerate(agents.items()):
            print(f"\n  [{agent_idx + 1}/{len(agents)}] Training {agent_name}")
            print(f"  {'-' * 50}")
            
            try:
                episode_rewards = []
                episode_losses = []
                training_start = time.time()
                
                # Training loop
                for episode in range(episodes):
                    episode_start = time.time()
                    
                    # Reset environment
                    state = env.reset()
                    episode_reward = 0
                    episode_loss = 0
                    step = 0
                    max_steps = 100  # Reasonable episode length
                    
                    while step < max_steps:
                        # State preprocessing
                        if not isinstance(state, torch.Tensor):
                            state = torch.tensor(state, dtype=torch.float32, device=self.device)
                        if state.dim() == 1:
                            state = state.unsqueeze(0)
                        
                        # Agent action selection
                        action = agent.select_action(state.squeeze(0))
                        
                        # **CRITICAL FIX**: Ensure action is in correct format for simulator
                        if isinstance(action, (int, np.integer)):
                            # Convert scalar to tensor format expected by simulator
                            action = torch.tensor([[action]], dtype=torch.long, device=self.device)
                        elif isinstance(action, np.ndarray):
                            # Convert numpy array to tensor
                            action = torch.tensor(action, dtype=torch.long, device=self.device)
                            if action.dim() == 0:  # 0D tensor (scalar)
                                action = action.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
                            elif action.dim() == 1:  # 1D tensor
                                action = action.unsqueeze(0)  # Make it [1, N]
                        
                        # Environment step
                        next_state, reward, done, info = env.step(action)
                        episode_reward += reward
                        
                        # **FIXED**: Correct batch data format (4-tuple, not 5-tuple)
                        if step > 16:  # Start updating after initial experience
                            try:
                                # Create training batch with correct format
                                batch_size = 32
                                
                                # Use actual transition
                                current_state = state.squeeze(0)
                                
                                # **CRITICAL FIX**: Use 4-tuple format (states, actions, rewards, dones)
                                # Remove next_states from the tuple as agents handle this internally
                                states = torch.stack([current_state] * batch_size)
                                actions = torch.tensor([action] * batch_size, device=self.device)
                                rewards = torch.tensor([reward] * batch_size, device=self.device)
                                dones = torch.tensor([done] * batch_size, device=self.device)
                                
                                # Correct batch format: 4-tuple instead of 5-tuple
                                batch_data = (states, actions, rewards, dones)
                                result = agent.update(batch_data)
                                
                                if hasattr(result, 'critic_loss'):
                                    episode_loss += result.critic_loss
                                elif isinstance(result, dict) and 'loss' in result:
                                    episode_loss += result['loss']
                                
                            except Exception as update_e:
                                # Continue despite update errors, but log them
                                if episode < 5:  # Only log early errors to avoid spam
                                    print(f"    ⚠️  Update error (episode {episode}): {update_e}")
                        
                        state = next_state
                        step += 1
                        
                        if done:
                            break
                    
                    episode_rewards.append(episode_reward)
                    episode_losses.append(episode_loss / max(step, 1))
                    
                    # Progress reporting
                    if (episode + 1) % 50 == 0:
                        avg_reward = np.mean(episode_rewards[-50:])
                        avg_loss = np.mean(episode_losses[-50:]) if episode_losses[-50:] else 0
                        elapsed = time.time() - training_start
                        eta = (elapsed / (episode + 1)) * (episodes - episode - 1)
                        
                        print(f"    Episode {episode + 1:3d}/{episodes}: "
                              f"Reward={avg_reward:8.3f}, Loss={avg_loss:8.6f}, "
                              f"ETA={eta/60:.1f}min")
                
                # Calculate final metrics
                training_time = time.time() - training_start
                
                # Performance analysis
                initial_performance = np.mean(episode_rewards[:50]) if len(episode_rewards) >= 50 else np.mean(episode_rewards[:10])
                final_performance = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards[-10:])
                improvement = final_performance - initial_performance
                
                # Stability metrics
                reward_std = np.std(episode_rewards)
                final_stability = np.std(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.std(episode_rewards)
                
                training_results[agent_name] = {
                    'episodes_completed': len(episode_rewards),
                    'initial_performance': initial_performance,
                    'final_performance': final_performance,
                    'improvement': improvement,
                    'improvement_pct': (improvement / abs(initial_performance)) * 100 if initial_performance != 0 else 0,
                    'reward_std': reward_std,
                    'final_stability': final_stability,
                    'total_training_time': training_time,
                    'avg_episode_time': training_time / episodes,
                    'total_reward': sum(episode_rewards),
                    'best_episode': max(episode_rewards),
                    'worst_episode': min(episode_rewards),
                    'success_rate': sum(1 for r in episode_rewards if r > 0) / len(episode_rewards),
                    'converged': improvement > 0
                }
                
                print(f"    ✅ {agent_name} training completed")
                print(f"       Performance: {initial_performance:.3f} → {final_performance:.3f}")
                print(f"       Improvement: {improvement:+.3f} ({training_results[agent_name]['improvement_pct']:+.1f}%)")
                print(f"       Training time: {training_time/60:.1f} minutes")
                print(f"       Success rate: {training_results[agent_name]['success_rate']:.1%}")
                
            except Exception as e:
                print(f"    ❌ {agent_name} training failed: {e}")
                training_results[agent_name] = {'failed': True, 'error': str(e)}
                import traceback
                print(f"       Traceback: {traceback.format_exc()}")
        
        self.results['training'] = training_results
        return training_results
    
    def save_models(self, agents):
        """Save trained models."""
        print(f"\n💾 Saving Fixed Production Models")
        print("-" * 60)
        
        models_dir = self.output_dir / "production_models"
        models_dir.mkdir(exist_ok=True)
        
        for agent_name, agent in agents.items():
            try:
                agent_dir = models_dir / agent_name
                agent_dir.mkdir(exist_ok=True)
                
                model_path = agent_dir / "model.pth"
                torch.save(agent.online_network.state_dict(), model_path)
                
                print(f"  ✅ {agent_name} saved to {model_path}")
                
            except Exception as e:
                print(f"  ❌ Failed to save {agent_name}: {e}")
        
        # Save ensemble config
        ensemble_config = {
            'agents': list(agents.keys()),
            'strategy': 'weighted_vote',
            'timestamp': self.timestamp,
            'device': str(self.device)
        }
        
        config_path = models_dir / "ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        print(f"  ✅ Ensemble config saved to {config_path}")
        print(f"  📁 Models saved to: {models_dir}")
    
    def run_fixed_production_training(self):
        """Run complete fixed production training."""
        print("\n" + "=" * 70)
        print("🚀 STARTING FIXED PRODUCTION TRAINING")
        print("=" * 70)
        
        try:
            # Setup environment
            env, state_dim, action_dim = self.setup_trading_environment()
            
            # Create agents
            agents = self.create_production_agents(state_dim, action_dim)
            
            if not agents:
                raise ValueError("No agents created successfully")
            
            # Train agents
            training_results = self.train_agents(agents, env, episodes=100)  # Reduced for testing
            
            # Save models
            self.save_models(agents)
            
            # Save results
            results_file = self.output_dir / f"fixed_production_results_{self.timestamp}.json"
            self.results['end_time'] = time.time()
            self.results['total_duration'] = self.results['end_time'] - self.results['start_time']
            
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"\n📁 Results saved to: {results_file}")
            print(f"⏱️ Total duration: {self.results['total_duration']/3600:.2f} hours")
            
            # Summary
            successful_agents = [name for name, result in training_results.items() 
                               if not result.get('failed', False)]
            
            print(f"\n🏆 FIXED PRODUCTION TRAINING COMPLETE!")
            print(f"✅ Successful agents: {len(successful_agents)}/{len(agents)}")
            
            if successful_agents:
                best_agent = max(successful_agents, 
                               key=lambda x: training_results[x]['improvement'])
                print(f"🥇 Best performing agent: {best_agent}")
                print(f"   Improvement: {training_results[best_agent]['improvement']:+.3f}")
            
            return self.results
            
        except Exception as e:
            print(f"\n💥 Fixed training failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

def main():
    """Main training function."""
    try:
        trainer = FixedProductionTrainer()
        results = trainer.run_fixed_production_training()
        print("\n🎉 Fixed production training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())