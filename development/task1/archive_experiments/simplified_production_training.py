#!/usr/bin/env python3
"""
Simplified Production Training - Using corrected GPU infrastructure
Bypasses complex ensemble frameworks to focus on core training functionality
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import random
from collections import deque

# Add src path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

class SimpleTradingAgent:
    """Simplified trading agent using corrected infrastructure."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-4, device='cuda'):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        ).to(self.device)
        
        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Training parameters
        self.gamma = 0.99
        self.epsilon = 0.1
        self.tau = 0.005  # Soft update rate
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Copy weights to target network
        self.update_target_network()
        
        print(f"🤖 Agent created with {sum(p.numel() for p in self.q_network.parameters()):,} parameters")
    
    def update_target_network(self):
        """Update target network with soft update."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """Update the network using experience replay."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_target_network()
        
        return loss.item()
    
    def save(self, filepath):
        """Save model state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, filepath)
        
    def load(self, filepath):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

class SimplifiedTrainingEnvironment:
    """Simplified trading environment using corrected data loading."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_data()
        self.reset()
    
    def load_data(self):
        """Load Bitcoin LOB data with corrected infrastructure."""
        print("📊 Loading Bitcoin LOB Data")
        print("-" * 40)
        
        # Load enhanced features v3 (41D)
        data_path = "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict_enhanced_v3.npy"
        price_path = "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec.csv"
        
        if os.path.exists(data_path):
            self.features = np.load(data_path).astype(np.float32)
            print(f"✅ Enhanced features loaded: {self.features.shape}")
        else:
            raise FileNotFoundError(f"Features not found: {data_path}")
        
        # Load price data for rewards
        import pandas as pd
        if os.path.exists(price_path):
            try:
                price_df = pd.read_csv(price_path)
                # Try different column names
                price_col = None
                for col in ['close', 'Close', 'price', 'Price', 'mid_price']:
                    if col in price_df.columns:
                        price_col = col
                        break
                
                if price_col:
                    self.prices = price_df[price_col].values.astype(np.float32)
                    print(f"✅ Price data loaded: {len(self.prices)} timesteps from '{price_col}' column")
                else:
                    # Use first numeric column
                    numeric_cols = price_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        self.prices = price_df[numeric_cols[0]].values.astype(np.float32)
                        print(f"✅ Price data loaded: {len(self.prices)} timesteps from '{numeric_cols[0]}' column")
                    else:
                        raise ValueError("No numeric columns found")
            except Exception as e:
                print(f"⚠️  CSV loading failed ({e}), using synthetic prices")
                self.prices = np.random.randn(len(self.features)).cumsum() + 60000
        else:
            # Generate synthetic prices for demonstration
            self.prices = np.random.randn(len(self.features)).cumsum() + 60000
            print("⚠️  Using synthetic price data")
        
        # Align data lengths
        min_len = min(len(self.features), len(self.prices))
        self.features = self.features[:min_len]
        self.prices = self.prices[:min_len]
        
        print(f"📈 Final dataset: {len(self.features)} timesteps, {self.features.shape[1]} features")
    
    def reset(self):
        """Reset environment state."""
        self.current_step = 0
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.cash = 10000.0
        self.portfolio_value = self.cash
        self.max_steps = len(self.features) - 100  # Leave buffer
        return self.get_state()
    
    def get_state(self):
        """Get current state (features + position)."""
        if self.current_step >= len(self.features):
            return np.zeros(self.features.shape[1] + 1, dtype=np.float32)
        
        # Features + position
        state = np.concatenate([
            self.features[self.current_step],
            [self.position]
        ]).astype(np.float32)
        
        return state
    
    def step(self, action):
        """Execute trading action."""
        if self.current_step >= self.max_steps:
            return self.get_state(), 0.0, True, {}
        
        # Actions: 0=Buy, 1=Hold, 2=Sell
        old_position = self.position
        current_price = self.prices[self.current_step]
        
        # Execute action
        if action == 0 and self.position != 1:  # Buy
            self.position = 1
        elif action == 2 and self.position != -1:  # Sell
            self.position = -1
        # else: Hold (action == 1 or no change)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward based on price movement and position
        if self.current_step < len(self.prices):
            next_price = self.prices[self.current_step]
            price_change = (next_price - current_price) / current_price
            
            # Reward is based on position alignment with price movement
            if self.position == 1:  # Long position
                reward = price_change * 100  # Scale reward
            elif self.position == -1:  # Short position
                reward = -price_change * 100
            else:  # Neutral position
                reward = -abs(price_change) * 10  # Small penalty for not taking position
            
            # Add small penalty for excessive trading
            if old_position != self.position:
                reward -= 0.1  # Transaction cost
        else:
            reward = 0.0
            
        done = self.current_step >= self.max_steps
        
        return self.get_state(), reward, done, {}

class SimplifiedProductionTrainer:
    """Simplified production trainer using corrected infrastructure."""
    
    def __init__(self, output_dir="simplified_production_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"🚀 Simplified Production Training - Session: {self.timestamp}")
        print(f"💻 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    def train_agent(self, agent, env, episodes=500, save_frequency=100):
        """Train a single agent."""
        print(f"\n🎓 Training Agent for {episodes} episodes")
        print("-" * 50)
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            while True:
                # Select action
                action = agent.select_action(state, training=True)
                
                # Environment step
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Update agent
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.update()
                    episode_loss += loss
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / max(steps, 1))
            
            # Reduce exploration over time
            if episode % 100 == 0:
                agent.epsilon = max(0.01, agent.epsilon * 0.995)
            
            # Progress reporting
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_loss = np.mean(episode_losses[-50:])
                print(f"Episode {episode + 1:4d}/{episodes}: "
                      f"Reward={avg_reward:8.3f}, Loss={avg_loss:8.6f}, "
                      f"ε={agent.epsilon:.3f}")
            
            # Save checkpoint
            if (episode + 1) % save_frequency == 0:
                model_path = self.output_dir / f"agent_checkpoint_{episode + 1}.pth"
                agent.save(model_path)
                print(f"📁 Checkpoint saved: {model_path}")
        
        return episode_rewards, episode_losses
    
    def run_production_training(self):
        """Run complete production training."""
        print("\n" + "=" * 70)
        print("🚀 STARTING SIMPLIFIED PRODUCTION TRAINING")
        print("=" * 70)
        
        # Create environment
        env = SimplifiedTrainingEnvironment()
        state_dim = env.get_state().shape[0]
        action_dim = 3  # Buy, Hold, Sell
        
        print(f"🌍 Environment: {state_dim}D state space, {action_dim} actions")
        
        # Create agents with different configurations
        agents = {
            'Conservative_Agent': SimpleTradingAgent(
                state_dim=state_dim, 
                action_dim=action_dim,
                hidden_dim=256,
                lr=1e-4,
                device=self.device
            ),
            'Aggressive_Agent': SimpleTradingAgent(
                state_dim=state_dim,
                action_dim=action_dim, 
                hidden_dim=512,
                lr=3e-4,
                device=self.device
            ),
            'Balanced_Agent': SimpleTradingAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=384,
                lr=2e-4,
                device=self.device
            )
        }
        
        # Train each agent
        results = {}
        
        for agent_name, agent in agents.items():
            print(f"\n🤖 Training {agent_name}")
            print("=" * 50)
            
            try:
                episode_rewards, episode_losses = self.train_agent(
                    agent, env, episodes=300, save_frequency=100
                )
                
                # Save final model
                final_model_path = self.output_dir / f"{agent_name}_final.pth"
                agent.save(final_model_path)
                
                # Calculate metrics
                initial_perf = np.mean(episode_rewards[:50])
                final_perf = np.mean(episode_rewards[-50:])
                improvement = final_perf - initial_perf
                
                results[agent_name] = {
                    'episodes_completed': len(episode_rewards),
                    'initial_performance': float(initial_perf),
                    'final_performance': float(final_perf),
                    'improvement': float(improvement),
                    'total_reward': float(np.sum(episode_rewards)),
                    'best_episode': float(np.max(episode_rewards)),
                    'worst_episode': float(np.min(episode_rewards)),
                    'avg_loss': float(np.mean(episode_losses)),
                    'model_path': str(final_model_path),
                    'converged': improvement > 0
                }
                
                print(f"✅ {agent_name} completed successfully!")
                print(f"   Performance: {initial_perf:.3f} → {final_perf:.3f}")
                print(f"   Improvement: {improvement:+.3f}")
                print(f"   Total reward: {np.sum(episode_rewards):,.0f}")
                
            except Exception as e:
                print(f"❌ {agent_name} training failed: {e}")
                results[agent_name] = {'failed': True, 'error': str(e)}
        
        # Save results
        results_file = self.output_dir / f"training_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
        
        # Summary
        successful_agents = [name for name, result in results.items() 
                           if not result.get('failed', False)]
        
        print(f"\n🏆 TRAINING COMPLETE!")
        print(f"✅ Successful agents: {len(successful_agents)}/{len(agents)}")
        print(f"📁 Models saved to: {self.output_dir}")
        
        if successful_agents:
            best_agent = max(successful_agents, 
                           key=lambda x: results[x]['improvement'])
            print(f"🥇 Best performing agent: {best_agent}")
            print(f"   Improvement: {results[best_agent]['improvement']:+.3f}")
        
        return results

def main():
    """Main training function."""
    try:
        trainer = SimplifiedProductionTrainer()
        results = trainer.run_production_training()
        print("\n🎉 Production training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())