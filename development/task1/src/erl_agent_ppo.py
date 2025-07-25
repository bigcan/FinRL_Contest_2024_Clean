#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Agent for Cryptocurrency Trading
Implements state-of-the-art policy gradient method with stable learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from collections import deque
import time

from erl_net import ActorDiscretePPO, CriticAdv
from erl_replay_buffer import ReplayBuffer


class AgentPPO:
    """
    Proximal Policy Optimization agent optimized for discrete trading actions
    
    Key improvements over DQN:
    1. Policy gradient method - directly optimizes policy
    2. Clipped objective prevents destructive policy updates
    3. Advantage estimation reduces variance
    4. On-policy learning with experience replay
    """
    
    def __init__(self, 
                 net_dims: tuple,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 args=None):
        """
        Initialize PPO agent
        
        Args:
            net_dims: Network architecture (hidden layer sizes)
            state_dim: State space dimension
            action_dim: Action space dimension (3 for Hold/Buy/Sell)
            gpu_id: GPU device ID (-1 for CPU)
            args: Training configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_dims = net_dims
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")
        
        # PPO-specific hyperparameters
        self.clip_ratio = getattr(args, 'ppo_clip_ratio', 0.2)  # Clipping parameter
        self.policy_epochs = getattr(args, 'ppo_policy_epochs', 4)  # Policy update epochs
        self.value_epochs = getattr(args, 'ppo_value_epochs', 4)  # Value update epochs
        self.gae_lambda = getattr(args, 'ppo_gae_lambda', 0.95)  # GAE parameter
        self.entropy_coeff = getattr(args, 'ppo_entropy_coeff', 0.01)  # Entropy bonus
        self.value_loss_coeff = getattr(args, 'ppo_value_loss_coeff', 0.5)  # Value loss weight
        self.max_grad_norm = getattr(args, 'ppo_max_grad_norm', 0.5)  # Gradient clipping
        
        # Standard RL hyperparameters
        self.learning_rate = getattr(args, 'learning_rate', 3e-4)
        self.gamma = getattr(args, 'gamma', 0.995)
        self.batch_size = getattr(args, 'batch_size', 512)
        
        # Networks
        self.act = ActorDiscretePPO(net_dims, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dims, state_dim, 1).to(self.device)  # Value function
        
        # Optimizers
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        
        # Learning rate schedulers (if enabled)
        self.use_lr_scheduler = getattr(args, 'use_lr_scheduler', False)
        if self.use_lr_scheduler:
            self.act_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.act_optimizer, T_max=getattr(args, 'break_step', 200)
            )
            self.cri_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.cri_optimizer, T_max=getattr(args, 'break_step', 200)
            )
        
        # Training metrics
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        self.clip_fractions = deque(maxlen=100)
        
        # State tracking
        self.last_state = None
        
        print(f"🤖 PPO Agent initialized:")
        print(f"   Device: {self.device}")
        print(f"   Networks: Actor{net_dims} + Critic{net_dims}")
        print(f"   Clip ratio: {self.clip_ratio}")
        print(f"   Policy epochs: {self.policy_epochs}")
        print(f"   GAE lambda: {self.gae_lambda}")
        print(f"   Learning rate: {self.learning_rate}")
    
    def select_actions(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select actions using current policy
        
        Returns:
            actions: Selected actions
            log_probs: Log probabilities of selected actions  
            values: State value estimates
        """
        with torch.no_grad():
            # Get action probabilities
            action_probs = self.act(states)
            
            # Sample actions from policy
            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
            
            # Get state values
            values = self.cri(states).squeeze(-1)
            
        return actions, log_probs, values
    
    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities (for compatibility with existing code)
        """
        return self.act.forward(state)
    
    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> List[torch.Tensor]:
        """
        Collect experience from environment
        
        Args:
            env: Trading environment
            horizon_len: Number of steps to collect
            if_random: Use random policy (for initial buffer warm-up)
            
        Returns:
            List of [states, actions, rewards, dones, next_states, log_probs, values]
        """
        states = torch.zeros((horizon_len, env.num_envs, self.state_dim), dtype=torch.float32, device=self.device)
        actions = torch.zeros((horizon_len, env.num_envs, 1), dtype=torch.int64, device=self.device)
        rewards = torch.zeros((horizon_len, env.num_envs), dtype=torch.float32, device=self.device)
        dones = torch.zeros((horizon_len, env.num_envs), dtype=torch.bool, device=self.device)
        log_probs = torch.zeros((horizon_len, env.num_envs), dtype=torch.float32, device=self.device)
        values = torch.zeros((horizon_len, env.num_envs), dtype=torch.float32, device=self.device)
        
        # Get initial state
        if self.last_state is None:
            state = env.reset()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = self.last_state
        
        # Collect experience
        for t in range(horizon_len):
            states[t] = state
            
            if if_random:
                # Random actions for exploration
                action = torch.randint(0, self.action_dim, (env.num_envs, 1), device=self.device)
                log_prob = torch.zeros(env.num_envs, device=self.device)
                value = torch.zeros(env.num_envs, device=self.device)
            else:
                # Policy actions
                action, log_prob, value = self.select_actions(state)
                action = action.unsqueeze(-1)  # Add batch dimension
            
            actions[t] = action
            log_probs[t] = log_prob
            values[t] = value
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if not isinstance(done, torch.Tensor):
                done = torch.tensor(done, dtype=torch.bool, device=self.device)
            
            rewards[t] = reward
            dones[t] = done
            
            # Update state (reset if done)
            if done.any():
                state = env.reset()
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                # Keep non-done states
                state = torch.where(done.unsqueeze(-1), state, next_state)
            else:
                state = next_state
        
        self.last_state = state.detach()
        
        # Return data in format expected by replay buffer
        # Format: [states, actions, rewards, dones, next_states, log_probs, values]
        next_states = torch.zeros_like(states)
        next_states[:-1] = states[1:]
        next_states[-1] = state  # Last next state is current state
        
        # Flatten all data
        flat_states = states.reshape(-1, self.state_dim)
        flat_actions = actions.reshape(-1, 1)
        flat_rewards = rewards.reshape(-1)
        flat_dones = dones.reshape(-1)
        flat_next_states = next_states.reshape(-1, self.state_dim)
        flat_log_probs = log_probs.reshape(-1)
        flat_values = values.reshape(-1)
        
        return [
            flat_states,
            flat_actions,
            flat_rewards,
            flat_dones,
            flat_next_states,
            flat_log_probs,
            flat_values
        ]
    
    def compute_gae_advantages(self, 
                             rewards: torch.Tensor,
                             values: torch.Tensor, 
                             next_values: torch.Tensor,
                             dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Compute advantages backward
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = next_values[t]
                next_non_terminal = 1.0 - dones[t].float()
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t].float()
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, float, float, float]:
        """
        Update PPO networks using collected experience
        
        Returns:
            Tuple of (policy_loss, value_loss, entropy_loss, clip_fraction)
        """
        # Get batch data
        with torch.no_grad():
            batch_size = buffer.now_len
            indices = torch.randint(0, batch_size, size=(self.batch_size,), device=self.device)
            
            states = buffer.state[indices]
            actions = buffer.action[indices].long().squeeze(-1)
            rewards = buffer.reward[indices] 
            dones = buffer.done[indices]
            
            # Get old policy data
            old_log_probs = buffer.log_prob[indices] if hasattr(buffer, 'log_prob') else None
            old_values = buffer.value[indices] if hasattr(buffer, 'value') else None
            
            # If buffer doesn't have log_probs/values, compute them
            if old_log_probs is None or old_values is None:
                with torch.no_grad():
                    action_probs = self.act(states)
                    old_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze(1)
                    old_values = self.cri(states).squeeze(-1)
            
            # Compute next values for GAE
            next_states = torch.roll(states, -1, dims=0)  # Approximate next states
            next_values = self.cri(next_states).squeeze(-1)
            
            # Compute advantages and returns
            advantages, returns = self.compute_gae_advantages(rewards, old_values, next_values, dones)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_clip_fraction = 0
        total_updates = 0
        
        # Policy updates
        for _ in range(self.policy_epochs):
            # Get current policy
            action_probs = self.act(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            
            # Compute ratio and clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coeff * entropy
            
            # Combined actor loss
            actor_loss = policy_loss + entropy_loss
            
            # Update actor
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), self.max_grad_norm)
            self.act_optimizer.step()
            
            # Track metrics
            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item() 
            
            # Clip fraction (fraction of ratios clipped)
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float()).item()
            total_clip_fraction += clip_fraction
            
            total_updates += 1
        
        # Value function updates
        for _ in range(self.value_epochs):
            # Current value estimates
            current_values = self.cri(states).squeeze(-1)
            
            # Value loss (MSE)
            value_loss = F.mse_loss(current_values, returns)
            
            # Update critic
            self.cri_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cri.parameters(), self.max_grad_norm)
            self.cri_optimizer.step()
            
            total_value_loss += value_loss.item()
        
        # Update learning rate schedulers
        if self.use_lr_scheduler:
            self.act_scheduler.step()
            self.cri_scheduler.step()
        
        # Average losses
        avg_policy_loss = total_policy_loss / self.policy_epochs
        avg_value_loss = total_value_loss / self.value_epochs
        avg_entropy_loss = total_entropy_loss / self.policy_epochs
        avg_clip_fraction = total_clip_fraction / self.policy_epochs
        
        # Store metrics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)
        self.clip_fractions.append(avg_clip_fraction)
        
        return avg_policy_loss, avg_value_loss, avg_entropy_loss, avg_clip_fraction
    
    def save_or_load_agent(self, cwd: str, if_save: bool):
        """Save or load agent networks"""
        
        act_path = f"{cwd}/actor_ppo.pth"
        cri_path = f"{cwd}/critic_ppo.pth"
        
        if if_save:
            torch.save(self.act.state_dict(), act_path)
            torch.save(self.cri.state_dict(), cri_path)
            print(f"✅ PPO agent saved to {cwd}")
        else:
            try:
                self.act.load_state_dict(torch.load(act_path, map_location=self.device))
                self.cri.load_state_dict(torch.load(cri_path, map_location=self.device))
                print(f"✅ PPO agent loaded from {cwd}")
            except FileNotFoundError:
                print(f"⚠️ PPO checkpoint not found at {cwd}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        
        return {
            'policy_loss': np.mean(list(self.policy_losses)) if self.policy_losses else 0.0,
            'value_loss': np.mean(list(self.value_losses)) if self.value_losses else 0.0,
            'entropy_loss': np.mean(list(self.entropy_losses)) if self.entropy_losses else 0.0,
            'clip_fraction': np.mean(list(self.clip_fractions)) if self.clip_fractions else 0.0,
            'actor_lr': self.act_optimizer.param_groups[0]['lr'],
            'critic_lr': self.cri_optimizer.param_groups[0]['lr']
        }
    
    def print_training_stats(self):
        """Print current training statistics"""
        
        stats = self.get_training_stats()
        
        print(f"📊 PPO Training Stats:")
        print(f"   Policy Loss: {stats['policy_loss']:.4f}")
        print(f"   Value Loss: {stats['value_loss']:.4f}")
        print(f"   Entropy Loss: {stats['entropy_loss']:.4f}")
        print(f"   Clip Fraction: {stats['clip_fraction']:.3f}")
        print(f"   Actor LR: {stats['actor_lr']:.2e}")
        print(f"   Critic LR: {stats['critic_lr']:.2e}")


def test_ppo_agent():
    """Test PPO agent with mock data"""
    
    print("🧪 Testing PPO Agent")
    print("=" * 50)
    
    # Mock configuration
    class MockArgs:
        learning_rate = 3e-4
        gamma = 0.995
        batch_size = 256
        ppo_clip_ratio = 0.2
        ppo_policy_epochs = 4
        ppo_value_epochs = 4
        ppo_gae_lambda = 0.95
        ppo_entropy_coeff = 0.01
        use_lr_scheduler = False
    
    # Create agent
    agent = AgentPPO(
        net_dims=(128, 64, 32),
        state_dim=8,
        action_dim=3,
        gpu_id=-1,  # CPU for testing
        args=MockArgs()
    )
    
    # Test action selection
    test_states = torch.randn(16, 8)
    actions, log_probs, values = agent.select_actions(test_states)
    
    print(f"\n🎯 Action Selection Test:")
    print(f"   Input shape: {test_states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Log probs shape: {log_probs.shape}")
    print(f"   Values shape: {values.shape}")
    print(f"   Action range: [{actions.min().item()}, {actions.max().item()}]")
    
    # Test training stats
    agent.print_training_stats()
    
    print(f"\n✅ PPO agent test completed successfully!")


if __name__ == "__main__":
    test_ppo_agent()