import os
import torch
import numpy as np
from typing import Tuple
from copy import deepcopy
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from erl_config import Config
from erl_replay_buffer import ReplayBuffer
from erl_per_buffer import PrioritizedReplayBuffer
from erl_net import QNetTwin, QNetTwinDuel
from erl_noisy_net import QNetTwinNoisy, QNetTwinDuelNoisy
from erl_exploration import ExplorationOrchestrator, AdaptiveEpsilonGreedy 


def get_optim_param(optimizer: torch.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


class AgentDoubleDQN:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        self.gamma = args.gamma  # discount factor of future rewards
        self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''network'''
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        '''optimizer'''
        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer
            
        from types import MethodType  # built-in package of Python3
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

        self.act_target = self.cri_target = deepcopy(self.act)
        self.act.explore_rate = getattr(args, "explore_rate", 1 / 32)
        
        # Enhanced exploration parameters
        self.min_explore_rate = getattr(args, "min_explore_rate", 0.01)
        self.exploration_decay_rate = getattr(args, "exploration_decay_rate", 0.995)
        self.exploration_warmup_steps = getattr(args, "exploration_warmup_steps", 5000)
        self.force_exploration_probability = getattr(args, "force_exploration_probability", 0.05)
        self.total_exploration_steps = 0
        
        # Action diversity tracking
        self.action_history = []
        self.action_diversity_window = 100
        
        # Initialize exploration orchestrator
        self.exploration_orchestrator = ExplorationOrchestrator(
            strategies=[
                AdaptiveEpsilonGreedy(
                    initial_epsilon=self.act.explore_rate,
                    min_epsilon=self.min_explore_rate,
                    decay_rate=self.exploration_decay_rate,
                    warmup_steps=self.exploration_warmup_steps
                )
            ],
            action_dim=self.action_dim
        )

    def get_obj_critic(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.
        Implements proper Double DQN: use online network for action selection, target network for evaluation.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)
            
            # CRITICAL FIX: Ensure ALL sampled tensors are on correct device immediately
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            undones = undones.to(self.device)
            next_ss = next_ss.to(self.device)

            # Double DQN: use online network for action selection
            next_q1_online, next_q2_online = self.act.get_q1_q2(next_ss)
            next_actions = torch.min(next_q1_online, next_q2_online).argmax(dim=1, keepdim=True)
            
            # Use target network for evaluation with selected actions
            next_q1_target, next_q2_target = self.cri_target.get_q1_q2(next_ss)
            
            # CRITICAL FIX: Ensure next_actions is on same device as target network tensors
            next_actions = next_actions.to(next_q1_target.device)
            
            next_qs = torch.min(next_q1_target, next_q2_target).gather(1, next_actions).squeeze(1)
            q_labels = rewards + undones * self.gamma * next_qs

        # CRITICAL FIX: Ensure actions is on same device as Q-network output
        q1_out, q2_out = self.act.get_q1_q2(states)
        actions = actions.to(q1_out.device)
        q1, q2 = [qs.gather(1, actions.long()).squeeze(1) for qs in [q1_out, q2_out]]
        
        # CRITICAL FIX: Ensure q_labels is on same device as Q-values for loss calculation
        q_labels = q_labels.to(q1.device)
        
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, q1

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_target', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            if if_save:
                torch.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, torch.load(file_path, map_location=self.device, weights_only=False))

    def update_exploration_rate(self):
        """Update exploration rate with adaptive scheduling"""
        if self.total_exploration_steps < self.exploration_warmup_steps:
            # Warmup phase: maintain high exploration
            return
            
        # Decay exploration rate
        current_rate = self.act.explore_rate
        new_rate = max(self.min_explore_rate, current_rate * self.exploration_decay_rate)
        
        # Check action diversity and adjust if needed
        if len(self.action_history) >= self.action_diversity_window:
            recent_actions = self.action_history[-self.action_diversity_window:]
            unique_actions = len(set(recent_actions))
            diversity_ratio = unique_actions / self.action_dim
            
            # If diversity is too low, boost exploration
            if diversity_ratio < 0.5:  # Less than 50% of actions being used
                new_rate = min(0.3, new_rate * 1.5)  # Boost exploration
                print(f"⚠️ Low action diversity detected ({diversity_ratio:.2f}), boosting exploration to {new_rate:.3f}")
                
        self.act.explore_rate = new_rate
        
    def should_force_exploration(self) -> bool:
        """Determine if we should force exploration based on recent behavior"""
        if len(self.action_history) < 50:
            return False
            
        recent_actions = self.action_history[-50:]
        hold_ratio = recent_actions.count(1) / len(recent_actions)  # Assuming 1 is hold
        
        # Force exploration if too conservative
        if hold_ratio > 0.8:  # More than 80% hold actions
            return np.random.random() < self.force_exploration_probability * 2  # Double probability
        else:
            return np.random.random() < self.force_exploration_probability

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (num_envs, state_dim) for a vectorized env.

        # Update exploration rate periodically
        if self.total_exploration_steps % 1000 == 0:
            self.update_exploration_rate()

        # Use online network for exploration to get most up-to-date policy
        get_action = self.act.get_action
        for t in range(horizon_len):
            # Force exploration if needed
            force_explore = self.should_force_exploration() and not if_random
            
            if if_random or force_explore:
                action = torch.randint(self.action_dim, size=(self.num_envs, 1), device=self.device)
                if force_explore and t == 0:  # Log only once per horizon
                    print(f"🎲 Forcing exploration due to conservative behavior")
            else:
                # CRITICAL DEBUG: Check state device before neural network call
                if not isinstance(state, torch.Tensor):
                    print(f"❌ ERROR: state is not a tensor, type: {type(state)}")
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                elif state.device != self.device:
                    print(f"❌ ERROR: state device mismatch - state: {state.device}, agent: {self.device}")
                    state = state.to(self.device)
                    print(f"✅ Fixed state device to: {state.device}")
                
                action = get_action(state).detach()  # different
                
            states[t] = state

            state, reward, done, _ = env.step(action)  # next_state
            
            # CRITICAL FIX: Ensure state tensor is on correct device after each env step
            # The environment may return state tensors on CPU, causing device mismatch in subsequent episodes
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            else:
                state = state.to(self.device)
            
            actions[t] = action
            rewards[t] = reward
            dones[t] = done
            
            # Track actions for diversity monitoring
            if self.num_envs == 1:
                self.action_history.append(action.item())
            else:
                self.action_history.extend(action.cpu().numpy().flatten().tolist())
            
            # Keep action history bounded
            if len(self.action_history) > self.action_diversity_window * 2:
                self.action_history = self.action_history[-self.action_diversity_window:]
                
            self.total_exploration_steps += self.num_envs

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)
            
            # CRITICAL FIX: Ensure ALL sampled tensors are on correct device immediately
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            undones = undones.to(self.device)
            next_ss = next_ss.to(self.device)  # next_ss: next states
            next_qs = self.cri_target(next_ss).max(dim=1, keepdim=True)[0].squeeze(1)  # next q_values
            q_labels = rewards + undones * self.gamma * next_qs

        # CRITICAL FIX: Ensure actions is on same device as critic network output
        q_output = self.cri(states)
        actions = actions.to(q_output.device)
        q_values = q_output.gather(1, actions.long()).squeeze(1)
        
        # CRITICAL FIX: Ensure q_labels is on same device as Q-values for loss calculation
        q_labels = q_labels.to(q_values.device)
        
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, q_values

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        
        # CRITICAL FIX: Ensure optimizer state is on correct device before step
        self._fix_optimizer_device_consistency(optimizer)
        
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()
        
    def _fix_optimizer_device_consistency(self, optimizer):
        """Ensure optimizer state tensors are on the same device as parameters"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            if value.device != param.device:
                                print(f"❌ OPTIMIZER STATE MISMATCH: {key} device={value.device}, param device={param.device}")
                                state[key] = value.to(param.device)
                                print(f"✅ Fixed optimizer state {key} device to: {state[key].device}")

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        
        # CRITICAL FIX: Ensure last_state is on correct device for neural network forward pass
        # This method is called during agent updates and can cause device mismatch in episode transitions
        if not isinstance(last_state, torch.Tensor):
            last_state = torch.tensor(last_state, dtype=torch.float32, device=self.device)
        else:
            last_state = last_state.to(self.device)
        
        # Fix: Use actual Q-value instead of action index for cumulative rewards
        next_q_values = self.act_target.get_q1_q2(last_state)
        next_value = torch.min(*next_q_values).max(dim=1, keepdim=True)[0].squeeze(1).detach()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def update_avg_std_for_normalization(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        
        # CRITICAL FIX: Ensure computed stats are on same device as network parameters
        state_avg = state_avg.to(self.act.state_avg.device)
        state_std = state_std.to(self.act.state_std.device)
        
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        
        # CRITICAL FIX: Ensure returns stats are on same device as network parameters
        returns_avg = returns_avg.to(self.cri.value_avg.device)
        returns_std = returns_std.to(self.cri.value_std.device)
        
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4


class AgentD3QN(AgentDoubleDQN):  # Dueling Double Deep Q Network. (D3QN)
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)

class AgentPrioritizedDQN(AgentDoubleDQN):
    """Double DQN with Prioritized Experience Replay for improved sample efficiency"""
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        # Use Dueling architecture for better value estimation
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)
        
        # PER hyperparameters
        self.per_alpha = getattr(args, 'per_alpha', 0.6)
        self.per_beta = getattr(args, 'per_beta', 0.4)
        self.per_beta_annealing_steps = getattr(args, 'per_beta_annealing_steps', 100000)
        
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        
    def get_obj_critic_per(self, buffer: PrioritizedReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculate loss with Prioritized Experience Replay
        
        Returns:
            obj_critic: Loss value
            q1: Q-values from first network
            indices: Sample indices for priority updates
            td_errors: TD errors for priority updates
        """
        # Sample from PER buffer
        states, actions, rewards, undones, next_ss, indices, weights = buffer.sample(batch_size)
        
        with torch.no_grad():
            # Double DQN: use online network for action selection
            next_q1_online, next_q2_online = self.act.get_q1_q2(next_ss)
            next_actions = torch.min(next_q1_online, next_q2_online).argmax(dim=1, keepdim=True)
            
            # Use target network for evaluation with selected actions
            next_q1_target, next_q2_target = self.cri_target.get_q1_q2(next_ss)
            
            # CRITICAL FIX: Ensure next_actions is on same device as target network tensors
            next_actions = next_actions.to(next_q1_target.device)
            
            next_qs = torch.min(next_q1_target, next_q2_target).gather(1, next_actions).squeeze(1)
            q_labels = rewards + undones * self.gamma * next_qs

        # Get current Q-values
        # CRITICAL FIX: Ensure actions is on same device as Q-network output
        q1_out, q2_out = self.act.get_q1_q2(states)
        actions = actions.to(q1_out.device)
        q1, q2 = [qs.gather(1, actions.long()).squeeze(1) for qs in [q1_out, q2_out]]
        
        # Calculate TD errors for priority updates
        td_errors = torch.abs(q1 - q_labels) + torch.abs(q2 - q_labels)
        
        # Apply importance sampling weights
        weights = weights.to(self.device)
        weighted_loss1 = (self.criterion(q1, q_labels) * weights).mean()
        weighted_loss2 = (self.criterion(q2, q_labels) * weights).mean()
        obj_critic = weighted_loss1 + weighted_loss2
        
        return obj_critic, q1, indices, td_errors
    
    def update_net(self, buffer) -> Tuple[float, ...]:
        """Update network using PER if available, otherwise fall back to standard replay"""
        if isinstance(buffer, PrioritizedReplayBuffer):
            return self.update_net_per(buffer)
        else:
            return super().update_net(buffer)
    
    def update_net_per(self, buffer: PrioritizedReplayBuffer) -> Tuple[float, ...]:
        """Update network using Prioritized Experience Replay"""
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        
        for _ in range(update_times):
            obj_critic, q_value, indices, td_errors = self.get_obj_critic_per(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            
            # Update network
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            
            # Update priorities in buffer
            buffer.update_priorities(indices, td_errors)
            
        return obj_critics / update_times, obj_actors / update_times


class AgentNoisyDQN(AgentDoubleDQN):
    """Double DQN with Noisy Networks for parameter space exploration"""
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        # Use Noisy Networks for exploration
        self.act_class = getattr(self, "act_class", QNetTwinNoisy)
        self.cri_class = getattr(self, "cri_class", None)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        
    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Exploration using Noisy Networks (no epsilon-greedy needed)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state

        # Reset noise in noisy layers before exploration
        if hasattr(self.act, 'reset_noise'):
            self.act.reset_noise()
            
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(self.num_envs, 1), device=self.device) if if_random \
                else get_action(state).detach()
            states[t] = state

            state, reward, done, _ = env.step(action)
            
            # CRITICAL FIX: Ensure state tensor is on correct device after each env step
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            else:
                state = state.to(self.device)
            
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones


class AgentNoisyDuelDQN(AgentDoubleDQN):
    """Double DQN with Noisy Dueling Networks for advanced exploration"""
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        # Use Noisy Dueling Networks
        self.act_class = getattr(self, "act_class", QNetTwinDuelNoisy)
        self.cri_class = getattr(self, "cri_class", None)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        
    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Exploration using Noisy Dueling Networks
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state

        # Reset noise in noisy layers before exploration
        if hasattr(self.act, 'reset_noise'):
            self.act.reset_noise()
            
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(self.num_envs, 1), device=self.device) if if_random \
                else get_action(state).detach()
            states[t] = state

            state, reward, done, _ = env.step(action)
            
            # CRITICAL FIX: Ensure state tensor is on correct device after each env step
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            else:
                state = state.to(self.device)
            
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones



class AgentRainbowDQN(AgentDoubleDQN):
    """Rainbow DQN with multi-step learning and other Rainbow components"""
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        # Multi-step learning parameters
        self.n_step = getattr(args, "n_step", 3)  # N-step returns
        
        # Use Noisy Dueling Networks
        self.act_class = getattr(self, "act_class", QNetTwinDuelNoisy)
        self.cri_class = getattr(self, "cri_class", None)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        
    def get_obj_critic_multistep(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Calculate loss with multi-step learning"""
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)
            
            # CRITICAL FIX: Ensure ALL sampled tensors are on correct device immediately
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            undones = undones.to(self.device)
            next_ss = next_ss.to(self.device)
            
            # Double DQN with multi-step
            next_q1_online, next_q2_online = self.act.get_q1_q2(next_ss)
            next_actions = torch.min(next_q1_online, next_q2_online).argmax(dim=1, keepdim=True)
            
            next_q1_target, next_q2_target = self.cri_target.get_q1_q2(next_ss)
            next_qs = torch.min(next_q1_target, next_q2_target).gather(1, next_actions).squeeze(1)
            
            # Multi-step discount
            gamma_n = self.gamma ** self.n_step
            q_labels = rewards + undones * gamma_n * next_qs

        # CRITICAL FIX: Ensure actions is on same device as Q-network output
        q1_out, q2_out = self.act.get_q1_q2(states)
        actions = actions.to(q1_out.device)
        q1, q2 = [qs.gather(1, actions.long()).squeeze(1) for qs in [q1_out, q2_out]]
        
        # CRITICAL FIX: Ensure q_labels is on same device as Q-values for loss calculation
        q_labels = q_labels.to(q1.device)
        
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, q1
    
    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """Exploration with noisy networks"""
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state

        # Reset noise in noisy layers
        if hasattr(self.act, "reset_noise"):
            self.act.reset_noise()
            
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(self.num_envs, 1), device=self.device) if if_random \
                else get_action(state).detach()
            states[t] = state

            state, reward, done, _ = env.step(action)
            
            # CRITICAL FIX: Ensure state tensor is on correct device after each env step
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
            else:
                state = state.to(self.device)
            
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones
    
    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """Update network with multi-step learning"""
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        
        for _ in range(update_times):
            # Use multi-step learning
            obj_critic, q_value = self.get_obj_critic_multistep(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            
        return obj_critics / update_times, obj_actors / update_times




class AgentAdaptiveDQN(AgentDoubleDQN):
    """Double DQN with Adaptive Learning Rate Scheduling and Advanced Optimization"""
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        # Use Dueling Networks
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)
        
        # Adaptive LR parameters
        self.lr_strategy = getattr(args, "lr_strategy", "cosine_annealing")
        self.adaptive_grad_clip = getattr(args, "adaptive_grad_clip", True)
        
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        
        # Initialize adaptive optimizers
        self._setup_adaptive_optimizers(args)
        
    def _setup_adaptive_optimizers(self, args):
        """Setup adaptive learning rate schedulers and optimizers"""
        from erl_lr_scheduler import AdaptiveLRScheduler, AdaptiveOptimizerWrapper
        
        # Create adaptive LR schedulers
        act_lr_scheduler = AdaptiveLRScheduler(
            self.act_optimizer, 
            strategy=self.lr_strategy,
            T_max=getattr(args, "lr_T_max", 10000),
            patience=getattr(args, "lr_patience", 100),
            factor=getattr(args, "lr_factor", 0.8)
        )
        
        if self.cri_optimizer != self.act_optimizer:
            cri_lr_scheduler = AdaptiveLRScheduler(
                self.cri_optimizer,
                strategy=self.lr_strategy,
                T_max=getattr(args, "lr_T_max", 10000),
                patience=getattr(args, "lr_patience", 100),
                factor=getattr(args, "lr_factor", 0.8)
            )
        else:
            cri_lr_scheduler = act_lr_scheduler
            
        # Wrap optimizers with adaptive features
        self.adaptive_act_optimizer = AdaptiveOptimizerWrapper(
            self.act_optimizer,
            act_lr_scheduler,
            grad_clip_norm=self.clip_grad_norm,
            adaptive_grad_clip=self.adaptive_grad_clip
        )
        
        self.adaptive_cri_optimizer = AdaptiveOptimizerWrapper(
            self.cri_optimizer,
            cri_lr_scheduler,
            grad_clip_norm=self.clip_grad_norm,
            adaptive_grad_clip=self.adaptive_grad_clip
        ) if self.cri_optimizer != self.act_optimizer else self.adaptive_act_optimizer
    
    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """Override optimizer update to use adaptive features"""
        # Calculate performance metric (negative loss for "higher is better")
        performance = -objective.item()
        
        optimizer.zero_grad()
        objective.backward()
        
        # Use adaptive optimizer if available
        if hasattr(self, "adaptive_cri_optimizer") and optimizer == self.cri_optimizer:
            self.adaptive_cri_optimizer.step(performance)
        elif hasattr(self, "adaptive_act_optimizer") and optimizer == self.act_optimizer:
            self.adaptive_act_optimizer.step(performance)
        else:
            # Fallback to standard optimization
            clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
            optimizer.step()
    
    def get_training_stats(self) -> dict:
        """Get training statistics including adaptive LR info"""
        stats = {}
        
        if hasattr(self, "adaptive_cri_optimizer"):
            stats["critic_lr"] = self.adaptive_cri_optimizer.get_lr()
            stats["critic_grad_norm"] = self.adaptive_cri_optimizer.get_grad_norm()
            
        if hasattr(self, "adaptive_act_optimizer"):
            stats["actor_lr"] = self.adaptive_act_optimizer.get_lr()
            stats["actor_grad_norm"] = self.adaptive_act_optimizer.get_grad_norm()
            
        return stats
    
    def save_or_load_agent(self, cwd: str, if_save: bool):
        """Enhanced save/load with adaptive optimizer states"""
        super().save_or_load_agent(cwd, if_save)
        
        # Save/load adaptive optimizer states
        if if_save:
            if hasattr(self, "adaptive_cri_optimizer"):
                torch.save(self.adaptive_cri_optimizer.state_dict(), f"{cwd}/adaptive_cri_optimizer.pth")
            if hasattr(self, "adaptive_act_optimizer") and self.adaptive_act_optimizer != self.adaptive_cri_optimizer:
                torch.save(self.adaptive_act_optimizer.state_dict(), f"{cwd}/adaptive_act_optimizer.pth")
        else:
            cri_path = f"{cwd}/adaptive_cri_optimizer.pth"
            if os.path.isfile(cri_path) and hasattr(self, "adaptive_cri_optimizer"):
                self.adaptive_cri_optimizer.load_state_dict(torch.load(cri_path, map_location=self.device))
                
            act_path = f"{cwd}/adaptive_act_optimizer.pth"
            if os.path.isfile(act_path) and hasattr(self, "adaptive_act_optimizer") and self.adaptive_act_optimizer != self.adaptive_cri_optimizer:
                self.adaptive_act_optimizer.load_state_dict(torch.load(act_path, map_location=self.device))

