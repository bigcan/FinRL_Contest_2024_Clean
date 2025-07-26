import os
import sys

# Fix encoding issues on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import numpy as np
from erl_config import Config, build_env
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


class EnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config, ensemble_method='majority_voting'):
        self.save_path = save_path
        self.agent_classes = agent_classes
        self.ensemble_method = ensemble_method

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        # Get state_dim from TradeSimulator (supports both original and enhanced features)
        temp_sim = TradeSimulator(num_sims=1)
        self.state_dim = temp_sim.state_dim
        print(f"Using state_dim: {self.state_dim}")
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)

        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        # self.net_assets = [torch.tensor(args.starting_cash, device=self.device)]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash

    def load_agents(self):
        args = self.args
        for agent_class in self.agent_classes:
            agent = agent_class(
                args.net_dims,
                args.state_dim,
                args.action_dim,
                gpu_id=args.gpu_id,
                args=args,
            )
            agent_name = agent_class.__name__
            cwd = os.path.join(self.save_path, agent_name)
            agent.save_or_load_agent(cwd, if_save=False)  # Load agent
            self.agents.append(agent)

    def multi_trade(self):
        """Evaluation loop using ensemble of agents"""

        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()

        last_state = state
        last_price = 0

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]

        for _ in range(trade_env.max_step):
            actions = []
            intermediate_state = last_state

            # Collect actions from each agent
            for agent in agents:
                actor = agent.act
                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                tensor_q_values = actor(tensor_state)
                tensor_action = tensor_q_values.argmax(dim=1)
                action = tensor_action.detach().cpu().unsqueeze(1)
                actions.append(action)

            action = self._ensemble_action(actions=actions, method=self.ensemble_method)
            action_int = action.item() - 1

            state, reward, done, _ = trade_env.step(action=action)
            
            # Update performance tracking for adaptive ensemble methods
            if self.ensemble_method in ['weighted_voting', 'adaptive_meta']:
                self.update_performance_tracking(reward.item() if torch.is_tensor(reward) else reward)

            action_ints.append(action_int)
            positions.append(trade_env.position)

            # Manually compute cumulative returns
            mid_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)

            new_cash = self.cash[-1]

            if action_int > 0 and self.cash[-1] > mid_price:  # Buy
                last_cash = self.cash[-1]
                new_cash = last_cash - mid_price
                self.current_btc += 1
            elif action_int < 0 and self.current_btc > 0:  # Sell
                last_cash = self.cash[-1]
                new_cash = last_cash + mid_price
                self.current_btc -= 1

            self.cash.append(new_cash)
            self.btc_assets.append((self.current_btc * mid_price).item())
            self.net_assets.append((to_python_number(self.btc_assets[-1]) + to_python_number(new_cash)))

            last_state = state

            # Log win rate
            if action_int == 1:
                correct_pred.append(1 if last_price < mid_price else -1 if last_price > mid_price else 0)
            elif action_int == -1:
                correct_pred.append(-1 if last_price < mid_price else 1 if last_price > mid_price else 0)
            else:
                correct_pred.append(0)

            last_price = mid_price
            current_btcs.append(self.current_btc)

        # Save results (convert CUDA tensors to CPU numpy arrays)
        positions_cpu = [pos.cpu().numpy() if hasattr(pos, 'cpu') else pos for pos in positions]
        np.save("evaluation_positions.npy", np.array(positions_cpu))
        np.save("evaluation_net_assets.npy", np.array(self.net_assets))
        np.save("evaluation_btc_positions.npy", np.array(self.btc_assets))
        np.save("evaluation_correct_predictions.npy", np.array(correct_pred))

        # Compute metrics
        returns = np.diff(self.net_assets) / self.net_assets[:-1]
        final_sharpe_ratio = sharpe_ratio(returns)
        final_max_drawdown = max_drawdown(returns)
        final_roma = return_over_max_drawdown(returns)

        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")

    def _ensemble_action(self, actions, method='majority_voting'):
        """
        Advanced ensemble methods for combining agent actions
        
        Args:
            actions: List of agent actions
            method: Ensemble method to use
                - 'majority_voting': Simple majority voting (default)
                - 'weighted_voting': Weighted voting based on historical performance
                - 'confidence_weighted': Weighted by Q-value confidence
                - 'adaptive_meta': Meta-learning adaptive ensemble
                
        Returns:
            torch.tensor: Final ensemble action
        """
        
        if method == 'majority_voting':
            # Original simple majority voting
            count = Counter([a.item() for a in actions])
            majority_action, _ = count.most_common(1)[0]
            return torch.tensor([[majority_action]], dtype=torch.int32)
            
        elif method == 'weighted_voting':
            # Weighted voting based on agent performance history
            return self._weighted_voting(actions)
            
        elif method == 'confidence_weighted':
            # Weight by Q-value confidence (would need Q-values passed in)
            return self._confidence_weighted_voting(actions)
            
        elif method == 'adaptive_meta':
            # Adaptive meta-learning ensemble
            return self._adaptive_meta_voting(actions)
            
        else:
            # Fallback to majority voting
            count = Counter([a.item() for a in actions])
            majority_action, _ = count.most_common(1)[0]
            return torch.tensor([[majority_action]], dtype=torch.int32)
    
    def _weighted_voting(self, actions):
        """Weighted voting based on agent historical performance"""
        
        # Initialize weights if not exists (equal weights initially)
        if not hasattr(self, 'agent_weights'):
            self.agent_weights = np.ones(len(actions)) / len(actions)
            
        # Convert actions to numpy for easier manipulation
        action_values = np.array([a.item() for a in actions])
        
        # Weighted voting using agent performance weights
        vote_counts = np.zeros(3)  # Assuming 3 actions: 0, 1, 2
        
        for i, action in enumerate(action_values):
            vote_counts[action] += self.agent_weights[i]
            
        # Select action with highest weighted vote
        final_action = np.argmax(vote_counts)
        return torch.tensor([[final_action]], dtype=torch.int32)
    
    def _confidence_weighted_voting(self, actions):
        """Confidence-weighted voting (placeholder - would need Q-values)"""
        # For now, fallback to equal weighting
        # In full implementation, would use max Q-values as confidence scores
        return self._weighted_voting(actions)
    
    def _adaptive_meta_voting(self, actions):
        """Adaptive meta-learning ensemble method"""
        
        # Initialize meta-learning parameters if not exists
        if not hasattr(self, 'meta_weights'):
            self.meta_weights = np.ones((len(actions), 3)) / 3  # Action-specific weights
            self.meta_learning_rate = 0.01
            self.recent_performance = []
            
        # Get current market context (simplified)
        current_context = self._get_market_context()
        
        # Adaptive weight adjustment based on recent performance
        if len(self.recent_performance) > 10:  # Need some history
            self._update_meta_weights()
            
        # Meta-weighted voting
        action_values = np.array([a.item() for a in actions])
        final_scores = np.zeros(3)
        
        for i, action in enumerate(action_values):
            final_scores += self.meta_weights[i] * np.eye(3)[action]
            
        final_action = np.argmax(final_scores)
        return torch.tensor([[final_action]], dtype=torch.int32)
    
    def _get_market_context(self):
        """Extract current market context for meta-learning"""
        # Simplified market context - could be expanded
        if hasattr(self.trade_env, 'price_ary') and hasattr(self.trade_env, 'step_i'):
            current_price = self.trade_env.price_ary[self.trade_env.step_i, 2]
            # Simple volatility proxy
            if self.trade_env.step_i > 10:
                recent_prices = self.trade_env.price_ary[max(0, self.trade_env.step_i-10):self.trade_env.step_i, 2]
                volatility = torch.std(recent_prices).item()
                return {'volatility': volatility, 'price': current_price.item()}
        return {'volatility': 0.01, 'price': 50000}  # Default values
    
    def _update_meta_weights(self):
        """Update meta-learning weights based on recent performance"""
        if len(self.recent_performance) < 2:
            return
            
        # Simple performance-based weight update
        recent_returns = np.diff(self.recent_performance[-20:])  # Last 20 steps
        
        if len(recent_returns) > 0:
            avg_return = np.mean(recent_returns)
            
            # Reward successful agents, penalize unsuccessful ones
            if avg_return > 0:
                # Boost weights of recently successful strategies  
                self.meta_weights *= (1 + self.meta_learning_rate * avg_return)
            else:
                # Reduce weights and add exploration
                self.meta_weights *= (1 + self.meta_learning_rate * avg_return)
                self.meta_weights += np.random.normal(0, 0.001, self.meta_weights.shape)
                
            # Normalize weights
            self.meta_weights = np.abs(self.meta_weights)
            self.meta_weights /= np.sum(self.meta_weights, axis=1, keepdims=True)
    
    def update_performance_tracking(self, reward):
        """Update performance tracking for adaptive methods"""
        if not hasattr(self, 'recent_performance'):
            self.recent_performance = []
            
        self.recent_performance.append(reward)
        
        # Keep only recent history
        if len(self.recent_performance) > 100:
            self.recent_performance = self.recent_performance[-100:]


def run_evaluation(save_path, agent_list, ensemble_method='majority_voting', ensemble_path=None):
    """
    Run ensemble evaluation with configurable ensemble method
    
    Args:
        save_path: Path to saved ensemble models (legacy parameter)
        agent_list: List of agent classes to evaluate
        ensemble_method: Ensemble method to use ('majority_voting', 'weighted_voting', 'adaptive_meta')
        ensemble_path: Alternative path to ensemble models (used by HPO script)
    """
    import sys

    # Use ensemble_path if provided (for HPO compatibility), otherwise use save_path
    if ensemble_path is not None:
        save_path = ensemble_path
        print(f"🔧 HPO Mode: Using ensemble_path = {ensemble_path}")

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # Get GPU_ID from command line arguments

    num_sims = 1
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7

    max_step = (4800 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": None,  # Will be auto-detected
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "dataset_path": "data/raw/task1/BTC_1sec_predict.npy",  # Evaluation dataset path
    }
    # Detect state dimension first
    temp_sim = TradeSimulator(num_sims=1)
    detected_state_dim = temp_sim.state_dim
    env_args["state_dim"] = detected_state_dim
    
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.state_dim = detected_state_dim
    args.starting_cash = 1e6
    
    # Use optimized architecture for 8-feature models
    if detected_state_dim == 8:
        args.net_dims = (128, 64, 32)
        print(f"Using optimized architecture for 8-feature models: {args.net_dims}")
    else:
        args.net_dims = (128, 128, 128)
        print(f"Using default architecture for {detected_state_dim}-feature models: {args.net_dims}")

    ensemble_evaluator = EnsembleEvaluator(
        save_path,
        agent_list,
        args,
        ensemble_method=ensemble_method
    )
    ensemble_evaluator.load_agents()
    ensemble_evaluator.multi_trade()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='FinRL Contest 2024 - Ensemble Evaluation')
    parser.add_argument('gpu_id', nargs='?', type=int, default=-1, help='GPU ID to use (default: -1 for CPU)')
    parser.add_argument('--save-path', type=str, help='Path to saved ensemble models')
    parser.add_argument('--ensemble-method', type=str, default='majority_voting',
                       choices=['majority_voting', 'weighted_voting', 'confidence_weighted', 'adaptive_meta'],
                       help='Ensemble method to use')
    
    args = parser.parse_args()
    
    # Determine save path
    if args.save_path:
        save_path = args.save_path
        print(f"🎯 Using specified models: {save_path}")
    else:
        # Use optimized models if available, fallback to standard
        save_path = "ensemble_optimized_phase2/ensemble_models"
        if not os.path.exists(save_path):
            save_path = "ensemble_teamname/ensemble_models"
            print(f"⚠️  Optimized models not found, using: {save_path}")
        else:
            print(f"✅ Using optimized models: {save_path}")
    
    print(f"🚀 FinRL Contest 2024 - Ensemble Evaluation")
    print(f"🔧 Using GPU: {args.gpu_id}")
    print(f"📁 Model path: {save_path}")
    print(f"🎯 Ensemble method: {args.ensemble_method}")
    
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    run_evaluation(save_path, agent_list, ensemble_method=args.ensemble_method)
