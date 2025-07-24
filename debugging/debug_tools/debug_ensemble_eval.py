import os
import torch
import numpy as np
from erl_config import Config, build_env
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


class DebugEnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config):
        self.save_path = save_path
        self.agent_classes = agent_classes
        
        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = 8 + 2
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)
        
        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash

    def load_agents(self):
        args = self.args
        print(f"Loading {len(self.agent_classes)} agent classes...")
        
        for i, agent_class in enumerate(self.agent_classes):
            print(f"Loading agent {i+1}: {agent_class.__name__}")
            agent = agent_class(
                args.net_dims,
                args.state_dim,
                args.action_dim,
                gpu_id=args.gpu_id,
                args=args,
            )
            agent_name = agent_class.__name__
            cwd = os.path.join(self.save_path, agent_name)
            print(f"  Model path: {cwd}")
            print(f"  Model files exist: {os.path.exists(cwd)}")
            if os.path.exists(cwd):
                files = os.listdir(cwd)
                print(f"  Files in directory: {files}")
            
            agent.save_or_load_agent(cwd, if_save=False)  # Load agent
            self.agents.append(agent)
            print(f"  Agent {agent_name} loaded successfully")
        
        print(f"Total agents loaded: {len(self.agents)}")

    def debug_trade(self, max_debug_steps=10):
        """Debug version with detailed logging"""
        
        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()
        
        print(f"=== DEBUGGING ENSEMBLE EVALUATION ===")
        print(f"Environment initialized: max_step={trade_env.max_step}")
        print(f"Initial state shape: {state.shape}")
        print(f"Initial cash: ${self.cash[0]:,.0f}")
        print(f"Initial BTC position: {self.current_btc}")
        print(f"Device: {self.device}")
        print(f"Number of agents: {len(agents)}")
        
        last_state = state
        last_price = 0
        
        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]
        
        for step in range(min(max_debug_steps, trade_env.max_step)):
            print(f"\n--- Step {step+1}/{min(max_debug_steps, trade_env.max_step)} ---")
            
            actions = []
            intermediate_state = last_state
            
            print(f"State: {intermediate_state}")
            
            # Collect actions from each agent
            for i, agent in enumerate(agents):
                actor = agent.act
                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                print(f"Agent {i+1} input tensor shape: {tensor_state.shape}")
                
                tensor_q_values = actor(tensor_state)
                print(f"Agent {i+1} Q-values: {tensor_q_values}")
                
                tensor_action = tensor_q_values.argmax(dim=1)
                print(f"Agent {i+1} selected action (tensor): {tensor_action}")
                
                action = tensor_action.detach().cpu().unsqueeze(1)
                print(f"Agent {i+1} final action: {action}")
                actions.append(action)
            
            print(f"All agent actions: {[a.item() for a in actions]}")
            
            action = self._ensemble_action(actions=actions)
            action_int = action.item() - 1  # Convert to {-1, 0, 1}
            
            print(f"Ensemble action (raw): {action.item()}")
            print(f"Ensemble action (interpreted): {action_int} ({'BUY' if action_int > 0 else 'SELL' if action_int < 0 else 'HOLD'})")
            
            # Get current price before step
            current_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)
            print(f"Current price: ${current_price}")
            print(f"Current cash: ${self.cash[-1]}")
            print(f"Current BTC: {self.current_btc}")
            
            state, reward, done, _ = trade_env.step(action=action)
            
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            
            action_ints.append(action_int)
            positions.append(trade_env.position)
            
            # Debug manual portfolio computation
            mid_price = current_price
            new_cash = self.cash[-1]
            
            print(f"Checking trade conditions:")
            print(f"  Buy condition: action_int > 0 ({action_int > 0}) AND cash > price ({self.cash[-1]} > {mid_price}) = {action_int > 0 and self.cash[-1] > mid_price}")
            print(f"  Sell condition: action_int < 0 ({action_int < 0}) AND btc > 0 ({self.current_btc > 0}) = {action_int < 0 and self.current_btc > 0}")
            
            if action_int > 0 and self.cash[-1] > mid_price:  # Buy
                print("  EXECUTING BUY")
                last_cash = self.cash[-1]
                new_cash = last_cash - mid_price
                self.current_btc += 1
                print(f"    Cash: ${last_cash} -> ${new_cash}")
                print(f"    BTC: {self.current_btc-1} -> {self.current_btc}")
            elif action_int < 0 and self.current_btc > 0:  # Sell
                print("  EXECUTING SELL")
                last_cash = self.cash[-1]
                new_cash = last_cash + mid_price
                self.current_btc -= 1
                print(f"    Cash: ${last_cash} -> ${new_cash}")
                print(f"    BTC: {self.current_btc+1} -> {self.current_btc}")
            else:
                print("  NO TRADE EXECUTED")
            
            self.cash.append(new_cash)
            self.btc_assets.append((self.current_btc * mid_price).item())
            net_asset_value = to_python_number(self.btc_assets[-1]) + to_python_number(new_cash)
            self.net_assets.append(net_asset_value)
            
            print(f"Updated portfolio:")
            print(f"  Cash: ${new_cash}")
            print(f"  BTC value: ${self.btc_assets[-1]}")
            print(f"  Net assets: ${net_asset_value}")
            
            last_state = state
            last_price = mid_price
            current_btcs.append(self.current_btc)
            
            if done:
                print("Environment terminated early")
                break
        
        print(f"\n=== SUMMARY AFTER {min(max_debug_steps, trade_env.max_step)} STEPS ===")
        print(f"Trades executed: {sum(1 for a in action_ints if a != 0)}")
        print(f"Buy orders: {sum(1 for a in action_ints if a > 0)}")
        print(f"Sell orders: {sum(1 for a in action_ints if a < 0)}")
        print(f"Hold actions: {sum(1 for a in action_ints if a == 0)}")
        print(f"Final cash: ${self.cash[-1]}")
        print(f"Final BTC: {self.current_btc}")
        print(f"Final net assets: ${self.net_assets[-1]}")
        print(f"Total return: {(self.net_assets[-1] / self.net_assets[0] - 1) * 100:.4f}%")

    def _ensemble_action(self, actions):
        """Returns the majority action among agents"""
        action_values = [a.item() for a in actions]
        count = Counter(action_values)
        majority_action, vote_count = count.most_common(1)[0]
        
        print(f"Voting: {dict(count)} -> majority: {majority_action} ({vote_count}/{len(actions)} votes)")
        
        return torch.tensor([[majority_action]], dtype=torch.int32)


def run_debug_evaluation(save_path, agent_list, debug_steps=10):
    import sys
    
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    
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
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "dataset_path": "data/raw/task1/BTC_1sec_predict.npy",
    }
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6
    
    debug_evaluator = DebugEnsembleEvaluator(
        save_path,
        agent_list,
        args,
    )
    debug_evaluator.load_agents()
    debug_evaluator.debug_trade(max_debug_steps=debug_steps)


if __name__ == "__main__":
    save_path = "ensemble_teamname/ensemble_models"
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    run_debug_evaluation(save_path, agent_list, debug_steps=20)