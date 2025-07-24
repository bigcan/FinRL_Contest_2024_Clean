import os
import torch
import numpy as np
from erl_config import Config, build_env
from reward_shaped_simulator import RewardShapedEvalTradeSimulator
from reward_shaping_config import RewardShapingConfig
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


class RewardShapedEnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config, reward_config=None):
        self.save_path = save_path
        self.agent_classes = agent_classes

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = 8 + 2
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        # Use reward-shaped environment
        self.reward_config = reward_config or RewardShapingConfig.balanced_config()
        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)

        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash

    def load_agents(self):
        args = self.args
        print(f"Loading {len(self.agent_classes)} agent classes...")
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
            print(f"Loaded {agent_name}")

    def multi_trade(self):
        """Evaluation loop using ensemble of agents with reward shaping"""

        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()

        last_state = state
        last_price = 0

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]
        
        print(f"Starting evaluation with reward shaping...")
        print(f"Reward config: {self.reward_config.get_summary()}")

        for step in range(trade_env.max_step):
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

            action = self._ensemble_action(actions=actions)
            action_int = action.item() - 1

            state, reward, done, _ = trade_env.step(action=action)

            action_ints.append(action_int)
            positions.append(trade_env.position)

            # Manually compute cumulative returns for portfolio tracking
            mid_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)

            new_cash = self.cash[-1]

            if action_int > 0 and self.cash[-1] > mid_price:  # Buy
                last_cash = self.cash[-1]
                new_cash = last_cash - mid_price
                self.current_btc += 1
                if step % 100 == 0 or step < 20:  # Log some trades
                    print(f"Step {step}: BUY at ${mid_price:.2f}, Cash: ${new_cash:.0f}, BTC: {self.current_btc}")
            elif action_int < 0 and self.current_btc > 0:  # Sell
                last_cash = self.cash[-1]
                new_cash = last_cash + mid_price
                self.current_btc -= 1
                if step % 100 == 0 or step < 20:  # Log some trades
                    print(f"Step {step}: SELL at ${mid_price:.2f}, Cash: ${new_cash:.0f}, BTC: {self.current_btc}")

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

        # Save results
        positions_cpu = [pos.cpu().numpy() if hasattr(pos, 'cpu') else pos for pos in positions]
        np.save("reward_shaped_evaluation_positions.npy", np.array(positions_cpu))
        np.save("reward_shaped_evaluation_net_assets.npy", np.array(self.net_assets))
        np.save("reward_shaped_evaluation_btc_positions.npy", np.array(self.btc_assets))
        np.save("reward_shaped_evaluation_correct_predictions.npy", np.array(correct_pred))

        # Compute metrics
        returns = np.diff(self.net_assets) / self.net_assets[:-1]
        final_sharpe_ratio = sharpe_ratio(returns)
        final_max_drawdown = max_drawdown(returns)
        final_roma = return_over_max_drawdown(returns)

        print(f"\n=== REWARD SHAPED EVALUATION RESULTS ===")
        print(f"Total trades: {sum(1 for a in action_ints if a != 0)}")
        print(f"Buy orders: {sum(1 for a in action_ints if a > 0)}")
        print(f"Sell orders: {sum(1 for a in action_ints if a < 0)}")
        print(f"Hold actions: {sum(1 for a in action_ints if a == 0)}")
        print(f"Starting net assets: ${self.net_assets[0]:,.0f}")
        print(f"Final net assets: ${self.net_assets[-1]:,.0f}")
        print(f"Total return: {(self.net_assets[-1] / self.net_assets[0] - 1) * 100:.4f}%")
        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")
        
        # Print reward analysis if available
        if hasattr(trade_env, 'print_reward_summary'):
            trade_env.print_reward_summary()

    def _ensemble_action(self, actions):
        """Returns the majority action among agents"""
        # Extract the action values correctly, handling the tensor dimensions
        action_values = []
        for a in actions:
            if a.numel() == 1:
                action_values.append(a.item())
            else:
                action_values.append(a[0].item())  # Take first element if batch dimension
        
        count = Counter(action_values)
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([[majority_action]], dtype=torch.int32)


def run_reward_shaped_evaluation(save_path, agent_list, config_type="balanced"):
    import sys

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1

    # Select reward shaping configuration
    if config_type == "conservative":
        reward_config = RewardShapingConfig.conservative_config()
    elif config_type == "aggressive":
        reward_config = RewardShapingConfig.aggressive_config()
    else:
        reward_config = RewardShapingConfig.balanced_config()

    print(f"Using {config_type} reward shaping configuration")

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
        "reward_config": reward_config  # Pass reward config to environment
    }
    
    args = Config(agent_class=None, env_class=RewardShapedEvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6

    ensemble_evaluator = RewardShapedEnsembleEvaluator(
        save_path,
        agent_list,
        args,
        reward_config
    )
    ensemble_evaluator.load_agents()
    ensemble_evaluator.multi_trade()


if __name__ == "__main__":
    save_path = "ensemble_teamname/ensemble_models"
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    
    # Test different configurations
    configurations = ["conservative", "balanced", "aggressive"]
    
    for config in configurations:
        print(f"\n" + "="*50)
        print(f"TESTING {config.upper()} CONFIGURATION")
        print("="*50)
        run_reward_shaped_evaluation(save_path, agent_list, config)
        print("\n" + "-"*50)
        print("Moving to next configuration...\n")