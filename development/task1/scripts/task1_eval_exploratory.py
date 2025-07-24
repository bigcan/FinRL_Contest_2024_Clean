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


class ExploratoryEnsembleEvaluator:
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
        
        # Add exploration parameters
        self.exploration_rate = 0.15  # 15% chance to take exploratory action
        self.force_trading_every_n_steps = 50  # Force a trade every N steps if no trades

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
        """Evaluation loop using ensemble of agents with exploration"""

        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()

        last_state = state
        last_price = 0
        last_trade_step = 0

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]

        for step in range(trade_env.max_step):
            actions = []
            intermediate_state = last_state

            # Collect actions from each agent
            for agent in agents:
                actor = agent.act
                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                tensor_q_values = actor(tensor_state)
                
                # Add epsilon-greedy exploration to individual agents
                if np.random.random() < self.exploration_rate:
                    # Random action
                    tensor_action = torch.randint(0, 3, (1,), device=agent.device)
                else:
                    # Best action
                    tensor_action = tensor_q_values.argmax(dim=1)
                
                action = tensor_action.detach().cpu().unsqueeze(1)
                actions.append(action)

            action = self._ensemble_action(actions=actions, step=step, last_trade_step=last_trade_step)
            action_int = action.item() - 1

            # Force trading if no trades for too long
            if step - last_trade_step >= self.force_trading_every_n_steps:
                # Force a buy or sell based on recent price trend
                mid_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)
                if self.current_btc == 0 and self.cash[-1] > mid_price:
                    action_int = 1  # Force buy
                    print(f"Step {step}: Forced BUY due to long inactivity")
                elif self.current_btc > 0:
                    action_int = -1  # Force sell
                    print(f"Step {step}: Forced SELL due to long inactivity")
                
                action = torch.tensor([[action_int + 1]], dtype=torch.int32)

            state, reward, done, _ = trade_env.step(action=action)

            action_ints.append(action_int)
            positions.append(trade_env.position)

            # Manually compute cumulative returns
            mid_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)

            new_cash = self.cash[-1]

            if action_int > 0 and self.cash[-1] > mid_price:  # Buy
                last_cash = self.cash[-1]
                new_cash = last_cash - mid_price
                self.current_btc += 1
                last_trade_step = step
                print(f"Step {step}: BUY at ${mid_price:.2f}, Cash: ${new_cash:.0f}, BTC: {self.current_btc}")
            elif action_int < 0 and self.current_btc > 0:  # Sell
                last_cash = self.cash[-1]
                new_cash = last_cash + mid_price
                self.current_btc -= 1
                last_trade_step = step
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
        np.save("exploratory_evaluation_positions.npy", np.array(positions_cpu))
        np.save("exploratory_evaluation_net_assets.npy", np.array(self.net_assets))
        np.save("exploratory_evaluation_btc_positions.npy", np.array(self.btc_assets))
        np.save("exploratory_evaluation_correct_predictions.npy", np.array(correct_pred))

        # Compute metrics
        returns = np.diff(self.net_assets) / self.net_assets[:-1]
        final_sharpe_ratio = sharpe_ratio(returns)
        final_max_drawdown = max_drawdown(returns)
        final_roma = return_over_max_drawdown(returns)

        print(f"\n=== EXPLORATORY EVALUATION RESULTS ===")
        print(f"Total trades: {sum(1 for a in action_ints if a != 0)}")
        print(f"Buy orders: {sum(1 for a in action_ints if a > 0)}")
        print(f"Sell orders: {sum(1 for a in action_ints if a < 0)}")
        print(f"Hold actions: {sum(1 for a in action_ints if a == 0)}")
        print(f"Final net assets: ${self.net_assets[-1]:,.0f}")
        print(f"Total return: {(self.net_assets[-1] / self.net_assets[0] - 1) * 100:.2f}%")
        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")

    def _ensemble_action(self, actions, step=0, last_trade_step=0):
        """Returns the majority action among agents with exploration"""
        count = Counter([a.item() for a in actions])
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([[majority_action]], dtype=torch.int32)


def run_exploratory_evaluation(save_path, agent_list):
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

    ensemble_evaluator = ExploratoryEnsembleEvaluator(
        save_path,
        agent_list,
        args,
    )
    ensemble_evaluator.load_agents()
    ensemble_evaluator.multi_trade()


if __name__ == "__main__":
    save_path = "ensemble_teamname/ensemble_models"
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    run_exploratory_evaluation(save_path, agent_list)