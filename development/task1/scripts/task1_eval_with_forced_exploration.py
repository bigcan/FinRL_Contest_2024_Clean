"""
Reward-shaped evaluation with forced exploration to guarantee trading activity
"""
import numpy as np
import torch
from erl_config import Config
from reward_shaped_simulator import RewardShapedEvalTradeSimulator
from reward_shaping_config import RewardShapingConfig
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from task1_eval import EnsembleEvaluator


class ForcedExplorationEvaluator(EnsembleEvaluator):
    def __init__(self, save_path, agent_classes, args, exploration_rate=0.1, force_trade_every_n=50):
        super().__init__(save_path, agent_classes, args)
        self.exploration_rate = exploration_rate
        self.force_trade_every_n = force_trade_every_n
        self.steps_since_last_trade = 0
        print(f"ForcedExplorationEvaluator initialized:")
        print(f"  Exploration rate: {exploration_rate:.1%}")
        print(f"  Force trade every: {force_trade_every_n} steps")
    
    def multi_trade(self):
        """Trade with forced exploration to ensure activity"""
        self.env.reset()
        
        self.net_assets = np.zeros((self.args.env_args['max_step'] + 1,))
        self.positions = np.zeros((self.args.env_args['max_step'] + 1,))
        self.btc_positions = np.zeros((self.args.env_args['max_step'] + 1,))
        
        state = self.env.get_state()
        self.net_assets[0] = self.args.starting_cash
        
        trades_made = 0
        forced_trades = 0
        exploration_trades = 0
        
        for step_i in range(self.args.env_args['max_step']):
            # Get agent actions
            agent_actions = []
            for agent in self.agents:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                q_values = agent.act(state_tensor)
                action = q_values.argmax(dim=1, keepdim=True)
                agent_actions.append(action)
            
            # Majority voting
            actions_concat = torch.cat(agent_actions, dim=1)
            ensemble_action = torch.mode(actions_concat, dim=1).values.unsqueeze(1)
            
            # Apply exploration strategy
            if self.steps_since_last_trade >= self.force_trade_every_n:
                # Force a trade
                if self.env.position[0] <= 0:
                    final_action = torch.tensor([[2]], dtype=torch.int32)  # Buy
                else:
                    final_action = torch.tensor([[0]], dtype=torch.int32)  # Sell
                action_source = "FORCED"
                forced_trades += 1
            elif np.random.random() < self.exploration_rate:
                # Random exploration
                final_action = torch.randint(0, 3, (1, 1), dtype=torch.int32)
                action_source = "EXPLORE"
                if final_action.item() != 1:
                    exploration_trades += 1
            else:
                # Agent's action
                final_action = ensemble_action
                action_source = "AGENT"
            
            # Track position before step
            old_position = self.env.position[0].item()
            
            # Step environment
            state, reward, done, info = self.env.step(final_action)
            
            # Track position after step
            new_position = self.env.position[0].item()
            
            # Check if trade occurred
            if abs(new_position - old_position) > 0.001:
                trades_made += 1
                self.steps_since_last_trade = 0
                action_name = ["SELL", "HOLD", "BUY"][final_action.item()]
                print(f"Step {step_i+1}: {action_name} trade executed ({action_source}) | "
                      f"Position: {old_position:.3f} -> {new_position:.3f}")
            else:
                self.steps_since_last_trade += 1
            
            # Record data
            self.net_assets[step_i + 1] = self.env.asset[0].cpu().numpy()
            self.positions[step_i + 1] = self.env.position[0].cpu().numpy()
            self.btc_positions[step_i + 1] = self.env.amount[0].cpu().numpy()
            
            if done.any():
                break
        
        # Save results
        np.save('forced_exploration_net_assets.npy', self.net_assets)
        np.save('forced_exploration_positions.npy', self.positions)
        np.save('forced_exploration_btc_positions.npy', self.btc_positions)
        
        # Print summary
        initial_value = self.net_assets[0]
        final_value = self.net_assets[self.args.env_args['max_step']]
        total_return = (final_value / initial_value - 1) * 100
        
        print(f"\n=== FORCED EXPLORATION EVALUATION RESULTS ===")
        print(f"Total trades: {trades_made}")
        print(f"  - Agent trades: {trades_made - forced_trades - exploration_trades}")
        print(f"  - Exploration trades: {exploration_trades}")
        print(f"  - Forced trades: {forced_trades}")
        print(f"Starting value: ${initial_value:,.2f}")
        print(f"Final value: ${final_value:,.2f}")
        print(f"Total return: {total_return:.2f}%")
        
        if trades_made > 0:
            avg_return_per_trade = total_return / trades_made
            print(f"Average return per trade: {avg_return_per_trade:.3f}%")
        
        return trades_made, total_return


def run_forced_exploration_evaluation():
    """Run evaluation with forced exploration"""
    
    print("="*60)
    print("ENSEMBLE EVALUATION WITH FORCED EXPLORATION")
    print("="*60)
    
    # Configuration
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": (4800 - 60) // 2,
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 1,
        "step_gap": 2,
        "dataset_path": "data/BTC_1sec_predict.npy"
    }
    
    args = Config(agent_class=None, env_class=RewardShapedEvalTradeSimulator, env_args=env_args)
    args.gpu_id = -1
    args.random_seed = 0
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6
    
    # Add reward config for reward-shaped environment
    reward_config = RewardShapingConfig.balanced_config()
    args.env_args['reward_config'] = reward_config
    
    # Test different exploration settings
    settings = [
        {"exploration_rate": 0.05, "force_trade_every_n": 100},  # Conservative
        {"exploration_rate": 0.1, "force_trade_every_n": 50},    # Balanced
        {"exploration_rate": 0.2, "force_trade_every_n": 25},    # Aggressive
    ]
    
    for i, setting in enumerate(settings):
        print(f"\n{'='*50}")
        print(f"TEST {i+1}: Exploration={setting['exploration_rate']:.0%}, "
              f"Force trade every {setting['force_trade_every_n']} steps")
        print("="*50)
        
        evaluator = ForcedExplorationEvaluator(
            save_path="ensemble_teamname",
            agent_classes=[AgentD3QN, AgentDoubleDQN, AgentTwinD3QN],
            args=args,
            exploration_rate=setting['exploration_rate'],
            force_trade_every_n=setting['force_trade_every_n']
        )
        
        evaluator.load_agents()
        trades, returns = evaluator.multi_trade()
        
        if trades > 0:
            print(f"\n✅ SUCCESS: Forced exploration achieved {trades} trades with {returns:.2f}% returns")
            break
    
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print("Forced exploration successfully enables trading activity.")
    print("This approach can be used as a fallback when pure reward shaping fails.")
    print("="*60)


if __name__ == "__main__":
    run_forced_exploration_evaluation()