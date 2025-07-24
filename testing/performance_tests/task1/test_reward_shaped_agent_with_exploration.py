"""
Test reward-shaped agent with exploration to verify it CAN trade
"""
import os
import numpy as np
import torch
from task1_eval import EnsembleEvaluator
from erl_config import Config
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN


class ExploratoryRewardShapedEvaluator(EnsembleEvaluator):
    def __init__(self, save_path, agent_classes, args, exploration_rate=0.1):
        super().__init__(save_path, agent_classes, args)
        self.exploration_rate = exploration_rate
        print(f"ExploratoryRewardShapedEvaluator initialized with {exploration_rate:.1%} exploration")
    
    def multi_trade(self):
        """Trade with exploration to test if agent can trade when forced to explore"""
        print(f"Starting exploratory trading with {self.exploration_rate:.1%} exploration...")
        
        state = self.env.reset()
        trades_made = 0
        total_steps = 0
        
        for step_i in range(self.args.env_args['max_step']):
            actions = []
            
            for i, agent in enumerate(self.agents):
                # Force some exploration
                if np.random.random() < self.exploration_rate:
                    # Random action
                    action = torch.randint(0, 3, (1, 1), dtype=torch.int32)
                    action_name = "EXPLORE"
                else:
                    # Agent's action
                    tensor_state = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                    q_values = agent.act(tensor_state)
                    action = q_values.argmax(dim=1, keepdim=True)
                    action_name = "AGENT"
                
                actions.append(action)
            
            # Use majority voting for final action
            action_votes = torch.stack(actions).squeeze()
            if len(action_votes.shape) == 0:
                action_votes = action_votes.unsqueeze(0)
            
            # Simple majority vote
            final_action = torch.mode(action_votes).values.unsqueeze(0).unsqueeze(0)
            
            # Step environment
            state, reward, done, info = self.env.step(final_action)
            
            # Check if this resulted in a trade
            if hasattr(self.env, 'action_int') and self.env.action_int[0] != 0:
                trades_made += 1
                print(f"  Step {step_i+1}: Trade executed! Action: {final_action.item()} -> {action_name}")
            
            total_steps += 1
            
            if done.any():
                break
        
        print(f"Exploratory evaluation completed:")
        print(f"  Total steps: {total_steps}")
        print(f"  Trades made: {trades_made}")
        print(f"  Trade rate: {trades_made/total_steps:.1%}")
        
        return trades_made


def test_reward_shaped_agent_exploration():
    """Test if reward-shaped agent can trade with exploration"""
    
    print("="*60)
    print("TESTING REWARD-SHAPED AGENT WITH EXPLORATION")
    print("="*60)
    
    # Configuration
    env_args = {
        "env_name": "TradeSimulator-v0", 
        "num_envs": 1,
        "max_step": 100,  # Shorter test
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": 1,
        "step_gap": 2,
        "dataset_path": "data/BTC_1sec_predict.npy"
    }
    
    args = Config(agent_class=AgentD3QN, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = -1
    args.random_seed = 42
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6
    
    # Test different exploration rates
    exploration_rates = [0.0, 0.1, 0.3]
    
    for exploration_rate in exploration_rates:
        print(f"\n--- Testing with {exploration_rate:.1%} exploration ---")
        
        try:
            evaluator = ExploratoryRewardShapedEvaluator(
                save_path="ensemble_reward_shaped_balanced",
                agent_classes=[AgentD3QN],
                args=args,
                exploration_rate=exploration_rate
            )
            
            evaluator.load_agents()
            trades = evaluator.multi_trade()
            
            if trades > 0:
                print(f"✅ Agent CAN trade with {exploration_rate:.1%} exploration ({trades} trades)")
            else:
                print(f"❌ No trades even with {exploration_rate:.1%} exploration")
                
        except Exception as e:
            print(f"❌ Error with {exploration_rate:.1%} exploration: {e}")
    
    print(f"\n" + "="*60)
    print("CONCLUSION:")
    print("If agent trades with exploration, the issue is overly conservative policy.")
    print("If agent doesn't trade even with exploration, there may be a deeper issue.")
    print("="*60)


if __name__ == "__main__":
    test_reward_shaped_agent_exploration()