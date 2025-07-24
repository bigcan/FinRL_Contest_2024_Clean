"""
Final test - force aggressive agent to explore and verify it CAN trade
"""
import os
import numpy as np
import torch
from erl_config import Config, build_env
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN


def force_trading_test():
    """Force the aggressive agent to trade and see results"""
    
    print("="*60)
    print("FINAL AGGRESSIVE AGENT FORCE TRADING TEST")
    print("="*60)
    
    # Configuration
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": 500,  # Longer test
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
    
    try:
        # Load aggressive agent
        agent = AgentD3QN(args.net_dims, args.env_args['state_dim'], 
                         args.env_args['action_dim'], gpu_id=args.gpu_id, args=args)
        
        # Load saved model
        agent_path = "ensemble_reward_shaped_aggressive/AgentD3QN"
        if os.path.exists(f"{agent_path}/act.pth"):
            agent.act_net.load_state_dict(torch.load(f"{agent_path}/act.pth", map_location='cpu'))
            print(f"✓ Loaded aggressive agent from {agent_path}")
        else:
            print(f"❌ Could not find agent at {agent_path}")
            return
        
        # Create environment
        env = build_env(EvalTradeSimulator, env_args, gpu_id=-1)
        
        # Test different exploration rates
        exploration_rates = [0.0, 0.2, 0.5, 0.8]
        
        for explore_rate in exploration_rates:
            print(f"\n--- Testing with {explore_rate:.1%} exploration ---")
            
            state = env.reset()
            trades_made = 0
            portfolio_values = [env.asset[0].item()]
            
            for step in range(200):  # Shorter test
                # Exploration vs exploitation
                if np.random.random() < explore_rate:
                    # Random action
                    action = torch.randint(0, 3, (1, 1), dtype=torch.int32)
                    decision = "EXPLORE"
                else:
                    # Agent's action
                    tensor_state = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                    q_values = agent.act(tensor_state)
                    action = q_values.argmax(dim=1, keepdim=True)
                    decision = "AGENT"
                
                # Step environment
                old_position = env.position[0].item()
                state, reward, done, info = env.step(action)
                new_position = env.position[0].item()
                
                # Check if trade occurred
                if abs(new_position - old_position) > 0.001:
                    trades_made += 1
                    print(f"  Step {step+1}: TRADE! {old_position:.3f} -> {new_position:.3f} ({decision})")
                
                portfolio_values.append(env.asset[0].item())
                
                if done.any():
                    break
            
            # Calculate results
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value / initial_value - 1) * 100
            
            print(f"  Results: {trades_made} trades, {total_return:.2f}% return")
            
            if trades_made > 0:
                print(f"  🎉 SUCCESS! Agent CAN trade with {explore_rate:.1%} exploration")
                avg_return_per_trade = total_return / trades_made if trades_made > 0 else 0
                print(f"  Average return per trade: {avg_return_per_trade:.3f}%")
                break
            else:
                print(f"  😐 No trades with {explore_rate:.1%} exploration")
        
        print(f"\n" + "="*60)
        print("CONCLUSION:")
        if any(explore_rate > 0 for explore_rate in exploration_rates):
            if trades_made > 0:
                print("✅ Agent CAN trade when forced to explore!")
                print("   Issue: Learned policy is overly conservative")
                print("   Solution: Need stronger activity incentives during training")
            else:
                print("❌ Agent won't trade even with high exploration")
                print("   Issue: May be deeper structural problem")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    force_trading_test()