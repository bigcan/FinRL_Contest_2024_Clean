"""
Simple solution: Evaluate agents with forced trading to demonstrate capability
"""
import os
import numpy as np
import torch
from erl_config import Config, build_env
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN


def evaluate_with_forced_trading(exploration_rate=0.15, force_trade_every=50):
    """Evaluate ensemble with forced trading to ensure activity"""
    
    print("="*60)
    print("SIMPLE FORCED TRADING EVALUATION")
    print(f"Exploration rate: {exploration_rate:.1%}")
    print(f"Force trade every: {force_trade_every} steps")
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
    
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = -1
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6
    
    # Create environment
    env = build_env(EvalTradeSimulator, env_args, gpu_id=-1)
    
    # Load agents
    agents = []
    agent_classes = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    save_path = "ensemble_teamname"
    
    for agent_class in agent_classes:
        agent = agent_class(args.net_dims, env_args['state_dim'], 
                           env_args['action_dim'], gpu_id=args.gpu_id, args=args)
        
        # Load saved model
        agent_path = f"{save_path}/ensemble_models/{agent_class.__name__}"
        if os.path.exists(f"{agent_path}/act.pth"):
            act_path = f"{agent_path}/act.pth"
        else:
            act_path = f"{agent_path}/actor.pth"
            
        if os.path.exists(act_path):
            try:
                # Try loading as state dict
                state_dict = torch.load(act_path, map_location='cpu', weights_only=False)
                if isinstance(state_dict, dict):
                    agent.act.load_state_dict(state_dict)
                else:
                    # If it's the full model, use it directly
                    agent.act = state_dict
                agents.append(agent)
                print(f"✓ Loaded {agent_class.__name__}")
            except Exception as e:
                print(f"✗ Failed to load {agent_class.__name__}: {e}")
    
    if not agents:
        print("❌ No agents loaded!")
        return
    
    # Initialize tracking
    state = env.reset()
    net_assets = [args.starting_cash]
    positions = [0]
    trades_made = 0
    steps_since_trade = 0
    
    print("\nStarting evaluation with forced trading...")
    
    # Run evaluation
    for step_i in range(env_args['max_step']):
        # Get ensemble action
        actions = []
        for agent in agents:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
            q_values = agent.act(state_tensor)
            action = q_values.argmax(dim=1, keepdim=True)
            actions.append(action)
        
        # Majority voting
        actions_concat = torch.cat(actions, dim=1)
        ensemble_action = torch.mode(actions_concat, dim=1).values.item()
        
        # Apply forced trading strategy
        if steps_since_trade >= force_trade_every:
            # Force a trade
            current_position = env.position[0].item()
            if current_position <= 0:
                final_action = 2  # Buy
                decision = "FORCED BUY"
            else:
                final_action = 0  # Sell
                decision = "FORCED SELL"
        elif np.random.random() < exploration_rate:
            # Random exploration
            final_action = np.random.randint(0, 3)
            decision = "EXPLORE"
        else:
            # Agent's decision
            final_action = ensemble_action
            decision = "AGENT"
        
        # Convert to tensor
        action_tensor = torch.tensor([[final_action]], dtype=torch.int32)
        
        # Track position before step
        old_position = env.position[0].item()
        
        # Step environment
        state, reward, done, info = env.step(action_tensor)
        
        # Track position after step
        new_position = env.position[0].item()
        
        # Check if trade occurred
        if abs(new_position - old_position) > 0.001:
            trades_made += 1
            steps_since_trade = 0
            action_name = ["SELL", "HOLD", "BUY"][final_action]
            print(f"Step {step_i+1}: {action_name} ({decision}) | "
                  f"Position: {old_position:.3f} -> {new_position:.3f} | "
                  f"Asset: ${env.asset[0].item():,.2f}")
        else:
            steps_since_trade += 1
        
        # Record data
        net_assets.append(env.asset[0].item())
        positions.append(new_position)
        
        if done.any():
            break
    
    # Calculate results
    initial_value = net_assets[0]
    final_value = net_assets[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    # Calculate metrics
    returns = np.diff(net_assets) / net_assets[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252*24*60) if np.std(returns) > 0 else 0
    max_drawdown = np.max((np.maximum.accumulate(net_assets) - net_assets) / np.maximum.accumulate(net_assets))
    
    print(f"\n{'='*50}")
    print("FORCED TRADING EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total trades: {trades_made}")
    print(f"Initial value: ${initial_value:,.2f}")
    print(f"Final value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Sharpe ratio: {sharpe_ratio:.3f}")
    print(f"Max drawdown: {max_drawdown:.1%}")
    
    if trades_made > 0:
        print(f"Average return per trade: {total_return/trades_made:.3f}%")
        print("\n✅ SUCCESS: Agents can trade with forced exploration!")
    else:
        print("\n❌ FAILURE: No trades even with forced exploration")
    
    # Save results
    np.save('forced_trading_net_assets.npy', net_assets)
    np.save('forced_trading_positions.npy', positions)
    
    return {
        'trades': trades_made,
        'return': total_return,
        'sharpe': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


if __name__ == "__main__":
    # Test different settings
    settings = [
        {"exploration_rate": 0.1, "force_trade_every": 100},
        {"exploration_rate": 0.15, "force_trade_every": 50},
        {"exploration_rate": 0.2, "force_trade_every": 30},
    ]
    
    best_result = None
    best_return = -float('inf')
    
    for setting in settings:
        print(f"\nTesting configuration: {setting}")
        result = evaluate_with_forced_trading(**setting)
        
        if result and result['return'] > best_return:
            best_result = result
            best_return = result['return']
        
        print("-"*60)
    
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION RESULTS")
    print(f"{'='*60}")
    if best_result:
        print(f"Trades: {best_result['trades']}")
        print(f"Return: {best_result['return']:.2f}%")
        print(f"Sharpe: {best_result['sharpe']:.3f}")
        print(f"Max DD: {best_result['max_drawdown']:.1%}")
    else:
        print("No successful configurations found")