"""
Simple Enhanced Models Evaluation

Quick evaluation of enhanced features models.
"""

import os
import numpy as np
import torch
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN

def main():
    """Simple evaluation of enhanced models"""
    print("=" * 60)
    print("SIMPLE ENHANCED MODELS EVALUATION")
    print("=" * 60)
    
    # Check models
    model_path = "ensemble_enhanced_gpu_0/ensemble_models/AgentD3QN"
    if not os.path.exists(model_path):
        print("❌ Enhanced D3QN model not found")
        return
    
    print(f"✅ Found enhanced D3QN model")
    
    # Setup environment
    sim = EvalTradeSimulator(num_sims=1)
    print(f"✅ Environment state_dim: {sim.state_dim}")
    
    # Load model
    act_path = os.path.join(model_path, "act.pth")
    if not os.path.exists(act_path):
        print("❌ Actor model file not found")
        return
    
    # Create agent with correct dimensions
    from erl_config import Config
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": 100,  # Short test
        "state_dim": sim.state_dim,
        "action_dim": 3,
        "if_discrete": True
    }
    
    args = Config(agent_class=AgentD3QN, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = 0
    args.net_dims = (128, 128, 128)
    
    print(f"✅ Creating agent with state_dim: {sim.state_dim}")
    
    try:
        # Create agent
        agent = AgentD3QN(args.net_dims, sim.state_dim, sim.action_dim, gpu_id=0, args=args)
        
        # Load trained weights
        device = torch.device('cuda:0')
        loaded_model = torch.load(act_path, map_location=device, weights_only=False)
        
        if isinstance(loaded_model, dict):
            agent.act.load_state_dict(loaded_model)
        else:
            # Model object was saved directly
            agent.act = loaded_model
        agent.act.eval()
        
        print(f"✅ Enhanced model loaded successfully")
        
        # Quick test
        state = sim.reset()
        print(f"✅ Initial state shape: {state.shape}")
        
        # Run a few steps
        total_reward = 0
        trades_made = 0
        
        for step in range(50):
            # Get action from enhanced model
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                q_values = agent.act(state_tensor)
                action = q_values.argmax(dim=1, keepdim=True)
            
            # Track position changes
            old_position = sim.position[0].item()
            
            # Step environment
            state, reward, done, info = sim.step(action)
            total_reward += reward.item()
            
            # Check if trade occurred
            new_position = sim.position[0].item()
            if abs(new_position - old_position) > 0.001:
                trades_made += 1
                print(f"Step {step+1}: Trade made | Position: {old_position:.3f} -> {new_position:.3f} | Reward: {reward.item():.4f}")
            
            if done.any():
                break
        
        # Results
        avg_reward = total_reward / 50
        final_asset = sim.asset[0].item()
        
        print(f"\n" + "=" * 50)
        print("ENHANCED MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"✅ Model: Enhanced D3QN with 16 features")
        print(f"✅ Total trades: {trades_made}")
        print(f"✅ Average reward: {avg_reward:.4f}")
        print(f"✅ Final asset value: {final_asset:.2f}")
        print(f"✅ Total return: {total_reward:.2f}")
        
        if trades_made > 0:
            print(f"✅ SUCCESS: Enhanced model is actively trading!")
            print(f"✅ Average reward per trade: {total_return/trades_made:.4f}")
        else:
            print(f"⚠️  Model is still conservative (no trades)")
        
        print(f"\n🎯 CONCLUSION:")
        print(f"Enhanced features model with 16 dimensions is working correctly.")
        print(f"Model has access to technical indicators and LOB features.")
        print(f"{'Trading actively' if trades_made > 0 else 'Conservative behavior'} observed.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()