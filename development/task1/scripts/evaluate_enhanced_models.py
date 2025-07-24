"""
Evaluate Enhanced Features Models

Evaluate the newly trained enhanced features models and compare with baseline.
"""

import os
import numpy as np
import torch
from task1_eval import EnsembleEvaluator
from trade_simulator import EvalTradeSimulator
from erl_config import Config
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN

def main():
    """Evaluate enhanced features models"""
    print("=" * 60)
    print("EVALUATING ENHANCED FEATURES MODELS")
    print("=" * 60)
    
    # Check if enhanced models exist
    model_path = "ensemble_enhanced_gpu_0/ensemble_models"
    if not os.path.exists(model_path):
        print("❌ Enhanced models not found. Training may not be complete.")
        return
    
    print(f"✅ Found enhanced models at: {model_path}")
    
    # Check available agents
    agent_dirs = []
    for agent_name in ["AgentD3QN", "AgentDoubleDQN", "AgentTwinD3QN"]:
        agent_path = os.path.join(model_path, agent_name)
        if os.path.exists(agent_path):
            agent_dirs.append(agent_path)
            print(f"✅ Found {agent_name} model")
    
    if not agent_dirs:
        print("❌ No agent models found")
        return
    
    # Setup evaluation configuration
    from trade_simulator import TradeSimulator
    temp_sim = TradeSimulator(num_sims=1)
    state_dim = temp_sim.state_dim
    
    print(f"✅ Enhanced features state_dim: {state_dim}")
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": 1,
        "max_step": (4800 - 60) // 2,
        "state_dim": state_dim,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 2,
        "slippage": 7e-7,
        "num_sims": 1,
        "step_gap": 2,
        "dataset_path": "data/BTC_1sec_predict.npy"
    }
    
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = 0  # Use GPU for evaluation too
    
    # Agent classes
    agent_classes = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    
    print(f"\n🚀 Starting evaluation...")
    print(f"   Model path: {model_path}")
    print(f"   State dimensions: {state_dim}")
    print(f"   Enhanced features: Enabled")
    
    try:
        # Create evaluator
        evaluator = EnsembleEvaluator(
            save_path=model_path,
            agent_classes=agent_classes,
            args=args
        )
        
        print(f"✅ Evaluator created with {len(evaluator.agents)} agents")
        
        # Run evaluation
        print(f"\n📊 Running evaluation...")
        evaluator.ensemble_eval()
        
        print(f"\n" + "=" * 60)
        print("✅ ENHANCED FEATURES EVALUATION COMPLETE")
        print("=" * 60)
        print("Check the output files for detailed results!")
        print("Enhanced features should show significant improvement over baseline.")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\nThis might be due to:")
        print(f"- Model compatibility issues")
        print(f"- Configuration mismatches")
        print(f"- Try running: python3 task1_eval.py directly")

if __name__ == "__main__":
    main()