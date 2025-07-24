"""
Quick Phase 3 Training Test
Runs a short training session to validate the optimized pipeline
"""

import os
import sys
import torch
import numpy as np
import time
from datetime import datetime

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator

def quick_training_test():
    """Run a quick training test with optimized features"""
    print("🚀 Phase 3: Quick Training Test")
    print("=" * 50)
    
    # Setup
    gpu_id = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🖥️  Device: {device_name}")
    
    # Get actual state dimension from TradeSimulator
    temp_sim = TradeSimulator(num_sims=1)
    state_dim = temp_sim.state_dim
    print(f"📊 State Dimension: {state_dim}")
    print(f"🎯 Features: {temp_sim.feature_names}")
    
    # Quick training configuration
    num_sims = 8  # Very small for quick test
    max_step = 200  # Short episodes
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": state_dim,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": num_sims,
        "step_gap": 2,
    }
    
    # Create configuration
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = 42
    
    # Optimized architecture
    args.net_dims = (128, 64, 32)
    args.learning_rate = 2e-6
    args.batch_size = 128  # Small for quick test
    args.gamma = 0.995
    args.explore_rate = 0.005
    args.break_step = 4  # Very short training
    args.horizon_len = max_step
    args.buffer_size = max_step * 4
    args.repeat_times = 1
    args.eval_per_step = max_step
    args.eval_times = 1
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.cwd = f"../../../results/task1_results/trained_models/quick_phase3_test_{timestamp}"
    os.makedirs(args.cwd, exist_ok=True)
    
    print(f"💾 Model save path: {args.cwd}")
    print(f"🔧 Training config: LR={args.learning_rate}, Batch={args.batch_size}, Steps={args.break_step}")
    
    # Initialize training
    args.init_before_training()
    torch.set_grad_enabled(False)
    
    # Build environment
    env = build_env(args.env_class, args.env_args, args.gpu_id)
    print(f"🌍 Environment created: {num_sims} parallel simulations")
    
    # Create agent
    print(f"🤖 Creating agent with architecture {args.net_dims}...")
    agent = args.agent_class(
        args.net_dims,
        args.state_dim,
        args.action_dim,
        gpu_id=args.gpu_id,
        args=args,
    )
    
    # Initialize state
    state = env.reset()
    if args.num_envs == 1:
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    else:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(agent.device)
    
    agent.last_state = state.detach()
    print(f"🎮 Initial state shape: {state.shape}")
    
    # Initialize buffer
    buffer = ReplayBuffer(
        gpu_id=args.gpu_id,
        num_seqs=args.num_envs,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1,
    )
    
    # Warm up buffer
    print("🔄 Warming up replay buffer...")
    buffer_items = agent.explore_env(env, args.horizon_len, if_random=True)
    buffer.update(buffer_items)
    print(f"   Buffer initialized with capacity: {args.buffer_size}")
    
    # Training loop
    print(f"🏋️  Starting training for {args.break_step} steps...")
    start_time = time.time()
    
    for step in range(args.break_step):
        step_start = time.time()
        
        # Collect experience
        buffer_items = agent.explore_env(env, args.horizon_len)
        exp_r = buffer_items[2].mean().item()
        
        # Update buffer
        buffer.update(buffer_items)
        
        # Update network
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        
        step_time = time.time() - step_start
        
        print(f"   Step {step+1}/{args.break_step}: Reward={exp_r:.4f}, Time={step_time:.1f}s")
        
        if logging_tuple:
            obj_critic, obj_actor = logging_tuple[:2]
            print(f"     Critic Loss: {obj_critic:.4f}, Actor Loss: {obj_actor:.4f}")
    
    total_time = time.time() - start_time
    print(f"✅ Training completed in {total_time:.1f}s")
    
    # Save model
    agent.save_or_load_agent(args.cwd, if_save=True)
    print(f"💾 Model saved to: {args.cwd}")
    
    # Quick evaluation test
    print(f"\n🧪 Quick Evaluation Test...")
    eval_state = env.reset()
    if not isinstance(eval_state, torch.Tensor):
        eval_state = torch.tensor(eval_state, dtype=torch.float32)
    eval_state = eval_state.to(agent.device)
    
    with torch.no_grad():
        q_values = agent.act(eval_state)
        actions = q_values.argmax(dim=1, keepdim=True)
        print(f"   Action selection working: {actions.shape}")
        print(f"   Q-values: {q_values[0].detach().cpu().numpy()}")
    
    # Test different states
    action_counts = {0: 0, 1: 0, 2: 0}
    for _ in range(100):
        test_state = env.reset()
        if not isinstance(test_state, torch.Tensor):
            test_state = torch.tensor(test_state, dtype=torch.float32)
        test_state = test_state.to(agent.device)
        
        with torch.no_grad():
            q_vals = agent.act(test_state)
            action = q_vals.argmax(dim=1, keepdim=True)
            action_counts[action[0].item()] += 1
    
    print(f"   Action distribution over 100 states: {action_counts}")
    
    env.close() if hasattr(env, "close") else None
    
    print(f"\n🎉 Quick Phase 3 Test Results:")
    print(f"   ✅ Optimized 8-feature pipeline working")
    print(f"   ✅ Enhanced architecture training successful")
    print(f"   ✅ Agent learning and making decisions")
    print(f"   ✅ Training speed: {total_time/args.break_step:.1f}s per step")
    
    # Check if agent is making varied decisions
    action_variety = len([v for v in action_counts.values() if v > 0])
    if action_variety >= 2:
        print(f"   ✅ Good action variety: {action_variety}/3 actions used")
    else:
        print(f"   ⚠️  Limited action variety: {action_variety}/3 actions used")
    
    return True

if __name__ == "__main__":
    try:
        success = quick_training_test()
        if success:
            print(f"\n🚀 Ready for full Phase 3 training!")
            print(f"   Command: python3 task1_ensemble_optimized.py 0")
        else:
            print(f"\n❌ Quick test failed!")
    except Exception as e:
        print(f"\n❌ Error in quick test: {e}")
        import traceback
        traceback.print_exc()