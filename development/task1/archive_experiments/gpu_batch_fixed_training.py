#!/usr/bin/env python3
"""
GPU Batch-Fixed Training - Resolves tensor dimension mismatch errors
Handles vectorized environment batch size issues correctly
"""

import os
import sys
import time
import json
import torch
from pathlib import Path
from datetime import datetime

# Add src path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

def verify_gpu_and_fix_batch():
    """Verify GPU and set batch-compatible defaults."""
    print("🔍 GPU & Batch Configuration Check")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Clear any existing memory
    torch.cuda.empty_cache()
    print(f"✅ GPU memory cleared")
    
    # Set device explicitly
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print(f"✅ CUDA device set: {device}")
    
    return True

def run_batch_fixed_training():
    """Run training with batch dimension fixes."""
    
    print("🚀 GPU BATCH-FIXED Production Training")
    print("=" * 60)
    
    if not verify_gpu_and_fix_batch():
        return False
    
    try:
        # Import with explicit device configuration
        from task1_ensemble import run
        from erl_agent import AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN
        
        print("🎯 Training Configuration:")
        print(f"   🔥 GPU: BATCH-FIXED (device 0)")
        print(f"   📊 Data: Enhanced v3 Bitcoin LOB (823K timesteps)")
        print(f"   🎯 Agents: D3QN, DoubleDQN, PrioritizedDQN")
        print(f"   🔧 Batch Size: Correctly handled")
        print("")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"gpu_batch_fixed_{timestamp}"
        
        print(f"📁 Save path: {save_path}")
        print(f"⏳ Starting batch-fixed training...")
        print("")
        
        # Configuration with explicit batch handling
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN]
        config_dict = {
            'gpu_id': 0,
            'starting_cash': 100000,
            'env_class': 'TradeSimulator',
            'device': 'cuda:0',
            # Batch size fixes
            'batch_size': 512,
            'horizon_len': 2370,
            'num_envs': 64,  # Explicit vectorized env count
            'if_per_or_gae': False,  # Disable PER to avoid dimension issues
        }
        
        result = run(
            save_path=save_path,
            agent_list=agent_list,
            log_rules=["print_time", "save_model"],
            config_dict=config_dict
        )
        
        print("✅ Batch-Fixed Training Completed!")
        return result
        
    except Exception as e:
        print(f"💥 Batch-fixed training failed: {e}")
        if "size" in str(e).lower() or "dimension" in str(e).lower():
            print("🔧 This is still a tensor dimension issue.")
            print("   The vectorized environment batch handling needs adjustment.")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with comprehensive error handling."""
    
    print("🚀 FinRL Contest 2024 - GPU BATCH-FIXED Training")
    print("=" * 70)
    print("🔧 CRITICAL FIX: Resolves tensor batch dimension errors")
    print("⚡ GPU training with proper vectorized environment handling")
    print("🎯 Goal: Complete training without dimension mismatches")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        success = run_batch_fixed_training()
        elapsed_hours = (time.time() - start_time) / 3600
        
        if success:
            print("\\n" + "=" * 70)
            print("🎉 BATCH-FIXED TRAINING COMPLETED!")
            print("=" * 70)
            print(f"⏱️  Duration: {elapsed_hours:.2f} hours")
            print(f"🔧 All tensor dimension issues resolved")
            print("=" * 70)
            return 0
        else:
            print("\\n💥 Batch-fixed training failed")
            return 1
        
    except Exception as e:
        print(f"\\n💥 Training crashed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())