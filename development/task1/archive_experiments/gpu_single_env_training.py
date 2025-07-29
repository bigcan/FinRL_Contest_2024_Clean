#!/usr/bin/env python3
"""
GPU Single Environment Training - Avoids vectorized batch issues
Uses single environment instead of vectorized to prevent dimension mismatches
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

def run_single_env_training():
    """Run training with single environment to avoid batch issues."""
    
    print("🚀 GPU SINGLE ENVIRONMENT Training")
    print("=" * 60)
    
    # Verify GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.set_device(0)
    
    try:
        from task1_ensemble import run
        from erl_agent import AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN
        
        print("🎯 Training Configuration:")
        print(f"   🔥 GPU: Single Environment (device 0)")
        print(f"   📊 Data: Enhanced v3 Bitcoin LOB")
        print(f"   🎯 Agents: D3QN, DoubleDQN, PrioritizedDQN")
        print(f"   🔧 Environment: Single (non-vectorized)")
        print("")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"gpu_single_env_{timestamp}"
        
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN]
        config_dict = {
            'gpu_id': 0,
            'starting_cash': 100000,
            'env_class': 'TradeSimulator',
            # CRITICAL: Force single environment
            'num_envs': 1,  # Single environment only
            'env_num': 1,   # Single environment 
            'batch_size': 256,  # Smaller batch for single env
            'horizon_len': 1000,  # Shorter horizon
            'max_step': 100000,  # Reduced steps
            'if_remove': True,  # Clean previous runs
        }
        
        print(f"📁 Save path: {save_path}")
        print(f"⏳ Starting single environment training...")
        
        result = run(
            save_path=save_path,
            agent_list=agent_list,
            log_rules=["print_time", "save_model"],
            config_dict=config_dict
        )
        
        print("✅ Single Environment Training Completed!")
        return result
        
    except Exception as e:
        print(f"💥 Single env training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_simple():
    """Simple monitoring without complex GPU queries."""
    import threading
    import subprocess
    
    def simple_monitor():
        for i in range(20):  # Monitor for 10 minutes (20 * 30s)
            try:
                # Simple GPU check
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_util, mem_used = result.stdout.strip().split(', ')
                    print(f"🔥 GPU: {gpu_util}% | Memory: {mem_used}MB")
            except:
                pass
            time.sleep(30)
    
    thread = threading.Thread(target=simple_monitor, daemon=True)
    thread.start()

def main():
    """Main single environment training."""
    
    print("🚀 FinRL Contest 2024 - GPU SINGLE ENVIRONMENT Training")
    print("=" * 70)
    print("🔧 AVOIDS: Vectorized environment batch dimension errors")
    print("⚡ Uses single environment for stable training")
    print("🎯 Goal: Complete successful GPU training run")
    print("=" * 70)
    
    monitor_simple()
    start_time = time.time()
    
    try:
        success = run_single_env_training()
        elapsed_hours = (time.time() - start_time) / 3600
        
        if success:
            print("\\n" + "=" * 70)
            print("🎉 SINGLE ENVIRONMENT TRAINING COMPLETED!")
            print("=" * 70)
            print(f"⏱️  Duration: {elapsed_hours:.2f} hours")
            print(f"🔧 No batch dimension errors encountered")
            print("=" * 70)
            return 0
        else:
            print("\\n💥 Single environment training failed")
            return 1
        
    except Exception as e:
        print(f"\\n💥 Training crashed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())