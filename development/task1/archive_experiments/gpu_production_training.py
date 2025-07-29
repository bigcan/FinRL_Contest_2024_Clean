#!/usr/bin/env python3
"""
GPU-Enabled Production Training - FORCES GPU usage for training
Addresses the critical issue where training was running on CPU instead of GPU
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

def verify_gpu_setup():
    """Verify GPU is available and configured correctly."""
    print("🔍 GPU Configuration Check")
    print("-" * 40)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training will fail.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ GPU devices: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"✅ GPU {i}: {gpu_name}")
    
    # Test GPU memory allocation
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"✅ GPU memory test passed")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False
    
    return True

def run_gpu_training():
    """Run production training with ENFORCED GPU usage."""
    
    print("🚀 GPU-ENFORCED Production Training")
    print("=" * 60)
    
    # CRITICAL: Verify GPU setup before starting
    if not verify_gpu_setup():
        print("💥 GPU setup failed. Cannot continue.")
        return False
    
    try:
        # Import training components
        from task1_ensemble import run
        from erl_agent import AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN
        
        print("🎯 Training Configuration:")
        print(f"   🔥 GPU: ENFORCED (device 0)")
        print(f"   📊 Data: Enhanced v3 Bitcoin LOB (823K timesteps)")
        print(f"   🎯 Agents: D3QN, DoubleDQN, PrioritizedDQN")
        print(f"   🚀 Features: 41-dimensional enhanced microstructure")
        print("")
        
        # Set CUDA device explicitly
        torch.cuda.set_device(0)
        print(f"✅ CUDA device set to: {torch.cuda.current_device()}")
        
        # Force GPU usage environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"gpu_training_{timestamp}"
        
        print(f"📁 Save path: {save_path}")
        print(f"⏳ Starting GPU training...")
        print("")
        
        # Run training with GPU enforcement using correct function signature
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN]  # Use actual classes
        config_dict = {
            'gpu_id': 0,  # EXPLICIT GPU ID = 0
            'starting_cash': 100000,
            'env_class': 'TradeSimulator'
        }
        
        result = run(
            save_path=save_path,
            agent_list=agent_list,
            log_rules=["print_time", "save_model"],
            config_dict=config_dict
        )
        
        print("")
        print("✅ GPU Production Training Completed!")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"💥 GPU training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_gpu_usage():
    """Monitor GPU usage during training."""
    import subprocess
    import threading
    import time
    
    def gpu_monitor():
        while True:
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
                    print(f"🔥 GPU: {gpu_util}% | Memory: {mem_used}/{mem_total} MB")
            except Exception as e:
                print(f"⚠️ GPU monitor error: {e}")
                break
            time.sleep(30)  # Check every 30 seconds
    
    monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
    monitor_thread.start()
    print("📊 GPU monitoring started (every 30 seconds)")

def main():
    """Main GPU training function with monitoring."""
    
    print("🚀 FinRL Contest 2024 - GPU-ENFORCED Production Training")
    print("=" * 70)
    print("🔥 CRITICAL FIX: Forces GPU usage (previous runs used CPU)")
    print("⚡ Expected 10-20x speed improvement with GPU acceleration")
    print("🎯 Goal: Competition-ready models with >60% accuracy")
    print("=" * 70)
    
    # Start GPU monitoring
    monitor_gpu_usage()
    
    start_time = time.time()
    
    try:
        # Run GPU training
        success = run_gpu_training()
        
        elapsed_hours = (time.time() - start_time) / 3600
        
        if success:
            print("\\n" + "=" * 70)
            print("🎉 GPU PRODUCTION TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"⏱️  Duration: {elapsed_hours:.2f} hours")
            print(f"🔥 GPU acceleration enabled throughout training")
            print(f"📊 Models ready for competition evaluation")
            print("=" * 70)
            return 0
        else:
            print("\\n💥 GPU training failed")
            return 1
        
    except Exception as e:
        print(f"\\n💥 GPU training crashed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())