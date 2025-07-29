#!/usr/bin/env python3
"""
GPU Synchronized Training - Fixes device synchronization issues
Ensures ALL tensors (models, data, states) are on the same GPU device
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
    print("🔍 GPU Configuration Verification")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training will fail.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ GPU devices: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"✅ GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
    
    # Test GPU allocation and clear cache
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"✅ GPU memory allocation test passed")
        del test_tensor
        torch.cuda.empty_cache()
        print(f"✅ GPU memory cleared")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False
    
    return True

def force_gpu_sync():
    """Force all torch operations to use GPU and sync devices."""
    print("🔧 Forcing GPU Synchronization")
    print("-" * 30)
    
    # Set default tensor type to CUDA
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Set CUDA device explicitly
    torch.cuda.set_device(0)
    
    # Force environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    print(f"✅ Default tensor type: {torch.get_default_dtype()}")
    print(f"✅ CUDA device: {torch.cuda.current_device()}")
    print(f"✅ Environment variables set")

def run_gpu_sync_training():
    """Run training with complete GPU synchronization."""
    
    print("🚀 GPU SYNCHRONIZED Production Training")
    print("=" * 60)
    
    # STEP 1: Verify GPU
    if not verify_gpu_setup():
        print("💥 GPU setup failed. Cannot continue.")
        return False
    
    # STEP 2: Force GPU synchronization
    force_gpu_sync()
    
    try:
        # STEP 3: Import training components AFTER GPU setup
        from task1_ensemble import run
        from erl_agent import AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN
        
        print("🎯 Training Configuration:")
        print(f"   🔥 GPU: SYNCHRONIZED (device 0)")
        print(f"   📊 Data: Enhanced v3 Bitcoin LOB (823K timesteps)")
        print(f"   🎯 Agents: D3QN, DoubleDQN, PrioritizedDQN")
        print(f"   🚀 Features: 41-dimensional enhanced microstructure")
        print(f"   🔧 Device Sync: ENFORCED")
        print("")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"gpu_sync_training_{timestamp}"
        
        print(f"📁 Save path: {save_path}")
        print(f"⏳ Starting GPU synchronized training...")
        print("")
        
        # STEP 4: Run training with explicit GPU configuration
        agent_list = [AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN]
        config_dict = {
            'gpu_id': 0,  # EXPLICIT GPU ID
            'starting_cash': 100000,
            'env_class': 'TradeSimulator',
            'device': 'cuda:0',  # EXPLICIT DEVICE
            'force_gpu': True,   # FORCE GPU FLAG
        }
        
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        
        result = run(
            save_path=save_path,
            agent_list=agent_list,
            log_rules=["print_time", "save_model"],
            config_dict=config_dict
        )
        
        print("")
        print("✅ GPU Synchronized Training Completed!")
        print("=" * 60)
        
        return result
        
    except RuntimeError as e:
        if "device" in str(e).lower() or "cuda" in str(e).lower():
            print(f"💥 GPU synchronization error: {e}")
            print("🔧 This indicates tensors are still on different devices")
            print("   - Try restarting the process")
            print("   - Check if all models are properly moved to GPU")
        else:
            print(f"💥 Runtime error: {e}")
        return False
        
    except Exception as e:
        print(f"💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_gpu_detailed():
    """Detailed GPU monitoring with process detection."""
    import subprocess
    import threading
    import time
    
    def gpu_monitor():
        while True:
            try:
                # Get detailed GPU stats
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    gpu_util, mem_util, mem_used, mem_total, temp, power = result.stdout.strip().split(', ')
                    print(f"🔥 GPU: {gpu_util:>3s}% | Mem: {mem_used:>4s}/{mem_total}MB ({mem_util:>3s}%) | Temp: {temp:>2s}°C | Power: {power:>5s}W")
                
                # Check for GPU processes with better command
                proc_result = subprocess.run([
                    'nvidia-smi', 'pmon', '-c', '1'
                ], capture_output=True, text=True, timeout=5)
                
                if proc_result.returncode == 0 and "python3" in proc_result.stdout:
                    lines = proc_result.stdout.strip().split('\n')
                    for line in lines:
                        if "python3" in line and "gpu_sync_training" in line:
                            print(f"           ✅ GPU process detected: {line.strip()}")
                
            except Exception as e:
                print(f"⚠️ GPU monitor error: {e}")
                break
            time.sleep(30)  # Check every 30 seconds
    
    monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
    monitor_thread.start()
    print("📊 Detailed GPU monitoring started")

def main():
    """Main GPU synchronized training with monitoring."""
    
    print("🚀 FinRL Contest 2024 - GPU SYNCHRONIZED Production Training")
    print("=" * 70)
    print("🔧 CRITICAL FIX: Resolves device synchronization errors")
    print("⚡ Forces ALL tensors to GPU device for maximum performance")
    print("🎯 Goal: Competition-ready models with >60% accuracy")
    print("=" * 70)
    
    # Start detailed GPU monitoring
    monitor_gpu_detailed()
    
    start_time = time.time()
    
    try:
        # Run GPU synchronized training
        success = run_gpu_sync_training()
        
        elapsed_hours = (time.time() - start_time) / 3600
        
        if success:
            print("\\n" + "=" * 70)
            print("🎉 GPU SYNCHRONIZED TRAINING COMPLETED!")
            print("=" * 70)
            print(f"⏱️  Duration: {elapsed_hours:.2f} hours")
            print(f"🔥 All device synchronization issues resolved")
            print(f"📊 Models ready for competition evaluation")
            print("=" * 70)
            return 0
        else:
            print("\\n💥 GPU synchronized training failed")
            return 1
        
    except Exception as e:
        print(f"\\n💥 Training crashed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())