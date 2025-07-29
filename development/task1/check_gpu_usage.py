#!/usr/bin/env python3
"""
GPU Usage Verification - Check if training is actually using GPU
"""

import subprocess
import time
import sys

def get_gpu_processes():
    """Get detailed GPU process information."""
    try:
        # Get GPU process list
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')
        return []
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return []

def get_gpu_utilization():
    """Get GPU utilization stats."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def check_training_process():
    """Check if our training process exists."""
    try:
        result = subprocess.run([
            'ps', 'aux'
        ], capture_output=True, text=True, timeout=10)
        
        training_processes = []
        for line in result.stdout.split('\n'):
            if 'gpu_production_training.py' in line and 'grep' not in line:
                training_processes.append(line.strip())
        
        return training_processes
    except Exception as e:
        print(f"Error checking processes: {e}")
        return []

def main():
    """Main GPU verification function."""
    print("🔍 GPU Usage Verification")
    print("=" * 50)
    
    # Check training process
    training_procs = check_training_process()
    if training_procs:
        print("✅ Training process found:")
        for proc in training_procs:
            print(f"   {proc}")
    else:
        print("❌ No training process found")
        return
    
    print("\n📊 GPU Status Check:")
    print("-" * 30)
    
    # Monitor for 60 seconds
    for i in range(12):  # 12 * 5 seconds = 1 minute
        # Get GPU utilization
        gpu_stats = get_gpu_utilization()
        if gpu_stats:
            gpu_util, mem_util, mem_used, mem_total, power = gpu_stats.split(', ')
            print(f"⏰ Check {i+1:2d}/12: GPU {gpu_util:>3s}% | Mem {mem_util:>3s}% | Used {mem_used:>4s}/{mem_total}MB | Power {power}W")
        
        # Check for GPU processes
        gpu_procs = get_gpu_processes()
        if gpu_procs:
            print(f"           🔥 GPU processes: {len(gpu_procs)}")
            for proc in gpu_procs:
                if proc.strip():
                    pid, name, memory = proc.split(', ')
                    print(f"              PID {pid}: {name} ({memory}MB)")
        else:
            print(f"           ⚠️  No GPU processes detected")
        
        if i < 11:  # Don't sleep on last iteration
            time.sleep(5)
    
    print("\n📋 Summary:")
    final_gpu_stats = get_gpu_utilization()
    final_gpu_procs = get_gpu_processes()
    
    if final_gpu_procs:
        print("✅ TRAINING IS USING GPU!")
        print(f"   Active GPU processes: {len(final_gpu_procs)}")
    else:
        print("❌ TRAINING IS NOT USING GPU!")
        print("   This explains why training is slow.")
    
    if final_gpu_stats:
        gpu_util, mem_util, mem_used, mem_total, power = final_gpu_stats.split(', ')
        if int(gpu_util) > 10:
            print(f"✅ GPU utilization: {gpu_util}% (actively training)")
        else:
            print(f"⚠️  GPU utilization: {gpu_util}% (likely not training)")

if __name__ == "__main__":
    main()