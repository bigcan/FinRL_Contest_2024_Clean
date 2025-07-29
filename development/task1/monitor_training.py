#!/usr/bin/env python3
"""
Real-time Multi-Episode Training Monitor
Tracks progress without interrupting the training process
"""

import time
import os
import subprocess
import psutil
from datetime import datetime

def get_process_info(pid):
    """Get process information"""
    try:
        process = psutil.Process(pid)
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'status': process.status(),
            'runtime': time.time() - process.create_time()
        }
    except psutil.NoSuchProcess:
        return None

def get_gpu_info():
    """Get GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, gpu_mem = result.stdout.strip().split(', ')
            return {'utilization': int(gpu_util), 'memory_mb': int(gpu_mem)}
    except Exception:
        pass
    return {'utilization': 0, 'memory_mb': 0}

def monitor_training():
    """Monitor the training process"""
    
    # Find training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'task1_ensemble.py' in ' '.join(proc.info['cmdline']):
                training_pid = proc.info['pid']
                break
        except (psutil.NoSuchProcess, TypeError):
            continue
    
    if not training_pid:
        print("❌ No training process found")
        return
    
    print(f"🔍 Monitoring Multi-Episode Training (PID: {training_pid})")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Get process info
        proc_info = get_process_info(training_pid)
        if not proc_info:
            print(f"\n❌ [{current_time}] Training process ended")
            break
            
        # Get GPU info
        gpu_info = get_gpu_info()
        
        # Calculate runtime
        runtime_min = int(proc_info['runtime'] // 60)
        runtime_sec = int(proc_info['runtime'] % 60)
        
        # Display status
        print(f"\r🚀 [{current_time}] Runtime: {runtime_min:02d}:{runtime_sec:02d} | "
              f"CPU: {proc_info['cpu_percent']:5.1f}% | "
              f"RAM: {proc_info['memory_mb']:6.0f}MB | "
              f"GPU: {gpu_info['utilization']:2d}% ({gpu_info['memory_mb']:4d}MB)", end="")
        
        # Check for completion indicators
        if os.path.exists('ensemble_teamname'):
            print(f"\n✅ [{current_time}] Training completed! Models saved.")
            break
            
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")