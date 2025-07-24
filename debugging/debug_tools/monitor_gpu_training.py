"""
Monitor GPU Training Progress

Real-time monitoring of enhanced features training on GPU.
"""

import time
import subprocess
import os

def get_gpu_info():
    """Get current GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, temp = result.stdout.strip().split(', ')
            return f"GPU: {gpu_util}% | Mem: {mem_used}MB | Temp: {temp}°C"
    except:
        pass
    return "GPU: N/A"

def check_training_process():
    """Check if training process is running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'train_enhanced_features_gpu.py' in line and 'grep' not in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                return f"PID: {pid} | CPU: {cpu}% | RAM: {mem}%"
    except:
        pass
    return "Process: Not found"

def check_training_files():
    """Check for new training output files"""
    files = []
    for item in os.listdir('.'):
        if 'ensemble_enhanced' in item or ('TradeSimulator-v0' in item and os.path.getmtime(item) > time.time() - 300):
            files.append(item)
    return files

def main():
    """Monitor training progress"""
    print("=" * 60)
    print("GPU TRAINING MONITOR")
    print("=" * 60)
    
    print("Monitoring enhanced features training...")
    print("Press Ctrl+C to stop monitoring\n")
    
    start_time = time.time()
    last_files = set()
    
    try:
        while True:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            
            # Get status
            gpu_info = get_gpu_info()
            process_info = check_training_process()
            training_files = check_training_files()
            
            # Clear screen (simple version)
            print(f"\r[{mins:02d}:{secs:02d}] {gpu_info} | {process_info}", end="", flush=True)
            
            # Check for new files
            current_files = set(training_files)
            new_files = current_files - last_files
            if new_files:
                print(f"\n✅ New training files: {list(new_files)}")
                last_files = current_files
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("MONITORING STOPPED")
        print("="*60)
        
        # Final status
        print(f"Training time: {elapsed//60}m {elapsed%60}s")
        print(f"GPU status: {get_gpu_info()}")
        print(f"Process: {check_training_process()}")
        
        if training_files:
            print(f"Training files: {training_files}")
            
        print("\nTraining continues in background...")

if __name__ == "__main__":
    main()