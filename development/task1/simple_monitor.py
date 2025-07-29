#!/usr/bin/env python3
"""
Simple Training Monitor - Quick status checks
"""

import subprocess
import os
import time
from datetime import datetime

def check_training_status():
    # Check if process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'task1_ensemble.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip()
            print(f"✅ Training RUNNING (PID: {pid})")
            
            # Check GPU usage
            gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
            if gpu_result.returncode == 0:
                gpu_util, gpu_mem = gpu_result.stdout.strip().split(', ')
                print(f"🎯 GPU Usage: {gpu_util}% utilization, {gpu_mem}MB memory")
            
            # Check latest log
            log_files = [f for f in os.listdir('.') if f.startswith('training_fresh_')]
            if log_files:
                latest_log = max(log_files, key=lambda x: os.path.getmtime(x))
                size_mb = os.path.getsize(latest_log) / (1024*1024)
                print(f"📊 Log: {latest_log} ({size_mb:.1f}MB)")
                
                # Get last few lines
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 5:
                        print("📝 Latest progress:")
                        for line in lines[-3:]:
                            if line.strip() and not line.startswith('/'):
                                print(f"   {line.strip()}")
            
            return True
        else:
            print("❌ Training NOT RUNNING")
            
            # Check for completion notification
            if os.path.exists('TRAINING_COMPLETE_NOTIFICATION.txt'):
                print("🎉 TRAINING COMPLETED!")
                with open('TRAINING_COMPLETE_NOTIFICATION.txt', 'r') as f:
                    print(f.read())
                return False
            
            # Check latest results
            model_dirs = ['ensemble_teamname', 'complete_production_results']
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    recent_files = []
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            if file.endswith('.pth') and 'model' in file.lower():
                                file_path = os.path.join(root, file)
                                if time.time() - os.path.getmtime(file_path) < 3600:  # Last hour
                                    recent_files.append(file_path)
                    if recent_files:
                        print(f"🔥 Recent models in {model_dir}: {len(recent_files)}")
            
            return False
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"🔍 Training Status Check - {timestamp}")
    print("=" * 50)
    
    is_running = check_training_status()
    
    if is_running:
        print("\n💡 Training is active! Use Ctrl+C to stop monitoring.")
        print("💡 Run this script again anytime to check status.")
    else:
        print("\n💡 Training appears to be complete or stopped.")
        print("💡 Check logs and model directories for results.")