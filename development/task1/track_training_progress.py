#!/usr/bin/env python3
"""
Continuous Training Progress Tracker
Monitors GPU training progress with detailed metrics
"""

import subprocess
import time
import os
from datetime import datetime
from pathlib import Path

def get_training_process():
    """Get training process details."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split('\n'):
            if 'gpu_sync_training.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    return {
                        'pid': parts[1],
                        'cpu': parts[2], 
                        'memory': parts[3],
                        'vsz': parts[4],
                        'rss': parts[5],
                        'time': parts[9]
                    }
        return None
    except Exception as e:
        print(f"Error getting process: {e}")
        return None

def get_gpu_stats():
    """Get detailed GPU statistics."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_util, mem_util, mem_used, mem_total, temp, power = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(gpu_util),
                'mem_util': int(mem_util), 
                'mem_used': int(mem_used),
                'mem_total': int(mem_total),
                'temperature': int(float(temp)),
                'power': float(power)
            }
        return None
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def check_training_outputs():
    """Check for training output files."""
    src_path = Path("src")
    outputs = {
        'model_files': 0,
        'log_files': 0,
        'recent_models': [],
        'training_dirs': []
    }
    
    if src_path.exists():
        # Count model files
        outputs['model_files'] = len(list(src_path.rglob("*.pth")))
        outputs['log_files'] = len(list(src_path.rglob("*.log")))
        
        # Find recent models (last 5 minutes)
        recent_time = time.time() - 300
        for model_file in src_path.rglob("*.pth"):
            if model_file.stat().st_mtime > recent_time:
                outputs['recent_models'].append(str(model_file))
        
        # Find training directories
        for pattern in ["*training_*", "TradeSimulator*"]:
            outputs['training_dirs'].extend(list(src_path.glob(pattern)))
    
    return outputs

def track_progress():
    """Main progress tracking function."""
    print("📊 Training Progress Tracker")
    print("=" * 60)
    print("🔥 Monitoring GPU-synchronized training")
    print("⏰ Updates every 30 seconds")
    print("=" * 60)
    
    start_time = time.time()
    iteration = 0
    
    while True:
        iteration += 1
        current_time = time.time()
        elapsed_hours = (current_time - start_time) / 3600
        
        print(f"\n📈 Progress Update #{iteration} ({datetime.now().strftime('%H:%M:%S')})")
        print(f"⏱️  Runtime: {elapsed_hours:.2f} hours")
        print("-" * 50)
        
        # Check training process
        process = get_training_process()
        if process:
            print(f"✅ Training Process (PID {process['pid']}):")
            print(f"   CPU: {process['cpu']:>6s}% | Memory: {process['memory']:>5s}% | Runtime: {process['time']}")
        else:
            print("❌ Training process not found - may have completed or crashed")
            break
        
        # Check GPU stats
        gpu = get_gpu_stats()
        if gpu:
            print(f"🔥 GPU Status:")
            print(f"   Utilization: {gpu['gpu_util']:>3d}% | Memory: {gpu['mem_used']:>4d}/{gpu['mem_total']}MB ({gpu['mem_util']:>3d}%)")
            print(f"   Temperature: {gpu['temperature']:>2d}°C | Power Draw: {gpu['power']:>5.1f}W")
            
            # Performance indicators
            if gpu['gpu_util'] > 50:
                print("   🚀 High GPU utilization - training actively running")
            elif gpu['gpu_util'] > 10:
                print("   ⚡ Moderate GPU usage - processing data")
            else:
                print("   ⏳ Low GPU usage - may be in I/O or initialization phase")
        else:
            print("⚠️  Could not get GPU stats")
        
        # Check training outputs
        outputs = check_training_outputs()
        print(f"📁 Training Outputs:")
        print(f"   Model files: {outputs['model_files']} | Log files: {outputs['log_files']}")
        print(f"   Training dirs: {len(outputs['training_dirs'])}")
        
        if outputs['recent_models']:
            print(f"   🆕 Recent models ({len(outputs['recent_models'])}):")
            for model in outputs['recent_models'][-3:]:  # Show last 3
                print(f"      📄 {model}")
        
        # Check log file for progress
        if os.path.exists("gpu_sync_training.log"):
            try:
                with open("gpu_sync_training.log", 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line and not last_line.startswith('/'):  # Ignore warning paths
                            print(f"📝 Latest log: {last_line[:80]}...")
            except Exception as e:
                print(f"⚠️  Could not read log: {e}")
        
        print("-" * 50)
        
        # Stop conditions
        if elapsed_hours > 8:
            print("⏰ Training has been running for 8+ hours - stopping monitor")
            break
        
        time.sleep(30)  # Update every 30 seconds
    
    print(f"\n🏁 Training monitoring completed after {elapsed_hours:.2f} hours")

if __name__ == "__main__":
    try:
        track_progress()
    except KeyboardInterrupt:
        print(f"\n⚠️  Training monitoring stopped by user")
    except Exception as e:
        print(f"💥 Monitor error: {e}")