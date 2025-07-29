#!/usr/bin/env python3
"""
Robust Training Monitor - Real-time monitoring of the training process
"""

import os
import time
import psutil
from datetime import datetime
from pathlib import Path

def monitor_training():
    """Monitor the robust training process."""
    print("🔍 Robust Training Monitor")
    print("=" * 50)
    
    # Find the training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'robust_production_training.py' in ' '.join(proc.info['cmdline'] or []):
                training_pid = proc.info['pid']
                print(f"✅ Found training process: PID {training_pid}")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not training_pid:
        print("❌ No robust training process found")
        return
    
    start_time = time.time()
    
    try:
        training_proc = psutil.Process(training_pid)
        
        while training_proc.is_running():
            current_time = time.time()
            elapsed_hours = (current_time - start_time) / 3600
            
            # Get process stats
            cpu_percent = training_proc.cpu_percent()
            memory_info = training_proc.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            print(f"\n📊 Training Status ({datetime.now().strftime('%H:%M:%S')})")
            print(f"⏱️  Runtime: {elapsed_hours:.2f} hours")
            print(f"🖥️  CPU: {cpu_percent:.1f}%")
            print(f"💾 Memory: {memory_mb:.1f} MB")
            
            # Check for training directories
            src_path = Path("src")
            if src_path.exists():
                # Look for new TradeSimulator directories
                trade_dirs = list(src_path.glob("TradeSimulator*"))
                robust_dirs = list(src_path.glob("robust_training_*")) 
                
                print(f"📁 TradeSimulator dirs: {len(trade_dirs)}")
                print(f"📁 Robust training dirs: {len(robust_dirs)}")
                
                # Check for recent model files
                recent_models = []
                for pattern in ["*.pth", "*.pkl"]:
                    models = list(src_path.rglob(pattern))
                    recent_models.extend([
                        m for m in models 
                        if (time.time() - m.stat().st_mtime) < 300  # Last 5 minutes
                    ])
                
                if recent_models:
                    print(f"🆕 Recent models: {len(recent_models)}")
                    for model in recent_models[-3:]:  # Show last 3
                        print(f"   📄 {model}")
            
            print("-" * 50)
            time.sleep(30)  # Check every 30 seconds
            
    except psutil.NoSuchProcess:
        elapsed_hours = (time.time() - start_time) / 3600
        print(f"\n✅ Training process completed after {elapsed_hours:.2f} hours")
    except KeyboardInterrupt:
        print(f"\n⚠️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    try:
        monitor_training()
    except Exception as e:
        print(f"💥 Monitor failed: {e}")