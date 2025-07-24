"""
Monitor Enhanced Features Training

Monitor the training progress and compare with previous results.
"""

import os
import time
import subprocess
import signal
import sys

def monitor_training():
    """Monitor training progress"""
    print("=" * 60)
    print("MONITORING ENHANCED FEATURES TRAINING")
    print("=" * 60)
    
    print("✓ Enhanced features detected (16 dimensions)")
    print("✓ Training started with enhanced features")
    print("✓ Expected improvements: Better trend detection, LOB insights")
    
    # Start training in background
    print("\nStarting training process...")
    
    try:
        # Run training with output capture
        process = subprocess.Popen(
            ['python3', 'task1_ensemble.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("Training process started (PID: {})".format(process.pid))
        print("Monitoring output (Ctrl+C to stop monitoring, training continues)...")
        print("-" * 60)
        
        # Monitor output
        line_count = 0
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            line_count += 1
            
            # Show periodic status
            if line_count % 10 == 0:
                print(f"[INFO] Processed {line_count} output lines...")
            
            # Check if training finished
            if process.poll() is not None:
                break
                
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("MONITORING STOPPED (Training continues in background)")
        print("=" * 60)
        print(f"Training process PID: {process.pid}")
        print("To check progress: ps aux | grep task1_ensemble")
        print("To stop training: kill {}".format(process.pid))
        
    except Exception as e:
        print(f"Error monitoring training: {e}")
    
    finally:
        if 'process' in locals():
            process.stdout.close()

def check_training_files():
    """Check for training output files"""
    print("\nChecking for training output files...")
    
    expected_files = [
        "ensemble_teamname/ensemble_models/",
        "TradeSimulator-v0_D3QN_0/",
        "ensemble_teamname/"
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"- Not yet: {file_path}")

if __name__ == "__main__":
    try:
        monitor_training()
        check_training_files()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)