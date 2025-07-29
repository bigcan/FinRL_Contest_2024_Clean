#!/usr/bin/env python3
"""
Training Completion Monitor & Notification System
Monitors the training process and sends notifications when complete.
"""

import time
import subprocess
import os
import sys
from datetime import datetime

def check_process_running(pid):
    """Check if process is still running"""
    try:
        # Check if process exists
        result = subprocess.run(['ps', '-p', str(pid)], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def get_training_pid():
    """Get the PID of the training process"""
    try:
        result = subprocess.run(['pgrep', '-f', 'task1_ensemble.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split('\n')[0])
        return None
    except:
        return None

def send_notification(title, message):
    """Send system notification (multiple methods for reliability)"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Method 1: Print to console with visual emphasis
    print("\n" + "="*60)
    print(f"🎉 TRAINING COMPLETE! {timestamp}")
    print("="*60)
    print(f"📊 {title}")
    print(f"💬 {message}")
    print("="*60)
    
    # Method 2: Write notification file
    with open('TRAINING_COMPLETE_NOTIFICATION.txt', 'w') as f:
        f.write(f"TRAINING COMPLETED AT {timestamp}\n")
        f.write(f"Title: {title}\n")
        f.write(f"Message: {message}\n")
        f.write("="*50 + "\n")
    
    # Method 3: Try system notification (if available)
    try:
        # Try notify-send (Linux)
        subprocess.run(['notify-send', title, message], 
                      capture_output=True, timeout=5)
    except:
        pass
    
    # Method 4: Audio beep (if available)
    try:
        subprocess.run(['echo', '-e', '\\a'], capture_output=True, timeout=2)
    except:
        pass

def analyze_training_results():
    """Quick analysis of training results"""
    results = {
        'log_file': None,
        'models_created': [],
        'completion_status': 'unknown'
    }
    
    # Find the latest training log
    try:
        log_files = [f for f in os.listdir('.') if f.startswith('training_fresh_')]
        if log_files:
            latest_log = max(log_files, key=lambda x: os.path.getmtime(x))
            results['log_file'] = latest_log
            
            # Check log for completion indicators
            with open(latest_log, 'r') as f:
                log_content = f.read()
                
            if 'Traceback' in log_content or 'Error' in log_content:
                results['completion_status'] = 'failed'
            elif 'Training complete' in log_content or 'Saved' in log_content:
                results['completion_status'] = 'success'
    except:
        pass
    
    # Check for new model files
    try:
        model_dirs = ['complete_production_results', 'src']
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith('.pth') and 'model' in file.lower():
                            file_path = os.path.join(root, file)
                            # Check if file was modified recently (last 10 minutes)
                            if time.time() - os.path.getmtime(file_path) < 600:
                                results['models_created'].append(file_path)
    except:
        pass
    
    return results

def main():
    print("🔍 Training Monitor Started")
    print("Monitoring for training completion...")
    
    # Get the training process PID
    training_pid = get_training_pid()
    if not training_pid:
        print("❌ No training process found!")
        print("Make sure training is running with: python3 src/task1_ensemble.py")
        return
    
    print(f"📍 Monitoring training process PID: {training_pid}")
    
    # Monitor the process
    start_time = time.time()
    check_interval = 10  # Check every 10 seconds
    
    while True:
        if not check_process_running(training_pid):
            # Process has finished!
            end_time = time.time()
            duration = end_time - start_time
            
            # Analyze results
            results = analyze_training_results()
            
            # Send notification
            title = "FinRL Training Complete!"
            message = f"Duration: {duration/60:.1f} minutes\n"
            message += f"Status: {results['completion_status']}\n"
            message += f"Models: {len(results['models_created'])} created\n"
            message += f"Log: {results['log_file']}"
            
            send_notification(title, message)
            
            # Print detailed results
            print(f"\n📊 TRAINING RESULTS:")
            print(f"   Duration: {duration/60:.1f} minutes")
            print(f"   Status: {results['completion_status']}")
            print(f"   Log file: {results['log_file']}")
            print(f"   Models created: {len(results['models_created'])}")
            for model in results['models_created']:
                print(f"     - {model}")
            
            break
        
        # Still running - show status
        elapsed = time.time() - start_time
        print(f"⏱️  Training running... {elapsed/60:.1f} minutes elapsed")
        time.sleep(check_interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        print("But training may still be running in background!")