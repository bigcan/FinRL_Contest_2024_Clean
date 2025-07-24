"""
Monitor Phase 3 Training Progress
Real-time monitoring of ensemble training progress
"""

import os
import time
import subprocess
import sys

def monitor_training():
    """Monitor the training progress"""
    log_file = "../../../logs/phase3_ensemble_training.log"
    
    print("🔍 Phase 3 Training Monitor")
    print("=" * 50)
    
    # Check if training is running
    try:
        result = subprocess.run(["pgrep", "-f", "task1_ensemble_optimized.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pid = result.stdout.strip()
            print(f"✅ Training process found: PID {pid}")
        else:
            print("⚠️  No training process found")
    except:
        print("⚠️  Could not check process status")
    
    # Monitor log file
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📊 Monitoring log file: {log_file}")
    print("=" * 50)
    
    # Show recent progress
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        if len(lines) == 0:
            print("📝 Log file is empty - training may be starting...")
            return
            
        print(f"📈 Training Progress ({len(lines)} log lines):")
        print("-" * 30)
        
        # Show last 20 lines
        recent_lines = lines[-20:] if len(lines) > 20 else lines
        for line in recent_lines:
            line = line.strip()
            if line:
                # Highlight important information
                if "Agent" in line or "Training" in line:
                    print(f"🤖 {line}")
                elif "Step" in line or "Reward" in line:
                    print(f"📊 {line}")
                elif "✅" in line or "completed" in line:
                    print(f"🎉 {line}")
                elif "ERROR" in line or "Failed" in line:
                    print(f"❌ {line}")
                else:
                    print(f"   {line}")
        
        print("-" * 30)
        print(f"📝 Total log lines: {len(lines)}")
        
        # Try to extract key metrics
        try:
            agent_mentions = [line for line in lines if "Agent" in line and "Training" in line]
            if agent_mentions:
                print(f"🤖 Agents processed: {len(agent_mentions)}")
                
            step_mentions = [line for line in lines if "Step" in line and "/" in line]
            if step_mentions:
                last_step = step_mentions[-1]
                print(f"📊 Latest step: {last_step.strip()}")
                
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error reading log file: {e}")

def continuous_monitor():
    """Continuously monitor training"""
    print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            monitor_training()
            print(f"\n⏰ Last updated: {time.strftime('%H:%M:%S')}")
            print("🔄 Refreshing in 10 seconds... (Ctrl+C to stop)")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        continuous_monitor()
    else:
        monitor_training()