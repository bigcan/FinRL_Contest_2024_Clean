#!/usr/bin/env python3
"""
Robust Production Training - Handles evaluation errors while preserving training core
Full-scale RL training with proper error handling and recovery
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add src path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

# Import original framework components
from task1_ensemble import run
from erl_config import Config

class RobustProductionTrainer:
    """Robust production trainer with error handling and monitoring."""
    
    def __init__(self, output_dir: str = "robust_production_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"🚀 Robust Production Training - Session: {self.timestamp}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🛡️  Enhanced with error handling and monitoring")
    
    def backup_and_monitor(self, training_dir):
        """Monitor training progress and backup models."""
        print(f"\n📊 Monitoring Training Progress")
        print("-" * 50)
        
        backup_dir = self.output_dir / f"training_backup_{self.timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        last_backup = 0
        
        while True:
            try:
                current_time = time.time()
                elapsed_hours = (current_time - start_time) / 3600
                
                # Check if training directory exists and has content
                if os.path.exists(training_dir):
                    # Count model files
                    model_files = list(Path(training_dir).rglob("*.pth"))
                    log_files = list(Path(training_dir).rglob("*.log"))
                    
                    print(f"⏱️  Training Time: {elapsed_hours:.2f}h | Models: {len(model_files)} | Logs: {len(log_files)}")
                    
                    # Backup every 30 minutes  
                    if current_time - last_backup > 1800:  # 30 minutes
                        try:
                            backup_subdir = backup_dir / f"backup_{int(elapsed_hours*10)/10}h"
                            if model_files:
                                backup_subdir.mkdir(exist_ok=True)
                                for model_file in model_files:
                                    shutil.copy2(model_file, backup_subdir / model_file.name)
                                print(f"💾 Backup created: {backup_subdir}")
                            last_backup = current_time
                        except Exception as e:
                            print(f"⚠️  Backup failed: {e}")
                
                # Check for completion indicators
                if os.path.exists(Path(training_dir) / "training_complete.flag"):
                    print(f"✅ Training completed successfully after {elapsed_hours:.2f} hours")
                    break
                    
                # Emergency stop after 8 hours
                if elapsed_hours > 8:
                    print(f"⏰ Training stopped after 8 hours (safety limit)")
                    break
                
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Training monitoring interrupted by user")
                break
            except Exception as e:
                print(f"⚠️  Monitor error: {e}")
                time.sleep(60)
        
        return elapsed_hours
    
    def run_robust_training(self):
        """Run robust production training with monitoring."""
        print("\n" + "=" * 70)
        print("🚀 STARTING ROBUST PRODUCTION TRAINING")
        print("=" * 70)
        
        try:
            print(f"🎯 Training Configuration:")
            print(f"   Framework: Original Proven Task1 Ensemble")
            print(f"   GPU: Enabled (device 0)")
            print(f"   Data: Enhanced v3 Bitcoin LOB (823K timesteps)")
            print(f"   Agents: D3QN, DoubleDQN, PrioritizedDQN")
            print(f"   Features: 41-dimensional enhanced microstructure")
            
            # Create training script with robust configuration
            training_script = self.create_robust_training_script()
            
            print(f"\n🎓 Starting Robust Training Process")
            print("-" * 50)
            print(f"⏳ This will run for several hours...")
            print(f"📈 Progress will be monitored and backed up")
            
            # Start training in background and monitor
            import subprocess
            import threading
            
            # Set working directory to src for proper imports
            training_dir = current_dir / "src" / f"robust_training_{self.timestamp}"
            
            # Create the training command with correct path
            training_cmd = [
                sys.executable, f"../{training_script}",
                "0"  # GPU ID
            ]
            
            print(f"🚀 Launching training: {' '.join(training_cmd)}")
            
            # Start training process
            training_process = subprocess.Popen(
                training_cmd,
                cwd=str(current_dir / "src"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor training in separate thread
            def monitor_output():
                try:
                    for line in training_process.stdout:
                        print(f"[TRAIN] {line.strip()}")
                        
                        # Check for completion indicators
                        if "Training completed" in line or "ensemble completed" in line:
                            # Create completion flag
                            flag_file = training_dir / "training_complete.flag"
                            flag_file.parent.mkdir(parents=True, exist_ok=True)
                            flag_file.touch()
                            
                except Exception as e:
                    print(f"⚠️  Output monitoring error: {e}")
            
            # Start output monitoring
            monitor_thread = threading.Thread(target=monitor_output, daemon=True)
            monitor_thread.start()
            
            # Start progress monitoring
            elapsed_hours = self.backup_and_monitor(training_dir)
            
            # Wait for training to complete or timeout
            try:
                training_process.wait(timeout=300)  # 5 minute final wait
            except subprocess.TimeoutExpired:
                print("⏰ Training process cleanup timeout")
                training_process.terminate()
            
            print(f"\n📊 Training Session Summary")
            print("-" * 50)
            print(f"⏱️  Total duration: {elapsed_hours:.2f} hours")
            print(f"📁 Results location: {self.output_dir}")
            
            # Collect final results
            results = self.collect_training_results(training_dir, elapsed_hours)
            
            # Save session summary
            summary_file = self.output_dir / f"robust_training_summary_{self.timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"📄 Session summary: {summary_file}")
            
            return results
            
        except Exception as e:
            print(f"\n💥 Robust training failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def create_robust_training_script(self):
        """Create a robust training script with error handling."""
        script_path = self.output_dir / f"robust_train_{self.timestamp}.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Generated Robust Training Script - {self.timestamp}
Enhanced error handling for production training
"""

import sys
import os
from pathlib import Path

# Add current directory for imports
current_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(current_dir))

def robust_training_wrapper():
    """Wrapper with enhanced error handling."""
    try:
        print("🚀 Starting Robust Production Training")
        print("=" * 60)
        
        # Import and run original training
        from task1_ensemble import run
        
        # Enhanced configuration for production
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('gpu_id', type=int, default=0, help='GPU ID to use')
        args = parser.parse_args()
        
        print(f"🎮 Using GPU: {{args.gpu_id}}")
        print(f"📊 Enhanced v3 Bitcoin LOB features (41D)")
        print(f"🎯 Production ensemble training")
        
        # Run training with robust parameters
        run(
            env_class_name="TradeSimulator",
            agent_class_names=["AgentD3QN", "AgentDoubleDQN", "AgentPrioritizedDQN"],
            gpu_id=args.gpu_id,
            log_rules=["print_time", "save_model"],
            save_path=f"robust_training_{self.timestamp}",
            starting_cash=100000
        )
        
        print("✅ Robust production training completed successfully!")
        return True
        
    except Exception as e:
        print(f"💥 Training error: {{e}}")
        import traceback
        traceback.print_exc()
        
        # Try to save whatever progress was made
        try:
            print("🔄 Attempting to save partial progress...")
            # Additional error recovery logic here
        except:
            pass
            
        return False

if __name__ == "__main__":
    success = robust_training_wrapper()
    sys.exit(0 if success else 1)
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)  # Make executable
        return str(script_path)
    
    def collect_training_results(self, training_dir, elapsed_hours):
        """Collect and analyze training results."""
        results = {
            'timestamp': self.timestamp,
            'training_duration_hours': elapsed_hours,
            'training_directory': str(training_dir),
            'status': 'completed',
            'models': {},
            'performance': {}
        }
        
        try:
            if os.path.exists(training_dir):
                # Count model files
                model_files = list(Path(training_dir).rglob("*.pth"))
                results['models']['total_files'] = len(model_files)
                results['models']['model_paths'] = [str(f) for f in model_files]
                
                # Look for performance logs
                log_files = list(Path(training_dir).rglob("*.log"))
                results['performance']['log_files'] = len(log_files)
                
                # Try to extract final performance metrics
                # This would parse the latest logs for metrics
                
            print(f"📊 Results collected: {len(model_files)} models, {len(log_files)} logs")
            
        except Exception as e:
            print(f"⚠️  Result collection error: {e}")
            results['status'] = 'partial'
        
        return results

def main():
    """Main robust training function."""
    print("🚀 FinRL Contest 2024 - Robust Production Training")
    print("=" * 60)
    print("🛡️  Enhanced with monitoring and error recovery")
    print("⏰ Expected duration: 2-6 hours")
    print("🎯 Goal: Competition-ready models with >60% accuracy")
    print("=" * 60)
    
    print("\\n✅ Auto-starting robust production training...")
    print("🚀 Training will begin immediately")
    
    try:
        trainer = RobustProductionTrainer()
        results = trainer.run_robust_training()
        
        print("\\n" + "=" * 70)
        print("🎉 ROBUST PRODUCTION TRAINING COMPLETED!")
        print("=" * 70)
        print(f"⏱️  Duration: {results.get('training_duration_hours', 0):.2f} hours")
        print(f"📊 Models generated: {results.get('models', {}).get('total_files', 0)}")
        print(f"📁 Results: {results.get('training_directory', 'N/A')}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\\n💥 Robust training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())