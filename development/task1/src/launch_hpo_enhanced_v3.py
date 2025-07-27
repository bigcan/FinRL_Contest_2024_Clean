#!/usr/bin/env python3
"""
Launch HPO specifically for enhanced v3 features
This ensures hyperparameters are optimized for the new feature distribution
"""

import os
import sys
import json
import subprocess
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run HPO for Enhanced V3 Features')
    parser.add_argument('--n-trials', type=int, default=200, help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default=f'finrl_enhanced_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}', 
                       help='Optuna study name')
    parser.add_argument('--metric', type=str, default='sharpe_ratio', 
                       choices=['sharpe_ratio', 'total_return', 'romad'], help='Optimization metric')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--timeout', type=int, default=172800, help='Timeout in seconds (default: 48 hours)')
    parser.add_argument('--db-name', type=str, default='task1_enhanced_v3_hpo.db', help='Database name for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FinRL Contest 2024 - HPO for Enhanced V3 Features")
    print("="*80)
    print(f"Study Name: {args.study_name}")
    print(f"Database: {args.db_name}")
    print(f"Optimization Metric: {args.metric}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Timeout: {args.timeout/3600:.1f} hours")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🔍 Feature Set: Enhanced V3 (41 dimensions)")
    print("   - Market microstructure features")
    print("   - Order flow imbalance")
    print("   - Volume-weighted metrics")
    print("   - Technical indicators")
    print("="*80)
    
    # Verify enhanced features are being used
    print("\nVerifying enhanced v3 features...")
    verify_cmd = [
        sys.executable, '-c',
        "from trade_simulator import TradeSimulator; "
        "env = TradeSimulator(num_sims=1); "
        "print(f'✅ Confirmed: Using {env.state_dim} dimensional features')"
    ]
    subprocess.run(verify_cmd, check=True)
    
    # Create HPO directories
    os.makedirs("archived_experiments/hpo_databases", exist_ok=True)
    os.makedirs("hpo_experiments/enhanced_v3_results", exist_ok=True)
    
    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Prepare HPO command
    cmd = [
        sys.executable,
        'task1_hpo.py',
        '--n-trials', str(args.n_trials),
        '--study-name', args.study_name,
        '--metric', args.metric,
        '--timeout', str(args.timeout),
        '--n-jobs', '1',  # Single job for GPU training
        '--sampler', 'tpe',  # Tree-structured Parzen Estimator
        '--pruner', 'median',  # Median pruning for early stopping
        '--save-dir', 'hpo_experiments/enhanced_v3_results',
        '--gpu-id', str(args.gpu_id),
        '--db-name', args.db_name
    ]
    
    # Log file specific to enhanced v3
    log_file = f"hpo_enhanced_v3_{args.metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create study info file
    study_info = {
        "study_name": args.study_name,
        "feature_set": "enhanced_v3",
        "feature_dimensions": 41,
        "metric": args.metric,
        "n_trials": args.n_trials,
        "start_time": datetime.now().isoformat(),
        "gpu_id": args.gpu_id,
        "database": args.db_name,
        "log_file": log_file
    }
    
    with open("hpo_experiments/enhanced_v3_results/study_info.json", 'w') as f:
        json.dump(study_info, f, indent=2)
    
    print(f"\nLaunching HPO process...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print("\n📊 You can monitor progress with:")
    print(f"  1. Real-time monitor: python3 monitor_hpo_realtime.py --db {args.db_name}")
    print(f"  2. Status check: python3 check_hpo_status.py")
    print(f"  3. Log tail: tail -f {log_file}")
    print("-"*80)
    
    # Start the HPO process
    with open(log_file, 'w') as log:
        log.write(f"HPO for Enhanced V3 Features Started at {datetime.now()}\n")
        log.write(f"Study Info: {json.dumps(study_info, indent=2)}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write("="*80 + "\n")
        log.flush()
        
        try:
            # Run HPO
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1  # Line buffered
            )
            
            print(f"\n🚀 HPO process started with PID: {process.pid}")
            print("Process is running in the background...")
            print("\n⚠️  IMPORTANT: This will take 24-48 hours to complete!")
            print("The process will continue even if you close this terminal.")
            
            # Option to wait or detach
            user_input = input("\nDo you want to wait for completion? (y/n): ").lower()
            
            if user_input == 'y':
                print("\nWaiting for HPO to complete... (Press Ctrl+C to detach)")
                try:
                    return_code = process.wait()
                    if return_code == 0:
                        print("\n✅ HPO completed successfully!")
                    else:
                        print(f"\n❌ HPO failed with return code: {return_code}")
                except KeyboardInterrupt:
                    print("\n\n📌 Detached from HPO process. It continues running in background.")
                    print(f"Monitor with: python3 monitor_hpo_realtime.py --db {args.db_name}")
            else:
                print("\n📌 HPO is running in background.")
                print(f"Monitor progress with: python3 monitor_hpo_realtime.py --db {args.db_name}")
                
        except Exception as e:
            print(f"\n❌ Error launching HPO: {str(e)}")
            return 1
    
    print(f"\n📝 Next steps after HPO completion:")
    print(f"1. Check final results: python3 check_hpo_status.py")
    print(f"2. Best parameters will be in: hpo_experiments/enhanced_v3_results/task1_best_params.json")
    print(f"3. Launch training with: python3 launch_optimized_training.py --config <best_params_config>")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())