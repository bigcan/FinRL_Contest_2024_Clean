#!/usr/bin/env python3
"""
Launch HPO training for FinRL Contest 2024
This script manages the hyperparameter optimization process
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run HPO for FinRL Contest 2024')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default='finrl_sharpe_v3', help='Optuna study name')
    parser.add_argument('--metric', type=str, default='sharpe_ratio', choices=['sharpe_ratio', 'total_return', 'romad'], help='Optimization metric')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--timeout', type=int, default=172800, help='Timeout in seconds (default: 48 hours)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FinRL Contest 2024 - Hyperparameter Optimization")
    print("="*80)
    print(f"Study Name: {args.study_name}")
    print(f"Optimization Metric: {args.metric}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Timeout: {args.timeout/3600:.1f} hours")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Prepare command
    cmd = [
        sys.executable,
        'task1_hpo.py',
        '--n-trials', str(args.n_trials),
        '--study-name', args.study_name,
        '--metric', args.metric,
        '--timeout', str(args.timeout),
        '--n-jobs', '1',  # Single job for GPU training
        '--sampler', 'tpe',  # Tree-structured Parzen Estimator
        '--pruner', 'median',  # Median pruning
        '--save-dir', 'hpo_experiments',
        '--gpu-id', str(args.gpu_id)
    ]
    
    # Log file
    log_file = f"hpo_{args.metric}_production.log"
    
    print(f"\nLaunching HPO process...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print("\nYou can monitor progress with:")
    print(f"  tail -f {log_file}")
    print(f"  python3 check_hpo_status.py")
    print("-"*80)
    
    # Start the HPO process
    with open(log_file, 'w') as log:
        log.write(f"HPO Started at {datetime.now()}\n")
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
            
            print(f"\nHPO process started with PID: {process.pid}")
            print("Process is running in the background...")
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                print("\n✅ HPO completed successfully!")
            else:
                print(f"\n❌ HPO failed with return code: {return_code}")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user. Terminating HPO process...")
            process.terminate()
            process.wait()
            print("HPO process terminated.")
        except Exception as e:
            print(f"\n❌ Error running HPO: {str(e)}")
            return 1
    
    print(f"\nHPO finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Check results with: python3 check_hpo_status.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())