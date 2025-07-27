#!/usr/bin/env python3
"""
Launch ensemble training with optimized hyperparameters
"""

import os
import sys
import json
import subprocess
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Launch optimized ensemble training')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--config', type=str, default='optimized_config.json', help='Configuration file')
    parser.add_argument('--save-path', type=str, default='ensemble_optimized', help='Path to save trained models')
    parser.add_argument('--agent-list', nargs='+', default=['AgentD3QN', 'AgentDoubleDQN', 'AgentTwinD3QN'], 
                       help='List of agents to train')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("="*80)
    print("FinRL Contest 2024 - Optimized Ensemble Training")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Save Path: {args.save_path}")
    print(f"Agents: {', '.join(args.agent_list)}")
    print("\nOptimized Parameters:")
    print(f"  Network Dims: {config['net_dims']}")
    print(f"  Learning Rate: {config['learning_rate']:.2e}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Num Sims: {config['num_sims']}")
    print(f"  Step Gap: {config['step_gap']}")
    print(f"  Break Step: {config['break_step']} million")
    print("="*80)
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Save config to training directory
    config_save_path = os.path.join(args.save_path, 'training_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare command
    cmd = [
        sys.executable,
        'task1_ensemble.py',
        str(args.gpu_id),
        '--config', args.config,
        '--save-path', args.save_path,
        '--log-rules'
    ]
    
    # Log file
    log_file = os.path.join(args.save_path, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print("\nTraining in progress...")
    print("You can monitor with:")
    print(f"  tail -f {log_file}")
    print("-"*80)
    
    # Run training
    with open(log_file, 'w') as log:
        log.write(f"Optimized Ensemble Training Started at {datetime.now()}\n")
        log.write(f"Configuration: {json.dumps(config, indent=2)}\n")
        log.write("="*80 + "\n")
        log.flush()
        
        try:
            process = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
            
            print(f"\n✅ Training completed successfully!")
            print(f"Models saved in: {args.save_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error code: {e.returncode}")
            print(f"Check log file for details: {log_file}")
            return 1
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user")
            return 1
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            return 1
    
    print(f"\nTraining finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext steps:")
    print(f"1. Evaluate the trained ensemble:")
    print(f"   python3 task1_eval.py {args.gpu_id}")
    print(f"2. Check training metrics in: {log_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())