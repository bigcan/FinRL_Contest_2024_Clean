#!/usr/bin/env python3
"""
HPO Status Checker - Monitor hyperparameter optimization progress
"""
import os
import sqlite3
import json
from datetime import datetime
import optuna
from hpo_config import create_sqlite_storage

def check_hpo_status():
    """Check the current status of HPO optimization"""
    
    # Database path
    db_path = "task1_production_hpo.db"
    
    if not os.path.exists(db_path):
        print("ERROR: HPO database not found. Optimization may not have started.")
        return
    
    try:
        # Connect to Optuna study
        storage_url = create_sqlite_storage(db_path)
        study_name = None
        
        # Find the most recent study
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT study_name FROM studies ORDER BY study_id DESC LIMIT 1")
        result = cursor.fetchone()
        if result:
            study_name = result[0]
        conn.close()
        
        if not study_name:
            print("ERROR: No studies found in database")
            return
            
        # Load study
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        # Get study statistics
        n_trials = len(study.trials)
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        # Check if study is finished
        is_finished = n_running == 0 and n_complete > 0
        
        print("="*80)
        print("HPO OPTIMIZATION STATUS")
        print("="*80)
        print(f"Study Name: {study_name}")
        print(f"Total Trials: {n_trials}")
        print(f"Completed: {n_complete}")
        print(f"Running: {n_running}")
        print(f"Failed: {n_failed}")
        print(f"Last Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if is_finished:
            print("STATUS: OPTIMIZATION COMPLETED!")
            
            if study.best_trial:
                print("\nBEST RESULTS:")
                print(f"Best Value: {study.best_value:.6f}")
                print(f"Best Trial: #{study.best_trial.number}")
                print("\nBest Parameters:")
                for param, value in study.best_params.items():
                    print(f"  {param}: {value}")
            else:
                print("WARNING: No successful trials found")
                
        else:
            print("STATUS: OPTIMIZATION IN PROGRESS...")
            
            # Show recent trials
            if study.trials:
                recent_trials = sorted(study.trials, key=lambda x: x.number)[-5:]
                print(f"\nRecent Trials (last 5):")
                for trial in recent_trials:
                    status_map = {"COMPLETE": "DONE", "RUNNING": "RUN", "FAIL": "FAIL"}
                    status_str = status_map.get(trial.state.name, "UNKNOWN")
                    value_str = f"{trial.value:.4f}" if trial.value is not None else "N/A"
                    print(f"  Trial {trial.number:3d}: {status_str} Value: {value_str}")
        
        print("="*80)
        
        return is_finished
        
    except Exception as e:
        print(f"ERROR: Error checking HPO status: {str(e)}")
        return False

def check_log_progress():
    """Check progress from log files"""
    log_file = "hpo_sharpe_production.log"
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # Count trial completions
            trial_lines = [line for line in lines if "Trial" in line and "finished with value" in line]
            last_trial = None
            
            if trial_lines:
                last_line = trial_lines[-1]
                # Extract trial number
                import re
                match = re.search(r'Trial (\d+) finished', last_line)
                if match:
                    last_trial = int(match.group(1))
            
            print(f"\nLog File Analysis:")
            print(f"  Last completed trial: {last_trial if last_trial is not None else 'Unknown'}")
            print(f"  Total completed trials in log: {len(trial_lines)}")
            
        except Exception as e:
            print(f"WARNING: Could not read log file: {str(e)}")

if __name__ == "__main__":
    # Check HPO status
    finished = check_hpo_status()
    
    # Also check log progress
    check_log_progress()
    
    # Provide guidance
    print("\nNEXT STEPS:")
    if finished:
        print("  - Use best parameters for final training")
        print("  - Run evaluation with optimized configuration") 
        print("  - Generate final submission")
    else:
        print("  - Wait for optimization to complete")
        print("  - Monitor progress with: python check_hpo_status.py")
        print("  - Check process: tasklist | findstr python")
        print("  - View logs: tail -f hpo_sharpe_production.log")