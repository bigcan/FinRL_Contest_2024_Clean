#!/usr/bin/env python3
"""
Analyze HPO results to find valid trials
"""
import optuna
import pandas as pd
from hpo_config import create_sqlite_storage

def analyze_hpo_results():
    # Load study
    storage_url = create_sqlite_storage("task1_production_hpo.db")
    study_name = "task1_production_hpo_20250725_161813"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    # Analyze all trials
    valid_trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None and trial.value != float('-inf'):
            valid_trials.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params
            })
    
    print(f"Total trials: {len(study.trials)}")
    print(f"Valid trials (non-inf values): {len(valid_trials)}")
    
    if valid_trials:
        # Sort by value (descending for Sharpe ratio)
        valid_trials.sort(key=lambda x: x['value'], reverse=True)
        
        print("\nTop 5 valid trials:")
        for i, trial in enumerate(valid_trials[:5]):
            print(f"\n{i+1}. Trial #{trial['trial_number']} - Sharpe Ratio: {trial['value']:.6f}")
            print("Parameters:")
            for param, value in trial['params'].items():
                print(f"  {param}: {value}")
    else:
        print("\nNo valid trials found. All trials returned -inf.")
        print("\nChecking trial user attributes for additional info...")
        
        # Check if any trials have stored metrics
        for trial in study.trials[:10]:  # Check first 10 trials
            if hasattr(trial, 'user_attrs') and trial.user_attrs:
                print(f"\nTrial {trial.number} attributes:")
                for key, value in trial.user_attrs.items():
                    print(f"  {key}: {value}")
    
    return valid_trials

if __name__ == "__main__":
    results = analyze_hpo_results()