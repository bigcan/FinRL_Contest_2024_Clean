"""
Quick HPO demonstration with reduced trials and episodes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from hpo_optimization import run_hpo_study

if __name__ == "__main__":
    # Run with minimal configuration for demonstration
    data_path = Path(__file__).parent.parent / "task1_data" / "BTC_1sec_predict.npy"
    
    # Quick test with just 3 trials
    study = run_hpo_study(
        data_path=data_path,
        n_trials=3,  # Just 3 trials for demo
        n_jobs=1,
        study_name="profit_hpo_demo",
        use_gpu=True
    )
    
    print("\nQuick HPO Demo Complete!")
    print(f"Best Sharpe: {-study.best_value:.3f}")
    print("Best parameters summary:")
    for param in ['profit_amplifier', 'learning_rate', 'max_speed_multiplier']:
        if param in study.best_params:
            print(f"  {param}: {study.best_params[param]}")