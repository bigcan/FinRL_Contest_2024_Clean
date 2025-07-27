#!/usr/bin/env python3
"""
Real-time HPO monitoring with progress visualization
"""

import os
import time
import sqlite3
import optuna
from datetime import datetime
import sys
from hpo_config import create_sqlite_storage

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_duration(seconds):
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

def monitor_hpo(db_path="task1_production_hpo.db", refresh_interval=5):
    """Monitor HPO progress in real-time"""
    
    # Check if database exists
    full_db_path = os.path.join("archived_experiments", "hpo_databases", db_path)
    if not os.path.exists(full_db_path):
        full_db_path = db_path  # Try current directory
        if not os.path.exists(full_db_path):
            print(f"❌ HPO database not found: {db_path}")
            print("Please ensure HPO is running or provide correct database path.")
            return
    
    print(f"Monitoring HPO database: {full_db_path}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    start_time = time.time()
    
    try:
        while True:
            clear_screen()
            
            # Connect to database
            try:
                storage_url = create_sqlite_storage(full_db_path)
                
                # Get study name
                conn = sqlite3.connect(full_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT study_name FROM studies ORDER BY study_id DESC LIMIT 1")
                result = cursor.fetchone()
                study_name = result[0] if result else None
                conn.close()
                
                if not study_name:
                    print("⏳ Waiting for study to start...")
                    time.sleep(refresh_interval)
                    continue
                
                # Load study
                study = optuna.load_study(study_name=study_name, storage=storage_url)
                
                # Get statistics
                n_trials = len(study.trials)
                n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                n_running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
                n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
                n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                
                # Display header
                print("="*80)
                print("📊 FinRL Contest 2024 - HPO Real-time Monitor")
                print("="*80)
                print(f"Study: {study_name}")
                print(f"Direction: {study.direction.name}")
                print(f"Running Time: {format_duration(time.time() - start_time)}")
                print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                
                # Trial statistics
                print("\n📈 Trial Statistics:")
                print(f"  Total Trials:    {n_trials:4d}")
                print(f"  ✅ Completed:    {n_complete:4d}")
                print(f"  🏃 Running:      {n_running:4d}")
                print(f"  ❌ Failed:       {n_failed:4d}")
                print(f"  ✂️  Pruned:       {n_pruned:4d}")
                
                # Progress bar
                if n_trials > 0:
                    progress = n_complete / n_trials * 100
                    bar_length = 40
                    filled = int(bar_length * n_complete / n_trials)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    print(f"\n  Progress: [{bar}] {progress:.1f}%")
                
                # Best results
                if study.best_trial:
                    print("\n🏆 Best Results:")
                    print(f"  Best Value: {study.best_value:.6f}")
                    print(f"  Best Trial: #{study.best_trial.number}")
                    print("\n  Best Parameters:")
                    for param, value in study.best_params.items():
                        if isinstance(value, float):
                            print(f"    {param}: {value:.6f}")
                        else:
                            print(f"    {param}: {value}")
                
                # Recent trials
                if study.trials:
                    print("\n📝 Recent Trials (last 5):")
                    print("  Trial | Status |    Value    | Duration")
                    print("  ------|--------|-------------|----------")
                    
                    recent_trials = sorted(study.trials, key=lambda x: x.number)[-5:]
                    for trial in recent_trials:
                        status_map = {
                            "COMPLETE": "✅ DONE",
                            "RUNNING": "🏃 RUN ",
                            "FAIL": "❌ FAIL",
                            "PRUNED": "✂️  SKIP"
                        }
                        status_str = status_map.get(trial.state.name, "❓ UNKN")
                        value_str = f"{trial.value:.6f}" if trial.value is not None else "   N/A   "
                        
                        # Calculate duration
                        if trial.datetime_complete and trial.datetime_start:
                            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
                            duration_str = format_duration(duration)
                        else:
                            duration_str = "   N/A   "
                        
                        print(f"  {trial.number:5d} | {status_str} | {value_str} | {duration_str}")
                
                # Performance trend
                if n_complete >= 5:
                    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    recent_values = [t.value for t in sorted(completed_trials, key=lambda x: x.number)[-10:]]
                    avg_recent = sum(recent_values) / len(recent_values)
                    
                    print(f"\n📊 Performance Trend:")
                    print(f"  Last 10 trials average: {avg_recent:.6f}")
                    
                    # Simple trend indicator
                    if len(recent_values) >= 2:
                        trend = recent_values[-1] - recent_values[0]
                        trend_symbol = "↗️" if trend > 0 else "↘️" if trend < 0 else "→"
                        print(f"  Trend: {trend_symbol} ({trend:+.6f})")
                
                print("\n" + "="*80)
                print("Press Ctrl+C to stop monitoring")
                
            except Exception as e:
                print(f"⚠️ Error reading database: {str(e)}")
                print("Retrying in a moment...")
            
            # Wait before refresh
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
        print(f"Total monitoring time: {format_duration(time.time() - start_time)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time HPO monitoring')
    parser.add_argument('--db', type=str, default='task1_production_hpo.db', help='Database filename')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    monitor_hpo(args.db, args.interval)