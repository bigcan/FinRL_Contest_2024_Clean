"""
Real-time HPO Monitoring Dashboard
Tracks action diversity, conservative behavior, and training progress
"""

import os
import time
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path


class RealtimeHPOMonitor:
    """
    Real-time monitoring of HPO progress with focus on conservative trading issues
    """
    
    def __init__(self,
                 hpo_results_dir: str,
                 db_path: Optional[str] = None,
                 refresh_interval: int = 30):
        """
        Initialize real-time HPO monitor
        
        Args:
            hpo_results_dir: Directory containing HPO results
            db_path: Path to HPO database (auto-detected if None)
            refresh_interval: Refresh interval in seconds
        """
        self.results_dir = hpo_results_dir
        self.refresh_interval = refresh_interval
        
        # Auto-detect database if not provided
        if db_path is None:
            db_files = list(Path(hpo_results_dir).glob("*.db"))
            if db_files:
                self.db_path = str(db_files[0])
            else:
                self.db_path = None
        else:
            self.db_path = db_path
            
        # Monitoring state
        self.last_trial_count = 0
        self.monitoring_started = datetime.now()
        self.trial_data = []
        self.conservative_alerts = []
        
        # Setup monitoring directory
        self.monitor_dir = os.path.join(hpo_results_dir, "realtime_monitoring")
        os.makedirs(self.monitor_dir, exist_ok=True)
        
        print(f"🔍 HPO Monitor initialized")
        print(f"   Results dir: {hpo_results_dir}")
        print(f"   Database: {self.db_path}")
        print(f"   Monitor dir: {self.monitor_dir}")
        
    def start_monitoring(self, duration_hours: Optional[float] = None):
        """
        Start real-time monitoring
        
        Args:
            duration_hours: Duration to monitor (None for indefinite)
        """
        print(f"\n🚀 Starting real-time HPO monitoring")
        print(f"   Refresh interval: {self.refresh_interval}s")
        if duration_hours:
            print(f"   Duration: {duration_hours:.1f} hours")
        print("   Press Ctrl+C to stop\n")
        
        end_time = None
        if duration_hours:
            end_time = datetime.now() + timedelta(hours=duration_hours)
            
        try:
            while True:
                # Check if we should stop
                if end_time and datetime.now() > end_time:
                    print("⏰ Monitoring duration reached")
                    break
                    
                # Update monitoring data
                self._update_data()
                
                # Generate monitoring report
                self._generate_monitoring_report()
                
                # Check for conservative behavior alerts
                self._check_conservative_alerts()
                
                # Generate plots
                self._generate_plots()
                
                # Wait for next update
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n⚠️ Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {str(e)}")
            
        # Generate final summary
        self._generate_final_summary()
        
    def _update_data(self):
        """Update monitoring data from various sources"""
        
        # Update trial data from database
        if self.db_path and os.path.exists(self.db_path):
            self._update_from_database()
            
        # Update from JSON files
        self._update_from_json_files()
        
        # Check for new trials
        current_trial_count = len(self.trial_data)
        if current_trial_count > self.last_trial_count:
            new_trials = current_trial_count - self.last_trial_count
            print(f"📈 {new_trials} new trial(s) detected (Total: {current_trial_count})")
            self.last_trial_count = current_trial_count
            
    def _update_from_database(self):
        """Update data from Optuna database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query trials
            query = """
            SELECT trial_id, value, state, datetime_start, datetime_complete
            FROM trials
            WHERE state = 'COMPLETE'
            ORDER BY trial_id
            """
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                # Update trial count and basic stats
                self.db_trials = len(df)
                self.best_value = df['value'].max() if 'value' in df.columns else None
                
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Database update error: {e}")
            
    def _update_from_json_files(self):
        """Update data from JSON result files"""
        json_files = list(Path(self.results_dir).glob("trial_*_results.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    trial_data = json.load(f)
                    
                # Check if this trial is already in our data
                trial_number = trial_data.get('trial_number', -1)
                if not any(t.get('trial_number') == trial_number for t in self.trial_data):
                    self.trial_data.append(trial_data)
                    
            except Exception as e:
                print(f"⚠️ Error reading {json_file}: {e}")
                
    def _check_conservative_alerts(self):
        """Check for conservative behavior alerts"""
        if len(self.trial_data) < 3:
            return
            
        # Check recent trials for conservative patterns
        recent_trials = self.trial_data[-3:]
        
        for trial in recent_trials:
            diversity_metrics = trial.get('diversity_metrics', {}).get('metrics', {})
            
            if not diversity_metrics:
                continue
                
            hold_ratio = diversity_metrics.get('hold_ratio', 0)
            buy_ratio = diversity_metrics.get('buy_ratio', 0)
            entropy = diversity_metrics.get('entropy', 0)
            
            # Conservative alert conditions
            alerts = []
            
            if hold_ratio > 0.8:
                alerts.append(f"High conservatism: {hold_ratio:.1%} hold actions")
                
            if buy_ratio < 0.05:
                alerts.append(f"No buying behavior: {buy_ratio:.1%} buy actions")
                
            if entropy < 0.3:
                alerts.append(f"Low action diversity: {entropy:.3f} entropy")
                
            if alerts:
                alert_info = {
                    'trial_number': trial.get('trial_number', -1),
                    'timestamp': datetime.now().isoformat(),
                    'alerts': alerts,
                    'metrics': diversity_metrics
                }
                
                self.conservative_alerts.append(alert_info)
                
                print(f"🚨 Conservative behavior detected in Trial {trial.get('trial_number', -1)}:")
                for alert in alerts:
                    print(f"   - {alert}")
                    
    def _generate_monitoring_report(self):
        """Generate current monitoring report"""
        
        if len(self.trial_data) == 0:
            return
            
        # Calculate statistics
        objective_values = [t.get('objective_value', -10) for t in self.trial_data]
        best_objective = max(objective_values) if objective_values else -10
        mean_objective = np.mean(objective_values) if objective_values else -10
        
        # Conservative behavior statistics
        conservative_count = 0
        total_hold_ratio = 0
        total_buy_ratio = 0
        total_entropy = 0
        valid_metrics = 0
        
        for trial in self.trial_data:
            diversity_metrics = trial.get('diversity_metrics', {}).get('metrics', {})
            if diversity_metrics:
                hold_ratio = diversity_metrics.get('hold_ratio', 0)
                buy_ratio = diversity_metrics.get('buy_ratio', 0)
                entropy = diversity_metrics.get('entropy', 0)
                
                total_hold_ratio += hold_ratio
                total_buy_ratio += buy_ratio
                total_entropy += entropy
                valid_metrics += 1
                
                if hold_ratio > 0.7 or buy_ratio < 0.1:
                    conservative_count += 1
                    
        # Generate report
        report = {
            'monitoring_summary': {
                'monitoring_duration': str(datetime.now() - self.monitoring_started),
                'total_trials': len(self.trial_data),
                'best_objective': best_objective,
                'mean_objective': mean_objective,
                'conservative_trials': conservative_count,
                'conservative_ratio': conservative_count / max(1, len(self.trial_data))
            },
            'action_statistics': {
                'mean_hold_ratio': total_hold_ratio / max(1, valid_metrics),
                'mean_buy_ratio': total_buy_ratio / max(1, valid_metrics),
                'mean_entropy': total_entropy / max(1, valid_metrics)
            } if valid_metrics > 0 else {},
            'recent_alerts': self.conservative_alerts[-5:],  # Last 5 alerts
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_file = os.path.join(self.monitor_dir, "current_monitoring_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        print(f"\n📊 Monitoring Summary ({datetime.now().strftime('%H:%M:%S')})")
        print(f"   Trials completed: {len(self.trial_data)}")
        print(f"   Best objective: {best_objective:.6f}")
        print(f"   Conservative trials: {conservative_count}/{len(self.trial_data)} ({conservative_count/max(1,len(self.trial_data)):.1%})")
        
        if valid_metrics > 0:
            print(f"   Avg hold ratio: {total_hold_ratio/valid_metrics:.1%}")
            print(f"   Avg buy ratio: {total_buy_ratio/valid_metrics:.1%}")
            print(f"   Avg entropy: {total_entropy/valid_metrics:.3f}")
            
    def _generate_plots(self):
        """Generate real-time monitoring plots"""
        if len(self.trial_data) < 3:
            return
            
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"HPO Real-time Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Extract data for plotting
            trial_numbers = [t.get('trial_number', i) for i, t in enumerate(self.trial_data)]
            objective_values = [t.get('objective_value', -10) for t in self.trial_data]
            
            hold_ratios = []
            buy_ratios = []
            entropies = []
            
            for trial in self.trial_data:
                diversity_metrics = trial.get('diversity_metrics', {}).get('metrics', {})
                hold_ratios.append(diversity_metrics.get('hold_ratio', 0))
                buy_ratios.append(diversity_metrics.get('buy_ratio', 0))
                entropies.append(diversity_metrics.get('entropy', 0))
                
            # Plot 1: Objective values over trials
            axes[0, 0].plot(trial_numbers, objective_values, 'b-', alpha=0.7)
            axes[0, 0].scatter(trial_numbers, objective_values, c=objective_values, cmap='viridis', alpha=0.8)
            axes[0, 0].set_title('Objective Values Over Trials')
            axes[0, 0].set_xlabel('Trial Number')
            axes[0, 0].set_ylabel('Objective Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add best value line
            if objective_values:
                best_value = max(objective_values)
                axes[0, 0].axhline(y=best_value, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_value:.3f}')
                axes[0, 0].legend()
            
            # Plot 2: Hold ratios (conservative behavior indicator)
            axes[0, 1].plot(trial_numbers, hold_ratios, 'r-', alpha=0.7, label='Hold Ratio')
            axes[0, 1].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Conservative Threshold')
            axes[0, 1].set_title('Hold Action Ratios')
            axes[0, 1].set_xlabel('Trial Number')
            axes[0, 1].set_ylabel('Hold Ratio')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Buy ratios (trading activity indicator)
            axes[1, 0].plot(trial_numbers, buy_ratios, 'g-', alpha=0.7, label='Buy Ratio')
            axes[1, 0].axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Minimum Target')
            axes[1, 0].set_title('Buy Action Ratios')
            axes[1, 0].set_xlabel('Trial Number')
            axes[1, 0].set_ylabel('Buy Ratio')
            axes[1, 0].set_ylim(0, max(0.5, max(buy_ratios) if buy_ratios else 0.5))
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Action entropy (diversity indicator)
            axes[1, 1].plot(trial_numbers, entropies, 'purple', alpha=0.7, label='Action Entropy')
            axes[1, 1].axhline(y=0.5, color='purple', linestyle='--', alpha=0.5, label='Diversity Threshold')
            axes[1, 1].set_title('Action Diversity (Entropy)')
            axes[1, 1].set_xlabel('Trial Number')
            axes[1, 1].set_ylabel('Normalized Entropy')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.monitor_dir, f"realtime_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            
            # Also save as latest
            latest_plot = os.path.join(self.monitor_dir, "latest_monitoring.png")
            plt.savefig(latest_plot, dpi=150, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Plot generation error: {e}")
            
    def _generate_final_summary(self):
        """Generate final monitoring summary"""
        
        if len(self.trial_data) == 0:
            print("📊 No trial data collected during monitoring")
            return
            
        # Calculate comprehensive statistics
        objective_values = [t.get('objective_value', -10) for t in self.trial_data]
        
        conservative_trials = []
        successful_trials = []
        
        for trial in self.trial_data:
            diversity_metrics = trial.get('diversity_metrics', {}).get('metrics', {})
            if diversity_metrics:
                hold_ratio = diversity_metrics.get('hold_ratio', 0)
                buy_ratio = diversity_metrics.get('buy_ratio', 0)
                
                if hold_ratio > 0.7 or buy_ratio < 0.1:
                    conservative_trials.append(trial)
                else:
                    successful_trials.append(trial)
                    
        summary = {
            'monitoring_session': {
                'start_time': self.monitoring_started.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': str(datetime.now() - self.monitoring_started),
                'total_trials_monitored': len(self.trial_data)
            },
            'performance_summary': {
                'best_objective': max(objective_values) if objective_values else -10,
                'worst_objective': min(objective_values) if objective_values else -10,
                'mean_objective': np.mean(objective_values) if objective_values else -10,
                'std_objective': np.std(objective_values) if objective_values else 0
            },
            'conservative_behavior_analysis': {
                'conservative_trials_count': len(conservative_trials),
                'successful_trials_count': len(successful_trials),
                'conservative_ratio': len(conservative_trials) / max(1, len(self.trial_data)),
                'total_alerts_generated': len(self.conservative_alerts)
            },
            'recommendations': self._generate_recommendations(conservative_trials, successful_trials)
        }
        
        # Save final summary
        summary_file = os.path.join(self.monitor_dir, "final_monitoring_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print final summary
        print("\n" + "="*60)
        print("📊 FINAL HPO MONITORING SUMMARY")
        print("="*60)
        print(f"Monitoring duration: {datetime.now() - self.monitoring_started}")
        print(f"Total trials: {len(self.trial_data)}")
        print(f"Best objective: {max(objective_values) if objective_values else 'N/A':.6f}")
        print(f"Conservative trials: {len(conservative_trials)}/{len(self.trial_data)} ({len(conservative_trials)/max(1,len(self.trial_data)):.1%})")
        print(f"Alerts generated: {len(self.conservative_alerts)}")
        print(f"\n📁 Full summary saved to: {summary_file}")
        
        # Print recommendations
        recommendations = summary['recommendations']
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
                
    def _generate_recommendations(self, conservative_trials: List, successful_trials: List) -> List[str]:
        """Generate recommendations based on monitoring results"""
        
        recommendations = []
        total_trials = len(conservative_trials) + len(successful_trials)
        
        if len(conservative_trials) / max(1, total_trials) > 0.5:
            recommendations.append("High conservative behavior detected - consider increasing exploration rates")
            
        if len(conservative_trials) > 10:
            recommendations.append("Many conservative trials - review reward function weights")
            
        if len(successful_trials) > 0:
            # Find best successful trial
            best_successful = max(successful_trials, key=lambda x: x.get('objective_value', -10))
            recommendations.append(f"Best non-conservative trial: #{best_successful.get('trial_number', -1)} - analyze parameters")
            
        if len(self.conservative_alerts) > len(self.trial_data) * 0.3:
            recommendations.append("Frequent conservative alerts - consider action masking or forced exploration")
            
        return recommendations


def main():
    """Main monitoring function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python realtime_hpo_monitor.py <hpo_results_directory> [duration_hours]")
        return
        
    results_dir = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
        
    # Create monitor
    monitor = RealtimeHPOMonitor(
        hpo_results_dir=results_dir,
        refresh_interval=30
    )
    
    # Start monitoring
    monitor.start_monitoring(duration_hours=duration)


if __name__ == "__main__":
    main()