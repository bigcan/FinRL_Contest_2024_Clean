"""
Shared utilities for Hyperparameter Optimization across Task 1 and Task 2
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class HPOAnalyzer:
    """Analysis utilities for HPO results"""
    
    def __init__(self, results_dir: str):
        """
        Initialize HPO analyzer
        
        Args:
            results_dir: Directory containing HPO results
        """
        self.results_dir = results_dir
        self.task1_results = self._load_task_results("task1")
        self.task2_results = self._load_task_results("task2")
    
    def _load_task_results(self, task_name: str) -> Dict[str, Any]:
        """Load results for a specific task"""
        results = {}
        
        # Load best parameters
        best_params_path = os.path.join(self.results_dir, f"{task_name}_best_params.json")
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                results['best_params'] = json.load(f)
        
        # Load study statistics
        stats_path = os.path.join(self.results_dir, f"{task_name}_study_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                results['stats'] = json.load(f)
        
        # Load trials history
        trials_path = os.path.join(self.results_dir, f"{task_name}_trials_history.json")
        if os.path.exists(trials_path):
            with open(trials_path, 'r') as f:
                results['trials'] = json.load(f)
        
        return results
    
    def create_optimization_plots(self, task_name: str, save_path: Optional[str] = None):
        """
        Create comprehensive optimization plots
        
        Args:
            task_name: Name of the task (task1 or task2)
            save_path: Path to save plots
        """
        if task_name == "task1":
            results = self.task1_results
        elif task_name == "task2":
            results = self.task2_results
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        if 'trials' not in results:
            print(f"No trial data found for {task_name}")
            return
        
        trials = results['trials']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'HPO Analysis: {task_name.upper()}', fontsize=16, fontweight='bold')
        
        # Prepare data
        trial_numbers = [t['number'] for t in trials if t['value'] is not None]
        trial_values = [t['value'] for t in trials if t['value'] is not None]
        
        # Plot 1: Optimization History
        axes[0, 0].plot(trial_numbers, trial_values, 'b-', alpha=0.6, linewidth=1)
        
        # Add best value line
        if trial_values:
            best_values = []
            current_best = trial_values[0]
            for val in trial_values:
                if val > current_best:  # Assuming maximization
                    current_best = val
                best_values.append(current_best)
            axes[0, 0].plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best So Far')
        
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Value Distribution
        if trial_values:
            axes[0, 1].hist(trial_values, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(np.mean(trial_values), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(trial_values):.4f}')
            axes[0, 1].axvline(np.median(trial_values), color='green', linestyle='--',
                              label=f'Median: {np.median(trial_values):.4f}')
        axes[0, 1].set_title('Objective Value Distribution')
        axes[0, 1].set_xlabel('Objective Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Plot 3: Trial Duration Distribution
        durations = [t['duration'] for t in trials if t['duration'] is not None]
        if durations:
            axes[0, 2].hist(durations, bins=15, alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Trial Duration Distribution')
            axes[0, 2].set_xlabel('Duration (seconds)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].axvline(np.mean(durations), color='red', linestyle='--',
                              label=f'Mean: {np.mean(durations):.1f}s')
            axes[0, 2].legend()
        
        # Plot 4: Parameter Importance (if available)
        self._plot_parameter_importance(axes[1, 0], trials, task_name)
        
        # Plot 5: Correlation Matrix
        self._plot_parameter_correlation(axes[1, 1], trials, task_name)
        
        # Plot 6: Top Trials Comparison
        self._plot_top_trials(axes[1, 2], trials, task_name)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
    
    def _plot_parameter_importance(self, ax, trials, task_name):
        """Plot parameter importance"""
        try:
            # Extract parameters and values
            param_data = {}
            values = []
            
            for trial in trials:
                if trial['value'] is not None and trial['params']:
                    values.append(trial['value'])
                    for param, value in trial['params'].items():
                        if param not in param_data:
                            param_data[param] = []
                        param_data[param].append(value)
            
            # Calculate correlations
            correlations = {}
            for param, param_values in param_data.items():
                if len(set(param_values)) > 1:  # Skip constant parameters
                    try:
                        # Convert to numeric if possible
                        numeric_values = []
                        for v in param_values:
                            if isinstance(v, (int, float)):
                                numeric_values.append(v)
                            elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
                                numeric_values.append(float(v))
                            else:
                                break
                        
                        if len(numeric_values) == len(param_values):
                            corr = np.corrcoef(numeric_values, values)[0, 1]
                            if not np.isnan(corr):
                                correlations[param] = abs(corr)
                    except:
                        pass
            
            if correlations:
                # Sort by importance
                sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                params, importances = zip(*sorted_params[:10])  # Top 10
                
                bars = ax.barh(range(len(params)), importances)
                ax.set_yticks(range(len(params)))
                ax.set_yticklabels(params)
                ax.set_xlabel('Absolute Correlation with Objective')
                ax.set_title('Parameter Importance')
                
                # Color bars
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            else:
                ax.text(0.5, 0.5, 'No numeric parameters\nfor importance analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Parameter Importance')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in parameter\nimportance analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Importance')
    
    def _plot_parameter_correlation(self, ax, trials, task_name):
        """Plot parameter correlation matrix"""
        try:
            # Extract numeric parameters
            param_data = {}
            
            for trial in trials:
                if trial['params']:
                    for param, value in trial['params'].items():
                        if isinstance(value, (int, float)):
                            if param not in param_data:
                                param_data[param] = []
                            param_data[param].append(value)
            
            # Create correlation matrix
            if len(param_data) > 1:
                df = pd.DataFrame(param_data)
                corr_matrix = df.corr()
                
                im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_yticks(range(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax.set_yticklabels(corr_matrix.columns)
                ax.set_title('Parameter Correlation Matrix')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Add correlation values
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Insufficient numeric\nparameters for correlation', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Parameter Correlation Matrix')
        except Exception as e:
            ax.text(0.5, 0.5, 'Error in correlation\nanalysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Correlation Matrix')
    
    def _plot_top_trials(self, ax, trials, task_name):
        """Plot comparison of top trials"""
        try:
            # Get top 5 trials
            valid_trials = [t for t in trials if t['value'] is not None]
            top_trials = sorted(valid_trials, key=lambda x: x['value'], reverse=True)[:5]
            
            if top_trials:
                trial_numbers = [f"Trial {t['number']}" for t in top_trials]
                trial_values = [t['value'] for t in top_trials]
                
                bars = ax.bar(range(len(trial_numbers)), trial_values)
                ax.set_xticks(range(len(trial_numbers)))
                ax.set_xticklabels(trial_numbers, rotation=45)
                ax.set_ylabel('Objective Value')
                ax.set_title('Top 5 Trials')
                
                # Color bars
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, trial_values)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(trial_values),
                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No completed trials', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Top 5 Trials')
        except Exception as e:
            ax.text(0.5, 0.5, 'Error in top trials\nanalysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Top 5 Trials')
    
    def generate_comparison_report(self) -> str:
        """Generate a comparison report between Task 1 and Task 2 HPO results"""
        report = """
# HPO Comparison Report: Task 1 vs Task 2

## Summary
"""
        
        # Task 1 Summary
        if self.task1_results and 'stats' in self.task1_results:
            stats1 = self.task1_results['stats']
            report += f"""
### Task 1 (Ensemble Trading)
- **Best Value**: {stats1.get('best_value', 'N/A')}
- **Total Trials**: {stats1.get('n_trials', 'N/A')}
- **Study Name**: {stats1.get('study_name', 'N/A')}
"""
        
        # Task 2 Summary
        if self.task2_results and 'stats' in self.task2_results:
            stats2 = self.task2_results['stats']
            report += f"""
### Task 2 (LLM Signal Generation)
- **Best Value**: {stats2.get('best_value', 'N/A')}
- **Total Trials**: {stats2.get('n_trials', 'N/A')}
- **Study Name**: {stats2.get('study_name', 'N/A')}
"""
        
        # Best Parameters Comparison
        report += "\n## Best Parameters\n"
        
        if self.task1_results and 'best_params' in self.task1_results:
            report += "\n### Task 1 Best Parameters\n"
            for param, value in self.task1_results['best_params'].items():
                report += f"- **{param}**: {value}\n"
        
        if self.task2_results and 'best_params' in self.task2_results:
            report += "\n### Task 2 Best Parameters\n"
            for param, value in self.task2_results['best_params'].items():
                report += f"- **{param}**: {value}\n"
        
        return report
    
    def export_best_configurations(self, export_dir: str):
        """
        Export best configurations for production use
        
        Args:
            export_dir: Directory to export configurations
        """
        os.makedirs(export_dir, exist_ok=True)
        
        # Export Task 1 configuration
        if self.task1_results and 'best_params' in self.task1_results:
            task1_config = {
                'best_params': self.task1_results['best_params'],
                'performance': self.task1_results.get('stats', {}),
                'export_timestamp': datetime.now().isoformat(),
                'task': 'task1_ensemble_trading'
            }
            
            task1_path = os.path.join(export_dir, 'task1_best_config.json')
            with open(task1_path, 'w') as f:
                json.dump(task1_config, f, indent=2)
        
        # Export Task 2 configuration
        if self.task2_results and 'best_params' in self.task2_results:
            task2_config = {
                'best_params': self.task2_results['best_params'],
                'performance': self.task2_results.get('stats', {}),
                'export_timestamp': datetime.now().isoformat(),
                'task': 'task2_llm_signal_generation'
            }
            
            task2_path = os.path.join(export_dir, 'task2_best_config.json')
            with open(task2_path, 'w') as f:
                json.dump(task2_config, f, indent=2)
        
        print(f"Best configurations exported to {export_dir}")


def load_study_from_db(db_path: str, study_name: str) -> optuna.Study:
    """
    Load an Optuna study from database
    
    Args:
        db_path: Path to SQLite database
        study_name: Name of the study
        
    Returns:
        Loaded Optuna study
    """
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    return study


def compare_multiple_studies(study_paths: List[Tuple[str, str]], metric_name: str = "objective"):
    """
    Compare multiple HPO studies
    
    Args:
        study_paths: List of (db_path, study_name) tuples
        metric_name: Name of the metric to compare
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(study_paths)))
    
    # Plot optimization histories
    for i, (db_path, study_name) in enumerate(study_paths):
        try:
            study = load_study_from_db(db_path, study_name)
            
            # Extract trial data
            trial_numbers = [t.number for t in study.trials if t.value is not None]
            trial_values = [t.value for t in study.trials if t.value is not None]
            
            if trial_values:
                axes[0].plot(trial_numbers, trial_values, 
                           color=colors[i], alpha=0.7, label=study_name)
        except Exception as e:
            print(f"Error loading study {study_name}: {e}")
    
    axes[0].set_title('Optimization Histories Comparison')
    axes[0].set_xlabel('Trial Number')
    axes[0].set_ylabel(metric_name)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot best values comparison
    study_names = []
    best_values = []
    
    for db_path, study_name in study_paths:
        try:
            study = load_study_from_db(db_path, study_name)
            study_names.append(study_name[:15])  # Truncate long names
            best_values.append(study.best_value)
        except Exception as e:
            print(f"Error loading study {study_name}: {e}")
    
    if best_values:
        bars = axes[1].bar(range(len(study_names)), best_values, color=colors[:len(study_names)])
        axes[1].set_xticks(range(len(study_names)))
        axes[1].set_xticklabels(study_names, rotation=45, ha='right')
        axes[1].set_title('Best Values Comparison')
        axes[1].set_ylabel(metric_name)
        
        # Add value labels
        for bar, value in zip(bars, best_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(best_values),
                        f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def create_hpo_dashboard(results_dir: str):
    """
    Create an interactive HPO dashboard
    
    Args:
        results_dir: Directory containing HPO results
    """
    analyzer = HPOAnalyzer(results_dir)
    
    print("="*80)
    print("🎯 HYPERPARAMETER OPTIMIZATION DASHBOARD")
    print("="*80)
    
    # Generate comparison report
    report = analyzer.generate_comparison_report()
    print(report)
    
    # Create plots for both tasks
    for task in ['task1', 'task2']:
        try:
            print(f"\n📊 Creating plots for {task.upper()}...")
            plot_path = os.path.join(results_dir, f"{task}_hpo_analysis.png")
            analyzer.create_optimization_plots(task, save_path=plot_path)
        except Exception as e:
            print(f"Error creating plots for {task}: {e}")
    
    # Export best configurations
    export_dir = os.path.join(results_dir, "best_configs")
    analyzer.export_best_configurations(export_dir)
    
    print("\n✅ Dashboard generation completed!")
    print(f"Results saved in: {results_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "hpo_experiments"
    
    create_hpo_dashboard(results_dir)