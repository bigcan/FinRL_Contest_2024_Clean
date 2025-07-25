# Hyperparameter Optimization (HPO) Integration Guide

## Overview

This guide covers the comprehensive HPO integration for the FinRL Contest 2024, providing systematic hyperparameter tuning for both Task 1 (Ensemble Cryptocurrency Trading) and Task 2 (LLM-based Signal Generation).

## 🚀 Features

- **Optuna-based Optimization**: Advanced hyperparameter optimization using TPE sampler
- **Intelligent Pruning**: Early stopping of unpromising trials to save computational resources
- **Persistent Storage**: SQLite-based study storage for resumable optimization
- **Comprehensive Analysis**: Detailed visualization and reporting of optimization results
- **Seamless Integration**: Drop-in replacement for existing training scripts
- **Multi-GPU Support**: Automatic GPU allocation for parallel trials
- **Production Ready**: Export optimized configurations for production deployment

## 📁 File Structure

```
development/
├── task1/src/
│   ├── hpo_config.py                    # Core HPO configuration classes
│   ├── task1_hpo.py                     # Task 1 HPO optimizer
│   ├── task1_ensemble_hpo_integrated.py # HPO-integrated training script
│   └── ...
├── task2/src/
│   ├── task2_hpo.py                     # Task 2 HPO optimizer
│   └── ...
├── shared/
│   └── hpo_utils.py                     # Shared HPO utilities and analysis
├── requirements_hpo.txt                 # HPO-specific dependencies
└── HPO_INTEGRATION_GUIDE.md            # This guide
```

## 🛠 Installation

1. **Install HPO dependencies**:
```bash
pip install -r requirements_hpo.txt
```

2. **Verify installation**:
```python
import optuna
import plotly
print("HPO framework ready!")
```

## 📊 Task 1: Ensemble Trading HPO

### Quick Start

```bash
# Quick HPO exploration (20 trials)
cd development/task1/src
python task1_hpo.py --config quick --metric sharpe_ratio

# Thorough optimization (200 trials)
python task1_hpo.py --config thorough --metric sharpe_ratio

# Production optimization (500 trials)
python task1_hpo.py --config production --metric romad
```

### Optimized Parameters

The HPO system optimizes the following parameters for Task 1:

**Network Architecture**:
- `net_dims_0`, `net_dims_1`, `net_dims_2`: Network layer dimensions (64-512)

**Learning Parameters**:
- `learning_rate`: Learning rate (1e-6 to 1e-3, log scale)
- `gamma`: Discount factor (0.99-0.999)
- `explore_rate`: Exploration rate (0.001-0.1, log scale)

**Training Parameters**:
- `batch_size`: Batch size (128, 256, 512, 1024)
- `buffer_size_multiplier`: Buffer size multiplier (4-16)
- `horizon_len_multiplier`: Horizon length multiplier (1-4)
- `repeat_times`: Gradient update repetitions (1-4)

**Environment Parameters**:
- `num_sims`: Parallel environments (32, 64, 128, 256)
- `step_gap`: Data step gap (1-5)
- `slippage`: Trading slippage (1e-8 to 1e-5, log scale)
- `max_position`: Maximum position size (1-3)

**Ensemble Configuration**:
- `n_ensemble_agents`: Number of agents in ensemble (2-4)
- `agent_0`, `agent_1`, etc.: Specific agent types

### Using HPO Results

```bash
# Train with HPO-optimized parameters
python task1_ensemble_hpo_integrated.py 0 \
    --hpo-results hpo_experiments/task1_hpo_results \
    --save-path ensemble_optimized

# Show HPO summary
python task1_ensemble_hpo_integrated.py \
    --hpo-results hpo_experiments/task1_hpo_results \
    --show-hpo-summary

# Override specific parameters
python task1_ensemble_hpo_integrated.py 0 \
    --hpo-results hpo_experiments/task1_hpo_results \
    --learning-rate 1e-5 \
    --batch-size 256
```

## 🤖 Task 2: LLM Signal Generation HPO

### Quick Start

```bash
# Quick HPO exploration (15 trials)
cd development/task2/src
python task2_hpo.py --config quick --metric cumulative_return

# Thorough optimization (100 trials)
python task2_hpo.py --config thorough --metric cumulative_return

# Production optimization (200 trials)
python task2_hpo.py --config production --metric sharpe_ratio
```

### Optimized Parameters

The HPO system optimizes the following parameters for Task 2:

**LoRA Parameters**:
- `lora_r`: LoRA rank (8-64, step 8)
- `lora_alpha`: LoRA alpha parameter (8-32, step 8)
- `lora_dropout`: LoRA dropout rate (0.05-0.3)

**Training Parameters**:
- `learning_rate`: Learning rate (1e-6 to 1e-4, log scale)
- `max_train_steps`: Maximum training steps (20-100)

**Signal Generation**:
- `signal_strength`: Signal strength parameter (5-20)
- `lookahead`: Lookahead days (1-7)

**Model Configuration**:
- `use_4bit`: Use 4-bit quantization (True/False)
- `quantization_type`: Quantization type ('fp4', 'nf4')

**Environment Parameters**:
- `max_env_steps`: Maximum environment steps (200-300)
- `reward_scaling`: Reward scaling factor (0.1-2.0)

## 📈 Analysis and Visualization

### Generate HPO Dashboard

```bash
# Create comprehensive analysis dashboard
cd development/shared
python hpo_utils.py hpo_experiments

# Or programmatically
from hpo_utils import create_hpo_dashboard
create_hpo_dashboard("hpo_experiments")
```

### Manual Analysis

```python
from shared.hpo_utils import HPOAnalyzer

# Initialize analyzer
analyzer = HPOAnalyzer("hpo_experiments/task1_hpo_results")

# Create optimization plots
analyzer.create_optimization_plots("task1", save_path="task1_analysis.png")

# Generate comparison report
report = analyzer.generate_comparison_report()
print(report)

# Export best configurations
analyzer.export_best_configurations("production_configs")
```

## 🔧 Advanced Configuration

### Custom HPO Configuration

```python
from hpo_config import HPOConfig, create_sqlite_storage

# Create custom configuration
custom_config = HPOConfig(
    study_name="custom_optimization",
    n_trials=50,
    n_jobs=2,
    timeout=7200,  # 2 hours
    sampler_type="tpe",
    pruner_type="median",
    storage_url=create_sqlite_storage("custom_hpo.db")
)

# Use with optimizer
from task1_hpo import Task1HPOOptimizer
optimizer = Task1HPOOptimizer(custom_config)
study = optimizer.run_optimization()
```

### Multi-Study Comparison

```python
from hpo_utils import compare_multiple_studies

# Compare multiple HPO runs
study_paths = [
    ("task1_hpo.db", "task1_thorough_hpo_20231120_143022"),
    ("task1_hpo.db", "task1_production_hpo_20231121_090000"),
]

compare_multiple_studies(study_paths, metric_name="Sharpe Ratio")
```

## 🎯 Optimization Strategies

### Metrics to Optimize

**Task 1 Options**:
- `sharpe_ratio`: Risk-adjusted returns (recommended)
- `total_return`: Cumulative returns
- `romad`: Return over Maximum Drawdown

**Task 2 Options**:
- `cumulative_return`: Final portfolio value (recommended)
- `mean_reward`: Average step reward
- `sharpe_ratio`: Risk-adjusted performance

### HPO Best Practices

1. **Start with Quick Exploration**: Use 'quick' configuration to understand parameter sensitivity
2. **Use Appropriate Metrics**: Choose metrics that align with contest evaluation criteria
3. **Enable Pruning**: Let Optuna stop unpromising trials early
4. **Monitor Progress**: Use Optuna dashboard for real-time monitoring
5. **Save Studies**: Use persistent storage for resumable optimization
6. **Validate Results**: Test optimized parameters on held-out data

## 🏭 Production Deployment

### Export Optimized Configuration

```python
# Export best configurations for production
from hpo_utils import HPOAnalyzer

analyzer = HPOAnalyzer("hpo_experiments")
analyzer.export_best_configurations("production_configs")

# Load in production
import json
with open("production_configs/task1_best_config.json", 'r') as f:
    best_config = json.load(f)
    
params = best_config['best_params']
```

### Integration with Existing Pipeline

```python
# Seamless integration with existing training
from task1_ensemble_hpo_integrated import HPOIntegratedTrainer

trainer = HPOIntegratedTrainer("hpo_experiments/task1_hpo_results")
trainer.train("ensemble_optimized")
```

## 📋 Monitoring and Debugging

### Real-time Monitoring

```bash
# Start Optuna dashboard
optuna-dashboard sqlite:///task1_hpo.db --host 0.0.0.0 --port 8080
```

### Debugging Failed Trials

```python
import optuna

# Load study and inspect failed trials
study = optuna.load_study(
    study_name="task1_production_hpo",
    storage="sqlite:///task1_hpo.db"
)

# Find failed trials
failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
print(f"Failed trials: {len(failed_trials)}")

# Inspect specific trial
if failed_trials:
    trial = failed_trials[0]
    print(f"Trial {trial.number} failed with params: {trial.params}")
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Slow Optimization**: Reduce `max_train_steps` for HPO trials
3. **Database Locks**: Ensure only one process writes to SQLite database
4. **CUDA Errors**: Set appropriate GPU allocation in multi-GPU setups

### Performance Tips

1. **Use Multiple GPUs**: Trials will automatically distribute across available GPUs
2. **Parallel Jobs**: Set `n_jobs=2` for Task 1, `n_jobs=1` for Task 2 (memory intensive)
3. **Early Stopping**: Enable pruning to stop unpromising trials
4. **Warm Start**: Use previous HPO results as starting point

## 📚 API Reference

### HPOConfig
Core configuration class for hyperparameter optimization.

### Task1HPOSearchSpace
Defines the search space for Task 1 parameters.

### Task2HPOSearchSpace  
Defines the search space for Task 2 parameters.

### HPOResultsManager
Manages saving and loading of HPO results.

### HPOAnalyzer
Provides analysis and visualization of HPO results.

## 🎉 Example Workflows

### Complete Task 1 Optimization Workflow

```bash
# 1. Run HPO optimization
python task1_hpo.py --config thorough --metric sharpe_ratio

# 2. Analyze results
python -c "from shared.hpo_utils import create_hpo_dashboard; create_hpo_dashboard('hpo_experiments')"

# 3. Train with optimized parameters
python task1_ensemble_hpo_integrated.py 0 \
    --hpo-results hpo_experiments/task1_hpo_results \
    --save-path ensemble_final

# 4. Evaluate results
python task1_eval.py 0 --ensemble-path ensemble_final
```

### Complete Task 2 Optimization Workflow

```bash
# 1. Run HPO optimization
python task2_hpo.py --config thorough --metric cumulative_return

# 2. Analyze results  
python -c "from shared.hpo_utils import create_hpo_dashboard; create_hpo_dashboard('hpo_experiments')"

# 3. Extract best parameters for manual training
python -c "
from shared.hpo_utils import HPOAnalyzer
analyzer = HPOAnalyzer('hpo_experiments/task2_hpo_results')
print(analyzer.task2_results['best_params'])
"
```

## 🔮 Future Enhancements

1. **Multi-Objective Optimization**: Optimize multiple metrics simultaneously
2. **Hyperband Integration**: More efficient resource allocation
3. **Neural Architecture Search**: Automatic network architecture optimization
4. **Distributed Optimization**: Scale across multiple machines
5. **Auto-ML Integration**: Full pipeline optimization

---

**Happy Optimizing! 🎯**

For questions or issues, please refer to the Optuna documentation or create an issue in the repository.