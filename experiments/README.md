# Experiments Directory

This directory manages all experimental work for both Task 1 and Task 2, providing structured experiment tracking and comparison capabilities.

## Structure

### `task1_experiments/` - Task 1 Cryptocurrency Trading Experiments
- **`baseline/`**: Baseline ensemble experiments using provided code
- **`ensemble_variations/`**: Different ensemble methods and algorithms
- **`hyperparameter_search/`**: Hyperparameter optimization results
- **`ablation_studies/`**: Component analysis and ablation studies

### `task2_experiments/` - Task 2 LLM Signal Generation Experiments
- **`baseline/`**: Baseline RLMF experiments using provided code
- **`prompt_engineering/`**: Different prompt templates and strategies
- **`lora_configurations/`**: LoRA hyperparameter variations
- **`reward_functions/`**: Different reward function designs

### `experiment_tracking/` - Experiment Management
- **`run_logs/`**: Individual experiment run records
- **`comparisons/`**: Cross-experiment comparison results
- **`best_configs/`**: Top-performing configuration files

## Experiment Naming Convention

Use descriptive names with the following format:
```
{task}_{experiment_type}_{date}_{description}
```

Examples:
- `task1_baseline_20240724_default_config`
- `task1_ensemble_20240724_voting_weights`
- `task2_baseline_20240724_llama3_standard`
- `task2_prompt_20240724_sentiment_v2`

## Running Experiments

### Task 1 Baseline Experiment
```bash
cd experiments/task1_experiments/baseline/

# Create experiment directory
mkdir task1_baseline_$(date +%Y%m%d)_default
cd task1_baseline_$(date +%Y%m%d)_default

# Copy configuration
cp ../../../../development/task1/configs/baseline_config.yaml .

# Run experiment
python ../../../../development/task1/src/task1_ensemble.py \
    --config baseline_config.yaml \
    --output-dir . \
    --experiment-name task1_baseline_$(date +%Y%m%d)_default
```

### Task 2 Baseline Experiment
```bash
cd experiments/task2_experiments/baseline/

# Create experiment directory
mkdir task2_baseline_$(date +%Y%m%d)_llama3
cd task2_baseline_$(date +%Y%m%d)_llama3

# Copy configuration
cp ../../../../development/task2/configs/baseline_config.yaml .

# Run experiment
python ../../../../development/task2/src/task2_train.py \
    --config baseline_config.yaml \
    --output-dir . \
    --experiment-name task2_baseline_$(date +%Y%m%d)_llama3
```

## Experiment Configuration

### Configuration File Format (YAML)
```yaml
# Task 1 Configuration Example
experiment:
  name: "task1_ensemble_voting"
  description: "Ensemble with voting mechanism"
  
model:
  agents: ["D3QN", "DoubleDQN", "TwinD3QN"]
  ensemble_method: "voting"
  
training:
  max_episodes: 1000
  learning_rate: 0.001
  batch_size: 32
  
environment:
  initial_cash: 1000000
  data_path: "data/raw/task1/BTC_1sec_predict.npy"
  
logging:
  level: "INFO"
  save_intermediate: true
```

```yaml
# Task 2 Configuration Example  
experiment:
  name: "task2_rlmf_sentiment"
  description: "RLMF with sentiment analysis"
  
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  lora_r: 30
  lora_alpha: 16
  lora_dropout: 0.1
  
training:
  max_steps: 100
  learning_rate: 1e-5
  signal_strength: 10
  
environment:
  lookahead: 3
  num_long: 3
  num_short: 3
  
logging:
  level: "INFO"
  save_intermediate: true
```

## Experiment Tracking

### Run Log Format
Each experiment creates a run log with:
- **Start/End Times**: Experiment duration
- **Configuration**: All hyperparameters and settings
- **Results**: Performance metrics and outcomes
- **Resource Usage**: CPU, GPU, memory consumption
- **Artifacts**: Saved models, plots, intermediate outputs

### Example Run Log
```json
{
  "experiment_id": "task1_baseline_20240724_default",
  "start_time": "2024-07-24T10:00:00Z",
  "end_time": "2024-07-24T12:30:00Z",
  "duration_minutes": 150,
  "config": {
    "model": {"agents": ["D3QN", "DoubleDQN"]},
    "training": {"max_episodes": 1000}
  },
  "results": {
    "sharpe_ratio": 1.23,
    "max_drawdown": 0.15,
    "cumulative_return": 0.45
  },
  "resource_usage": {
    "peak_memory_gb": 8.5,
    "gpu_utilization": 0.85
  },
  "artifacts": {
    "models": ["ensemble_model.pkl"],
    "plots": ["performance.png", "training_curve.png"],
    "logs": ["training.log"]
  }
}
```

## Performance Comparison

### Generate Comparison Report
```bash
cd experiments/experiment_tracking/comparisons/

python ../../../development/shared/evaluation/compare_experiments.py \
    --experiments task1_baseline_20240724 task1_ensemble_20240724 \
    --metrics sharpe_ratio max_drawdown cumulative_return \
    --output task1_comparison_$(date +%Y%m%d).html
```

### Best Configuration Tracking
The system automatically tracks top-performing configurations:
```bash
cd experiments/experiment_tracking/best_configs/

# View current best configurations
cat task1_best_sharpe.yaml
cat task2_best_return.yaml
```

## Hyperparameter Search

### Grid Search Example
```bash
cd experiments/task1_experiments/hyperparameter_search/

python ../../../development/shared/utils/grid_search.py \
    --config_template grid_search_config.yaml \
    --param_grid param_grid.json \
    --output_dir grid_search_$(date +%Y%m%d)
```

### Random Search Example
```bash
cd experiments/task2_experiments/hyperparameter_search/

python ../../../development/shared/utils/random_search.py \
    --config_template random_search_config.yaml \
    --n_trials 50 \
    --output_dir random_search_$(date +%Y%m%d)
```

## Ablation Studies

### Task 1 Component Ablation
```bash
cd experiments/task1_experiments/ablation_studies/

# Test individual agents
python ablation_single_agents.py

# Test ensemble methods
python ablation_ensemble_methods.py

# Test feature sets
python ablation_features.py
```

### Task 2 Component Ablation
```bash
cd experiments/task2_experiments/ablation_studies/

# Test prompt variations
python ablation_prompts.py

# Test LoRA configurations
python ablation_lora.py

# Test reward functions
python ablation_rewards.py
```

## Best Practices

1. **Use descriptive experiment names** with dates and descriptions
2. **Save all configurations** in YAML format for reproducibility
3. **Document experiment motivation** and expected outcomes
4. **Compare against established baselines** before claiming improvements
5. **Archive completed experiments** to save disk space
6. **Track resource usage** for cost optimization
7. **Validate results** before drawing conclusions