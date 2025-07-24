# Development Directory

This directory contains all active development work for both Task 1 and Task 2.

## Structure

### `task1/` - Cryptocurrency Trading Development
- **`src/`**: Modified source code from Task_1_starter_kit
- **`configs/`**: Experiment configurations and hyperparameters
- **`models/`**: Trained ensemble models and checkpoints
- **`scripts/`**: Training and evaluation scripts

### `task2/` - LLM Signal Generation Development  
- **`src/`**: Modified source code from Task_2_starter_kit
- **`configs/`**: Model configs, LoRA settings, prompt templates
- **`models/`**: Fine-tuned LLM models and LoRA adapters
- **`scripts/`**: Training and evaluation scripts

### `shared/` - Common Utilities
- **`data_processing/`**: Data preprocessing and feature engineering
- **`evaluation/`**: Shared evaluation metrics and tools
- **`utils/`**: Common helper functions and utilities

### `environments/` - Python Environments
- **`task1_env/`**: Isolated environment for Task 1 dependencies
- **`task2_env/`**: Isolated environment for Task 2 dependencies

## Usage

### Task 1 Development
```bash
# Activate Task 1 environment
source environments/task1_env/bin/activate

# Work in Task 1 source
cd task1/src/

# Run training
python task1_ensemble.py
```

### Task 2 Development
```bash
# Activate Task 2 environment  
source environments/task2_env/bin/activate

# Work in Task 2 source
cd task2/src/

# Run training
python task2_train.py
```

## Configuration Management

Store all experiment configurations in the respective `configs/` directories:
- Use YAML format for readability
- Include all hyperparameters and settings
- Version control configuration files
- Document configuration changes

## Model Storage

Save trained models in the `models/` directories:
- Use descriptive filenames with timestamps
- Include model metadata and performance metrics
- Organize by experiment type or date
- Maintain model registry for tracking