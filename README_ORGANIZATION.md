# FinRL Contest 2024 - File Organization System

## 🎯 Overview

This document describes the comprehensive file organization system designed for the FinRL Contest 2024 project. The system supports debugging, testing, verification, and experiment management while preserving original files and maintaining clean separation of concerns.

## 📁 Directory Structure

```
FinRL_Contest_2024/
├── 🔒 original/                        # PRESERVED ORIGINAL FILES
│   ├── Task_1_starter_kit/            # Original Task 1 files (untouched)
│   ├── Task_2_starter_kit/            # Original Task 2 files (untouched)
│   ├── Submission/                    # Example submissions
│   └── Tutorials/                     # Reference materials
│
├── 🛠️ development/                     # OUR DEVELOPMENT WORK
│   ├── task1/                         # Task 1 Development
│   │   ├── src/                       # Modified/enhanced source code
│   │   ├── configs/                   # Configuration files & experiments
│   │   ├── models/                    # Trained model storage
│   │   └── scripts/                   # Training/evaluation scripts
│   │
│   ├── task2/                         # Task 2 Development
│   │   ├── src/                       # Modified/enhanced source code
│   │   ├── configs/                   # Configuration files & experiments
│   │   ├── models/                    # Fine-tuned LLM storage
│   │   └── scripts/                   # Training/evaluation scripts
│   │
│   ├── shared/                        # Common utilities
│   │   ├── data_processing/           # Data preprocessing tools
│   │   ├── evaluation/                # Shared evaluation metrics
│   │   └── utils/                     # Common helper functions
│   │
│   └── environments/                  # Virtual environments
│       ├── task1_env/                 # Task 1 Python environment
│       └── task2_env/                 # Task 2 Python environment
│
├── 🧪 testing/                        # COMPREHENSIVE TESTING
│   ├── unit_tests/                    # Unit tests for individual components
│   ├── integration_tests/             # Integration tests
│   ├── performance_tests/             # Performance benchmarks
│   ├── data_validation/               # Data integrity tests  
│   └── test_configs/                  # Test-specific configurations
│
├── 🐛 debugging/                      # DEBUG SUPPORT
│   ├── logs/                          # All log files
│   ├── intermediate_outputs/          # Debug outputs
│   ├── profiling/                     # Performance profiling
│   └── debug_tools/                   # Custom debugging utilities
│
├── 🔬 experiments/                    # EXPERIMENT MANAGEMENT
│   ├── task1_experiments/             # Task 1 experiments
│   ├── task2_experiments/             # Task 2 experiments
│   └── experiment_tracking/           # Experiment metadata
│
├── ✅ verification/                   # VERIFICATION & VALIDATION
│   ├── reproducibility/               # Reproducibility checks
│   ├── results_validation/            # Results validation
│   ├── baseline_comparisons/          # Compare against baselines
│   └── submission_validation/         # Pre-submission checks
│
├── 📊 data/                           # DATA MANAGEMENT
│   ├── raw/                           # Original datasets
│   ├── processed/                     # Preprocessed data
│   ├── synthetic/                     # Generated test data
│   └── data_splits/                   # Train/val/test splits
│
├── 📈 results/                        # RESULTS & OUTPUTS
│   ├── task1_results/                 # Task 1 outputs
│   ├── task2_results/                 # Task 2 outputs
│   └── final_submissions/             # Contest submissions
│
└── 📚 documentation/                  # COMPREHENSIVE DOCS
    ├── setup_guides/                  # Installation & setup
    ├── api_documentation/             # Code documentation
    ├── experiment_logs/               # Detailed experiment notes
    ├── troubleshooting/               # Common issues & solutions
    └── methodology_notes/             # Approach explanations
```

## 🚀 Getting Started

### 1. Data Setup
```bash
# For Task 1: Download Bitcoin LOB data to data/raw/task1/
# - BTC_1sec.csv
# - BTC_1sec_predict.npy

# For Task 2: Dataset already extracted to development/task2/src/task2_dsets/
```

### 2. Environment Setup
```bash
# Task 1 environment is already set up in development/environments/task1_env/
# For Task 2, create new environment:
cd development/environments/
python3 -m venv task2_env
source task2_env/bin/activate
pip install -r ../task2/src/requirements_simplified.txt
```

### 3. Development Workflow
```bash
# Work in development directories
cd development/task1/src/    # For Task 1 work
cd development/task2/src/    # For Task 2 work

# Run experiments
cd experiments/task1_experiments/baseline/
cd experiments/task2_experiments/baseline/
```

## 🔍 Key Features

### 🔒 **Original File Preservation**
- All starter kit files preserved in `original/` directory
- Easy reference and comparison with baseline
- No accidental modifications to contest-provided code

### 🛠️ **Clean Development Structure**
- Separate directories for Task 1 and Task 2
- Isolated virtual environments
- Shared utilities for common functionality
- Configuration management for experiments

### 🧪 **Comprehensive Testing**
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmarking and optimization
- **Data Validation**: Integrity and format checks

### 🐛 **Advanced Debugging**
- **Structured Logging**: Multi-level logging system
- **Intermediate Outputs**: Capture states, actions, rewards
- **Performance Profiling**: Memory, CPU, GPU usage
- **Custom Tools**: Data inspectors, model analyzers

### 🔬 **Experiment Management**
- **Organized Tracking**: Structured experiment storage
- **Hyperparameter Search**: HPO result organization
- **Ablation Studies**: Component analysis
- **Cross-Experiment Comparisons**: Performance analysis

### ✅ **Verification Systems**
- **Reproducibility**: Environment and result validation
- **Baseline Comparisons**: Against provided benchmarks
- **Results Validation**: Cross-validation and testing
- **Submission Checks**: Pre-contest validation

## 📝 Usage Examples

### Running Experiments
```bash
# Task 1 Baseline Experiment
cd experiments/task1_experiments/baseline/
python ../../development/task1/src/task1_ensemble.py --config baseline_config.yaml

# Task 2 Baseline Experiment  
cd experiments/task2_experiments/baseline/
python ../../development/task2/src/task2_train.py --config baseline_config.yaml
```

### Debugging Sessions
```bash
# Enable debug logging
export FINRL_DEBUG=True
export FINRL_LOG_LEVEL=DEBUG

# Run with intermediate output capture
python task1_ensemble.py --debug --save-intermediates debugging/intermediate_outputs/task1/
```

### Testing
```bash
# Run unit tests
cd testing/unit_tests/task1/
python -m pytest test_agents.py

# Run integration tests
cd testing/integration_tests/task1/
python test_full_pipeline.py
```

### Results Analysis
```bash
# View experiment results
cd results/task1_results/performance_metrics/
python analyze_results.py --experiment baseline

# Compare multiple experiments
python compare_experiments.py --experiments baseline ensemble_v2 ensemble_v3
```

## 🛡️ Best Practices

### Development
1. **Always work in `development/` directories**
2. **Use separate virtual environments for each task**
3. **Save all configurations in `configs/` directories**
4. **Document experiments in `experiment_logs/`**

### Debugging
1. **Enable logging for all training runs**
2. **Save intermediate outputs for analysis**
3. **Use profiling for performance optimization**
4. **Create custom debug tools as needed**

### Testing
1. **Write unit tests for all new components**
2. **Run integration tests before experiments**
3. **Validate data integrity regularly**
4. **Benchmark performance improvements**

### Experiments
1. **Use descriptive experiment names**
2. **Save all hyperparameters and configs**
3. **Document methodology and results**
4. **Compare against established baselines**

## 🔧 Maintenance

### Regular Tasks
- **Clean old log files**: `debugging/logs/`
- **Archive completed experiments**: Move to archive directory
- **Update documentation**: Keep READMEs current
- **Backup important results**: Use version control

### Before Submission
1. **Run verification tests**: `verification/submission_validation/`
2. **Check reproducibility**: `verification/reproducibility/`
3. **Validate file formats**: Ensure contest compliance
4. **Test on clean environment**: Fresh installation test

## 📊 Monitoring

### Key Metrics to Track
- **Training Progress**: Loss, rewards, convergence
- **Performance Metrics**: Sharpe ratio, returns, win/loss
- **Resource Usage**: GPU memory, CPU, disk space
- **Experiment Status**: Running, completed, failed

### Alerting
- **Long Running Jobs**: Monitor training duration
- **Resource Exhaustion**: GPU memory, disk space
- **Failed Experiments**: Automatic error notification
- **Performance Regression**: Metric degradation

This organization system ensures clean, debuggable, and reproducible development while maintaining professional standards for the FinRL Contest 2024.