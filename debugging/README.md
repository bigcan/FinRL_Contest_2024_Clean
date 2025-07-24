# Debugging Directory

This directory provides comprehensive debugging support for both Task 1 and Task 2 development.

## Structure

### `logs/` - Logging System
- **`task1/`**: Task 1 training and evaluation logs
- **`task2/`**: Task 2 training and evaluation logs  
- **`errors/`**: Error logs and stack traces
- **`system/`**: System resource usage logs

### `intermediate_outputs/` - Debug Outputs
- **`task1/`**: Agent states, actions, rewards, environment data
- **`task2/`**: LLM outputs, signals, embeddings, attention maps
- **`visualizations/`**: Debug plots, charts, and visual analysis

### `profiling/` - Performance Analysis
- **`task1/`**: Memory, CPU profiles for Task 1 components
- **`task2/`**: GPU memory, inference speed profiles for Task 2

### `debug_tools/` - Custom Debugging Utilities
- **`data_inspectors/`**: Tools to inspect and validate data
- **`model_analyzers/`**: Model analysis and interpretation tools
- **`environment_testers/`**: Environment validation and testing

## Logging Configuration

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information messages
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical errors that stop execution

### Environment Variables
```bash
# Enable debug mode
export FINRL_DEBUG=True

# Set log level
export FINRL_LOG_LEVEL=DEBUG

# Set log directory
export FINRL_LOG_DIR=debugging/logs/
```

## Usage Examples

### Enable Debugging for Task 1
```bash
# Set environment variables
export FINRL_DEBUG=True
export FINRL_LOG_LEVEL=DEBUG

# Run with debug output
cd development/task1/src/
python task1_ensemble.py --debug --save-intermediates ../../../debugging/intermediate_outputs/task1/
```

### Enable Debugging for Task 2
```bash
# Set environment variables
export FINRL_DEBUG=True
export FINRL_LOG_LEVEL=DEBUG

# Run with debug output
cd development/task2/src/
python task2_train.py --debug --save-intermediates ../../../debugging/intermediate_outputs/task2/
```

### View Logs
```bash
# View latest Task 1 logs
tail -f debugging/logs/task1/training_$(date +%Y%m%d).log

# View error logs
tail -f debugging/logs/errors/error_$(date +%Y%m%d).log

# View system resource logs
tail -f debugging/logs/system/system_$(date +%Y%m%d).log
```

## Intermediate Output Analysis

### Task 1 Debug Outputs
- **Agent States**: Current observations and internal states
- **Actions**: Decision sequences and action probabilities
- **Rewards**: Reward signals and cumulative returns
- **Environment Data**: Market state, positions, portfolio values

### Task 2 Debug Outputs
- **LLM Outputs**: Generated text and token probabilities
- **Signals**: Sentiment scores and confidence measures
- **Embeddings**: Token embeddings and attention weights
- **Training Data**: Input-output pairs and gradient information

## Performance Profiling

### Memory Profiling
```bash
# Profile memory usage
python -m memory_profiler task1_ensemble.py

# GPU memory profiling for Task 2
python -c "import torch; print(torch.cuda.memory_summary())"
```

### CPU Profiling
```bash
# Profile CPU usage
python -m cProfile -o profile_output.prof task1_ensemble.py

# Analyze profile
python -m pstats profile_output.prof
```

## Custom Debug Tools

### Data Inspector
```python
from debugging.debug_tools.data_inspectors import DataInspector

inspector = DataInspector()
inspector.validate_btc_data("data/raw/task1/BTC_1sec.csv")
inspector.analyze_news_data("data/raw/task2/task2_news_train.csv")
```

### Model Analyzer  
```python
from debugging.debug_tools.model_analyzers import ModelAnalyzer

analyzer = ModelAnalyzer()
analyzer.analyze_ensemble_weights("development/task1/models/")
analyzer.analyze_llm_attention("development/task2/models/")
```

### Environment Tester
```python
from debugging.debug_tools.environment_testers import EnvironmentTester

tester = EnvironmentTester()
tester.validate_trading_env()
tester.test_rlmf_environment()
```

## Best Practices

1. **Always enable logging** for training runs
2. **Save intermediate outputs** for failed experiments
3. **Profile performance** before optimization attempts
4. **Use custom debug tools** to validate assumptions
5. **Clean old logs** regularly to save disk space
6. **Document debugging sessions** for future reference