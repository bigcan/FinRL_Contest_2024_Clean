# Hyperparameter Optimization (HPO) Documentation

## Overview

Phase 5 implements systematic hyperparameter optimization using Optuna to find the optimal combination of all parameters introduced in previous phases. This GPU-accelerated optimization searches through millions of possible configurations to maximize the Sharpe ratio.

## HPO Architecture

### Search Space

The HPO explores a comprehensive parameter space:

#### 1. Reward Function Parameters
- **profit_amplifier**: 2.0 to 10.0 (how much to amplify profits)
- **loss_multiplier**: 0.5 to 1.5 (penalty for losses)
- **trade_completion_bonus**: 0.01 to 0.05 (bonus for closing trades)
- **opportunity_cost_penalty**: 0.0005 to 0.005 (penalty for holding)
- **momentum_bonus**: 0.2 to 1.0 (bonus for momentum trades)
- **blend_factor**: 0.6 to 0.95 (profit vs original reward blend)

#### 2. Profit Speed Parameters
- **max_speed_multiplier**: 3.0 to 10.0 (multiplier for ultra-fast profits)
- **speed_decay_rate**: 0.01 to 0.05 (exponential decay rate)
- **min_holding_time**: 3 to 10 steps (minimum hold to avoid noise)

#### 3. Agent Hyperparameters
- **learning_rate**: 5e-5 to 5e-4 (log scale)
- **batch_size**: [128, 256, 512]
- **horizon_len**: [1024, 2048, 4096]
- **explore_rate**: 0.05 to 0.2 (initial exploration)
- **explore_decay**: 0.98 to 0.995 (exploration decay)
- **gamma**: 0.99 to 0.999 (discount factor)
- **entropy_coef**: 0.001 to 0.1 (exploration bonus)

#### 4. Network Architecture
- **medium**: [256, 256, 256]
- **large**: [512, 512, 256]
- **xlarge**: [512, 512, 512, 256]

#### 5. Environment Parameters
- **max_position**: 2 to 5 (maximum position size)
- **transaction_cost**: 0.0005 to 0.002
- **slippage**: 1e-5 to 1e-4 (log scale)
- **max_holding_time**: 600 to 3600 seconds

#### 6. Regime Detection Parameters
- **short_lookback**: 10 to 30 periods
- **medium_lookback**: 30 to 70 periods
- **long_lookback**: 70 to 150 periods

### Optimization Strategy

#### Objective Function
- **Primary metric**: Sharpe ratio (risk-adjusted returns)
- **Secondary considerations**: Win rate, action diversity
- **Evaluation**: 5 episodes per trial with different data samples

#### Optuna Configuration
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Pruner**: Hyperband (aggressive early stopping)
- **Storage**: SQLite database for persistence
- **Default trial**: Current best parameters from manual tuning

#### GPU Acceleration
- All neural network training on GPU
- Batch processing for efficiency
- Memory management with garbage collection
- Single job execution to avoid GPU memory conflicts

### Expected Search Process

1. **Initial exploration** (Trials 1-20)
   - Wide parameter exploration
   - Identify promising regions

2. **Refinement** (Trials 21-50)
   - Focus on high-performing regions
   - Fine-tune critical parameters

3. **Exploitation** (Trials 51-100)
   - Narrow search around best parameters
   - Validate optimal configuration

## Implementation Details

### HPOObjective Class
Core objective function that:
- Loads and manages data
- Creates environments with suggested parameters
- Trains agents for evaluation
- Calculates Sharpe ratio
- Reports intermediate results for pruning

### Trial Execution Flow
1. Suggest parameters from search space
2. Create trading environment with parameters
3. Initialize agent with suggested architecture
4. Train for 5 episodes (5000 steps each)
5. Calculate average Sharpe ratio
6. Report for pruning decision
7. Return negative Sharpe (for minimization)

### Results Analysis
- Parameter importance ranking
- Optimization history visualization
- Parallel coordinate plots
- Best parameter extraction
- Production config generation

## Usage

### Basic HPO Run
```bash
# Ensure GPU is available
python src/hpo_optimization.py

# Custom configuration
python src/hpo_optimization.py --trials 200 --episodes 10
```

### Testing HPO Setup
```bash
# Quick test to verify GPU and setup
python src/test_hpo_gpu.py
```

### Analyzing Results
```python
# Load completed study
study = optuna.load_study("profit_maximization_hpo", storage="sqlite:///profit_maximization_hpo.db")

# Get best parameters
best_params = study.best_params
best_sharpe = -study.best_value

# Generate production config
create_production_config(best_params, "production_config.json")
```

## Expected Outcomes

### Performance Targets
- **Sharpe Ratio**: > 2.0 (from baseline ~0.5)
- **Win Rate**: > 60%
- **Profit Factor**: > 2.0
- **Average Trade Duration**: < 60 seconds

### Key Parameter Insights
Based on the search space, we expect:
- High profit amplifiers (5-8x) for aggressive profit-seeking
- Fast learning rates (1e-4 to 3e-4) for quick adaptation
- Large networks for complex pattern recognition
- Moderate exploration (0.1-0.15) with slow decay

### Resource Requirements
- **GPU**: 8GB+ VRAM recommended
- **Time**: ~30 minutes per trial
- **Total**: 25-50 hours for 50-100 trials
- **Storage**: ~1GB for study database

## Integration with Previous Phases

The HPO optimizes across all previous enhancements:

1. **Phase 1**: Feature set is fixed (15 reduced features)
2. **Phase 2**: Profit reward weights are optimized
3. **Phase 3**: Network architecture and learning rates are tuned
4. **Phase 4**: Regime detection parameters are adjusted

## Production Deployment

After HPO completion:

1. **Extract best parameters**
   ```python
   best_params = study.best_params
   ```

2. **Generate production config**
   ```python
   create_production_config(best_params, "configs/production_hpo.json")
   ```

3. **Train final model**
   ```bash
   python src/train_production.py --config configs/production_hpo.json
   ```

4. **Validate performance**
   - Backtest on holdout data
   - Verify Sharpe ratio improvement
   - Check for overfitting

## Monitoring HPO Progress

During optimization:
- Watch `optimization_history.html` for convergence
- Check `param_importance.html` for key parameters
- Monitor GPU usage and memory
- Review pruned trials for patterns

## Common Issues and Solutions

1. **GPU Out of Memory**
   - Reduce batch size search space
   - Use smaller network architectures
   - Enable gradient checkpointing

2. **Slow Convergence**
   - Increase trials to 200+
   - Use more aggressive pruning
   - Start with narrower search space

3. **Unstable Results**
   - Check for data leakage
   - Verify environment consistency
   - Use fixed random seeds

## Next Steps

After completing HPO:
1. Analyze parameter importance
2. Run ablation studies on key parameters
3. Ensemble top 5 configurations
4. Deploy to production with monitoring