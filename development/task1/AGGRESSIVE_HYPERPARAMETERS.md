# Aggressive Hyperparameter Documentation

## Overview

This document explains the rationale behind the aggressive hyperparameter choices in Phase 3 of the profitability enhancement plan. These parameters are designed to create a highly profit-seeking agent that takes calculated risks for maximum returns.

## Key Hyperparameter Changes

### 1. Network Architecture
**Change**: `[256, 256, 256]` → `[512, 512, 256, 256]`

**Rationale**:
- **Increased capacity**: Larger networks can learn more complex profit patterns
- **Deeper architecture**: 4 layers instead of 3 for hierarchical feature learning
- **Gradual reduction**: 512→512→256→256 preserves information flow
- **Trade-off**: Higher compute cost but better pattern recognition

### 2. Learning Rate
**Change**: `3e-5` → `1e-4` (3.3x increase)

**Rationale**:
- **Faster adaptation**: Quick response to profitable patterns
- **Aggressive updates**: Larger steps toward profit maximization
- **Risk management**: Coupled with gradient clipping to prevent instability
- **Warmup**: 1000 steps warmup prevents early divergence

### 3. Exploration Strategy
**Change**: 
- Initial rate: `0.1` → `0.15` (50% increase)
- Decay rate: `0.995` → `0.99` (slower decay)
- Minimum: `0.01` → `0.005` (lower floor)

**Rationale**:
- **More exploration**: Discover profitable strategies early
- **Slower decay**: Maintain exploration longer for better coverage
- **Lower minimum**: Still explore even in late training
- **Action diversity**: Prevents getting stuck in suboptimal patterns

### 4. Batch Size & Horizon
**Change**:
- Batch size: `128` → `256` (2x)
- Horizon length: `1024` → `2048` (2x)

**Rationale**:
- **Stable gradients**: Larger batches reduce variance
- **Better credit assignment**: Longer horizons capture full trades
- **Efficiency**: Fewer but larger updates
- **GPU utilization**: Better hardware efficiency

### 5. Reward Amplification
**Change**:
- Profit amplifier: `3.0` → `5.0`
- Speed multiplier: `5.0` → `7.0`
- Blend factor: `0.7` → `0.85`

**Rationale**:
- **Extreme profit focus**: 5x base + 7x speed = up to 35x for ultra-fast profits
- **Higher blend**: 85% profit-focused vs 15% original rewards
- **Behavioral shift**: Force agent away from conservative strategies

### 6. Position & Risk Limits
**Change**:
- Max position: `2` → `3`
- Max holding time: `3600` → `1800` (halved)
- Transaction cost: `0.001` → `0.0008` (20% reduction)

**Rationale**:
- **Larger positions**: More profit potential per trade
- **Shorter holds**: Force quicker decisions
- **Lower friction**: Encourage more trading activity
- **Risk controls**: Stop loss (3%) and take profit (5%) limits

### 7. Training Configuration
**Change**:
- Episodes: `65` → `100`
- Samples per episode: `10,000` → `15,000`
- Update frequency: `512` steps
- Target Sharpe: `1.0` → `1.5`

**Rationale**:
- **More training**: Learn complex profit patterns
- **Larger episodes**: See more market conditions
- **Frequent updates**: Faster convergence
- **Higher target**: Push for better risk-adjusted returns

### 8. Optimizer Enhancements
**New additions**:
- Optimizer: AdamW with weight decay
- Cosine learning rate schedule
- Gradient accumulation (2 steps)
- Gradient clipping (max norm: 10.0)

**Rationale**:
- **AdamW**: Better generalization with weight decay
- **Cosine schedule**: Smooth learning rate decay
- **Accumulation**: Effective batch size of 512
- **Clipping**: Prevent gradient explosions

## Expected Impact

### Positive Effects
1. **Higher Returns**: 5-10x profit amplification drives aggressive profit-seeking
2. **Faster Learning**: 3x higher learning rate accelerates convergence
3. **Better Exploration**: Discovers more profitable strategies
4. **Improved Capacity**: Larger networks capture complex patterns

### Potential Risks
1. **Overfitting**: Mitigated by weight decay and larger datasets
2. **Instability**: Controlled by gradient clipping and warmup
3. **Excessive Risk**: Limited by position caps and stop-loss
4. **High Variance**: Reduced by larger batch sizes

## Implementation Notes

### GPU Requirements
- Minimum: 8GB VRAM (GTX 1070 or better)
- Recommended: 16GB+ VRAM for optimal batch sizes
- Training time: ~2-3 hours for 100 episodes

### Monitoring Metrics
Track these metrics during training:
- **Sharpe Ratio**: Target > 1.5
- **Win Rate**: Target > 55%
- **Profit Factor**: Target > 1.8
- **Average Trade Duration**: Target < 60s
- **Action Diversity**: Should remain balanced

### Hyperparameter Tuning Order
If results are suboptimal, adjust in this order:
1. Learning rate (try 5e-5 or 2e-4)
2. Profit amplifier (try 4.0 or 6.0)
3. Network size (try [384, 384, 192])
4. Exploration decay (try 0.985 or 0.995)

## Usage

### Training Command
```bash
python src/train_aggressive_profit.py
```

### Configuration Override
```python
# In code, override specific parameters:
config = load_aggressive_config()
config["agent_config"]["learning_rate"] = 5e-5  # Adjust if unstable
```

### Evaluation
After training, evaluate with:
```bash
python src/evaluate_aggressive_model.py --model-path aggressive_results/[timestamp]/best_model
```

## Comparison with Conservative Baseline

| Parameter | Conservative | Aggressive | Change |
|-----------|--------------|------------|---------|
| Network Size | [128, 128] | [512, 512, 256, 256] | 8x neurons |
| Learning Rate | 1e-5 | 1e-4 | 10x |
| Profit Amplifier | 1.5 | 5.0 | 3.3x |
| Max Position | 1 | 3 | 3x |
| Exploration | 0.05→0.01 | 0.15→0.005 | 3x range |
| Batch Size | 64 | 256 | 4x |

## Next Steps

After training with aggressive hyperparameters:
1. Analyze trading patterns for profit-seeking behavior
2. Fine-tune based on specific weaknesses
3. Implement market regime-specific adjustments (Phase 4)
4. Run systematic HPO for optimal weights (Phase 5)