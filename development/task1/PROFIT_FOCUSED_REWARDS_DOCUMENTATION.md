# Profit-Focused Reward Function Documentation

## Overview

This document describes the profit-focused reward function implemented to address the core issue: **the agent minimizes losses but doesn't generate profits**.

## Problem Statement

The original reward function was too conservative, leading to:
- Agents that avoid losses but rarely seek profits
- Excessive holding behavior (70%+ hold actions)
- Near-zero returns despite stable training

## Solution: Profit-Focused Rewards

### Key Components

#### 1. **Asymmetric Profit Amplification** (3x multiplier)
```python
if position_pnl > 0:
    base_reward = position_pnl * 3.0  # Amplify profits
else:
    base_reward = position_pnl * 1.0  # Normal losses
```
- Profits are rewarded 3x more than losses are penalized
- Encourages profit-seeking behavior
- Maintains risk awareness without excessive conservatism

#### 2. **Trade Completion Bonuses**
```python
if trade_completed and trade_return > 0:
    completion_bonus = 0.02 * (1 + trade_return)
```
- Rewards closing profitable trades
- Scales with trade profitability
- Encourages active trading over passive holding

#### 3. **Opportunity Cost Penalties**
```python
if position != 0:
    holding_penalty = 0.001 * min(holding_time / 100, 2.0)
```
- Escalating penalty for holding positions too long
- Prevents indefinite holding
- Encourages position turnover

#### 4. **Momentum Bonuses**
```python
if consecutive_positive_returns:
    momentum_bonus = position_pnl * 0.5
```
- 50% extra reward for riding momentum
- Rewards trend-following behavior
- Encourages staying with winners

#### 5. **Action Encouragement**
```python
if action != hold:
    action_bonus = 0.001
    if counter_trend_action:
        action_bonus *= 2
```
- Small bonus for decisive actions
- Double bonus for contrarian trades
- Reduces holding bias

#### 6. **Market Regime Sensitivity**
```python
regime_multipliers = {
    "trending": 1.2,   # Amplify in trends
    "volatile": 0.8,   # Reduce in volatility
    "ranging": 1.0     # Normal in ranges
}
```
- Adapts rewards to market conditions
- Encourages appropriate strategies per regime
- Integrates with existing MarketRegimeDetector

## Implementation Details

### ProfitFocusedRewardCalculator Class

Main class implementing the profit-focused logic:
- Tracks position entry prices
- Monitors holding times
- Calculates trade returns
- Maintains performance metrics

### MetaRewardCalculator Class

Parameterized version for HPO:
- Customizable component weights
- Systematic optimization support
- Easy experimentation with different configurations

### Integration Function

`integrate_profit_rewards()` seamlessly integrates with existing RewardCalculator:
- Preserves original functionality
- Blends profit-focused and original rewards (70/30 default)
- Maintains backward compatibility

## Configuration

### Default Parameters
```json
{
  "profit_amplifier": 3.0,
  "loss_multiplier": 1.0,
  "trade_completion_bonus": 0.02,
  "opportunity_cost_penalty": 0.001,
  "momentum_window": 10,
  "regime_sensitivity": true
}
```

### Training Configuration
- Network: [256, 256, 256] (increased capacity)
- Learning rate: 5e-5 (faster adaptation)
- Batch size: 128
- Features: 15 (reduced from 41)

## Performance Metrics

The calculator tracks:
- **Win rate**: Percentage of profitable trades
- **Profit factor**: Gross profits / Gross losses
- **Average trade return**: Mean return per completed trade
- **Largest win/loss**: Risk metrics
- **Total trades**: Activity level
- **Holding time**: Average position duration

## Usage

### Basic Usage
```python
from reward_functions import create_reward_calculator

# Create profit-focused calculator
calc = create_reward_calculator(
    reward_type="profit_focused",
    device="cuda"
)
```

### Training Script
```bash
python train_profit_focused.py
```

### Configuration File
Edit `configs/profit_focused_config.json` to customize parameters.

## Expected Improvements

1. **Higher Returns**: 3x profit amplification should drive positive returns
2. **More Active Trading**: Reduced holding from 70% to ~40-50%
3. **Better Win Rate**: Focus on profitable trades
4. **Faster Learning**: Stronger profit signals accelerate learning

## Next Steps

1. **Phase 3**: Implement aggressive hyperparameters
2. **Phase 4**: Enhance market regime integration
3. **Phase 5**: Run HPO for optimal weight discovery

## Testing

Run test script to validate reward calculations:
```bash
python test_profit_rewards.py
```

The test covers:
- Basic profit/loss scenarios
- Trade completion bonuses
- Opportunity cost penalties
- Market regime adjustments
- Episode simulation

## Monitoring

During training, monitor:
- `profit_win_rate`: Should increase over time
- `profit_total_trades`: Should show active trading
- `profit_profit_factor`: Target > 1.5
- `reward_mean`: Should be positive on average