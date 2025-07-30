# Profitability Enhancement Branch

**Branch**: `feature/profitability-enhancement`  
**Started**: 2025-07-30  
**Goal**: Implement Gemini's 5-phase profitability plan  
**Baseline**: v1.0-stable-baseline  

## 📋 Implementation Plan

### Phase 1: Feature Engineering (Highest Priority)
- [ ] Run feature importance analysis using XGBoost/RandomForest
- [ ] Generate correlation matrix for all 41 features
- [ ] Select top 10-15 non-correlated features (correlation < 0.7)
- [ ] Create new reduced feature set configuration

### Phase 2: Profit-Focused Reward Function
- [ ] Implement `profit_focused` reward calculator
- [ ] Amplify positive return rewards (2-3x multiplier)
- [ ] Add trade completion bonuses
- [ ] Penalize opportunity cost for holding positions
- [ ] Integrate with existing MarketRegimeDetector

### Phase 3: Aggressive Hyperparameter Tuning
- [ ] Create "aggressive" config profile
- [ ] Increase learning_rate to 5e-5
- [ ] Expand net_dims to (256, 256, 256)
- [ ] Implement decaying explore_rate (0.1 → 0.01)
- [ ] Adjust batch_size and horizon_len for faster adaptation

### Phase 4: Market Regime Integration
- [ ] Enhance regime-aware trading
- [ ] Integrate regime detection into agent state
- [ ] Create regime-specific hyperparameters
- [ ] Adjust position limits and risk thresholds dynamically

### Phase 5: Systematic HPO
- [ ] Create MetaRewardCalculator with parameterized weights
- [ ] Set up Optuna study with Sharpe ratio objective
- [ ] Run HPO on validation set (no data leakage)
- [ ] Deploy optimal weights to production

## 🎯 Success Criteria

1. **Primary**: Achieve positive returns (profit > 0)
2. **Secondary**: Improve Sharpe ratio > 1.0
3. **Tertiary**: Reduce maximum drawdown < 15%

## 📊 Tracking Progress

| Phase | Status | Start Date | Completion | Notes |
|-------|--------|------------|------------|-------|
| Feature Engineering | 🔄 In Progress | 2025-07-30 | - | Starting with feature analysis |
| Profit Rewards | ⏳ Pending | - | - | - |
| Aggressive Hyperparams | ⏳ Pending | - | - | - |
| Market Regime | ⏳ Pending | - | - | - |
| HPO Optimization | ⏳ Pending | - | - | - |

## 🔍 Key Insights from Analysis

1. **Current Issue**: Agent minimizes losses but doesn't seek profits
2. **Root Cause**: 41 features create noise; conservative reward function
3. **Solution Path**: Reduce features → Reward profits → Tune aggressively

## 📚 References

- [Gemini's Profitability Report](./PROFITABILITY_IMPROVEMENT_REPORT.md)
- [Current Training Framework](./development/task1/README.md)
- [Baseline Performance Metrics](./complete_production_results/)