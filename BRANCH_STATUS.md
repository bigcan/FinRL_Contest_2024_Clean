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
| Feature Engineering | ✅ Completed | 2025-07-30 | 2025-07-30 | Reduced from 41 to 15 features (63.4% reduction) |
| Profit Rewards | ✅ Completed | 2025-07-30 | 2025-07-30 | Implemented 3x profit amplification + trade bonuses |
| Aggressive Hyperparams | 🔄 In Progress | 2025-07-30 | - | Next: Larger networks, faster learning |
| Market Regime | ⏳ Pending | - | - | - |
| HPO Optimization | ⏳ Pending | - | - | - |

## 📈 Phase 1 Results: Feature Engineering

### Key Achievements:
- ✅ Analyzed 41 features using XGBoost, Random Forest, and Mutual Information
- ✅ Identified and removed redundant features (correlation > 0.7)
- ✅ Selected 15 most predictive, non-redundant features
- ✅ Generated reduced feature files (`BTC_1sec_predict_reduced.npy`)
- ✅ Created comprehensive documentation and visualizations

### Selected Features:
1. **existing_2** - Top importance score (0.5812 XGBoost)
2. **existing_5** - High predictive power
3. **existing_0** - Important base feature
4. **zscore_spread_z_score_60** - Spread dynamics
5. **regime_minus_di** - Market regime indicator
6. **regime_plus_di** - Directional movement
7. **transform_frac_diff** - Transformed feature
8. **micro_amihud_illiquidity** - Liquidity measure
9. **micro_kyles_lambda** - Price impact
10. **zscore_midpoint_z_score_60** - Price normalization
11. **zscore_midpoint_z_score_20** - Short-term price
12. **time_dow_cos** - Day of week cyclical
13. **vol_vol_persistence** - Volatility clustering
14. **time_hour_sin** - Hour of day cyclical
15. **regime_adx** - Trend strength

### Performance Impact:
- **Training speedup**: Expected 2-3x faster due to fewer features
- **Memory reduction**: 63.4% less memory usage
- **Overfitting reduction**: Removed 26 redundant features
- **Max correlation**: 0.686 (below 0.7 threshold)

## 🔍 Key Insights from Analysis

1. **Current Issue**: Agent minimizes losses but doesn't seek profits
2. **Root Cause**: 41 features create noise; conservative reward function
3. **Solution Path**: Reduce features → Reward profits → Tune aggressively

## 📚 References

- [Gemini's Profitability Report](./PROFITABILITY_IMPROVEMENT_REPORT.md)
- [Current Training Framework](./development/task1/README.md)
- [Baseline Performance Metrics](./complete_production_results/)