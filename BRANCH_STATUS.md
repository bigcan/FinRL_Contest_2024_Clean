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
| Profit Rewards | ✅ Completed | 2025-07-30 | 2025-07-30 | Implemented 3x profit amp + speed multiplier (up to 5x for fast profits) |
| Aggressive Hyperparams | ✅ Completed | 2025-07-30 | 2025-07-30 | 7/7 aggressiveness score: [512,512,256,256] network, 10x LR increase |
| Market Regime | ✅ Completed | 2025-07-30 | 2025-07-30 | 9 regime classification, dynamic params, state integration (+12 features) |
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

## 📈 Phase 2 Results: Profit-Focused Rewards

### Key Achievements:
- ✅ Implemented asymmetric profit amplification (3x for profits, 1x for losses)
- ✅ Added trade completion bonuses for profitable exits
- ✅ Created opportunity cost penalties for excessive holding
- ✅ **NEW**: Implemented profit speed multiplier with exponential decay
- ✅ Integrated seamlessly with existing reward system

### Profit Speed Innovation:
The profit speed multiplier rewards faster profits exponentially:
- **Ultra-fast (5-10s)**: Up to 5x multiplier → 15x total reward
- **Fast (30s)**: ~2.7x multiplier → 8.2x total reward
- **Normal (60s)**: ~1.5x multiplier → 4.5x total reward
- **Slow (300s+)**: 1.0x multiplier → 3x total reward (base only)

This creates strong incentives for:
- Quick profit-taking over prolonged holding
- Efficient capital utilization
- Reduced exposure to market risk
- Higher trade frequency with positive expectancy

### Expected Impact:
- **Profit-seeking behavior**: 3x base + up to 5x speed = massive profit incentive
- **Reduced holding time**: From 70%+ holds to targeted 30-40%
- **Faster capital turnover**: Quick trades preferred over slow gains
- **Better risk management**: Less time in market = less risk exposure

## 📈 Phase 3 Results: Aggressive Hyperparameters

### Key Achievements:
- ✅ Designed ultra-aggressive configuration (7/7 aggressiveness score)
- ✅ Increased network capacity: [512, 512, 256, 256] (4 layers, 458K parameters)
- ✅ Boosted learning rate: 1e-4 (10x increase from baseline)
- ✅ Enhanced exploration: 0.15→0.005 with slower decay (0.99)
- ✅ Optimized training efficiency: 2x batch size, 2x horizon length

### Configuration Highlights:
- **Network**: Deep architecture for complex pattern recognition
- **Learning**: Fast adaptation with AdamW optimizer + cosine schedule
- **Rewards**: 5x profit amp + 7x speed = up to 35x for ultra-fast profits
- **Positions**: Max 3 units, stop-loss 3%, take-profit 5%
- **Training**: 100 episodes, 1.5M samples total

### Expected Performance Boost:
- **Convergence**: 2-3x faster due to higher learning rate
- **Capacity**: Can learn more complex profit patterns
- **Risk-taking**: Larger positions with safety limits
- **Adaptation**: Quick response to profitable opportunities

## 📈 Phase 4 Results: Market Regime Integration

### Key Achievements:
- ✅ Expanded to 9 detailed market regime classifications
- ✅ Multi-metric regime detection (7 metrics analyzed)
- ✅ Extended agent state with 12 regime features
- ✅ Dynamic parameter adjustment per regime
- ✅ Regime-aware reward modifications

### Regime-Specific Optimizations:
- **Trending**: 1.5x position size, 1.3x profit amp, directional bias
- **Ranging**: 0.5-0.8x position, 1.5-2.0x profit amp, quick trades
- **Breakout**: 2.0x position size, 8% take profit target
- **Choppy**: 0.7x position, increased exploration

### Expected Impact:
- **Adaptive Strategy**: Different behaviors for different markets
- **Risk Reduction**: Dynamic position sizing and stops
- **Profit Enhancement**: Regime-appropriate reward amplification
- **State Awareness**: Agent sees market context explicitly

## 🔍 Key Insights from Analysis

1. **Current Issue**: Agent minimizes losses but doesn't seek profits
2. **Root Cause**: 41 features create noise; conservative reward function
3. **Solution Path**: Reduce features → Reward profits → Tune aggressively
4. **Innovation**: Profit speed multiplier creates urgency for quick wins
5. **Acceleration**: Aggressive hyperparameters enable rapid profit learning
6. **Adaptation**: Market regime awareness for context-appropriate trading

## 📚 References

- [Gemini's Profitability Report](./PROFITABILITY_IMPROVEMENT_REPORT.md)
- [Current Training Framework](./development/task1/README.md)
- [Baseline Performance Metrics](./complete_production_results/)