# Feature Selection Report

Generated: 2025-07-30 18:52:52

## Executive Summary

- **Original Features**: 41
- **Selected Features**: 15
- **Reduction**: 63.4%

## Selected Features

The following 15 features were selected based on:
1. High predictive power (XGBoost, Random Forest, Mutual Information)
2. Low redundancy (correlation < 0.7 with other selected features)
3. Coverage across key categories (microstructure, price action, volume, position)

### Final Feature Set:

1. **existing_2** (score: 1.0000)
2. **existing_5** (score: 0.2073)
3. **existing_0** (score: 0.1494)
4. **zscore_spread_z_score_60** (score: 0.1248)
5. **regime_minus_di** (score: 0.1168)
6. **regime_plus_di** (score: 0.1002)
7. **transform_frac_diff** (score: 0.0645)
8. **micro_amihud_illiquidity** (score: 0.0356)
9. **micro_kyles_lambda** (score: 0.0187)
10. **zscore_midpoint_z_score_60** (score: 0.0178)
11. **zscore_midpoint_z_score_20** (score: 0.0177)
12. **time_dow_cos** (score: 0.0165)
13. **vol_vol_persistence** (score: 0.0154)
14. **time_hour_sin** (score: 0.0147)
15. **regime_adx** (score: 0.0138)

## Correlation Analysis

Maximum pairwise correlation among selected features: 0.686

## Feature Importance Rankings

### XGBoost

| Rank | Feature | Score |
|------|---------|-------|
| 1 | existing_2 ✓ | 0.5812 |
| 2 | existing_3  | 0.1461 |
| 3 | time_dow_cos ✓ | 0.0163 |
| 4 | existing_0 ✓ | 0.0160 |
| 5 | existing_4  | 0.0153 |
| 6 | transform_frac_diff ✓ | 0.0133 |
| 7 | time_hour_cos  | 0.0132 |
| 8 | time_hour_sin ✓ | 0.0128 |
| 9 | regime_minus_di ✓ | 0.0117 |
| 10 | existing_6  | 0.0116 |
| 11 | time_dow_sin  | 0.0114 |
| 12 | regime_plus_di ✓ | 0.0112 |
| 13 | regime_adx ✓ | 0.0110 |
| 14 | existing_5 ✓ | 0.0109 |
| 15 | existing_7  | 0.0108 |
| 16 | volatility_garch  | 0.0102 |
| 17 | micro_kyles_lambda ✓ | 0.0099 |
| 18 | zscore_spread_z_score_60 ✓ | 0.0097 |
| 19 | existing_1  | 0.0092 |
| 20 | zscore_midpoint_z_score_60 ✓ | 0.0083 |

### RandomForest

| Rank | Feature | Score |
|------|---------|-------|
| 1 | existing_2 ✓ | 0.2674 |
| 2 | existing_3  | 0.2420 |
| 3 | existing_4  | 0.1300 |
| 4 | existing_1  | 0.1011 |
| 5 | existing_0 ✓ | 0.0450 |
| 6 | transform_frac_diff ✓ | 0.0418 |
| 7 | existing_5 ✓ | 0.0398 |
| 8 | existing_6  | 0.0310 |
| 9 | existing_7  | 0.0157 |
| 10 | micro_kyles_lambda ✓ | 0.0069 |
| 11 | volatility_garch  | 0.0062 |
| 12 | regime_adx ✓ | 0.0060 |
| 13 | regime_minus_di ✓ | 0.0053 |
| 14 | zscore_spread_z_score_60 ✓ | 0.0053 |
| 15 | regime_plus_di ✓ | 0.0053 |
| 16 | zscore_midpoint_z_score_60 ✓ | 0.0052 |
| 17 | vol_vol_persistence ✓ | 0.0051 |
| 18 | zscore_midpoint_z_score_20 ✓ | 0.0042 |
| 19 | micro_amihud_illiquidity ✓ | 0.0042 |
| 20 | zscore_spread_z_score_20  | 0.0039 |

### MutualInformation

| Rank | Feature | Score |
|------|---------|-------|
| 1 | existing_2 ✓ | 0.1715 |
| 2 | existing_3  | 0.1638 |
| 3 | existing_4  | 0.1099 |
| 4 | existing_1  | 0.1026 |
| 5 | existing_5 ✓ | 0.0779 |
| 6 | zscore_spread_z_score_60 ✓ | 0.0580 |
| 7 | regime_minus_di ✓ | 0.0532 |
| 8 | regime_plus_di ✓ | 0.0448 |
| 9 | existing_0 ✓ | 0.0432 |
| 10 | existing_6  | 0.0416 |
| 11 | existing_7  | 0.0198 |
| 12 | zscore_spread_z_score_20  | 0.0169 |
| 13 | micro_amihud_illiquidity ✓ | 0.0132 |
| 14 | zscore_midpoint_z_score_20 ✓ | 0.0042 |
| 15 | vol_vol_regime_high  | 0.0040 |
| 16 | micro_order_arrival_imbalance  | 0.0037 |
| 17 | vol_vol_regime_medium  | 0.0035 |
| 18 | zscore_midpoint_z_score_60 ✓ | 0.0034 |
| 19 | vol_vol_persistence ✓ | 0.0032 |
| 20 | micro_ask_cancellation_rate  | 0.0028 |

