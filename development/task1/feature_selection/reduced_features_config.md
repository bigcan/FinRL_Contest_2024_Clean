
# Configuration Update for Reduced Features

To use the reduced feature set, update the following in your training scripts:

## In trade_simulator.py:

Replace the feature loading priority with:
```python
# Priority loading: reduced > enhanced_v3 > optimized > enhanced > original
reduced_path = args.predict_ary_path.replace('.npy', '_reduced.npy')

if os.path.exists(reduced_path):
    print(f"Loading reduced features from {reduced_path}")
    self.factor_ary = np.load(reduced_path)
    
    # Load metadata for reduced features
    metadata_path = reduced_path.replace('.npy', '_metadata.npy')
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.feature_names = metadata.get('feature_names', [])
        print(f"Reduced features loaded: {len(self.feature_names)} features")
```

## Selected Features (15):
- existing_2
- existing_5
- existing_0
- zscore_spread_z_score_60
- regime_minus_di
- regime_plus_di
- transform_frac_diff
- micro_amihud_illiquidity
- micro_kyles_lambda
- zscore_midpoint_z_score_60
- zscore_midpoint_z_score_20
- time_dow_cos
- vol_vol_persistence
- time_hour_sin
- regime_adx

## Performance Benefits:
- Reduced training time (fewer features)
- Less overfitting (removed redundant features)
- Better generalization (focused on most predictive features)
- Lower memory usage
