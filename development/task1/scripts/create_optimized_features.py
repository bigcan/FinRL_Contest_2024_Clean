"""
Create Optimized Feature Array
Generates optimized 8-feature array based on Phase 1 analysis results
"""

import numpy as np
import os
import sys

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.append(os.path.join(project_root, 'development', 'task1', 'src'))

def create_optimized_features():
    """Create optimized feature array from Phase 1 analysis"""
    print("🚀 Phase 2: Creating Optimized Features")
    print("=" * 50)
    
    # Load enhanced features
    data_path = os.path.join(project_root, 'data', 'raw', 'task1')
    enhanced_path = os.path.join(data_path, 'BTC_1sec_predict_enhanced.npy')
    
    if not os.path.exists(enhanced_path):
        raise FileNotFoundError(f"Enhanced features not found: {enhanced_path}")
    
    print(f"📂 Loading enhanced features from: {enhanced_path}")
    enhanced_features = np.load(enhanced_path)
    print(f"   Original shape: {enhanced_features.shape}")
    
    # Load metadata
    metadata_path = enhanced_path.replace('.npy', '_metadata.npy')
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        feature_names = metadata.get('feature_names', [])
    else:
        feature_names = [f'feature_{i}' for i in range(enhanced_features.shape[1])]
    
    print(f"   Original features: {feature_names}")
    
    # Optimal feature selection from Phase 1 analysis
    optimal_indices = [2, 4, 6, 7, 8, 9, 11, 14]  # ema_20, rsi_14, momentum_20, spread_norm, trade_imbalance, order_flow_5, original_0, original_4
    optimal_names = [
        'ema_20',           # Index 2 - exponential moving average  
        'rsi_14',           # Index 4 - relative strength index
        'momentum_20',      # Index 6 - medium-term momentum
        'spread_norm',      # Index 7 - bid-ask spread dynamics
        'trade_imbalance',  # Index 8 - buy vs sell pressure
        'order_flow_5',     # Index 9 - rolling order flow
        'original_0',       # Index 11 - highest importance (52%)
        'original_4'        # Index 14 - good importance, uncorrelated
    ]
    
    print(f"\n🎯 Selected Features ({len(optimal_indices)}):")
    for i, (idx, name) in enumerate(zip(optimal_indices, optimal_names)):
        print(f"   {i}: {name} (original index {idx})")
    
    # Create optimized feature array
    optimized_features = enhanced_features[:, optimal_indices]
    print(f"\n✅ Optimized shape: {optimized_features.shape}")
    print(f"   Reduction: {enhanced_features.shape[1]} → {optimized_features.shape[1]} features")
    
    # Save optimized features
    optimized_path = os.path.join(data_path, 'BTC_1sec_predict_optimized.npy')
    np.save(optimized_path, optimized_features)
    print(f"💾 Saved optimized features: {optimized_path}")
    
    # Save metadata
    optimized_metadata = {
        'feature_names': optimal_names,
        'feature_indices': optimal_indices,
        'original_shape': enhanced_features.shape,
        'optimized_shape': optimized_features.shape,
        'creation_date': '2025-07-24',
        'phase': 'Phase 2 - Model Architecture Upgrade',
        'selection_criteria': 'Phase 1 correlation + importance + ablation analysis',
        'expected_accuracy': '90.6%+',
        'removed_features': [
            'position_norm', 'holding_norm', 'ema_50', 'momentum_5', 
            'ema_crossover', 'original_1', 'original_2', 'original_5'
        ]
    }
    
    metadata_path = optimized_path.replace('.npy', '_metadata.npy')
    np.save(metadata_path, optimized_metadata)
    print(f"📋 Saved metadata: {metadata_path}")
    
    # Validation checks
    print(f"\n🔍 Validation Checks:")
    print(f"   ✓ No NaN values: {not np.isnan(optimized_features).any()}")
    print(f"   ✓ No infinite values: {not np.isinf(optimized_features).any()}")
    print(f"   ✓ Shape consistency: {optimized_features.shape[0] == enhanced_features.shape[0]}")
    print(f"   ✓ Feature count: {optimized_features.shape[1] == 8}")
    
    # Feature statistics
    print(f"\n📊 Feature Statistics:")
    for i, name in enumerate(optimal_names):
        feature_data = optimized_features[:, i]
        print(f"   {name}:")
        print(f"     Mean: {np.mean(feature_data):.4f}, Std: {np.std(feature_data):.4f}")
        print(f"     Range: [{np.min(feature_data):.4f}, {np.max(feature_data):.4f}]")
    
    print(f"\n🎉 Optimized features created successfully!")
    print(f"📈 Expected improvements:")
    print(f"   • 50% fewer features (16 → 8)")
    print(f"   • Higher signal-to-noise ratio")
    print(f"   • Faster training and inference")
    print(f"   • Better generalization")
    
    return optimized_path, optimal_names

def update_trade_simulator():
    """Update TradeSimulator to use optimized features"""
    print(f"\n🔧 Updating TradeSimulator Integration...")
    
    simulator_path = os.path.join(project_root, 'development', 'task1', 'src', 'trade_simulator.py')
    
    # Read current simulator
    with open(simulator_path, 'r') as f:
        content = f.read()
    
    # Check if already updated
    if 'BTC_1sec_predict_optimized.npy' in content:
        print("   ⚠️  TradeSimulator already updated for optimized features")
        return
    
    # Update the feature loading logic
    old_logic = '''        # Try to load enhanced features first
        enhanced_path = args.predict_ary_path.replace('.npy', '_enhanced.npy')
        if os.path.exists(enhanced_path):
            print(f"Loading enhanced features from {enhanced_path}")
            self.factor_ary = np.load(enhanced_path)
            
            # Load metadata for enhanced features
            metadata_path = enhanced_path.replace('.npy', '_metadata.npy')
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True).item()
                self.feature_names = metadata.get('feature_names', [])
                print(f"Enhanced features loaded: {len(self.feature_names)} features")
            else:
                self.feature_names = []
        else:
            print(f"Loading original features from {args.predict_ary_path}")
            self.factor_ary = np.load(args.predict_ary_path)
            self.feature_names = []'''
    
    new_logic = '''        # Try to load optimized features first (Phase 2)
        optimized_path = args.predict_ary_path.replace('.npy', '_optimized.npy')
        if os.path.exists(optimized_path):
            print(f"Loading optimized features from {optimized_path}")
            self.factor_ary = np.load(optimized_path)
            
            # Load metadata for optimized features
            metadata_path = optimized_path.replace('.npy', '_metadata.npy')
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True).item()
                self.feature_names = metadata.get('feature_names', [])
                print(f"Optimized features loaded: {len(self.feature_names)} features")
            else:
                self.feature_names = []
        else:
            # Fallback to enhanced features
            enhanced_path = args.predict_ary_path.replace('.npy', '_enhanced.npy')
            if os.path.exists(enhanced_path):
                print(f"Loading enhanced features from {enhanced_path}")
                self.factor_ary = np.load(enhanced_path)
                
                # Load metadata for enhanced features
                metadata_path = enhanced_path.replace('.npy', '_metadata.npy')
                if os.path.exists(metadata_path):
                    metadata = np.load(metadata_path, allow_pickle=True).item()
                    self.feature_names = metadata.get('feature_names', [])
                    print(f"Enhanced features loaded: {len(self.feature_names)} features")
                else:
                    self.feature_names = []
            else:
                print(f"Loading original features from {args.predict_ary_path}")
                self.factor_ary = np.load(args.predict_ary_path)
                self.feature_names = []'''
    
    # Replace the logic
    if old_logic in content:
        content = content.replace(old_logic, new_logic)
        
        # Save updated simulator
        with open(simulator_path, 'w') as f:
            f.write(content)
        
        print("   ✅ TradeSimulator updated to prioritize optimized features")
    else:
        print("   ⚠️  Could not find expected pattern in TradeSimulator - manual update needed")

def main():
    """Main execution"""
    try:
        # Create optimized features
        optimized_path, feature_names = create_optimized_features()
        
        # Update TradeSimulator
        update_trade_simulator()
        
        print(f"\n🎯 Phase 2 Setup Complete!")
        print(f"   📁 Optimized features: {os.path.basename(optimized_path)}")
        print(f"   🧠 State dimension: 8 (vs 16 original)")
        print(f"   ⚡ Ready for enhanced architecture training")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Update ensemble training configuration")
        print(f"   2. Use enhanced neural networks (128, 64, 32)")
        print(f"   3. Run quick validation test")
        print(f"   4. Full ensemble retraining")
        
    except Exception as e:
        print(f"❌ Error in Phase 2 setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()