"""
Check Enhanced Features Status

Quick status check of enhanced features implementation and training.
"""

import os
import numpy as np
from trade_simulator import TradeSimulator

def main():
    """Check enhanced features status"""
    print("=" * 60)
    print("ENHANCED FEATURES STATUS CHECK")
    print("=" * 60)
    
    # 1. Check enhanced features exist
    enhanced_path = "./data/raw/task1/BTC_1sec_predict_enhanced.npy"
    original_path = "./data/raw/task1/BTC_1sec_predict.npy"
    
    print("1. FEATURE FILES STATUS:")
    if os.path.exists(enhanced_path):
        enhanced_data = np.load(enhanced_path)
        print(f"   ✓ Enhanced features: {enhanced_data.shape}")
    else:
        print("   ❌ Enhanced features not found")
        return
    
    if os.path.exists(original_path):
        original_data = np.load(original_path)
        print(f"   ✓ Original features: {original_data.shape}")
    else:
        print("   ❌ Original features not found")
    
    # 2. Check TradeSimulator detection
    print("\n2. SIMULATOR STATUS:")
    sim = TradeSimulator(num_sims=1)
    print(f"   ✓ Auto-detected state_dim: {sim.state_dim}")
    print(f"   ✓ Feature names loaded: {len(sim.feature_names)}")
    
    if sim.state_dim == 16:
        print("   ✅ ENHANCED FEATURES ACTIVE")
    elif sim.state_dim == 10:
        print("   ⚠️  Using original features (enhanced not detected)")
    else:
        print(f"   ❓ Unexpected state_dim: {sim.state_dim}")
    
    # 3. Check training directories
    print("\n3. TRAINING STATUS:")
    training_dirs = [
        "TradeSimulator-v0_D3QN_-1",
        "ensemble_teamname",
        "ensemble_teamname/ensemble_models"
    ]
    
    for dir_path in training_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"   ✓ {dir_path}/ ({len(files)} items)")
        else:
            print(f"   - {dir_path}/ (not found)")
    
    # 4. Feature comparison
    print("\n4. FEATURE IMPROVEMENT:")
    print(f"   Original dimensions: {original_data.shape[1]}")
    print(f"   Enhanced dimensions: {enhanced_data.shape[1]}")
    print(f"   Improvement factor: {enhanced_data.shape[1] / original_data.shape[1]:.1f}x")
    
    # 5. Expected benefits
    print("\n5. EXPECTED BENEFITS:")
    if sim.state_dim == 16:
        print("   ✅ Technical indicators (trend detection)")
        print("   ✅ LOB features (market microstructure)")
        print("   ✅ Selected original features (best predictors)")
        print("   ✅ Normalized feature values")
        print("   ✅ Ready for improved trading performance")
    else:
        print("   ⚠️  Enhanced features not active")
    
    # 6. Next steps
    print("\n6. CURRENT STATUS:")
    if sim.state_dim == 16:
        print("   🎯 Enhanced features successfully implemented")
        print("   🚀 Training can use 16 features instead of 10")
        print("   📈 Expected: Better trend detection and trading signals")
        print("   ⏳ Run training to see performance improvements")
    else:
        print("   ⚠️  Enhanced features not being used")
        print("   🔧 Check enhanced feature file exists and is accessible")
    
    print("\n" + "=" * 60)
    print("STATUS CHECK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()