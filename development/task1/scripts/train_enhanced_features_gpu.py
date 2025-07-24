"""
Train Enhanced Features on GPU

Clean training script that ensures GPU usage and handles enhanced features correctly.
"""

import sys
import os
import torch
from task1_ensemble import run
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN

def main():
    """Train ensemble with enhanced features on GPU"""
    print("=" * 60)
    print("TRAINING ENHANCED FEATURES ON GPU")
    print("=" * 60)
    
    # Force GPU usage
    gpu_id = 0  # Use GPU 0
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, falling back to CPU")
        gpu_id = -1
    else:
        device_name = torch.cuda.get_device_name(gpu_id)
        print(f"✅ Using GPU {gpu_id}: {device_name}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f}GB")
    
    # Check enhanced features
    enhanced_path = "./data/raw/task1/BTC_1sec_predict_enhanced.npy"
    if os.path.exists(enhanced_path):
        import numpy as np
        enhanced_data = np.load(enhanced_path)
        print(f"✅ Enhanced features loaded: {enhanced_data.shape}")
        print(f"✅ State dimension will be: {enhanced_data.shape[1]}")
    else:
        print("❌ Enhanced features not found")
        return
    
    # Clean up any existing models that might have wrong dimensions
    old_dirs = [
        "TradeSimulator-v0_D3QN_0",
        "TradeSimulator-v0_D3QN_-1",
        "ensemble_teamname"
    ]
    
    for dir_path in old_dirs:
        if os.path.exists(dir_path):
            print(f"⚠️  Found existing directory: {dir_path}")
            print("   (Models may have old dimensions - training will create new ones)")
    
    # Set up agent list
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    
    # Create save path with GPU identifier
    save_path = f"ensemble_enhanced_gpu_{gpu_id}"
    
    print(f"\n🚀 Starting training...")
    print(f"   Save path: {save_path}")
    print(f"   GPU ID: {gpu_id}")
    print(f"   Agents: {[agent.__name__ for agent in agent_list]}")
    print(f"   Enhanced features: 16 dimensions")
    
    # Override sys.argv to pass GPU ID
    original_argv = sys.argv.copy()
    sys.argv = ['train_enhanced_features_gpu.py', str(gpu_id)]
    
    try:
        # Run training
        run(save_path=save_path, agent_list=agent_list, log_rules=True)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Models saved to: {save_path}")
        print("Enhanced features training with GPU acceleration complete!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()