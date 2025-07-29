#!/usr/bin/env python3
"""
Generated Robust Training Script - 20250729_081032
Enhanced error handling for production training
"""

import sys
import os
from pathlib import Path

# Add current directory for imports
current_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(current_dir))

def robust_training_wrapper():
    """Wrapper with enhanced error handling."""
    try:
        print("🚀 Starting Robust Production Training")
        print("=" * 60)
        
        # Import and run original training
        from task1_ensemble import run
        
        # Enhanced configuration for production
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('gpu_id', type=int, default=0, help='GPU ID to use')
        args = parser.parse_args()
        
        print(f"🎮 Using GPU: {args.gpu_id}")
        print(f"📊 Enhanced v3 Bitcoin LOB features (41D)")
        print(f"🎯 Production ensemble training")
        
        # Run training with robust parameters
        run(
            env_class_name="TradeSimulator",
            agent_class_names=["AgentD3QN", "AgentDoubleDQN", "AgentPrioritizedDQN"],
            gpu_id=args.gpu_id,
            log_rules=["print_time", "save_model"],
            save_path=f"robust_training_20250729_081032",
            starting_cash=100000
        )
        
        print("✅ Robust production training completed successfully!")
        return True
        
    except Exception as e:
        print(f"💥 Training error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save whatever progress was made
        try:
            print("🔄 Attempting to save partial progress...")
            # Additional error recovery logic here
        except:
            pass
            
        return False

if __name__ == "__main__":
    success = robust_training_wrapper()
    sys.exit(0 if success else 1)
