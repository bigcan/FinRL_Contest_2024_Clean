"""
Test Extended Training - Quick validation run
Quick test of the extended training framework with reduced parameters
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from task1_ensemble_extended_training import run_extended_training
from erl_agent import AgentD3QN

def test_extended_training():
    """Test extended training with minimal configuration"""
    
    print("🧪 Testing Extended Training Framework")
    print("   Quick validation with minimal configuration")
    
    try:
        # Test with single agent and short training
        agent_list = [AgentD3QN]  # Single agent for testing
        
        result = run_extended_training(
            save_path="test_extended_training",
            agent_list=agent_list,
            max_episodes=10,  # Very short for testing  
            min_episodes=5    # Minimal episodes
        )
        
        if result:
            print("✅ Extended training test completed successfully\!")
            return True
        else:
            print("❌ Extended training test failed")
            return False
            
    except Exception as e:
        print(f"❌ Extended training test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_extended_training()
EOF < /dev/null
