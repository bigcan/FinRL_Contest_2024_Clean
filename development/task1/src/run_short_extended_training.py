#!/usr/bin/env python3
"""
Short Extended Training - Test run with reduced parameters
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def main():
    """Run short extended training test"""
    
    print("🚀 Short Extended Training Test")
    print("   Testing with reduced parameters for validation")
    
    try:
        from task1_ensemble_extended_training import run_extended_training
        from erl_agent import AgentD3QN, AgentDoubleDQN
        
        # Test with two agents and short training
        agent_list = [AgentD3QN, AgentDoubleDQN]
        
        print("📊 Configuration:")
        print(f"   Agents: {len(agent_list)}")
        print(f"   Max Episodes: 20")
        print(f"   Min Episodes: 10")
        print()
        
        # Run extended training
        results = run_extended_training(
            save_path="test_extended_short",
            agent_list=agent_list,
            max_episodes=20,  # Short for testing
            min_episodes=10   # Minimal episodes before early stopping
        )
        
        if results:
            print("🎉 Short extended training completed successfully!")
            
            # Print basic results
            for i, agent in enumerate(results):
                agent_name = agent_list[i].__name__
                episodes = getattr(agent, 'episodes_trained', 'unknown')
                final_perf = getattr(agent, 'final_performance', 'unknown')
                print(f"   {agent_name}: {episodes} episodes, final score: {final_perf}")
            
            return True
        else:
            print("❌ Extended training returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Extended training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()