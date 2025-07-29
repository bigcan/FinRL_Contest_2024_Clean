#!/usr/bin/env python3
"""
Quick debug test for device consistency issue
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from task1_ensemble import run, AgentD3QN

# Quick 2-episode test to debug device issue
debug_config = {
    'gpu_id': 0,
    'num_sims': 64,  # Original value
    'num_episodes': 2,  # Just 2 episodes to catch the error
    'data_length': 10000,  # Original value
    'break_step': 20000,  # Total steps for 2 episodes
    'batch_size': 512,  # Original value
}

print("🔍 Running 2-episode device debug test...")
print(f"Configuration: {debug_config}")

try:
    run(
        save_path='device_debug_test',
        agent_list=[AgentD3QN],  # Single agent for clarity
        log_rules=False,
        config_dict=debug_config
    )
    print("✅ Episodes completed successfully!")
except Exception as e:
    print(f"❌ Error caught: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()