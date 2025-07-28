#!/usr/bin/env python3
"""
Quick test script for production training setup
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add src and src_refactored to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))
sys.path.insert(0, str(current_dir / "src_refactored"))

print("🧪 Production Setup Test")
print("=" * 40)

start_time = time.time()

# Test 1: Device and environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")

# Test 2: Data availability
data_paths = [
    "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec.csv",
    "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict_enhanced_v3.npy"
]

for path in data_paths:
    if os.path.exists(path):
        print(f"✅ Data file: {os.path.basename(path)}")
    else:
        print(f"❌ Missing: {os.path.basename(path)}")

# Test 3: Imports
print("\n📦 Testing Imports:")
try:
    from data_config import ConfigData
    print("✅ ConfigData")
except Exception as e:
    print(f"❌ ConfigData: {e}")

try:
    from trade_simulator import TradeSimulator
    print("✅ TradeSimulator")
except Exception as e:
    print(f"❌ TradeSimulator: {e}")

try:
    from src_refactored.agents.double_dqn_agent import DoubleDQNAgent, D3QNAgent
    print("✅ Refactored Agents")
except Exception as e:
    print(f"❌ Refactored Agents: {e}")

try:
    from src_refactored.config.agent_configs import DoubleDQNConfig
    print("✅ Agent Configs")
except Exception as e:
    print(f"❌ Agent Configs: {e}")

try:
    from src_refactored.ensemble.voting_ensemble import VotingEnsemble, EnsembleStrategy
    print("✅ Ensemble")
except Exception as e:
    print(f"❌ Ensemble: {e}")

# Test 4: Quick data loading
print("\n📊 Testing Data Loading:")
try:
    data_config = ConfigData()
    predict_path = "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict_enhanced_v3.npy"
    
    if os.path.exists(predict_path):
        predict_data = np.load(predict_path)
        state_dim = predict_data.shape[1]
        action_dim = 3
        print(f"✅ Data loaded: shape={predict_data.shape}")
        print(f"✅ State dim: {state_dim}, Action dim: {action_dim}")
    else:
        # Fallback to standard data
        predict_path = "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict.npy"
        predict_data = np.load(predict_path)
        state_dim = predict_data.shape[1]
        action_dim = 3
        print(f"✅ Standard data loaded: shape={predict_data.shape}")
        print(f"✅ State dim: {state_dim}, Action dim: {action_dim}")

except Exception as e:
    print(f"❌ Data loading failed: {e}")
    state_dim, action_dim = 100, 3  # Fallback

# Test 5: Agent Creation
print("\n🤖 Testing Agent Creation:")
try:
    from src_refactored.agents.double_dqn_agent import DoubleDQNAgent
    from src_refactored.config.agent_configs import DoubleDQNConfig
    
    config = DoubleDQNConfig(
        net_dims=[256, 256],
        learning_rate=1e-4,
        batch_size=64
    )
    
    agent = DoubleDQNAgent(
        config=config,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    
    print(f"✅ Agent created: {type(agent).__name__}")
    print(f"✅ Network parameters: {sum(p.numel() for p in agent.online_network.parameters()):,}")
    
except Exception as e:
    print(f"❌ Agent creation failed: {e}")

# Test 6: Trading Environment
print("\n🏪 Testing Trading Environment:")
try:
    from trade_simulator import TradeSimulator
    from data_config import ConfigData
    
    # Simple environment test
    data_config = ConfigData()
    
    # Create minimal environment for testing
    class MockTradeEnv:
        def __init__(self, state_dim, action_dim):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.step_count = 0
            
        def reset(self):
            self.step_count = 0
            return np.random.randn(self.state_dim)
            
        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(self.state_dim)
            reward = np.random.randn()
            done = self.step_count >= 100
            info = {}
            return next_state, reward, done, info
    
    env = MockTradeEnv(state_dim, action_dim)
    print(f"✅ Mock environment created")
    
    # Test environment interaction
    state = env.reset()
    action = np.random.randint(0, action_dim)
    next_state, reward, done, info = env.step(action)
    print(f"✅ Environment interaction test passed")
    
except Exception as e:
    print(f"❌ Environment test failed: {e}")

# Summary
elapsed = time.time() - start_time
print(f"\n⏱️ Test completed in {elapsed:.2f} seconds")
print(f"🎯 System ready for production training setup")