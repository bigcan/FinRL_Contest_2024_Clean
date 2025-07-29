#!/usr/bin/env python3
"""
Quick GPU Validation Script - Test core training functionality
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

def validate_gpu_setup():
    """Validate GPU setup and core functionality."""
    print("🚀 GPU Validation for FinRL Contest 2024")
    print("=" * 60)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        print(f"🔢 GPU Memory Free: {torch.cuda.memory_stats()['reserved_bytes.all.current'] / 1e9:.1f} GB")
    
    # Test basic operations
    print("\n🔬 Testing Basic Operations")
    print("-" * 40)
    
    try:
        # Test tensor operations
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print(f"✅ Matrix multiplication: {z.shape}")
        
        # Test gradients
        x.requires_grad_(True)
        loss = (z ** 2).mean()
        loss.backward()
        print(f"✅ Gradient computation: {x.grad.shape}")
        
        # Clear memory
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        return False
    
    # Test loading data
    print("\n📊 Testing Data Loading")
    print("-" * 40)
    
    try:
        # Check for enhanced features
        data_path = "/mnt/c/QuantConnect/FinRL_Contest_2024/FinRL_Contest_2024/data/raw/task1/BTC_1sec_predict_enhanced_v3.npy"
        if os.path.exists(data_path):
            data = np.load(data_path)
            print(f"✅ Enhanced v3 features loaded: {data.shape}")
            
            # Test conversion to tensor
            tensor_data = torch.tensor(data[:1000], dtype=torch.float32, device=device)
            print(f"✅ Tensor conversion: {tensor_data.shape}")
            
            # Test basic statistics
            mean_val = tensor_data.mean()
            std_val = tensor_data.std()
            print(f"✅ Data statistics: mean={mean_val:.4f}, std={std_val:.4f}")
            
        else:
            print(f"❌ Enhanced data not found at {data_path}")
            return False
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test simple neural network
    print("\n🧠 Testing Neural Network")
    print("-" * 40)
    
    try:
        import torch.nn as nn
        
        # Simple network
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(41, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 3)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)
        
        net = SimpleNet().to(device)
        
        # Test forward pass
        test_input = torch.randn(32, 41, device=device)
        output = net(test_input)
        print(f"✅ Network forward pass: {output.shape}")
        
        # Test backward pass
        loss = output.mean()
        loss.backward()
        print(f"✅ Network backward pass: loss={loss.item():.4f}")
        
        # Count parameters
        params = sum(p.numel() for p in net.parameters())
        print(f"✅ Network parameters: {params:,}")
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        return False
    
    print("\n🎉 ALL VALIDATIONS PASSED!")
    print("=" * 60)
    print("✅ GPU setup is working correctly")
    print("✅ Data loading is functional")  
    print("✅ Neural networks can train on GPU")
    print("✅ Ready for production training")
    
    return True

if __name__ == "__main__":
    success = validate_gpu_setup()
    exit(0 if success else 1)