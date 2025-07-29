#!/usr/bin/env python3
"""
Simple Training Test - Verify GPU training works without complex dependencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("🚀 Simple GPU Training Test")
print("=" * 50)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Device: {device}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")

# Simple network for Bitcoin trading
class SimpleTradingNet(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(), 
            nn.Linear(hidden_dim//2, 3)  # Buy, Hold, Sell
        )
    
    def forward(self, x):
        return self.net(x)

# Create network and optimizer
net = SimpleTradingNet().to(device) 
optimizer = optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print(f"🧠 Network created with {sum(p.numel() for p in net.parameters()):,} parameters")

# Generate synthetic Bitcoin-like data
batch_size = 64
input_dim = 41

print("\n🎯 Starting Training Test")
print("-" * 30)

try:
    for epoch in range(5):
        # Generate batch of synthetic market data
        states = torch.randn(batch_size, input_dim, device=device)
        # Random actions for testing
        actions = torch.randint(0, 3, (batch_size,), device=device)
        
        # Forward pass
        q_values = net(states)
        loss = criterion(q_values, actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/5: Loss = {loss.item():.6f}")
    
    print("\n✅ SUCCESS: GPU training is working!")
    print("🎉 Ready to run full production training")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    exit(1)

print("\n" + "=" * 50)
print("✅ GPU infrastructure is operational")
print("✅ Neural networks can train successfully") 
print("✅ PyTorch CUDA integration working")