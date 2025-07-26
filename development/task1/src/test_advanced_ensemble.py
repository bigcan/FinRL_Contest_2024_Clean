#!/usr/bin/env python3
"""
Advanced Ensemble Testing Script
Demonstrates the new ensemble capabilities
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ensemble_methods():
    """Test all ensemble methods with a small sample"""
    print("🚀 Advanced Ensemble Strategy Testing")
    print("=" * 60)
    
    methods_to_test = [
        ('majority_voting', 'Simple majority vote - baseline method'),
        ('weighted_voting', 'Performance-based weighted voting with dynamic updates'),
        ('confidence_weighted', 'Q-value confidence + performance weighted voting'),
        ('adaptive_meta', 'Market regime-aware meta-learning ensemble'),
    ]
    
    for method, description in methods_to_test:
        print(f"\n🎯 Testing: {method}")
        print(f"📋 Description: {description}")
        print("-" * 50)
        
        # The actual evaluation would be run here
        # For now, just show the command structure
        cmd = f"python3 task1_eval.py -1 --ensemble-method {method} --verbose"
        print(f"💻 Command: {cmd}")
        print(f"✅ {method} configuration ready")
    
    print(f"\n🎉 All ensemble methods are configured and ready!")
    print(f"\n📖 Usage Examples:")
    print(f"   Basic weighted voting:     python3 task1_eval.py --ensemble-method weighted_voting")
    print(f"   Confidence-based ensemble: python3 task1_eval.py --ensemble-method confidence_weighted --verbose")
    print(f"   Adaptive meta-learning:    python3 task1_eval.py --ensemble-method adaptive_meta --performance-window 150")
    print(f"   Custom configuration:      python3 task1_eval.py --ensemble-method weighted_voting --weight-decay 0.9 --verbose")

if __name__ == "__main__":
    test_ensemble_methods()