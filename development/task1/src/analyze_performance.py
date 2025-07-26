#!/usr/bin/env python3
"""
Analyze performance of trial 5 models to understand why Sharpe ratio is so low
"""
import numpy as np
import matplotlib.pyplot as plt

def analyze_trial_performance():
    """Analyze the performance data from trial 5 evaluation"""
    
    try:
        # Load trial 5 evaluation results
        net_assets = np.load('evaluation_net_assets.npy')
        positions = np.load('evaluation_positions.npy')
        btc_positions = np.load('evaluation_btc_positions.npy')
        correct_predictions = np.load('evaluation_correct_predictions.npy')
        
        print("="*60)
        print("TRIAL 5 PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Basic performance metrics
        print(f"Net assets trajectory:")
        print(f"  Starting value: ${net_assets[0]:,.2f}")
        print(f"  Final value: ${net_assets[-1]:,.2f}")
        print(f"  Total return: {((net_assets[-1] / net_assets[0]) - 1) * 100:.4f}%")
        print(f"  Max value: ${net_assets.max():,.2f}")
        print(f"  Min value: ${net_assets.min():,.2f}")
        
        # Return statistics
        returns = np.diff(net_assets) / net_assets[:-1]
        print(f"\nReturn Statistics:")
        print(f"  Mean return: {returns.mean():.8f}")
        print(f"  Return volatility: {returns.std():.8f}")
        print(f"  Sharpe ratio (approx): {returns.mean() / returns.std() if returns.std() > 0 else 'N/A':.6f}")
        
        # Trading activity
        position_changes = np.diff(positions.flatten())
        n_trades = np.count_nonzero(position_changes)
        print(f"\nTrading Activity:")
        print(f"  Number of position changes: {n_trades}")
        print(f"  Trading frequency: {n_trades / len(positions):.4f}")
        
        # Position analysis
        unique_positions, position_counts = np.unique(positions, return_counts=True)
        print(f"\nPosition Distribution:")
        for pos, count in zip(unique_positions, position_counts):
            print(f"  Position {pos:.1f}: {count} steps ({count/len(positions)*100:.1f}%)")
            
        # Prediction accuracy
        correct_preds = correct_predictions[correct_predictions != 0]
        if len(correct_preds) > 0:
            accuracy = (correct_preds > 0).sum() / len(correct_preds)
            print(f"\nPrediction Accuracy:")
            print(f"  Prediction accuracy: {accuracy:.4f}")
            print(f"  Total predictions made: {len(correct_preds)}")
        
        # Identify the problem
        print(f"\n" + "="*60)
        print("PROBLEM DIAGNOSIS")
        print("="*60)
        
        # Check if agent is actually trading
        if n_trades < 10:
            print("❌ PROBLEM: Very low trading activity")
            print("   - Agent is barely trading, staying mostly in cash")
            print("   - This explains the very low Sharpe ratio")
        
        # Check return magnitude
        total_return_pct = ((net_assets[-1] / net_assets[0]) - 1) * 100
        if abs(total_return_pct) < 0.1:
            print("❌ PROBLEM: Extremely small returns")
            print(f"   - Total return of {total_return_pct:.4f}% is essentially flat")
            print("   - Agent is not generating meaningful alpha")
        
        # Check volatility
        if returns.std() > abs(returns.mean()) * 100:
            print("❌ PROBLEM: High volatility relative to returns")
            print("   - Volatility is much higher than mean return")
            print("   - This severely hurts the Sharpe ratio")
            
        return {
            'net_assets': net_assets,
            'returns': returns,
            'total_return_pct': total_return_pct,
            'n_trades': n_trades,
            'sharpe_approx': returns.mean() / returns.std() if returns.std() > 0 else 0
        }
        
    except Exception as e:
        print(f"Error analyzing performance: {e}")
        return None

if __name__ == "__main__":
    results = analyze_trial_performance()