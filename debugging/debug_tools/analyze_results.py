#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Load evaluation results
net_assets = np.load('evaluation_net_assets.npy')
btc_positions = np.load('evaluation_btc_positions.npy')
correct_predictions = np.load('evaluation_correct_predictions.npy')

print(f'Net assets shape: {net_assets.shape}')
print(f'Starting cash: ${net_assets[0]:,.0f}')
print(f'Final net assets: ${net_assets[-1]:,.0f}')
print(f'Total return: {(net_assets[-1] / net_assets[0] - 1) * 100:.2f}%')

# Calculate returns
returns = np.diff(net_assets) / net_assets[:-1]
print(f'Number of positive returns: {(returns > 0).sum()}')
print(f'Number of negative returns: {(returns < 0).sum()}')
print(f'Number of zero returns: {(returns == 0).sum()}')

# Win rate from predictions
nonzero_predictions = correct_predictions[correct_predictions != 0]
win_rate = (nonzero_predictions > 0).sum() / len(nonzero_predictions) if len(nonzero_predictions) > 0 else 0
print(f'Prediction accuracy: {win_rate:.2%}')

print(f'Max net assets: ${net_assets.max():,.0f}')
print(f'Min net assets: ${net_assets.min():,.0f}')

# Check why Sharpe ratio is infinite
returns_std = returns.std()
returns_mean = returns.mean()
print(f'Average return per step: {returns_mean:.6f}')
print(f'Return standard deviation: {returns_std:.6f}')
print(f'Sharpe ratio calculation: {returns_mean / returns_std if returns_std > 0 else "inf (std=0)"}')

# Check for constant portfolio value
unique_assets = np.unique(net_assets)
print(f'Number of unique asset values: {len(unique_assets)}')
if len(unique_assets) <= 5:
    print(f'Unique values: {unique_assets}')