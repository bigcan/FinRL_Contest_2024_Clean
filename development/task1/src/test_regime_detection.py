"""
Test script for advanced market regime detection
Validates regime classification and parameter adjustment
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from advanced_market_regime import (
    AdvancedMarketRegimeDetector, 
    MarketRegime,
    RegimeAwareEnvironment
)

def generate_synthetic_market_data(n_points=1000):
    """Generate synthetic market data with different regimes"""
    
    # Initialize arrays
    prices = np.zeros(n_points)
    volumes = np.zeros(n_points)
    regimes_true = []
    
    # Starting price
    price = 100.0
    
    # Generate data with regime switches
    regime_lengths = [200, 150, 250, 200, 200]
    regime_types = [
        "uptrend", "ranging_low", "downtrend", "ranging_high", "breakout"
    ]
    
    idx = 0
    for length, regime in zip(regime_lengths, regime_types):
        end_idx = min(idx + length, n_points)
        
        if regime == "uptrend":
            # Strong uptrend with low volatility
            trend = np.linspace(0, 0.1, end_idx - idx)
            noise = np.random.normal(0, 0.002, end_idx - idx)
            prices[idx:end_idx] = price * (1 + trend + noise).cumprod()
            volumes[idx:end_idx] = np.random.normal(1.0, 0.2, end_idx - idx)
            regimes_true.extend([MarketRegime.STRONG_UPTREND] * (end_idx - idx))
            
        elif regime == "ranging_low":
            # Low volatility ranging
            for i in range(idx, end_idx):
                prices[i] = price + np.sin((i - idx) * 0.1) * 0.5 + np.random.normal(0, 0.1)
            volumes[idx:end_idx] = np.random.normal(0.8, 0.1, end_idx - idx)
            regimes_true.extend([MarketRegime.RANGING_LOW_VOL] * (end_idx - idx))
            
        elif regime == "downtrend":
            # Strong downtrend
            trend = np.linspace(0, -0.08, end_idx - idx)
            noise = np.random.normal(0, 0.003, end_idx - idx)
            prices[idx:end_idx] = price * (1 + trend + noise).cumprod()
            volumes[idx:end_idx] = np.random.normal(1.2, 0.3, end_idx - idx)
            regimes_true.extend([MarketRegime.STRONG_DOWNTREND] * (end_idx - idx))
            
        elif regime == "ranging_high":
            # High volatility ranging
            for i in range(idx, end_idx):
                prices[i] = price + np.sin((i - idx) * 0.05) * 2 + np.random.normal(0, 0.5)
            volumes[idx:end_idx] = np.random.normal(1.5, 0.4, end_idx - idx)
            regimes_true.extend([MarketRegime.RANGING_HIGH_VOL] * (end_idx - idx))
            
        elif regime == "breakout":
            # Breakout pattern
            consolidation = int(length * 0.6)
            breakout = length - consolidation
            
            # Consolidation phase
            for i in range(idx, idx + consolidation):
                prices[i] = price + np.random.normal(0, 0.2)
            volumes[idx:idx+consolidation] = np.random.normal(0.7, 0.1, consolidation)
            
            # Breakout phase
            trend = np.linspace(0, 0.15, breakout)
            noise = np.random.normal(0, 0.004, breakout)
            prices[idx+consolidation:end_idx] = prices[idx+consolidation-1] * (1 + trend + noise).cumprod()
            volumes[idx+consolidation:end_idx] = np.random.normal(2.0, 0.5, breakout)
            
            regimes_true.extend([MarketRegime.RANGING_LOW_VOL] * consolidation)
            regimes_true.extend([MarketRegime.BREAKOUT] * breakout)
        
        price = prices[end_idx - 1]
        idx = end_idx
    
    return prices, volumes, regimes_true

def test_regime_detection():
    """Test regime detection accuracy"""
    print("="*60)
    print("Testing Advanced Market Regime Detection")
    print("="*60)
    
    # Generate synthetic data
    prices, volumes, true_regimes = generate_synthetic_market_data(1000)
    
    # Initialize detector
    detector = AdvancedMarketRegimeDetector(
        short_lookback=20,
        medium_lookback=50,
        long_lookback=100
    )
    
    # Detect regimes
    detected_regimes = []
    confidences = []
    regime_params = []
    
    print("\nProcessing market data...")
    for i in range(len(prices)):
        result = detector.detect_regime(prices[i], volumes[i])
        detected_regimes.append(result["regime"])
        confidences.append(result["confidence"])
        regime_params.append(result["parameters"])
        
        # Print regime changes
        if i > 0 and detected_regimes[i] != detected_regimes[i-1]:
            print(f"Step {i}: Regime change detected: {detected_regimes[i-1].value} → {detected_regimes[i].value} (confidence: {result['confidence']:.2f})")
    
    # Calculate accuracy (after warmup period)
    warmup = 100
    if len(true_regimes) > warmup:
        matches = sum(1 for i in range(warmup, min(len(detected_regimes), len(true_regimes)))
                     if detected_regimes[i] == true_regimes[i])
        accuracy = matches / (min(len(detected_regimes), len(true_regimes)) - warmup)
        print(f"\nRegime detection accuracy (after warmup): {accuracy:.2%}")
    
    # Analyze regime parameters
    print("\nRegime-specific parameters:")
    unique_regimes = set(detected_regimes[warmup:])
    for regime in unique_regimes:
        params = detector.get_regime_parameters(regime)
        print(f"\n{regime.value}:")
        print(f"  Position size multiplier: {params['position_size_multiplier']}")
        print(f"  Stop loss: {params['stop_loss']:.3f}")
        print(f"  Take profit: {params['take_profit']:.3f}")
        print(f"  Profit amplifier: {params['profit_amplifier']}")
    
    # Plot results
    plot_regime_detection(prices, detected_regimes, confidences, true_regimes)
    
    return detector, detected_regimes, confidences

def plot_regime_detection(prices, detected_regimes, confidences, true_regimes=None):
    """Plot regime detection results"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Price with regime colors
        ax1 = axes[0]
        
        # Define colors for each regime
        regime_colors = {
            MarketRegime.STRONG_UPTREND: 'darkgreen',
            MarketRegime.WEAK_UPTREND: 'lightgreen',
            MarketRegime.STRONG_DOWNTREND: 'darkred',
            MarketRegime.WEAK_DOWNTREND: 'lightcoral',
            MarketRegime.RANGING_HIGH_VOL: 'orange',
            MarketRegime.RANGING_LOW_VOL: 'yellow',
            MarketRegime.BREAKOUT: 'blue',
            MarketRegime.BREAKDOWN: 'purple',
            MarketRegime.CHOPPY: 'gray'
        }
        
        # Plot price
        ax1.plot(prices, 'k-', alpha=0.7, linewidth=1)
        ax1.set_ylabel('Price')
        ax1.set_title('Price with Detected Regimes')
        ax1.grid(True, alpha=0.3)
        
        # Color background by regime
        for i in range(1, len(detected_regimes)):
            if detected_regimes[i] != detected_regimes[i-1] or i == 1:
                # Find end of this regime
                end = i + 1
                while end < len(detected_regimes) and detected_regimes[end] == detected_regimes[i]:
                    end += 1
                
                color = regime_colors.get(detected_regimes[i], 'gray')
                ax1.axvspan(i, end, alpha=0.2, color=color)
        
        # Plot 2: Regime confidence
        ax2 = axes[1]
        ax2.plot(confidences, 'b-', label='Confidence')
        ax2.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='High confidence threshold')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regime timeline
        ax3 = axes[2]
        
        # Convert regimes to numeric values
        regime_values = [list(MarketRegime).index(r) for r in detected_regimes]
        ax3.plot(regime_values, 'g-', label='Detected', alpha=0.7)
        
        if true_regimes:
            true_values = [list(MarketRegime).index(r) for r in true_regimes[:len(detected_regimes)]]
            ax3.plot(true_values, 'r--', label='True', alpha=0.7)
        
        ax3.set_ylabel('Regime Index')
        ax3.set_xlabel('Time Step')
        ax3.set_yticks(range(len(MarketRegime)))
        ax3.set_yticklabels([r.value for r in MarketRegime], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path(__file__).parent.parent / "regime_analysis"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "regime_detection_test.png", dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_dir / 'regime_detection_test.png'}")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"\nError generating plot: {e}")

def test_regime_features():
    """Test regime feature generation for agent state"""
    print("\n" + "="*60)
    print("Testing Regime Feature Generation")
    print("="*60)
    
    detector = AdvancedMarketRegimeDetector()
    
    # Simulate some market data
    for i in range(150):
        price = 100 + np.sin(i * 0.1) * 5 + np.random.normal(0, 0.5)
        volume = 1.0 + np.random.normal(0, 0.2)
        
        result = detector.detect_regime(price, volume)
        
        if i % 50 == 0:
            features = detector.get_regime_features()
            print(f"\nStep {i}:")
            print(f"  Regime: {result['regime'].value}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Feature shape: {features.shape}")
            print(f"  Feature values: {features}")
            print(f"  Non-zero features: {np.sum(features != 0)}")

def test_regime_parameters():
    """Test regime-specific parameter retrieval"""
    print("\n" + "="*60)
    print("Testing Regime Parameter Configuration")
    print("="*60)
    
    detector = AdvancedMarketRegimeDetector()
    
    print("\nAll regime parameters:")
    print("-" * 40)
    
    for regime in MarketRegime:
        params = detector.get_regime_parameters(regime)
        print(f"\n{regime.value}:")
        for key, value in params.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Run all tests
    detector, regimes, confidences = test_regime_detection()
    test_regime_features()
    test_regime_parameters()
    
    print("\n" + "="*60)
    print("Regime detection tests completed!")
    print("="*60)