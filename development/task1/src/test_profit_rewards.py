"""
Test script for profit-focused reward function
Validates the new reward calculations with sample scenarios
"""

import numpy as np
import torch as th
import sys
sys.path.append('.')
from profit_focused_rewards import ProfitFocusedRewardCalculator, create_profit_focused_calculator
from reward_functions import create_reward_calculator

def test_basic_scenarios():
    """Test basic reward calculation scenarios"""
    print("="*60)
    print("Testing Profit-Focused Reward Calculator")
    print("="*60)
    
    # Initialize calculator
    calc = ProfitFocusedRewardCalculator(
        profit_amplifier=3.0,
        loss_multiplier=1.0,
        trade_completion_bonus=0.02,
        opportunity_cost_penalty=0.001
    )
    
    # Test scenarios
    scenarios = [
        {
            "name": "Profitable long position",
            "action": 1,  # hold
            "current_price": 105.0,
            "previous_price": 100.0,
            "position": 1,
            "previous_position": 1,
            "expected": "positive (amplified)"
        },
        {
            "name": "Losing long position",
            "action": 1,  # hold
            "current_price": 95.0,
            "previous_price": 100.0,
            "position": 1,
            "previous_position": 1,
            "expected": "negative"
        },
        {
            "name": "Closing profitable trade",
            "action": 0,  # sell
            "current_price": 110.0,
            "previous_price": 105.0,
            "position": 0,
            "previous_position": 1,
            "expected": "positive + completion bonus"
        },
        {
            "name": "Opening new position",
            "action": 2,  # buy
            "current_price": 100.0,
            "previous_price": 99.0,
            "position": 1,
            "previous_position": 0,
            "expected": "small positive (action bonus)"
        },
        {
            "name": "Holding for too long",
            "action": 1,  # hold
            "current_price": 100.0,
            "previous_price": 100.0,
            "position": 1,
            "previous_position": 1,
            "expected": "negative (opportunity cost)"
        }
    ]
    
    # Simulate position entry for trade completion test
    calc.position_entry_price = 100.0
    
    print("\nScenario Results:")
    print("-" * 60)
    
    for scenario in scenarios:
        reward = calc.calculate_reward(
            action=scenario["action"],
            current_price=scenario["current_price"],
            previous_price=scenario["previous_price"],
            position=scenario["position"],
            previous_position=scenario["previous_position"]
        )
        
        print(f"\n{scenario['name']}:")
        print(f"  Price: {scenario['previous_price']} → {scenario['current_price']}")
        print(f"  Position: {scenario['previous_position']} → {scenario['position']}")
        print(f"  Reward: {reward:.6f}")
        print(f"  Expected: {scenario['expected']}")
        
        # Simulate holding time for opportunity cost test
        if scenario["name"] == "Holding for too long":
            calc.position_holding_time = 100
    
    # Print metrics
    print("\n" + "="*60)
    print("Performance Metrics:")
    print("-" * 60)
    metrics = calc.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

def test_integration():
    """Test integration with existing reward calculator"""
    print("\n" + "="*60)
    print("Testing Integration with Existing System")
    print("="*60)
    
    # Create standard calculator
    standard_calc = create_reward_calculator(
        reward_type="multi_objective",
        device="cpu"
    )
    
    # Create profit-focused calculator
    profit_calc = create_reward_calculator(
        reward_type="profit_focused",
        device="cpu"
    )
    
    print("\nCalculator types created successfully")
    
    # Test reward calculation
    test_params = {
        "current_asset": 1050000,
        "initial_asset": 1000000,
        "action": 1,
        "current_price": 105.0,
        "previous_price": 100.0,
        "position": 1,
        "previous_position": 1,
        "current_volatility": 0.02
    }
    
    print("\nTest scenario: 5% price increase with long position")
    
    # Calculate rewards
    standard_reward = standard_calc.calculate_reward(**test_params)
    
    # For profit calculator, we need to ensure it has the profit_calculator attribute
    if hasattr(profit_calc, 'profit_calculator'):
        profit_reward = profit_calc.calculate_reward(**test_params)
        print(f"Standard reward: {standard_reward:.6f}")
        print(f"Profit-focused reward: {profit_reward:.6f}")
        print(f"Amplification factor: {profit_reward/standard_reward:.2f}x")
    else:
        print("Profit calculator integration not properly set up")

def test_market_regimes():
    """Test market regime sensitivity"""
    print("\n" + "="*60)
    print("Testing Market Regime Sensitivity")
    print("="*60)
    
    calc = ProfitFocusedRewardCalculator(regime_sensitivity=True)
    
    regimes = ["trending", "volatile", "ranging"]
    
    # Same scenario, different regimes
    base_params = {
        "action": 1,
        "current_price": 105.0,
        "previous_price": 100.0,
        "position": 1,
        "previous_position": 1
    }
    
    print("\nReward adjustments by market regime:")
    print("-" * 40)
    
    for regime in regimes:
        reward = calc.calculate_reward(
            **base_params,
            market_regime=regime
        )
        print(f"{regime:10s}: {reward:.6f}")

def test_performance_over_episode():
    """Simulate performance over a trading episode"""
    print("\n" + "="*60)
    print("Testing Performance Over Episode")
    print("="*60)
    
    calc = ProfitFocusedRewardCalculator()
    
    # Simulate price series
    np.random.seed(42)
    prices = [100.0]
    for _ in range(100):
        change = np.random.normal(0.0001, 0.01)  # Small random walk
        prices.append(prices[-1] * (1 + change))
    
    # Simulate trading
    position = 0
    total_reward = 0.0
    actions = []
    
    for i in range(1, len(prices)):
        # Simple strategy: buy on dips, sell on rises
        price_change = (prices[i] - prices[i-1]) / prices[i-1]
        
        if price_change < -0.005 and position == 0:
            action = 2  # buy
            new_position = 1
        elif price_change > 0.005 and position == 1:
            action = 0  # sell
            new_position = 0
        else:
            action = 1  # hold
            new_position = position
        
        reward = calc.calculate_reward(
            action=action,
            current_price=prices[i],
            previous_price=prices[i-1],
            position=new_position,
            previous_position=position
        )
        
        total_reward += reward
        actions.append(action)
        position = new_position
    
    # Print episode summary
    print(f"\nEpisode Summary:")
    print(f"  Total steps: {len(prices)-1}")
    print(f"  Total reward: {total_reward:.6f}")
    print(f"  Actions: Buy={actions.count(2)}, Sell={actions.count(0)}, Hold={actions.count(1)}")
    
    metrics = calc.get_metrics()
    print(f"\nTrading Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run all tests
    test_basic_scenarios()
    test_integration()
    test_market_regimes()
    test_performance_over_episode()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)