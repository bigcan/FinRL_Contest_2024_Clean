#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing for Optimal Risk-Adjusted Returns
Implements dynamic position sizing based on historical win rate and expected returns
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, List, Tuple, Optional


class KellyPositionSizer:
    """
    Kelly Criterion position sizing for cryptocurrency trading
    
    The Kelly Criterion determines optimal bet size to maximize long-term growth:
    f* = (bp - q) / b
    
    Where:
    - f* = fraction of capital to bet
    - b = odds received on the wager (average win / average loss)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """
    
    def __init__(self, 
                 lookback_window: int = 100,
                 min_trades: int = 20,
                 max_kelly_fraction: float = 0.25,
                 kelly_multiplier: float = 0.5,
                 min_position_size: float = 0.01,
                 max_position_size: float = 1.0):
        """
        Initialize Kelly position sizer
        
        Args:
            lookback_window: Number of recent trades to consider
            min_trades: Minimum trades needed before using Kelly sizing
            max_kelly_fraction: Maximum Kelly fraction to prevent over-leverage
            kelly_multiplier: Conservative multiplier (0.5 = half-Kelly)
            min_position_size: Minimum position size
            max_position_size: Maximum position size
        """
        self.lookback_window = lookback_window
        self.min_trades = min_trades
        self.max_kelly_fraction = max_kelly_fraction
        self.kelly_multiplier = kelly_multiplier
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        
        # Trade history tracking
        self.trade_history = deque(maxlen=lookback_window)
        self.returns_history = deque(maxlen=lookback_window)
        
        # Performance metrics cache
        self._cached_metrics = {}
        self._cache_valid = False
        
        print(f"🎯 Kelly Position Sizer initialized:")
        print(f"   Lookback window: {lookback_window} trades")
        print(f"   Kelly multiplier: {kelly_multiplier} (conservative)")
        print(f"   Max Kelly fraction: {max_kelly_fraction}")
        print(f"   Position range: {min_position_size:.1%} - {max_position_size:.1%}")
    
    def add_trade_result(self, action: int, return_pct: float, success: bool = None):
        """
        Add a trade result to the history
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            return_pct: Percentage return from the trade
            success: Whether trade was profitable (auto-calculated if None)
        """
        if success is None:
            success = return_pct > 0
        
        self.trade_history.append({
            'action': action,
            'return_pct': return_pct,
            'success': success,
            'timestamp': len(self.trade_history)
        })
        
        if action != 0:  # Only track returns for non-hold actions
            self.returns_history.append(return_pct)
        
        # Invalidate cache
        self._cache_valid = False
    
    def calculate_kelly_metrics(self) -> Dict[str, float]:
        """Calculate Kelly Criterion metrics from trade history"""
        
        if self._cache_valid and self._cached_metrics:
            return self._cached_metrics
        
        if len(self.returns_history) < self.min_trades:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_loss_ratio': 0.0,
                'kelly_fraction': 0.0,
                'confidence': 0.0
            }
        
        returns = np.array(list(self.returns_history))
        
        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        # Calculate metrics
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.01  # Avoid division by zero
        
        # Win/Loss ratio (b in Kelly formula)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Kelly fraction: f* = (bp - q) / b = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        if win_loss_ratio > 0:
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        else:
            kelly_fraction = 0.0
        
        # Apply conservative constraints
        kelly_fraction = np.clip(kelly_fraction, 0.0, self.max_kelly_fraction)
        
        # Confidence based on sample size
        confidence = min(1.0, len(returns) / (self.min_trades * 2))
        
        metrics = {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'kelly_fraction': kelly_fraction,
            'confidence': confidence,
            'total_trades': len(returns),
            'total_wins': len(wins),
            'total_losses': len(losses)
        }
        
        # Cache results
        self._cached_metrics = metrics
        self._cache_valid = True
        
        return metrics
    
    def get_position_size(self, 
                         base_action: int, 
                         confidence: float = 1.0,
                         market_volatility: float = 1.0) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            base_action: 0=Hold, 1=Buy, 2=Sell
            confidence: Agent confidence in prediction (0-1)
            market_volatility: Current market volatility multiplier
            
        Returns:
            Position size as fraction of capital (0-1)
        """
        
        # Hold action gets zero position
        if base_action == 0:
            return 0.0
        
        # Get Kelly metrics
        metrics = self.calculate_kelly_metrics()
        
        # Start with Kelly fraction
        kelly_fraction = metrics['kelly_fraction']
        
        # Apply conservative multiplier (half-Kelly is common)
        position_size = kelly_fraction * self.kelly_multiplier
        
        # Adjust for confidence
        position_size *= confidence
        
        # Adjust for sample confidence (reduce positions with limited data)
        position_size *= metrics['confidence']
        
        # Adjust for market volatility (reduce positions in high volatility)
        volatility_adjustment = 1.0 / max(1.0, market_volatility)
        position_size *= volatility_adjustment
        
        # Apply min/max constraints
        position_size = np.clip(position_size, self.min_position_size, self.max_position_size)
        
        return position_size
    
    def get_dynamic_position_size(self, 
                                 action_probabilities: torch.Tensor,
                                 market_volatility: float = 1.0) -> Tuple[int, float]:
        """
        Get both action and position size based on action probabilities
        
        Args:
            action_probabilities: Softmax probabilities for [Hold, Buy, Sell]
            market_volatility: Current market volatility
            
        Returns:
            Tuple of (action, position_size)
        """
        
        # Convert to numpy
        if isinstance(action_probabilities, torch.Tensor):
            probs = action_probabilities.detach().cpu().numpy()
        else:
            probs = action_probabilities
        
        # Select action (could be probabilistic or deterministic)
        action = np.argmax(probs)
        
        # Get confidence from probability
        confidence = probs[action]
        
        # Calculate position size
        position_size = self.get_position_size(
            base_action=action,
            confidence=confidence,
            market_volatility=market_volatility
        )
        
        return action, position_size
    
    def print_statistics(self):
        """Print current Kelly statistics"""
        
        metrics = self.calculate_kelly_metrics()
        
        print(f"\n📊 Kelly Position Sizing Statistics:")
        print(f"   Total trades: {metrics.get('total_trades', 0)}")
        print(f"   Win rate: {metrics.get('win_rate', 0.0):.1%}")
        print(f"   Avg win: {metrics.get('avg_win', 0.0):.3%}")
        print(f"   Avg loss: {metrics.get('avg_loss', 0.0):.3%}")
        print(f"   Win/Loss ratio: {metrics.get('win_loss_ratio', 0.0):.2f}")
        print(f"   Kelly fraction: {metrics.get('kelly_fraction', 0.0):.3f}")
        print(f"   Confidence: {metrics.get('confidence', 0.0):.1%}")
        
        # Position sizing example
        if metrics.get('total_trades', 0) >= self.min_trades:
            example_high_conf = self.get_position_size(1, confidence=0.9)
            example_low_conf = self.get_position_size(1, confidence=0.3)
            
            print(f"\n💡 Position Size Examples:")
            print(f"   High confidence (90%): {example_high_conf:.1%}")
            print(f"   Low confidence (30%): {example_low_conf:.1%}")
        else:
            print(f"\n⚠️  Need {self.min_trades - metrics.get('total_trades', 0)} more trades for full Kelly sizing")


class EnsembleKellyManager:
    """
    Manages Kelly position sizing for multiple agents in an ensemble
    """
    
    def __init__(self, agent_names: List[str], **kelly_kwargs):
        """
        Initialize Kelly managers for each agent
        
        Args:
            agent_names: List of agent names
            **kelly_kwargs: Arguments passed to KellyPositionSizer
        """
        self.agent_names = agent_names
        self.kelly_sizers = {
            name: KellyPositionSizer(**kelly_kwargs) 
            for name in agent_names
        }
        
        # Ensemble-level position sizing
        self.ensemble_sizer = KellyPositionSizer(**kelly_kwargs)
        
        print(f"🎯 Ensemble Kelly Manager initialized for {len(agent_names)} agents")
    
    def add_agent_trade(self, agent_name: str, action: int, return_pct: float):
        """Add trade result for specific agent"""
        if agent_name in self.kelly_sizers:
            self.kelly_sizers[agent_name].add_trade_result(action, return_pct)
    
    def add_ensemble_trade(self, action: int, return_pct: float):
        """Add trade result for ensemble"""
        self.ensemble_sizer.add_trade_result(action, return_pct)
    
    def get_agent_position_sizes(self, 
                                agent_actions: Dict[str, int],
                                agent_confidences: Dict[str, float] = None,
                                market_volatility: float = 1.0) -> Dict[str, float]:
        """Get position sizes for all agents"""
        
        if agent_confidences is None:
            agent_confidences = {name: 1.0 for name in agent_actions.keys()}
        
        position_sizes = {}
        for agent_name, action in agent_actions.items():
            if agent_name in self.kelly_sizers:
                confidence = agent_confidences.get(agent_name, 1.0)
                position_size = self.kelly_sizers[agent_name].get_position_size(
                    base_action=action,
                    confidence=confidence,
                    market_volatility=market_volatility
                )
                position_sizes[agent_name] = position_size
        
        return position_sizes
    
    def get_ensemble_position_size(self, 
                                  ensemble_action: int,
                                  ensemble_confidence: float = 1.0,
                                  market_volatility: float = 1.0) -> float:
        """Get position size for ensemble decision"""
        
        return self.ensemble_sizer.get_position_size(
            base_action=ensemble_action,
            confidence=ensemble_confidence,
            market_volatility=market_volatility
        )
    
    def print_all_statistics(self):
        """Print statistics for all agents"""
        
        print(f"\n🎯 ENSEMBLE KELLY STATISTICS")
        print("=" * 60)
        
        # Ensemble statistics
        print(f"\n📊 Ensemble Performance:")
        self.ensemble_sizer.print_statistics()
        
        # Individual agent statistics
        for agent_name in self.agent_names:
            print(f"\n📊 {agent_name} Performance:")
            self.kelly_sizers[agent_name].print_statistics()


def test_kelly_position_sizing():
    """Test Kelly position sizing with sample data"""
    
    print("🧪 Testing Kelly Position Sizing")
    print("=" * 60)
    
    # Create Kelly sizer
    kelly = KellyPositionSizer(
        lookback_window=50,
        min_trades=10,
        kelly_multiplier=0.5
    )
    
    # Simulate some trades
    np.random.seed(42)
    
    # Simulate 30 trades with 55% win rate
    for i in range(30):
        action = np.random.choice([1, 2])  # Buy or Sell
        
        # 55% win rate with wins averaging 2% and losses averaging 1.5%
        if np.random.random() < 0.55:
            return_pct = np.random.normal(0.02, 0.01)  # Win
        else:
            return_pct = np.random.normal(-0.015, 0.008)  # Loss
        
        kelly.add_trade_result(action, return_pct)
        
        # Print progress every 10 trades
        if (i + 1) % 10 == 0:
            print(f"\n--- After {i+1} trades ---")
            kelly.print_statistics()
    
    # Test position sizing
    print(f"\n🎯 Position Sizing Tests:")
    test_cases = [
        (1, 0.9, 1.0, "High confidence buy, normal volatility"),
        (2, 0.6, 1.0, "Medium confidence sell, normal volatility"),
        (1, 0.9, 2.0, "High confidence buy, high volatility"),
        (1, 0.3, 1.0, "Low confidence buy, normal volatility"),
        (0, 1.0, 1.0, "Hold action"),
    ]
    
    for action, confidence, volatility, description in test_cases:
        pos_size = kelly.get_position_size(action, confidence, volatility)
        print(f"   {description}: {pos_size:.1%}")
    
    print(f"\n✅ Kelly position sizing test complete!")


if __name__ == "__main__":
    test_kelly_position_sizing()