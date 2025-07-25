#!/usr/bin/env python3
"""
Risk-Managed Ensemble Integration
Integrates dynamic risk management with the ultimate ensemble trading system
"""

import os
import sys
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dynamic_risk_manager import DynamicRiskManager, RiskLimits, RiskMetrics
from enhanced_ensemble_manager import EnhancedEnsembleManager
from trade_simulator import TradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from erl_agent_ppo import AgentPPO
from erl_agent_rainbow import AgentRainbow


class RiskManagedEnsemble:
    """
    Ensemble trading system with integrated dynamic risk management
    Combines ultimate ensemble with real-time risk controls
    """
    
    def __init__(self, 
                 ensemble_manager: EnhancedEnsembleManager,
                 initial_capital: float = 100000.0,
                 risk_limits: Optional[RiskLimits] = None,
                 enable_risk_override: bool = True):
        """
        Initialize risk-managed ensemble
        
        Args:
            ensemble_manager: Pre-configured ensemble manager
            initial_capital: Starting capital for risk calculations
            risk_limits: Custom risk limits (uses defaults if None)
            enable_risk_override: Allow risk manager to override trading decisions
        """
        self.ensemble_manager = ensemble_manager
        self.initial_capital = initial_capital
        self.enable_risk_override = enable_risk_override
        
        # Set up conservative risk limits for live trading
        if risk_limits is None:
            risk_limits = RiskLimits(
                max_position_size=0.8,      # 80% max position
                max_daily_loss=0.05,        # 5% max daily loss
                max_drawdown=0.10,          # 10% max drawdown  
                var_limit=0.03,             # 3% VaR limit
                min_sharpe_ratio=-0.5,      # Stop if Sharpe < -0.5
                volatility_threshold=0.025   # 2.5% volatility alert
            )
        
        self.risk_manager = DynamicRiskManager(
            initial_capital=initial_capital,
            risk_limits=risk_limits,
            lookback_window=100
        )
        
        # Trading state
        self.current_position = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0
        
        # Performance tracking
        self.trading_history = []
        self.risk_overrides = 0
        self.total_trades = 0
        
        print(f"🛡️ RISK-MANAGED ENSEMBLE INITIALIZED")
        print(f"   Initial capital: ${initial_capital:,.0f}")
        print(f"   Risk override enabled: {enable_risk_override}")
        print(f"   Max position size: {risk_limits.max_position_size:.1%}")
        print(f"   Max drawdown limit: {risk_limits.max_drawdown:.1%}")
    
    def get_trading_action(self, 
                          state: torch.Tensor,
                          current_price: float,
                          confidence_weights: Optional[Dict[str, float]] = None) -> Tuple[int, Dict]:
        """
        Get trading action with integrated risk management
        
        Args:
            state: Current market state
            current_price: Current asset price
            confidence_weights: Optional agent confidence weights
            
        Returns:
            Tuple of (action, risk_info)
        """
        self.current_price = current_price
        
        # Update unrealized PnL
        if self.current_position != 0 and self.entry_price > 0:
            self.unrealized_pnl = self.current_position * (current_price - self.entry_price)
        
        # Update risk metrics
        risk_metrics = self.risk_manager.update_portfolio_state(
            current_price=current_price,
            position=self.current_position,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl
        )
        
        # Get ensemble recommendation
        ensemble_action, ensemble_info = self.ensemble_manager.get_ensemble_action(
            state=state,
            confidence_weights=confidence_weights
        )
        
        # Check if risk manager should override
        risk_override = False
        final_action = ensemble_action
        
        if self.enable_risk_override:
            # Emergency stop override
            if self.risk_manager.emergency_stop:
                final_action = 0  # Close position
                risk_override = True
                self.risk_overrides += 1
            
            # Force position closure if risk is too high
            elif self.risk_manager.should_close_position(risk_metrics):
                final_action = 0  # Close position
                risk_override = True
                self.risk_overrides += 1
            
            # Position size adjustment based on risk
            elif ensemble_action != 0:  # Only for buy/sell actions
                # Calculate optimal position size
                target_position = 1.0 if ensemble_action == 1 else -1.0  # Buy or sell
                optimal_position = self.risk_manager.get_optimal_position_size(
                    current_metrics=risk_metrics,
                    base_position=target_position,
                    confidence=ensemble_info.get('confidence', 0.5)
                )
                
                # If optimal position is too small, don't trade
                if abs(optimal_position) < 0.1:
                    final_action = 0
                    risk_override = True
                    self.risk_overrides += 1
        
        # Execute position changes and update PnL
        self._execute_action(final_action, current_price)
        
        # Compile comprehensive trading info
        trading_info = {
            'ensemble_action': ensemble_action,
            'final_action': final_action,
            'risk_override': risk_override,
            'risk_metrics': risk_metrics,
            'ensemble_info': ensemble_info,
            'current_position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'portfolio_value': risk_metrics.portfolio_value,
            'market_regime': risk_metrics.market_regime,
            'confidence_score': risk_metrics.confidence_score,
            'emergency_stop': self.risk_manager.emergency_stop
        }
        
        # Log trading decision
        self.trading_history.append({
            'timestamp': time.time(),
            'price': current_price,
            'ensemble_action': ensemble_action,
            'final_action': final_action,
            'risk_override': risk_override,
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'portfolio_value': risk_metrics.portfolio_value,
            'market_regime': risk_metrics.market_regime,
            'drawdown': risk_metrics.max_drawdown,
            'volatility': risk_metrics.volatility
        })
        
        return final_action, trading_info
    
    def _execute_action(self, action: int, current_price: float):
        """Execute trading action and update position/PnL"""
        
        previous_position = self.current_position
        
        if action == 1:  # Buy
            if self.current_position <= 0:  # Close short or enter long
                if self.current_position < 0:
                    # Close short position
                    self.realized_pnl += -self.current_position * (self.entry_price - current_price)
                # Enter long position
                self.current_position = 1.0
                self.entry_price = current_price
                self.unrealized_pnl = 0.0
                self.total_trades += 1
        
        elif action == 2:  # Sell
            if self.current_position >= 0:  # Close long or enter short
                if self.current_position > 0:
                    # Close long position
                    self.realized_pnl += self.current_position * (current_price - self.entry_price)
                # Enter short position
                self.current_position = -1.0
                self.entry_price = current_price
                self.unrealized_pnl = 0.0
                self.total_trades += 1
        
        elif action == 0:  # Hold/Close
            if self.current_position != 0:
                # Close position
                if self.current_position > 0:
                    self.realized_pnl += self.current_position * (current_price - self.entry_price)
                else:
                    self.realized_pnl += -self.current_position * (self.entry_price - current_price)
                
                self.current_position = 0.0
                self.entry_price = 0.0
                self.unrealized_pnl = 0.0
                self.total_trades += 1
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance and risk summary"""
        
        if not self.trading_history:
            return {"status": "no_trading_data"}
        
        # Get current risk dashboard
        risk_dashboard = self.risk_manager.get_risk_dashboard()
        
        # Calculate trading statistics
        total_pnl = self.realized_pnl + self.unrealized_pnl
        pnl_pct = total_pnl / self.initial_capital
        
        # Action distribution
        actions = [h['final_action'] for h in self.trading_history]
        action_counts = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}
        trading_activity = (action_counts[1] + action_counts[2]) / len(actions) * 100
        
        # Risk override statistics
        risk_override_rate = self.risk_overrides / len(self.trading_history) * 100
        
        # Performance trend
        if len(self.trading_history) >= 10:
            recent_pnl = [h['realized_pnl'] + h['unrealized_pnl'] for h in self.trading_history[-10:]]
            performance_trend = "improving" if recent_pnl[-1] > recent_pnl[0] else "declining"
        else:
            performance_trend = "stable"
        
        summary = {
            "timestamp": time.time(),
            "performance": {
                "total_pnl": total_pnl,
                "pnl_percentage": pnl_pct,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "portfolio_value": self.initial_capital + total_pnl,
                "performance_trend": performance_trend
            },
            "trading": {
                "total_decisions": len(self.trading_history),
                "total_trades": self.total_trades,
                "trading_activity_pct": trading_activity,
                "action_distribution": action_counts,
                "current_position": self.current_position,
                "entry_price": self.entry_price
            },
            "risk_management": {
                "risk_status": risk_dashboard["risk_status"],
                "emergency_stop": risk_dashboard["emergency_stop"],
                "max_drawdown": risk_dashboard["risk_metrics"]["max_drawdown"],
                "current_volatility": risk_dashboard["risk_metrics"]["volatility"],
                "sharpe_ratio": risk_dashboard["risk_metrics"]["sharpe_ratio"],
                "market_regime": risk_dashboard["market"]["regime"],
                "risk_overrides": self.risk_overrides,
                "risk_override_rate": risk_override_rate,
                "total_alerts": risk_dashboard["alerts"]["total_alerts"]
            },
            "ensemble": {
                "agent_count": len(self.ensemble_manager.agents),
                "voting_method": "performance_weighted",
                "kelly_position_sizing": True
            }
        }
        
        return summary
    
    def print_status_update(self):
        """Print real-time status update"""
        
        summary = self.get_performance_summary()
        
        if summary.get("status") == "no_trading_data":
            print("📊 No trading data available yet")
            return
        
        perf = summary["performance"]
        trading = summary["trading"]
        risk = summary["risk_management"]
        
        # Status emoji based on performance
        if perf["pnl_percentage"] > 0.05:
            status_emoji = "🚀"
        elif perf["pnl_percentage"] > 0.01:
            status_emoji = "📈"
        elif perf["pnl_percentage"] > -0.01:
            status_emoji = "➡️"
        else:
            status_emoji = "📉"
        
        # Risk emoji
        risk_emoji = {
            "LOW": "🟢",
            "MEDIUM": "🟡", 
            "HIGH": "🟠",
            "CRITICAL": "🔴",
            "EMERGENCY_STOP": "🛑"
        }.get(risk["risk_status"], "⚪")
        
        print(f"\n{status_emoji} RISK-MANAGED ENSEMBLE STATUS:")
        print(f"   💰 Portfolio: ${perf['portfolio_value']:,.0f} (PnL: {perf['pnl_percentage']:+.2%})")
        print(f"   📊 Position: {trading['current_position']:+.2f} @ ${self.current_price:,.0f}")
        print(f"   🎯 Trading: {trading['trading_activity_pct']:.1f}% activity, {trading['total_trades']} trades")
        print(f"   {risk_emoji} Risk: {risk['risk_status']}, DD: {risk['max_drawdown']:.2%}, Vol: {risk['current_volatility']:.3f}")
        print(f"   🌍 Market: {risk['market_regime']}, Overrides: {risk['risk_override_rate']:.1f}%")
        
        if risk["emergency_stop"]:
            print(f"   🛑 EMERGENCY STOP ACTIVE - NO NEW POSITIONS")
    
    def save_trading_data(self, filename: str = None):
        """Save complete trading and risk data"""
        
        if filename is None:
            filename = f"risk_managed_trading_data_{int(time.time())}.json"
        
        data = {
            "configuration": {
                "initial_capital": self.initial_capital,
                "enable_risk_override": self.enable_risk_override,
                "risk_limits": {
                    "max_position_size": self.risk_manager.risk_limits.max_position_size,
                    "max_daily_loss": self.risk_manager.risk_limits.max_daily_loss,
                    "max_drawdown": self.risk_manager.risk_limits.max_drawdown,
                    "var_limit": self.risk_manager.risk_limits.var_limit
                }
            },
            "performance_summary": self.get_performance_summary(),
            "trading_history": self.trading_history,
            "timestamp": time.time()
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Also save risk manager data
        risk_filename = filename.replace('.json', '_risk_data.json')
        self.risk_manager.save_risk_data(risk_filename)
        
        print(f"💾 Risk-managed trading data saved to {filename}")
        print(f"💾 Risk management data saved to {risk_filename}")
        
        return filename, risk_filename


def test_risk_managed_ensemble():
    """Test the risk-managed ensemble system"""
    
    print("🧪 Testing Risk-Managed Ensemble")
    print("=" * 60)
    
    # Create a mock ensemble manager for testing
    class MockEnsembleManager:
        def __init__(self):
            self.agents = {"mock_agent": None}
        
        def get_ensemble_action(self, state, confidence_weights=None):
            # Simulate ensemble decision making
            action = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
            confidence = np.random.uniform(0.3, 0.9)
            
            info = {
                'confidence': confidence,
                'voting_results': {0: 0.6, 1: 0.2, 2: 0.2},
                'agent_votes': {"mock_agent": action}
            }
            
            return action, info
    
    # Create risk-managed ensemble
    mock_ensemble = MockEnsembleManager()
    
    risk_managed_ensemble = RiskManagedEnsemble(
        ensemble_manager=mock_ensemble,
        initial_capital=100000.0,
        enable_risk_override=True
    )
    
    print("\n📊 Simulating risk-managed trading...")
    
    # Simulate trading session
    base_price = 50000.0
    state = torch.randn(1, 8)  # Mock state
    
    for step in range(50):
        # Simulate price movement with some volatility
        price_change = np.random.normal(0, 0.015) * base_price
        current_price = base_price + price_change * (step + 1) / 50
        
        # Get trading action with risk management
        action, trading_info = risk_managed_ensemble.get_trading_action(
            state=state,
            current_price=current_price
        )
        
        # Print periodic updates
        if step % 10 == 0:
            risk_managed_ensemble.print_status_update()
        
        # Simulate state changes
        state = torch.randn(1, 8)
    
    # Final performance summary
    print(f"\n📋 FINAL PERFORMANCE SUMMARY:")
    summary = risk_managed_ensemble.get_performance_summary()
    
    print(f"   💰 Final Portfolio Value: ${summary['performance']['portfolio_value']:,.0f}")
    print(f"   📈 Total PnL: ${summary['performance']['total_pnl']:,.0f} ({summary['performance']['pnl_percentage']:+.2%})")
    print(f"   🎯 Trading Activity: {summary['trading']['trading_activity_pct']:.1f}%")
    print(f"   🛡️ Risk Status: {summary['risk_management']['risk_status']}")
    print(f"   🔄 Risk Overrides: {summary['risk_management']['risk_overrides']} ({summary['risk_management']['risk_override_rate']:.1f}%)")
    print(f"   📊 Max Drawdown: {summary['risk_management']['max_drawdown']:.2%}")
    print(f"   🌍 Market Regime: {summary['risk_management']['market_regime']}")
    print(f"   ⚡ Sharpe Ratio: {summary['risk_management']['sharpe_ratio']:.3f}")
    
    # Save test data
    risk_managed_ensemble.save_trading_data("test_risk_managed_ensemble.json")
    
    print(f"\n✅ Risk-managed ensemble test completed!")
    print(f"   Processed {len(risk_managed_ensemble.trading_history)} trading decisions")
    print(f"   Risk management integration successful")
    
    return summary


if __name__ == "__main__":
    test_risk_managed_ensemble()