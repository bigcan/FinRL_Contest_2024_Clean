#!/usr/bin/env python3
"""
Dynamic Risk Management System
Implements real-time risk controls with adaptive position sizing and market regime awareness
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
import json

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


@dataclass
class RiskMetrics:
    """Real-time risk metrics container"""
    timestamp: float
    portfolio_value: float
    unrealized_pnl: float
    realized_pnl: float
    current_position: float
    volatility: float
    var_95: float  # Value at Risk (95%)
    max_drawdown: float
    sharpe_ratio: float
    kelly_fraction: float
    market_regime: str
    confidence_score: float


@dataclass
class RiskLimits:
    """Dynamic risk limits configuration"""
    max_position_size: float = 1.0
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.10   # 10% max drawdown
    var_limit: float = 0.03      # 3% VaR limit
    min_sharpe_ratio: float = 0.0
    max_correlation: float = 0.8
    position_decay_rate: float = 0.95  # Reduce position if risk increases
    volatility_threshold: float = 0.02  # 2% volatility threshold


class MarketRegimeDetector:
    """
    Detects market regimes for adaptive risk management
    Uses volatility clustering and trend analysis
    """
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.price_history = deque(maxlen=lookback_window)
        self.return_history = deque(maxlen=lookback_window)
        self.volatility_history = deque(maxlen=lookback_window)
        
        # Regime thresholds
        self.low_vol_threshold = 0.01    # 1% daily vol
        self.high_vol_threshold = 0.03   # 3% daily vol
        self.trend_threshold = 0.02      # 2% trend strength
        
        print(f"📊 Market Regime Detector initialized:")
        print(f"   Lookback window: {lookback_window}")
        print(f"   Volatility thresholds: {self.low_vol_threshold:.1%} - {self.high_vol_threshold:.1%}")
    
    def update(self, price: float) -> str:
        """Update regime detection with new price"""
        self.price_history.append(price)
        
        if len(self.price_history) < 2:
            return "unknown"
        
        # Calculate return
        current_return = (price - self.price_history[-2]) / self.price_history[-2]
        self.return_history.append(current_return)
        
        if len(self.return_history) < 20:
            return "unknown"
        
        # Calculate rolling volatility
        recent_returns = list(self.return_history)[-20:]
        volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
        self.volatility_history.append(volatility)
        
        # Calculate trend strength
        if len(self.price_history) >= 50:
            recent_prices = list(self.price_history)[-50:]
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            trend_strength = abs(trend)
        else:
            trend_strength = 0
        
        # Determine regime
        if volatility < self.low_vol_threshold:
            if trend_strength > self.trend_threshold:
                regime = "low_vol_trending"
            else:
                regime = "low_vol_ranging"
        elif volatility > self.high_vol_threshold:
            if trend_strength > self.trend_threshold:
                regime = "high_vol_trending" 
            else:
                regime = "high_vol_ranging"
        else:
            if trend_strength > self.trend_threshold:
                regime = "medium_vol_trending"
            else:
                regime = "medium_vol_ranging"
        
        return regime
    
    def get_regime_risk_multiplier(self, regime: str) -> float:
        """Get risk adjustment multiplier for current regime"""
        regime_multipliers = {
            "low_vol_trending": 1.2,    # Can take larger positions
            "low_vol_ranging": 1.0,     # Normal position sizing
            "medium_vol_trending": 0.8, # Reduce positions slightly
            "medium_vol_ranging": 0.7,  # Reduce positions more
            "high_vol_trending": 0.5,   # Significantly reduce positions
            "high_vol_ranging": 0.3,    # Very conservative
            "unknown": 0.5              # Conservative default
        }
        return regime_multipliers.get(regime, 0.5)


class DynamicRiskManager:
    """
    Dynamic risk management system with real-time monitoring and adaptive controls
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 risk_limits: Optional[RiskLimits] = None,
                 lookback_window: int = 100):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()
        self.lookback_window = lookback_window
        
        # Risk tracking
        self.risk_metrics_history = deque(maxlen=lookback_window)
        self.pnl_history = deque(maxlen=lookback_window)
        self.position_history = deque(maxlen=lookback_window)
        self.drawdown_history = deque(maxlen=lookback_window)
        
        # Components
        self.regime_detector = MarketRegimeDetector(lookback_window)
        
        # Risk alerts
        self.risk_alerts = []
        self.emergency_stop = False
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.max_capital = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown_seen = 0.0
        
        print(f"🛡️ Dynamic Risk Manager initialized:")
        print(f"   Initial capital: ${initial_capital:,.0f}")
        print(f"   Max position size: {risk_limits.max_position_size:.1%}")
        print(f"   Max daily loss: {risk_limits.max_daily_loss:.1%}")
        print(f"   Max drawdown: {risk_limits.max_drawdown:.1%}")
        print(f"   VaR limit: {risk_limits.var_limit:.1%}")
    
    def update_portfolio_state(self, 
                              current_price: float,
                              position: float,
                              unrealized_pnl: float = 0.0,
                              realized_pnl: float = 0.0) -> RiskMetrics:
        """Update portfolio state and calculate risk metrics"""
        
        timestamp = time.time()
        
        # Update capital tracking
        total_pnl = unrealized_pnl + realized_pnl
        portfolio_value = self.initial_capital + total_pnl
        
        # Track maximum capital for drawdown calculation
        if portfolio_value > self.max_capital:
            self.max_capital = portfolio_value
        
        # Calculate current drawdown
        self.current_drawdown = (self.max_capital - portfolio_value) / self.max_capital
        if self.current_drawdown > self.max_drawdown_seen:
            self.max_drawdown_seen = self.current_drawdown
        
        # Update histories
        self.pnl_history.append(total_pnl)
        self.position_history.append(position)
        self.drawdown_history.append(self.current_drawdown)
        
        # Calculate volatility
        if len(self.pnl_history) >= 20:
            recent_returns = []
            pnl_values = list(self.pnl_history)
            for i in range(1, min(21, len(pnl_values))):
                if pnl_values[i-1] != 0:
                    ret = (pnl_values[i] - pnl_values[i-1]) / abs(pnl_values[i-1])
                    recent_returns.append(ret)
            
            volatility = np.std(recent_returns) if recent_returns else 0.0
        else:
            volatility = 0.0
        
        # Calculate VaR (95th percentile)
        if len(self.pnl_history) >= 20:
            pnl_changes = np.diff(list(self.pnl_history)[-20:])
            var_95 = np.percentile(pnl_changes, 5) if len(pnl_changes) > 0 else 0.0
        else:
            var_95 = 0.0
        
        # Calculate Sharpe ratio
        if len(self.pnl_history) >= 20 and volatility > 0:
            recent_returns = np.diff(list(self.pnl_history)[-20:])
            avg_return = np.mean(recent_returns)
            sharpe_ratio = avg_return / (volatility * np.sqrt(252)) if volatility > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate Kelly fraction (simplified)
        if len(self.pnl_history) >= 20:
            recent_returns = []
            pnl_values = list(self.pnl_history)
            for i in range(1, min(21, len(pnl_values))):
                if pnl_values[i-1] != 0:
                    ret = (pnl_values[i] - pnl_values[i-1]) / abs(pnl_values[i-1])
                    recent_returns.append(ret)
            
            if recent_returns and volatility > 0:
                win_rate = len([r for r in recent_returns if r > 0]) / len(recent_returns)
                avg_win = np.mean([r for r in recent_returns if r > 0]) if any(r > 0 for r in recent_returns) else 0
                avg_loss = abs(np.mean([r for r in recent_returns if r < 0])) if any(r < 0 for r in recent_returns) else 0.01
                
                if avg_loss > 0:
                    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(1, kelly_fraction))  # Clamp between 0 and 1
                else:
                    kelly_fraction = 0.5
            else:
                kelly_fraction = 0.5
        else:
            kelly_fraction = 0.5
        
        # Get market regime
        market_regime = self.regime_detector.update(current_price)
        
        # Calculate confidence score based on recent performance
        if len(self.pnl_history) >= 5:
            recent_pnl = list(self.pnl_history)[-5:]
            positive_periods = len([p for p in recent_pnl if p > recent_pnl[0]])
            confidence_score = positive_periods / 4.0  # 0 to 1 scale
        else:
            confidence_score = 0.5
        
        # Create risk metrics
        risk_metrics = RiskMetrics(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            current_position=position,
            volatility=volatility,
            var_95=var_95,
            max_drawdown=self.current_drawdown,
            sharpe_ratio=sharpe_ratio,
            kelly_fraction=kelly_fraction,
            market_regime=market_regime,
            confidence_score=confidence_score
        )
        
        self.risk_metrics_history.append(risk_metrics)
        
        # Check risk limits
        self._check_risk_limits(risk_metrics)
        
        return risk_metrics
    
    def _check_risk_limits(self, metrics: RiskMetrics):
        """Check if any risk limits are breached"""
        
        alerts = []
        
        # Check position size limit
        if abs(metrics.current_position) > self.risk_limits.max_position_size:
            alerts.append({
                'type': 'POSITION_SIZE_BREACH',
                'message': f'Position size {metrics.current_position:.2f} exceeds limit {self.risk_limits.max_position_size:.2f}',
                'severity': 'HIGH',
                'timestamp': metrics.timestamp
            })
        
        # Check daily loss limit
        daily_loss_pct = abs(metrics.unrealized_pnl + metrics.realized_pnl) / self.initial_capital
        if daily_loss_pct > self.risk_limits.max_daily_loss and (metrics.unrealized_pnl + metrics.realized_pnl) < 0:
            alerts.append({
                'type': 'DAILY_LOSS_BREACH',
                'message': f'Daily loss {daily_loss_pct:.2%} exceeds limit {self.risk_limits.max_daily_loss:.2%}',
                'severity': 'CRITICAL',
                'timestamp': metrics.timestamp
            })
            self.emergency_stop = True
        
        # Check maximum drawdown
        if metrics.max_drawdown > self.risk_limits.max_drawdown:
            alerts.append({
                'type': 'MAX_DRAWDOWN_BREACH',
                'message': f'Drawdown {metrics.max_drawdown:.2%} exceeds limit {self.risk_limits.max_drawdown:.2%}',
                'severity': 'CRITICAL',
                'timestamp': metrics.timestamp
            })
            self.emergency_stop = True
        
        # Check VaR limit
        var_pct = abs(metrics.var_95) / self.initial_capital
        if var_pct > self.risk_limits.var_limit:
            alerts.append({
                'type': 'VAR_BREACH',
                'message': f'VaR {var_pct:.2%} exceeds limit {self.risk_limits.var_limit:.2%}',
                'severity': 'MEDIUM',
                'timestamp': metrics.timestamp
            })
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.risk_limits.min_sharpe_ratio:
            alerts.append({
                'type': 'POOR_PERFORMANCE',
                'message': f'Sharpe ratio {metrics.sharpe_ratio:.3f} below minimum {self.risk_limits.min_sharpe_ratio:.3f}',
                'severity': 'LOW',
                'timestamp': metrics.timestamp
            })
        
        # Add alerts to history
        for alert in alerts:
            self.risk_alerts.append(alert)
            print(f"🚨 RISK ALERT [{alert['severity']}]: {alert['message']}")
    
    def get_optimal_position_size(self, 
                                 current_metrics: RiskMetrics,
                                 base_position: float,
                                 confidence: float = 1.0) -> float:
        """Calculate optimal position size considering all risk factors"""
        
        if self.emergency_stop:
            return 0.0  # Emergency stop - no new positions
        
        # Start with base position
        optimal_size = base_position
        
        # Apply Kelly Criterion scaling
        kelly_multiplier = min(1.0, max(0.1, current_metrics.kelly_fraction))
        optimal_size *= kelly_multiplier
        
        # Apply market regime adjustment
        regime_multiplier = self.regime_detector.get_regime_risk_multiplier(current_metrics.market_regime)
        optimal_size *= regime_multiplier
        
        # Apply confidence scaling
        confidence_multiplier = max(0.2, min(1.0, confidence))
        optimal_size *= confidence_multiplier
        
        # Apply volatility adjustment
        if current_metrics.volatility > 0.02:  # High volatility
            vol_multiplier = max(0.3, 1.0 - (current_metrics.volatility - 0.02) * 5)
            optimal_size *= vol_multiplier
        
        # Apply drawdown adjustment
        if current_metrics.max_drawdown > 0.05:  # Significant drawdown
            dd_multiplier = max(0.2, 1.0 - current_metrics.max_drawdown * 2)
            optimal_size *= dd_multiplier
        
        # Apply hard limits
        optimal_size = max(-self.risk_limits.max_position_size, 
                          min(self.risk_limits.max_position_size, optimal_size))
        
        return optimal_size
    
    def should_close_position(self, current_metrics: RiskMetrics) -> bool:
        """Determine if position should be closed due to risk"""
        
        if self.emergency_stop:
            return True
        
        # Close if in high-risk regime with poor performance
        if (current_metrics.market_regime in ["high_vol_ranging", "high_vol_trending"] and 
            current_metrics.confidence_score < 0.3):
            return True
        
        # Close if Sharpe ratio is very poor
        if current_metrics.sharpe_ratio < -1.0:
            return True
        
        # Close if VaR indicates high risk
        var_pct = abs(current_metrics.var_95) / self.initial_capital
        if var_pct > self.risk_limits.var_limit * 2:
            return True
        
        return False
    
    def get_risk_dashboard(self) -> Dict:
        """Generate comprehensive risk dashboard"""
        
        if not self.risk_metrics_history:
            return {"status": "no_data"}
        
        current_metrics = self.risk_metrics_history[-1]
        
        # Recent alerts (last 10)
        recent_alerts = self.risk_alerts[-10:] if self.risk_alerts else []
        critical_alerts = [a for a in recent_alerts if a['severity'] == 'CRITICAL']
        
        # Performance summary
        if len(self.risk_metrics_history) >= 2:
            performance_trend = "improving" if current_metrics.portfolio_value > self.risk_metrics_history[-2].portfolio_value else "declining"
        else:
            performance_trend = "stable"
        
        # Risk status
        if self.emergency_stop:
            risk_status = "EMERGENCY_STOP"
        elif len(critical_alerts) > 0:
            risk_status = "CRITICAL"
        elif current_metrics.max_drawdown > self.risk_limits.max_drawdown * 0.8:
            risk_status = "HIGH"
        elif current_metrics.volatility > 0.02:
            risk_status = "MEDIUM"
        else:
            risk_status = "LOW"
        
        dashboard = {
            "timestamp": current_metrics.timestamp,
            "risk_status": risk_status,
            "emergency_stop": self.emergency_stop,
            "portfolio": {
                "current_value": current_metrics.portfolio_value,
                "initial_capital": self.initial_capital,
                "total_pnl": current_metrics.unrealized_pnl + current_metrics.realized_pnl,
                "pnl_pct": (current_metrics.portfolio_value - self.initial_capital) / self.initial_capital,
                "performance_trend": performance_trend
            },
            "risk_metrics": {
                "current_position": current_metrics.current_position,
                "volatility": current_metrics.volatility,
                "var_95": current_metrics.var_95,
                "max_drawdown": current_metrics.max_drawdown,
                "sharpe_ratio": current_metrics.sharpe_ratio,
                "kelly_fraction": current_metrics.kelly_fraction
            },
            "market": {
                "regime": current_metrics.market_regime,
                "confidence_score": current_metrics.confidence_score,
                "regime_multiplier": self.regime_detector.get_regime_risk_multiplier(current_metrics.market_regime)
            },
            "alerts": {
                "total_alerts": len(self.risk_alerts),
                "recent_alerts": len(recent_alerts),
                "critical_alerts": len(critical_alerts),
                "latest_alerts": recent_alerts[-3:] if recent_alerts else []
            },
            "limits": {
                "max_position_size": self.risk_limits.max_position_size,
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "max_drawdown": self.risk_limits.max_drawdown,
                "var_limit": self.risk_limits.var_limit
            }
        }
        
        return dashboard
    
    def reset_emergency_stop(self):
        """Reset emergency stop (use with caution)"""
        self.emergency_stop = False
        print("⚠️ Emergency stop reset - monitor closely")
    
    def save_risk_data(self, filename: str = None):
        """Save risk management data to file"""
        
        if filename is None:
            filename = f"risk_management_data_{int(time.time())}.json"
        
        # Prepare data for JSON serialization
        metrics_data = []
        for metrics in self.risk_metrics_history:
            metrics_data.append({
                'timestamp': metrics.timestamp,
                'portfolio_value': metrics.portfolio_value,
                'unrealized_pnl': metrics.unrealized_pnl,
                'realized_pnl': metrics.realized_pnl,
                'current_position': metrics.current_position,
                'volatility': metrics.volatility,
                'var_95': metrics.var_95,
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'kelly_fraction': metrics.kelly_fraction,
                'market_regime': metrics.market_regime,
                'confidence_score': metrics.confidence_score
            })
        
        data = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'emergency_stop': self.emergency_stop,
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_daily_loss': self.risk_limits.max_daily_loss,
                'max_drawdown': self.risk_limits.max_drawdown,
                'var_limit': self.risk_limits.var_limit
            },
            'metrics_history': metrics_data,
            'risk_alerts': self.risk_alerts,
            'max_drawdown_seen': self.max_drawdown_seen,
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Risk management data saved to {filename}")
        return filename


def test_dynamic_risk_manager():
    """Test the dynamic risk management system"""
    
    print("🧪 Testing Dynamic Risk Management System")
    print("=" * 60)
    
    # Create risk manager
    risk_limits = RiskLimits(
        max_position_size=0.8,
        max_daily_loss=0.03,
        max_drawdown=0.08,
        var_limit=0.025
    )
    
    risk_manager = DynamicRiskManager(
        initial_capital=100000.0,
        risk_limits=risk_limits,
        lookback_window=50
    )
    
    print("\n📊 Simulating trading scenario...")
    
    # Simulate trading scenario
    base_price = 50000.0
    position = 0.0
    realized_pnl = 0.0
    
    for step in range(100):
        # Simulate price movement
        price_change = np.random.normal(0, 0.01) * base_price
        current_price = base_price + price_change * (step + 1) / 100
        
        # Simulate position changes
        if step % 10 == 0:  # Change position every 10 steps
            position = np.random.uniform(-0.5, 0.5)
        
        # Calculate unrealized PnL
        price_diff = current_price - base_price
        unrealized_pnl = position * price_diff
        
        # Update risk metrics
        metrics = risk_manager.update_portfolio_state(
            current_price=current_price,
            position=position,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl
        )
        
        # Get optimal position recommendation
        optimal_position = risk_manager.get_optimal_position_size(
            current_metrics=metrics,
            base_position=0.5,
            confidence=0.7
        )
        
        # Print periodic updates
        if step % 20 == 0:
            print(f"   Step {step:2d}: Price=${current_price:,.0f}, Pos={position:.3f}, "
                  f"PnL=${unrealized_pnl:,.0f}, Regime={metrics.market_regime}")
            print(f"           Optimal pos={optimal_position:.3f}, DD={metrics.max_drawdown:.2%}, "
                  f"Vol={metrics.volatility:.3f}")
    
    # Generate final dashboard
    dashboard = risk_manager.get_risk_dashboard()
    
    print(f"\n📋 FINAL RISK DASHBOARD:")
    print(f"   Risk Status: {dashboard['risk_status']}")
    print(f"   Portfolio Value: ${dashboard['portfolio']['current_value']:,.0f}")
    print(f"   Total PnL: ${dashboard['portfolio']['total_pnl']:,.0f} ({dashboard['portfolio']['pnl_pct']:.2%})")
    print(f"   Max Drawdown: {dashboard['risk_metrics']['max_drawdown']:.2%}")
    print(f"   Current Position: {dashboard['risk_metrics']['current_position']:.3f}")
    print(f"   Market Regime: {dashboard['market']['regime']}")
    print(f"   Confidence Score: {dashboard['market']['confidence_score']:.3f}")
    print(f"   Volatility: {dashboard['risk_metrics']['volatility']:.3f}")
    print(f"   Sharpe Ratio: {dashboard['risk_metrics']['sharpe_ratio']:.3f}")
    print(f"   Kelly Fraction: {dashboard['risk_metrics']['kelly_fraction']:.3f}")
    print(f"   Total Alerts: {dashboard['alerts']['total_alerts']}")
    
    # Save test data
    risk_manager.save_risk_data("test_risk_management.json")
    
    print(f"\n✅ Dynamic risk management system test completed!")
    print(f"   Processed {len(risk_manager.risk_metrics_history)} data points")
    print(f"   Detected {len(risk_manager.risk_alerts)} risk alerts")
    print(f"   Emergency stop: {risk_manager.emergency_stop}")
    
    return dashboard


if __name__ == "__main__":
    test_dynamic_risk_manager()