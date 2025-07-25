"""
Production Monitoring System for FinRL Contest 2024
Real-time monitoring, alerting, and performance tracking
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import logging
from dataclasses import dataclass, asdict
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    level: str  # 'critical', 'warning', 'info'
    category: str  # 'performance', 'risk', 'system', 'market'
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    action_required: Optional[str] = None

@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot"""
    timestamp: datetime
    pnl: float
    sharpe_ratio: float
    drawdown: float
    win_rate: float
    position_size: float
    market_regime: str
    model_confidence: float
    system_latency: float
    error_rate: float

class MetricsCollector:
    """Collects and aggregates trading metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.returns_buffer = deque(maxlen=window_size)
        self.trades_buffer = deque(maxlen=window_size)
        self.latency_buffer = deque(maxlen=100)
        self.error_buffer = deque(maxlen=1000)
        
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.peak_value = 1.0
        self.current_drawdown = 0
        
    def add_trade(self, pnl: float, is_winner: bool):
        """Add trade result"""
        self.total_trades += 1
        self.total_pnl += pnl
        self.trades_buffer.append({'pnl': pnl, 'is_winner': is_winner})
        
        if is_winner:
            self.winning_trades += 1
            
        # Update peak and drawdown
        current_value = 1.0 + self.total_pnl
        if current_value > self.peak_value:
            self.peak_value = current_value
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        
    def add_latency(self, latency_ms: float):
        """Add latency measurement"""
        self.latency_buffer.append(latency_ms)
        
    def add_error(self, error_type: str):
        """Add error occurrence"""
        self.error_buffer.append({
            'timestamp': datetime.now(),
            'type': error_type
        })
        
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        recent_trades = list(self.trades_buffer)
        
        if recent_trades:
            recent_pnls = [t['pnl'] for t in recent_trades]
            recent_winners = [t['is_winner'] for t in recent_trades]
            
            # Calculate Sharpe ratio (simplified)
            returns = np.array(recent_pnls)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
                
            win_rate = np.mean(recent_winners) if recent_winners else 0
        else:
            sharpe = 0
            win_rate = 0
            
        # Calculate error rate
        recent_errors = [e for e in self.error_buffer 
                        if e['timestamp'] > datetime.now() - timedelta(minutes=5)]
        error_rate = len(recent_errors) / max(self.total_trades, 1)
        
        return {
            'total_pnl': self.total_pnl,
            'sharpe_ratio': sharpe,
            'current_drawdown': self.current_drawdown,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'avg_latency': np.mean(self.latency_buffer) if self.latency_buffer else 0,
            'p99_latency': np.percentile(self.latency_buffer, 99) if self.latency_buffer else 0,
            'error_rate': error_rate
        }

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Alert thresholds
        self.thresholds = {
            'critical': {
                'drawdown': 0.03,
                'error_rate': 0.01,
                'latency_p99': 100,
                'model_confidence': 0.3
            },
            'warning': {
                'drawdown': 0.02,
                'error_rate': 0.005,
                'latency_p99': 50,
                'model_confidence': 0.5,
                'sharpe_ratio': 0.5
            }
        }
        
    def check_metrics(self, metrics: Dict) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        timestamp = datetime.now()
        
        # Check drawdown
        if metrics.get('current_drawdown', 0) > self.thresholds['critical']['drawdown']:
            alerts.append(Alert(
                timestamp=timestamp,
                level='critical',
                category='risk',
                message=f"Critical drawdown: {metrics['current_drawdown']:.1%}",
                metric_value=metrics['current_drawdown'],
                threshold=self.thresholds['critical']['drawdown'],
                action_required="Reduce position sizes immediately"
            ))
        elif metrics.get('current_drawdown', 0) > self.thresholds['warning']['drawdown']:
            alerts.append(Alert(
                timestamp=timestamp,
                level='warning',
                category='risk',
                message=f"High drawdown: {metrics['current_drawdown']:.1%}",
                metric_value=metrics['current_drawdown'],
                threshold=self.thresholds['warning']['drawdown']
            ))
            
        # Check error rate
        if metrics.get('error_rate', 0) > self.thresholds['critical']['error_rate']:
            alerts.append(Alert(
                timestamp=timestamp,
                level='critical',
                category='system',
                message=f"Critical error rate: {metrics['error_rate']:.1%}",
                metric_value=metrics['error_rate'],
                threshold=self.thresholds['critical']['error_rate'],
                action_required="Investigate system errors immediately"
            ))
            
        # Check latency
        if metrics.get('p99_latency', 0) > self.thresholds['critical']['latency_p99']:
            alerts.append(Alert(
                timestamp=timestamp,
                level='critical',
                category='system',
                message=f"Critical latency: {metrics['p99_latency']:.0f}ms",
                metric_value=metrics['p99_latency'],
                threshold=self.thresholds['critical']['latency_p99'],
                action_required="Check system performance"
            ))
            
        # Check Sharpe ratio
        if metrics.get('sharpe_ratio', 1) < self.thresholds['warning']['sharpe_ratio']:
            alerts.append(Alert(
                timestamp=timestamp,
                level='warning',
                category='performance',
                message=f"Low Sharpe ratio: {metrics['sharpe_ratio']:.2f}",
                metric_value=metrics['sharpe_ratio'],
                threshold=self.thresholds['warning']['sharpe_ratio']
            ))
            
        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
            self._trigger_callbacks(alert)
            
        return alerts
        
    def _trigger_callbacks(self, alert: Alert):
        """Trigger registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
    def register_callback(self, callback):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
        
    def get_recent_alerts(self, minutes: int = 60) -> List[Alert]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self.alert_history if a.timestamp > cutoff]

class MarketRegimeDetector:
    """Detects current market regime"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.price_buffer = deque(maxlen=lookback_period)
        self.volume_buffer = deque(maxlen=lookback_period)
        
    def update(self, price: float, volume: float):
        """Update with new market data"""
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
    def get_regime(self) -> Tuple[str, float]:
        """Get current market regime and confidence"""
        if len(self.price_buffer) < 20:
            return 'unknown', 0.0
            
        prices = np.array(self.price_buffer)
        returns = np.diff(np.log(prices))
        
        # Calculate regime indicators
        volatility = np.std(returns) * np.sqrt(252)
        trend = np.mean(returns[-20:]) * 252
        
        # Classify regime
        if abs(trend) < 0.005:
            if volatility > 0.03:
                regime = 'high_vol_sideways'
            else:
                regime = 'low_vol_sideways'
        elif trend > 0.01:
            if volatility > 0.03:
                regime = 'high_vol_bull'
            else:
                regime = 'low_vol_bull'
        else:
            if volatility > 0.03:
                regime = 'high_vol_bear'
            else:
                regime = 'low_vol_bear'
                
        # Calculate confidence (simplified)
        confidence = min(len(self.price_buffer) / self.lookback_period, 1.0)
        
        return regime, confidence

class ProductionMonitor:
    """Main production monitoring system"""
    
    def __init__(self, model_path: str, update_interval: int = 10):
        self.model_path = model_path
        self.update_interval = update_interval
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.regime_detector = MarketRegimeDetector()
        
        # State
        self.is_running = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=10000)
        
        # Register default alert handler
        self.alert_manager.register_callback(self._handle_alert)
        
    def start(self):
        """Start monitoring"""
        if self.is_running:
            logger.warning("Monitor already running")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logger.info("Production monitor started")
        
    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Production monitor stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Check for alerts
                alerts = self.alert_manager.check_metrics(metrics)
                
                # Store snapshot
                snapshot = self._create_snapshot(metrics)
                self.metrics_history.append(snapshot)
                
                # Log status
                if len(alerts) > 0:
                    logger.warning(f"Generated {len(alerts)} alerts")
                else:
                    logger.info(f"System healthy - PnL: ${metrics['total_pnl']:.2f}, "
                              f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                              f"Drawdown: {metrics['current_drawdown']:.1%}")
                
                # Sleep
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self.metrics_collector.add_error('monitor_loop_error')
                
    def _collect_metrics(self) -> Dict:
        """Collect current metrics"""
        # Get base metrics
        metrics = self.metrics_collector.get_metrics()
        
        # Add regime information
        regime, confidence = self.regime_detector.get_regime()
        metrics['market_regime'] = regime
        metrics['regime_confidence'] = confidence
        
        # Add model confidence (placeholder - would come from actual model)
        metrics['model_confidence'] = 0.85
        
        return metrics
        
    def _create_snapshot(self, metrics: Dict) -> MetricSnapshot:
        """Create metric snapshot"""
        return MetricSnapshot(
            timestamp=datetime.now(),
            pnl=metrics.get('total_pnl', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            drawdown=metrics.get('current_drawdown', 0),
            win_rate=metrics.get('win_rate', 0),
            position_size=0,  # Would come from actual trading
            market_regime=metrics.get('market_regime', 'unknown'),
            model_confidence=metrics.get('model_confidence', 0),
            system_latency=metrics.get('avg_latency', 0),
            error_rate=metrics.get('error_rate', 0)
        )
        
    def _handle_alert(self, alert: Alert):
        """Default alert handler"""
        if alert.level == 'critical':
            logger.critical(f"🚨 CRITICAL ALERT: {alert.message}")
            # In production, this would trigger notifications
        elif alert.level == 'warning':
            logger.warning(f"⚠️ WARNING: {alert.message}")
        else:
            logger.info(f"ℹ️ INFO: {alert.message}")
            
    def add_trade_result(self, pnl: float, is_winner: bool):
        """Add trade result to monitoring"""
        self.metrics_collector.add_trade(pnl, is_winner)
        
    def add_latency_measurement(self, latency_ms: float):
        """Add latency measurement"""
        self.metrics_collector.add_latency(latency_ms)
        
    def update_market_data(self, price: float, volume: float):
        """Update market data for regime detection"""
        self.regime_detector.update(price, volume)
        
    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard display"""
        recent_metrics = self.metrics_collector.get_metrics()
        recent_alerts = self.alert_manager.get_recent_alerts(60)
        
        # Get recent history for charts
        recent_snapshots = list(self.metrics_history)[-100:]
        
        return {
            'current_metrics': recent_metrics,
            'recent_alerts': [asdict(a) for a in recent_alerts],
            'metrics_history': [asdict(s) for s in recent_snapshots],
            'system_status': 'healthy' if len([a for a in recent_alerts if a.level == 'critical']) == 0 else 'critical'
        }
        
    def save_monitoring_data(self, filepath: str):
        """Save monitoring data to file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics_history': [asdict(s) for s in self.metrics_history],
            'alert_history': [asdict(a) for a in self.alert_manager.alert_history],
            'current_metrics': self.metrics_collector.get_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"Monitoring data saved to {filepath}")

class DashboardServer:
    """Simple dashboard server for monitoring visualization"""
    
    def __init__(self, monitor: ProductionMonitor, port: int = 8080):
        self.monitor = monitor
        self.port = port
        
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        data = self.monitor.get_dashboard_data()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FinRL Production Monitor</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metric-card {{ background: white; padding: 20px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
                .metric-label {{ color: #666; margin-bottom: 5px; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .alert-critical {{ background-color: #ffcccc; border-left: 4px solid #ff0000; }}
                .alert-warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                .alert-info {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; }}
                .status-healthy {{ color: #28a745; }}
                .status-critical {{ color: #dc3545; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>FinRL Production Monitor</h1>
                <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="metric-card">
                    <h2>System Status: <span class="status-{data['system_status']}">{data['system_status'].upper()}</span></h2>
                </div>
                
                <h2>Current Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total PnL</div>
                        <div class="metric-value">${data['current_metrics']['total_pnl']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{data['current_metrics']['sharpe_ratio']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Current Drawdown</div>
                        <div class="metric-value">{data['current_metrics']['current_drawdown']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{data['current_metrics']['win_rate']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Latency</div>
                        <div class="metric-value">{data['current_metrics']['avg_latency']:.1f}ms</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Error Rate</div>
                        <div class="metric-value">{data['current_metrics']['error_rate']:.2%}</div>
                    </div>
                </div>
                
                <h2>Recent Alerts</h2>
                <div class="metric-card">
        """
        
        # Add alerts
        if data['recent_alerts']:
            for alert in data['recent_alerts'][-10:]:  # Show last 10 alerts
                html += f"""
                    <div class="alert alert-{alert['level']}">
                        <strong>[{alert['timestamp']}]</strong> {alert['message']}
                        {f" - Action: {alert['action_required']}" if alert.get('action_required') else ""}
                    </div>
                """
        else:
            html += "<p>No recent alerts</p>"
            
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

def demo_monitoring_system():
    """Demonstrate the monitoring system"""
    print("🚀 Starting Production Monitoring System Demo")
    print("=" * 60)
    
    # Create monitor
    monitor = ProductionMonitor("ensemble_optimized_phase2", update_interval=5)
    
    # Start monitoring
    monitor.start()
    
    # Simulate trading activity
    print("\n📊 Simulating trading activity...")
    
    for i in range(20):
        # Simulate trade
        pnl = np.random.normal(0.001, 0.01)  # Random PnL
        is_winner = pnl > 0
        monitor.add_trade_result(pnl, is_winner)
        
        # Simulate latency
        latency = np.random.normal(20, 5)
        monitor.add_latency_measurement(max(0, latency))
        
        # Simulate market data
        price = 50000 + np.random.normal(0, 100)
        volume = np.random.uniform(0.1, 1.0)
        monitor.update_market_data(price, volume)
        
        # Occasionally simulate errors
        if np.random.random() < 0.05:
            monitor.metrics_collector.add_error('connection_error')
        
        time.sleep(1)
        
    # Get dashboard data
    print("\n📈 Current Dashboard Data:")
    dashboard_data = monitor.get_dashboard_data()
    
    print(f"System Status: {dashboard_data['system_status']}")
    print(f"Total PnL: ${dashboard_data['current_metrics']['total_pnl']:.2f}")
    print(f"Sharpe Ratio: {dashboard_data['current_metrics']['sharpe_ratio']:.2f}")
    print(f"Win Rate: {dashboard_data['current_metrics']['win_rate']:.1%}")
    print(f"Recent Alerts: {len(dashboard_data['recent_alerts'])}")
    
    # Save monitoring data
    monitor.save_monitoring_data(f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Stop monitoring
    monitor.stop()
    
    print("\n✅ Monitoring system demo complete!")


if __name__ == "__main__":
    demo_monitoring_system()