"""
Transaction Cost Analysis Module
Realistic modeling of trading costs, slippage, and market impact
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class TradingCost:
    """Individual trading cost component"""
    cost_type: str
    amount: float
    percentage: float
    description: str

@dataclass
class OrderExecution:
    """Order execution details with costs"""
    order_id: str
    timestamp: int
    side: OrderSide
    order_type: OrderType
    requested_quantity: float
    executed_quantity: float
    requested_price: float
    executed_price: float
    
    # Cost components
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    market_impact_cost: float = 0.0
    timing_cost: float = 0.0
    
    # Execution quality metrics
    fill_rate: float = 1.0
    price_improvement: float = 0.0
    execution_shortfall: float = 0.0
    
    # Market conditions at execution
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    volatility: float = 0.0

@dataclass
class CostModel:
    """Transaction cost model configuration"""
    # Commission structure
    commission_rate: float = 0.001  # 0.1% commission
    min_commission: float = 1.0
    max_commission: float = 100.0
    
    # Spread costs
    base_spread_bps: float = 5.0  # 5 basis points base spread
    spread_volatility_factor: float = 2.0  # Spread increases with volatility
    
    # Market impact model
    market_impact_factor: float = 0.01  # Impact factor
    sqrt_time_decay: float = 0.5  # Square root decay
    temporary_impact_factor: float = 0.005
    permanent_impact_factor: float = 0.002
    
    # Slippage model
    base_slippage_bps: float = 2.0  # 2 basis points base slippage
    volume_slippage_factor: float = 1.0  # Slippage increases with order size
    volatility_slippage_factor: float = 1.5  # Slippage increases with volatility
    
    # Timing costs
    delay_cost_factor: float = 0.001  # Cost of execution delay
    
    # Liquidity constraints
    max_participation_rate: float = 0.1  # Max 10% of volume
    min_order_size: float = 0.01
    max_order_size: float = 1000.0

class TransactionCostAnalyzer:
    """Comprehensive transaction cost analysis"""
    
    def __init__(self, cost_model: CostModel = None):
        self.cost_model = cost_model or CostModel()
        self.execution_history = []
        
    def calculate_execution_costs(self, 
                                order_side: OrderSide,
                                order_type: OrderType,
                                quantity: float,
                                target_price: float,
                                market_data: Dict[str, float],
                                order_id: str = None) -> OrderExecution:
        """Calculate realistic execution costs for an order"""
        
        if order_id is None:
            order_id = f"order_{len(self.execution_history)}"
        
        # Extract market data
        bid_price = market_data.get('bid', target_price * 0.999)
        ask_price = market_data.get('ask', target_price * 1.001)
        volume = market_data.get('volume', 1000.0)
        volatility = market_data.get('volatility', 0.02)
        
        # Calculate spread
        spread = ask_price - bid_price
        spread_bps = (spread / target_price) * 10000
        
        # Determine execution price based on order type
        execution_details = self._simulate_execution(
            order_side, order_type, quantity, target_price,
            bid_price, ask_price, volume, volatility
        )
        
        executed_price = execution_details['executed_price']
        executed_quantity = execution_details['executed_quantity']
        fill_rate = execution_details['fill_rate']
        
        # Calculate cost components
        costs = self._calculate_cost_components(
            order_side, quantity, executed_quantity, target_price, 
            executed_price, volume, volatility, spread
        )
        
        # Create execution record
        execution = OrderExecution(
            order_id=order_id,
            timestamp=len(self.execution_history),
            side=order_side,
            order_type=order_type,
            requested_quantity=quantity,
            executed_quantity=executed_quantity,
            requested_price=target_price,
            executed_price=executed_price,
            
            commission=costs['commission'],
            spread_cost=costs['spread_cost'],
            slippage_cost=costs['slippage_cost'],
            market_impact_cost=costs['market_impact_cost'],
            timing_cost=costs['timing_cost'],
            
            fill_rate=fill_rate,
            price_improvement=costs['price_improvement'],
            execution_shortfall=costs['execution_shortfall'],
            
            bid_price=bid_price,
            ask_price=ask_price,
            spread=spread,
            volume=volume,
            volatility=volatility
        )
        
        self.execution_history.append(execution)
        return execution
    
    def _simulate_execution(self, order_side: OrderSide, order_type: OrderType,
                          quantity: float, target_price: float,
                          bid_price: float, ask_price: float,
                          volume: float, volatility: float) -> Dict[str, float]:
        """Simulate order execution"""
        
        # Calculate maximum quantity based on liquidity constraints
        max_quantity = volume * self.cost_model.max_participation_rate
        effective_quantity = min(quantity, max_quantity)
        
        # Determine fill rate based on market conditions and order type
        if order_type == OrderType.MARKET:
            fill_rate = min(1.0, max_quantity / quantity)
            
            # Market orders execute at current market prices
            if order_side == OrderSide.BUY:
                base_price = ask_price
            else:
                base_price = bid_price
                
        elif order_type == OrderType.LIMIT:
            # Limit orders may not fill completely
            if order_side == OrderSide.BUY:
                if target_price >= ask_price:
                    fill_rate = min(1.0, max_quantity / quantity)
                    base_price = ask_price
                else:
                    # Probability of fill based on how aggressive the limit price is
                    aggressiveness = (target_price - bid_price) / (ask_price - bid_price)
                    fill_rate = min(1.0, max_quantity / quantity * max(0, aggressiveness))
                    base_price = target_price
            else:  # SELL
                if target_price <= bid_price:
                    fill_rate = min(1.0, max_quantity / quantity)
                    base_price = bid_price
                else:
                    aggressiveness = (ask_price - target_price) / (ask_price - bid_price)
                    fill_rate = min(1.0, max_quantity / quantity * max(0, aggressiveness))
                    base_price = target_price
        else:
            # Simplified for stop orders
            fill_rate = min(1.0, max_quantity / quantity)
            base_price = target_price
        
        # Apply market impact to execution price
        market_impact = self._calculate_market_impact(
            effective_quantity, volume, volatility, order_side
        )
        
        if order_side == OrderSide.BUY:
            executed_price = base_price + market_impact
        else:
            executed_price = base_price - market_impact
        
        # Add slippage
        slippage = self._calculate_slippage(effective_quantity, volume, volatility)
        executed_price += np.random.normal(0, slippage)
        
        return {
            'executed_price': executed_price,
            'executed_quantity': effective_quantity * fill_rate,
            'fill_rate': fill_rate
        }
    
    def _calculate_cost_components(self, order_side: OrderSide, requested_quantity: float,
                                 executed_quantity: float, target_price: float,
                                 executed_price: float, volume: float,
                                 volatility: float, spread: float) -> Dict[str, float]:
        """Calculate individual cost components"""
        
        costs = {}
        notional = executed_quantity * target_price
        
        # 1. Commission costs
        commission_amount = max(
            self.cost_model.min_commission,
            min(self.cost_model.max_commission, notional * self.cost_model.commission_rate)
        )
        costs['commission'] = commission_amount
        
        # 2. Spread costs (half spread for market orders)
        spread_cost = executed_quantity * spread * 0.5
        costs['spread_cost'] = spread_cost
        
        # 3. Slippage costs
        price_difference = abs(executed_price - target_price)
        slippage_cost = executed_quantity * price_difference
        costs['slippage_cost'] = slippage_cost
        
        # 4. Market impact costs
        market_impact_cost = self._calculate_market_impact_cost(
            executed_quantity, volume, volatility, target_price
        )
        costs['market_impact_cost'] = market_impact_cost
        
        # 5. Timing costs (opportunity cost of delay)
        timing_cost = executed_quantity * target_price * self.cost_model.delay_cost_factor
        costs['timing_cost'] = timing_cost
        
        # 6. Price improvement (negative cost)
        if order_side == OrderSide.BUY:
            improvement = max(0, target_price - executed_price)
        else:
            improvement = max(0, executed_price - target_price)
        costs['price_improvement'] = -improvement * executed_quantity
        
        # 7. Execution shortfall
        unfilled_quantity = requested_quantity - executed_quantity
        shortfall_cost = unfilled_quantity * target_price * 0.001  # Opportunity cost
        costs['execution_shortfall'] = shortfall_cost
        
        return costs
    
    def _calculate_market_impact(self, quantity: float, volume: float, 
                               volatility: float, order_side: OrderSide) -> float:
        """Calculate market impact using square root model"""
        
        if volume <= 0:
            return 0
        
        # Participation rate
        participation_rate = quantity / volume
        
        # Base impact using square root model
        base_impact = self.cost_model.market_impact_factor * np.sqrt(participation_rate)
        
        # Adjust for volatility
        volatility_adjustment = 1 + volatility * 10  # Scale volatility
        
        # Temporary vs permanent impact
        temporary_impact = base_impact * self.cost_model.temporary_impact_factor * volatility_adjustment
        permanent_impact = base_impact * self.cost_model.permanent_impact_factor * volatility_adjustment
        
        return temporary_impact + permanent_impact
    
    def _calculate_market_impact_cost(self, quantity: float, volume: float,
                                    volatility: float, price: float) -> float:
        """Calculate monetary cost of market impact"""
        
        impact_per_share = self._calculate_market_impact(quantity, volume, volatility, OrderSide.BUY)
        return quantity * impact_per_share
    
    def _calculate_slippage(self, quantity: float, volume: float, volatility: float) -> float:
        """Calculate expected slippage"""
        
        base_slippage = self.cost_model.base_slippage_bps / 10000
        
        # Volume-based slippage
        if volume > 0:
            volume_factor = (quantity / volume) * self.cost_model.volume_slippage_factor
        else:
            volume_factor = 1.0
        
        # Volatility-based slippage
        volatility_factor = volatility * self.cost_model.volatility_slippage_factor
        
        total_slippage = base_slippage * (1 + volume_factor + volatility_factor)
        
        return total_slippage
    
    def calculate_total_cost(self, execution: OrderExecution) -> float:
        """Calculate total cost for an execution"""
        
        total_cost = (execution.commission + 
                     execution.spread_cost + 
                     execution.slippage_cost + 
                     execution.market_impact_cost + 
                     execution.timing_cost + 
                     execution.price_improvement +  # This is negative for improvements
                     execution.execution_shortfall)
        
        return total_cost
    
    def calculate_cost_basis_points(self, execution: OrderExecution) -> float:
        """Calculate total cost in basis points"""
        
        total_cost = self.calculate_total_cost(execution)
        notional = execution.executed_quantity * execution.executed_price
        
        if notional == 0:
            return 0
        
        return (total_cost / notional) * 10000  # Convert to basis points
    
    def analyze_execution_quality(self, lookback_period: int = 100) -> Dict[str, float]:
        """Analyze execution quality over recent history"""
        
        if not self.execution_history:
            return {}
        
        # Get recent executions
        recent_executions = self.execution_history[-lookback_period:]
        
        # Calculate aggregate metrics
        total_costs = [self.calculate_total_cost(ex) for ex in recent_executions]
        cost_bps = [self.calculate_cost_basis_points(ex) for ex in recent_executions]
        fill_rates = [ex.fill_rate for ex in recent_executions]
        
        # Separate by order type and side
        market_orders = [ex for ex in recent_executions if ex.order_type == OrderType.MARKET]
        limit_orders = [ex for ex in recent_executions if ex.order_type == OrderType.LIMIT]
        
        buys = [ex for ex in recent_executions if ex.side == OrderSide.BUY]
        sells = [ex for ex in recent_executions if ex.side == OrderSide.SELL]
        
        analysis = {
            'total_executions': len(recent_executions),
            'avg_cost_bps': np.mean(cost_bps) if cost_bps else 0,
            'median_cost_bps': np.median(cost_bps) if cost_bps else 0,
            'avg_fill_rate': np.mean(fill_rates) if fill_rates else 0,
            'total_cost': sum(total_costs),
            
            # By order type
            'market_order_count': len(market_orders),
            'limit_order_count': len(limit_orders),
            'market_order_avg_cost_bps': np.mean([self.calculate_cost_basis_points(ex) for ex in market_orders]) if market_orders else 0,
            'limit_order_avg_cost_bps': np.mean([self.calculate_cost_basis_points(ex) for ex in limit_orders]) if limit_orders else 0,
            
            # By side
            'buy_order_count': len(buys),
            'sell_order_count': len(sells),
            'buy_avg_cost_bps': np.mean([self.calculate_cost_basis_points(ex) for ex in buys]) if buys else 0,
            'sell_avg_cost_bps': np.mean([self.calculate_cost_basis_points(ex) for ex in sells]) if sells else 0,
            
            # Cost breakdown
            'avg_commission_bps': np.mean([(ex.commission / (ex.executed_quantity * ex.executed_price)) * 10000 
                                         for ex in recent_executions if ex.executed_quantity > 0]) if recent_executions else 0,
            'avg_spread_cost_bps': np.mean([(ex.spread_cost / (ex.executed_quantity * ex.executed_price)) * 10000 
                                          for ex in recent_executions if ex.executed_quantity > 0]) if recent_executions else 0,
            'avg_slippage_bps': np.mean([(ex.slippage_cost / (ex.executed_quantity * ex.executed_price)) * 10000 
                                       for ex in recent_executions if ex.executed_quantity > 0]) if recent_executions else 0,
            'avg_market_impact_bps': np.mean([(ex.market_impact_cost / (ex.executed_quantity * ex.executed_price)) * 10000 
                                            for ex in recent_executions if ex.executed_quantity > 0]) if recent_executions else 0,
        }
        
        return analysis
    
    def optimize_execution_strategy(self, total_quantity: float, 
                                  target_price: float, 
                                  market_data: Dict[str, float],
                                  time_horizon: int = 10) -> List[Dict]:
        """Suggest optimal execution strategy"""
        
        # TWAP strategy - split into smaller orders
        num_slices = min(time_horizon, max(1, int(total_quantity / self.cost_model.min_order_size)))
        slice_size = total_quantity / num_slices
        
        strategy = []
        
        for i in range(num_slices):
            # Determine optimal order type based on market conditions
            spread = market_data.get('ask', target_price * 1.001) - market_data.get('bid', target_price * 0.999)
            volatility = market_data.get('volatility', 0.02)
            
            # Use limit orders in low volatility, market orders in high volatility
            if volatility < 0.015 and spread < target_price * 0.001:
                order_type = OrderType.LIMIT
                # Aggressive limit price
                limit_price = target_price * (1.0005 if i == 0 else 1.0001)  # More aggressive for first slice
            else:
                order_type = OrderType.MARKET
                limit_price = target_price
            
            strategy.append({
                'slice': i + 1,
                'quantity': slice_size,
                'order_type': order_type,
                'limit_price': limit_price,
                'delay_seconds': i * (time_horizon / num_slices) * 60,  # Space out orders
                'expected_cost_bps': self._estimate_execution_cost_bps(slice_size, market_data, order_type)
            })
        
        return strategy
    
    def _estimate_execution_cost_bps(self, quantity: float, 
                                   market_data: Dict[str, float],
                                   order_type: OrderType) -> float:
        """Estimate execution cost in basis points"""
        
        # Simplified cost estimation
        volume = market_data.get('volume', 1000)
        volatility = market_data.get('volatility', 0.02)
        price = market_data.get('mid', 100)
        
        # Base costs
        commission_bps = (self.cost_model.commission_rate * 10000)
        spread_bps = self.cost_model.base_spread_bps * (0.5 if order_type == OrderType.MARKET else 0.2)
        
        # Market impact
        participation_rate = quantity / volume if volume > 0 else 0.1
        impact_bps = self.cost_model.market_impact_factor * np.sqrt(participation_rate) * 10000
        
        # Slippage
        slippage_bps = self.cost_model.base_slippage_bps * (1 + volatility * 10)
        
        return commission_bps + spread_bps + impact_bps + slippage_bps
    
    def generate_cost_analysis_report(self, lookback_period: int = 500) -> str:
        """Generate comprehensive cost analysis report"""
        
        analysis = self.analyze_execution_quality(lookback_period)
        
        if not analysis:
            return "No execution history available for analysis."
        
        report = f"""
        
{'='*70}
TRANSACTION COST ANALYSIS REPORT
{'='*70}

EXECUTION SUMMARY
{'─'*30}
Total Executions:           {analysis['total_executions']:,}
Average Cost:               {analysis['avg_cost_bps']:.1f} bps
Median Cost:                {analysis['median_cost_bps']:.1f} bps
Average Fill Rate:          {analysis['avg_fill_rate']:.1%}
Total Cost:                 ${analysis['total_cost']:,.2f}

ORDER TYPE ANALYSIS
{'─'*30}
Market Orders:              {analysis['market_order_count']:,} ({analysis['market_order_count']/analysis['total_executions']*100:.1f}%)
  Average Cost:             {analysis['market_order_avg_cost_bps']:.1f} bps

Limit Orders:               {analysis['limit_order_count']:,} ({analysis['limit_order_count']/analysis['total_executions']*100:.1f}%)
  Average Cost:             {analysis['limit_order_avg_cost_bps']:.1f} bps

ORDER SIDE ANALYSIS
{'─'*30}
Buy Orders:                 {analysis['buy_order_count']:,}
  Average Cost:             {analysis['buy_avg_cost_bps']:.1f} bps

Sell Orders:                {analysis['sell_order_count']:,}
  Average Cost:             {analysis['sell_avg_cost_bps']:.1f} bps

COST BREAKDOWN (Average per Trade)
{'─'*30}
Commission:                 {analysis['avg_commission_bps']:.1f} bps
Spread Cost:                {analysis['avg_spread_cost_bps']:.1f} bps
Slippage:                   {analysis['avg_slippage_bps']:.1f} bps
Market Impact:              {analysis['avg_market_impact_bps']:.1f} bps

COST MODEL PARAMETERS
{'─'*30}
Commission Rate:            {self.cost_model.commission_rate*100:.3f}%
Base Spread:                {self.cost_model.base_spread_bps:.1f} bps
Market Impact Factor:       {self.cost_model.market_impact_factor:.4f}
Base Slippage:              {self.cost_model.base_slippage_bps:.1f} bps
Max Participation:          {self.cost_model.max_participation_rate*100:.1f}%

{'='*70}
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
        
        """
        
        return report

def main():
    """Example usage of transaction cost analyzer"""
    
    print("🚀 Transaction Cost Analysis Demo")
    print("=" * 50)
    
    # Initialize cost analyzer
    cost_model = CostModel(
        commission_rate=0.001,  # 0.1%
        base_spread_bps=5.0,
        market_impact_factor=0.01
    )
    
    analyzer = TransactionCostAnalyzer(cost_model)
    
    # Simulate some trades
    np.random.seed(42)
    
    for i in range(50):
        # Generate random market data
        market_data = {
            'bid': 100 - np.random.uniform(0.01, 0.05),
            'ask': 100 + np.random.uniform(0.01, 0.05),
            'volume': np.random.uniform(500, 2000),
            'volatility': np.random.uniform(0.01, 0.04),
            'mid': 100
        }
        
        # Random order parameters
        side = OrderSide.BUY if np.random.random() > 0.5 else OrderSide.SELL
        order_type = OrderType.MARKET if np.random.random() > 0.3 else OrderType.LIMIT
        quantity = np.random.uniform(10, 100)
        target_price = 100 + np.random.normal(0, 0.1)
        
        # Execute order
        execution = analyzer.calculate_execution_costs(
            side, order_type, quantity, target_price, market_data
        )
        
        print(f"Order {i+1}: {side.value} {quantity:.1f} @ {target_price:.2f} "
              f"-> Cost: {analyzer.calculate_cost_basis_points(execution):.1f} bps")
    
    # Generate analysis
    print("\n📊 Execution Quality Analysis:")
    analysis = analyzer.analyze_execution_quality()
    
    print(f"Average Cost: {analysis['avg_cost_bps']:.1f} bps")
    print(f"Average Fill Rate: {analysis['avg_fill_rate']:.1%}")
    print(f"Market vs Limit Orders: {analysis['market_order_count']} vs {analysis['limit_order_count']}")
    
    # Generate optimization strategy
    print("\n🎯 Execution Strategy Optimization:")
    strategy = analyzer.optimize_execution_strategy(
        total_quantity=500,
        target_price=100,
        market_data={'bid': 99.98, 'ask': 100.02, 'volume': 1000, 'volatility': 0.02}
    )
    
    for slice_info in strategy[:3]:  # Show first 3 slices
        print(f"Slice {slice_info['slice']}: {slice_info['quantity']:.1f} units "
              f"({slice_info['order_type'].value}) -> Est. {slice_info['expected_cost_bps']:.1f} bps")
    
    # Generate full report
    print("\n📝 Full Cost Analysis Report:")
    report = analyzer.generate_cost_analysis_report()
    print(report)

if __name__ == "__main__":
    main()