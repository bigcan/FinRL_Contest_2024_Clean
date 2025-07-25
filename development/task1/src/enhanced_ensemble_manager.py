#!/usr/bin/env python3
"""
Enhanced Ensemble Manager with Kelly Position Sizing and Performance Weighting
Integrates Kelly Criterion position sizing with performance-weighted ensemble voting
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from kelly_position_sizing import KellyPositionSizer, EnsembleKellyManager
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN


class EnhancedEnsembleManager:
    """
    Advanced ensemble manager combining:
    1. Performance-weighted voting
    2. Kelly Criterion position sizing
    3. Confidence-based decision making
    4. Dynamic agent selection
    """
    
    def __init__(self, 
                 agents: List,
                 agent_names: List[str],
                 performance_window: int = 50,
                 min_confidence_threshold: float = 0.1,
                 volatility_lookback: int = 20):
        """
        Initialize enhanced ensemble manager
        
        Args:
            agents: List of trained agent objects
            agent_names: Names corresponding to agents
            performance_window: Window for tracking agent performance
            min_confidence_threshold: Minimum confidence for making trades
            volatility_lookback: Window for calculating market volatility
        """
        self.agents = agents
        self.agent_names = agent_names
        self.performance_window = performance_window
        self.min_confidence_threshold = min_confidence_threshold
        self.volatility_lookback = volatility_lookback
        
        # Performance tracking
        self.agent_performance = {name: deque(maxlen=performance_window) for name in agent_names}
        self.agent_trade_counts = {name: 0 for name in agent_names}
        self.agent_weights = {name: 1.0 / len(agent_names) for name in agent_names}
        
        # Kelly position sizing
        self.kelly_manager = EnsembleKellyManager(
            agent_names=agent_names,
            lookback_window=100,
            min_trades=20,
            kelly_multiplier=0.5,
            max_kelly_fraction=0.25
        )
        
        # Market volatility tracking
        self.price_history = deque(maxlen=volatility_lookback)
        self.current_volatility = 1.0
        
        # Decision tracking
        self.decision_history = deque(maxlen=200)
        
        print(f"🎯 Enhanced Ensemble Manager initialized:")
        print(f"   Agents: {len(agents)} ({', '.join(agent_names)})")
        print(f"   Performance window: {performance_window}")
        print(f"   Kelly position sizing: Enabled")
        print(f"   Performance weighting: Enabled")
    
    def update_market_volatility(self, current_price: float):
        """Update market volatility estimate"""
        self.price_history.append(current_price)
        
        if len(self.price_history) >= 10:
            prices = np.array(list(self.price_history))
            returns = np.diff(np.log(prices))
            self.current_volatility = np.std(returns) * np.sqrt(len(returns))
        
        # Normalize volatility (1.0 = normal, >1.0 = high volatility)
        self.current_volatility = max(0.5, min(3.0, self.current_volatility * 100))
    
    def get_agent_predictions(self, state: torch.Tensor) -> Dict[str, Dict]:
        """
        Get predictions from all agents with confidence scores
        
        Returns:
            Dict mapping agent_name to {action, q_values, confidence}
        """
        predictions = {}
        
        for agent, agent_name in zip(self.agents, self.agent_names):
            try:
                with torch.no_grad():
                    q_values = agent.act(state)
                    
                    # Get action
                    action = q_values.argmax(dim=1, keepdim=True)[0].item()
                    
                    # Calculate confidence (normalized Q-value spread)
                    q_array = q_values[0].cpu().numpy()
                    q_max = q_array.max()
                    q_mean = q_array.mean()
                    
                    # Confidence based on how much the best action stands out
                    confidence = max(0.0, min(1.0, (q_max - q_mean) / (abs(q_mean) + 1e-6)))
                    
                    predictions[agent_name] = {
                        'action': action,
                        'q_values': q_array,
                        'confidence': confidence,
                        'raw_q_values': q_values
                    }
                    
            except Exception as e:
                print(f"   ⚠️ Error getting prediction from {agent_name}: {e}")
                # Default prediction
                predictions[agent_name] = {
                    'action': 0,  # Hold
                    'q_values': np.array([0.0, 0.0, 0.0]),
                    'confidence': 0.0,
                    'raw_q_values': None
                }
        
        return predictions
    
    def calculate_performance_weights(self) -> Dict[str, float]:
        """Calculate performance-based weights for ensemble voting"""
        
        weights = {}
        
        for agent_name in self.agent_names:
            performance_data = list(self.agent_performance[agent_name])
            
            if len(performance_data) < 5:
                # Insufficient data, use equal weight
                weights[agent_name] = 1.0 / len(self.agent_names)
            else:
                # Calculate performance metrics
                returns = np.array(performance_data)
                
                # Sharpe ratio (risk-adjusted performance)
                if np.std(returns) > 1e-6:
                    sharpe = np.mean(returns) / np.std(returns)
                else:
                    sharpe = 0.0
                
                # Win rate
                win_rate = np.mean(returns > 0)
                
                # Recent performance (last 10 trades weighted more)
                recent_performance = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
                
                # Combined performance score
                performance_score = (
                    0.4 * sharpe +           # Risk-adjusted returns
                    0.3 * win_rate +         # Consistency
                    0.3 * recent_performance # Recent performance
                )
                
                weights[agent_name] = max(0.01, performance_score)  # Minimum weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in weights.items()}
        else:
            weights = {name: 1.0 / len(self.agent_names) for name in self.agent_names}
        
        self.agent_weights = weights
        return weights
    
    def make_ensemble_decision(self, 
                             predictions: Dict[str, Dict],
                             use_performance_weighting: bool = True) -> Tuple[int, float, Dict]:
        """
        Make ensemble decision using performance weighting and confidence
        
        Returns:
            Tuple of (action, confidence, decision_info)
        """
        
        # Calculate performance weights
        if use_performance_weighting:
            weights = self.calculate_performance_weights()
        else:
            weights = {name: 1.0 / len(self.agent_names) for name in self.agent_names}
        
        # Weighted voting for each action
        action_votes = {0: 0.0, 1: 0.0, 2: 0.0}
        total_confidence = 0.0
        valid_agents = 0
        
        decision_breakdown = {}
        
        for agent_name, pred in predictions.items():
            agent_weight = weights.get(agent_name, 0.0)
            agent_confidence = pred['confidence']
            agent_action = pred['action']
            
            # Weight the vote by both performance and confidence
            vote_strength = agent_weight * (0.5 + 0.5 * agent_confidence)
            action_votes[agent_action] += vote_strength
            
            total_confidence += agent_confidence * agent_weight
            valid_agents += 1
            
            decision_breakdown[agent_name] = {
                'action': agent_action,
                'confidence': agent_confidence,
                'weight': agent_weight,
                'vote_strength': vote_strength
            }
        
        # Select winning action
        ensemble_action = max(action_votes, key=action_votes.get)
        ensemble_confidence = total_confidence / max(1, valid_agents)
        
        # Additional confidence factors
        vote_margin = sorted(action_votes.values(), reverse=True)
        if len(vote_margin) >= 2:
            # Higher confidence when there's a clear winner
            margin_bonus = (vote_margin[0] - vote_margin[1]) / (vote_margin[0] + 1e-6)
            ensemble_confidence = min(1.0, ensemble_confidence + 0.2 * margin_bonus)
        
        decision_info = {
            'action_votes': action_votes,
            'agent_weights': weights,
            'decision_breakdown': decision_breakdown,
            'vote_margin': vote_margin[0] - vote_margin[1] if len(vote_margin) >= 2 else 0.0,
            'participating_agents': valid_agents
        }
        
        return ensemble_action, ensemble_confidence, decision_info
    
    def get_kelly_position_size(self, 
                               action: int, 
                               confidence: float) -> float:
        """Get Kelly-optimal position size for the ensemble decision"""
        
        return self.kelly_manager.get_ensemble_position_size(
            ensemble_action=action,
            ensemble_confidence=confidence,
            market_volatility=self.current_volatility
        )
    
    def execute_trading_decision(self, 
                               state: torch.Tensor,
                               current_price: float = None) -> Dict:
        """
        Complete trading decision pipeline:
        1. Get agent predictions
        2. Make ensemble decision
        3. Calculate Kelly position size
        4. Return trading instruction
        """
        
        decision_start = time.time()
        
        # Update market volatility if price provided
        if current_price is not None:
            self.update_market_volatility(current_price)
        
        # Get predictions from all agents
        predictions = self.get_agent_predictions(state)
        
        # Make ensemble decision
        action, confidence, decision_info = self.make_ensemble_decision(predictions)
        
        # Get Kelly position size
        position_size = self.get_kelly_position_size(action, confidence)
        
        # Check minimum confidence threshold
        if confidence < self.min_confidence_threshold:
            action = 0  # Force hold
            position_size = 0.0
            decision_info['forced_hold'] = True
            decision_info['reason'] = f'Confidence {confidence:.3f} below threshold {self.min_confidence_threshold}'
        
        # Create trading instruction
        trading_instruction = {
            'action': action,
            'position_size': position_size,
            'confidence': confidence,
            'market_volatility': self.current_volatility,
            'decision_time': time.time() - decision_start,
            'decision_info': decision_info,
            'agent_predictions': predictions
        }
        
        # Store decision
        self.decision_history.append(trading_instruction)
        
        return trading_instruction
    
    def update_performance(self, 
                          agent_results: Dict[str, float],
                          ensemble_result: float):
        """
        Update performance tracking for agents and ensemble
        
        Args:
            agent_results: Dict mapping agent_name to return
            ensemble_result: Ensemble return for this trade
        """
        
        # Update individual agent performance
        for agent_name, result in agent_results.items():
            if agent_name in self.agent_performance:
                self.agent_performance[agent_name].append(result)
                self.agent_trade_counts[agent_name] += 1
                
                # Add to Kelly manager
                # Note: We'd need the action from the decision history
                if len(self.decision_history) > 0:
                    last_decision = self.decision_history[-1]
                    agent_pred = last_decision['agent_predictions'].get(agent_name, {})
                    agent_action = agent_pred.get('action', 0)
                    
                    self.kelly_manager.add_agent_trade(agent_name, agent_action, result)
        
        # Update ensemble performance
        if len(self.decision_history) > 0:
            last_decision = self.decision_history[-1]
            ensemble_action = last_decision['action']
            self.kelly_manager.add_ensemble_trade(ensemble_action, ensemble_result)
    
    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        
        print(f"\\n🎯 ENHANCED ENSEMBLE PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # Agent weights
        print(f"\\n📊 Current Agent Weights (Performance-Based):")
        weights = self.calculate_performance_weights()
        for agent_name, weight in weights.items():
            trade_count = self.agent_trade_counts[agent_name]
            print(f"   {agent_name:15}: {weight:.1%} ({trade_count} trades)")
        
        # Recent decisions
        if len(self.decision_history) > 0:
            recent_decisions = list(self.decision_history)[-10:]
            print(f"\\n📈 Recent Decisions (Last {len(recent_decisions)}):")
            
            actions_count = {0: 0, 1: 0, 2: 0}
            avg_confidence = 0.0
            avg_position_size = 0.0
            
            for decision in recent_decisions:
                actions_count[decision['action']] += 1
                avg_confidence += decision['confidence']
                avg_position_size += decision['position_size']
            
            total_decisions = len(recent_decisions)
            print(f"   Hold: {actions_count[0]/total_decisions:.1%}")
            print(f"   Buy:  {actions_count[1]/total_decisions:.1%}")
            print(f"   Sell: {actions_count[2]/total_decisions:.1%}")
            print(f"   Avg Confidence: {avg_confidence/total_decisions:.1%}")
            print(f"   Avg Position Size: {avg_position_size/total_decisions:.1%}")
            print(f"   Market Volatility: {self.current_volatility:.2f}x")
        
        # Kelly statistics
        print(f"\\n💰 Kelly Position Sizing Statistics:")
        self.kelly_manager.print_all_statistics()


def test_enhanced_ensemble():
    """Test the enhanced ensemble manager"""
    
    print("🧪 Testing Enhanced Ensemble Manager")
    print("=" * 60)
    
    # Mock agents for testing (normally these would be trained agents)
    class MockAgent:
        def __init__(self, name, bias=0.0):
            self.name = name
            self.bias = bias
        
        def act(self, state):
            # Return random Q-values with some bias
            base_q = torch.randn(1, 3) + self.bias
            return base_q
    
    # Create mock agents
    agents = [
        MockAgent("AgentD3QN", bias=0.1),
        MockAgent("AgentDoubleDQN", bias=-0.1),
        MockAgent("AgentTwinD3QN", bias=0.05)
    ]
    agent_names = [agent.name for agent in agents]
    
    # Create enhanced ensemble
    ensemble = EnhancedEnsembleManager(
        agents=agents,
        agent_names=agent_names,
        performance_window=20
    )
    
    # Simulate trading decisions
    state = torch.randn(1, 8)  # Mock state
    
    print(f"\\n🎮 Simulating 10 trading decisions...")
    
    for i in range(10):
        # Make trading decision
        decision = ensemble.execute_trading_decision(state, current_price=50000 + i * 100)
        
        print(f"\\n   Decision {i+1}:")
        print(f"     Action: {decision['action']} (0=Hold, 1=Buy, 2=Sell)")
        print(f"     Position Size: {decision['position_size']:.1%}")
        print(f"     Confidence: {decision['confidence']:.1%}")
        print(f"     Volatility: {decision['market_volatility']:.2f}x")
        
        # Simulate performance results
        agent_results = {}
        for name in agent_names:
            # Random performance with some correlation to decision quality
            performance = np.random.normal(0.002, 0.01) * decision['confidence']
            agent_results[name] = performance
        
        ensemble_result = np.mean(list(agent_results.values()))
        
        # Update performance tracking
        ensemble.update_performance(agent_results, ensemble_result)
    
    # Print final summary
    ensemble.print_performance_summary()
    
    print(f"\\n✅ Enhanced ensemble manager test complete!")


if __name__ == "__main__":
    test_enhanced_ensemble()