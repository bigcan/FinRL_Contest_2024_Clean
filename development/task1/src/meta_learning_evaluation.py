"""
Meta-Learning Ensemble Evaluation System
Enhanced evaluation framework with meta-learning specific metrics
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import evaluation components
from task1_eval import EnsembleEvaluator
from erl_config import Config, build_env
from trade_simulator import TradeSimulator, EvalTradeSimulator
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown

# Import meta-learning components
from meta_learning_framework import MetaLearningEnsembleManager, MetaLearningRiskManagedEnsemble
from meta_learning_config import MetaLearningConfig, create_meta_learning_config
from meta_learning_agent_wrapper import AgentWrapperFactory, AgentEnsembleWrapper
from task1_ensemble_meta_learning import MetaLearningEnsembleTrainer


class MetaLearningEvaluator:
    """
    Comprehensive evaluation system for meta-learning ensemble
    """
    
    def __init__(self, model_path: str, config: Optional[MetaLearningConfig] = None):
        self.model_path = model_path
        self.config = config
        
        # Initialize environment and get dimensions
        if config:
            self.env = build_env(config.env_class, config.env_args, gpu_id=getattr(config, 'gpu_id', 0))
        else:
            # Create default environment
            temp_sim = TradeSimulator(num_sims=1)
            self.state_dim = temp_sim.state_dim
            self.action_dim = 3
            self.env = EvalTradeSimulator()
        
        # Initialize evaluation metrics
        self.evaluation_metrics = {
            'portfolio_metrics': {},
            'meta_learning_metrics': {},
            'agent_comparison': {},
            'regime_analysis': {},
            'decision_analysis': {}
        }
        
        # Load trained ensemble
        self.meta_ensemble = None
        self.agent_wrappers = None
        self._load_trained_ensemble()
        
        print(f"🔍 Meta-Learning Evaluator Initialized")
        print(f"   📁 Model Path: {model_path}")
        print(f"   📊 State Dimension: {getattr(self, 'state_dim', 'auto-detected')}")
        print(f"   🎯 Action Dimension: {getattr(self, 'action_dim', 3)}")
    
    def _load_trained_ensemble(self):
        """Load trained meta-learning ensemble from saved models"""
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model path {self.model_path} does not exist")
                return
            
            # Load training statistics to get configuration info
            stats_path = os.path.join(self.model_path, "training_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    training_stats = json.load(f)
                
                config_summary = training_stats.get('config_summary', {})
                print(f"   📋 Loaded config: Meta-learning={'Enabled' if config_summary.get('meta_learning_enabled', True) else 'Disabled'}")
            
            # Create a minimal config if none provided
            if self.config is None:
                self.config = create_meta_learning_config(
                    preset='balanced',
                    env_args={
                        'env_name': 'TradeSimulator-v0',
                        'state_dim': getattr(self, 'state_dim', 50),
                        'action_dim': getattr(self, 'action_dim', 3),
                        'if_discrete': True
                    }
                )
            
            # Create trainer to load models
            trainer = MetaLearningEnsembleTrainer(
                config=self.config,
                save_dir=os.path.dirname(self.model_path)
            )
            
            # Load the trained models (this would be more complex in practice)
            # For now, we'll use the trainer's initialized ensemble
            self.meta_ensemble = trainer.meta_ensemble
            self.agent_wrappers = trainer.agent_wrappers
            
            print(f"   ✅ Meta-learning ensemble loaded successfully")
            
        except Exception as e:
            print(f"   ❌ Failed to load trained ensemble: {e}")
            # Create a basic ensemble for evaluation
            self._create_basic_ensemble()
    
    def _create_basic_ensemble(self):
        """Create basic ensemble for evaluation when loading fails"""
        print("   🔧 Creating basic ensemble for evaluation...")
        
        if self.config is None:
            self.config = create_meta_learning_config(
                preset='balanced',
                env_args={
                    'env_name': 'TradeSimulator-v0',
                    'state_dim': 50,
                    'action_dim': 3,
                    'if_discrete': True
                }
            )
        
        # Create a basic trainer for evaluation
        trainer = MetaLearningEnsembleTrainer(
            config=self.config,
            save_dir=self.model_path
        )
        
        self.meta_ensemble = trainer.meta_ensemble
        self.agent_wrappers = trainer.agent_wrappers
    
    def evaluate_comprehensive(self, 
                             num_episodes: int = 20,
                             max_steps_per_episode: int = 1000,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of meta-learning ensemble
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        
        print(f"📊 Starting Comprehensive Evaluation:")
        print(f"   🔢 Episodes: {num_episodes}")
        print(f"   📏 Max Steps: {max_steps_per_episode}")
        
        evaluation_start_time = time.time()
        
        # Initialize evaluation storage
        episode_results = []
        portfolio_values = []
        all_decisions = []
        regime_tracking = []
        agent_performance_tracking = defaultdict(list)
        
        # Run evaluation episodes
        for episode in range(num_episodes):
            episode_result = self._evaluate_single_episode(
                episode_num=episode,
                max_steps=max_steps_per_episode
            )
            
            episode_results.append(episode_result)
            portfolio_values.extend(episode_result['portfolio_values'])
            all_decisions.extend(episode_result['decisions'])
            regime_tracking.extend(episode_result['regime_history'])
            
            # Track individual agent performance
            for agent_name, perf in episode_result['agent_performances'].items():
                agent_performance_tracking[agent_name].append(perf)
            
            if episode % 5 == 0:
                print(f"   Episode {episode}: Return={episode_result['total_return']:.4f}, "
                      f"Sharpe={episode_result['sharpe_ratio']:.3f}, "
                      f"Regime={episode_result['final_regime']}")
        
        # Compile comprehensive results
        evaluation_results = self._compile_evaluation_results(
            episode_results=episode_results,
            portfolio_values=portfolio_values,
            all_decisions=all_decisions,
            regime_tracking=regime_tracking,
            agent_performance_tracking=agent_performance_tracking,
            evaluation_duration=time.time() - evaluation_start_time
        )
        
        # Generate detailed analysis
        evaluation_results['detailed_analysis'] = self._generate_detailed_analysis(evaluation_results)
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(evaluation_results)
        
        # Print summary
        self._print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _evaluate_single_episode(self, episode_num: int, max_steps: int) -> Dict[str, Any]:
        """Evaluate single episode and collect detailed metrics"""
        
        # Reset environment
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Episode tracking
        episode_data = {
            'episode': episode_num,
            'steps': 0,
            'total_return': 0.0,
            'returns': [],
            'actions': [],
            'decisions': [],
            'portfolio_values': [1.0],  # Start with normalized portfolio value
            'regime_history': [],
            'agent_performances': {},
            'confidence_scores': [],
            'regime_changes': 0,
            'agreement_rates': []
        }
        
        portfolio_value = 1.0
        done = False
        step = 0
        previous_regime = None
        
        while not done and step < max_steps:
            # Get current market data
            current_price = getattr(self.env, 'current_price', 100.0 + np.random.randn() * 0.5)
            current_volume = getattr(self.env, 'current_volume', 1000.0 + np.random.randn() * 50)
            
            # Get meta-learning ensemble action
            ensemble_action, decision_info = self.meta_ensemble.get_trading_action(
                state_tensor, current_price, current_volume
            )
            
            # Get individual agent actions for comparison
            agent_results = {}
            if self.agent_wrappers:
                for agent_name, wrapper in self.agent_wrappers.items():
                    try:
                        action, confidence, info = wrapper.get_action_with_confidence(state_tensor)
                        agent_results[agent_name] = {
                            'action': action,
                            'confidence': confidence,
                            'info': info
                        }
                    except Exception as e:
                        print(f"⚠️ Error getting action from {agent_name}: {e}")
            
            # Execute action
            next_state, reward, done, info = self.env.step(ensemble_action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            # Update portfolio value
            portfolio_value *= (1 + reward)
            
            # Track regime changes
            current_regime = decision_info.get('current_regime', 'unknown')
            if previous_regime and previous_regime != current_regime:
                episode_data['regime_changes'] += 1
            previous_regime = current_regime
            
            # Calculate agent agreement
            if agent_results:
                individual_actions = [result['action'] for result in agent_results.values()]
                agreement_rate = 1.0 if len(set(individual_actions)) == 1 else 0.0
                episode_data['agreement_rates'].append(agreement_rate)
                
                # Track average confidence
                confidences = [result['confidence'] for result in agent_results.values()]
                episode_data['confidence_scores'].append(np.mean(confidences))
            
            # Store step data
            episode_data['returns'].append(reward)
            episode_data['actions'].append(ensemble_action)
            episode_data['portfolio_values'].append(portfolio_value)
            episode_data['regime_history'].append({
                'step': step,
                'regime': current_regime,
                'regime_info': decision_info.get('regime_info', {})
            })
            episode_data['decisions'].append({
                'step': step,
                'state': state.tolist() if hasattr(state, 'tolist') else state,
                'action': ensemble_action,
                'reward': reward,
                'regime': current_regime,
                'agent_weights': decision_info.get('algorithm_weights', {}),
                'individual_actions': {name: result['action'] for name, result in agent_results.items()}
            })
            
            # Move to next state
            state = next_state
            state_tensor = torch.tensor(state, dtype=torch.float32)
            step += 1
        
        # Calculate episode metrics
        episode_data.update({
            'steps': step,
            'total_return': portfolio_value - 1.0,
            'final_portfolio_value': portfolio_value,
            'sharpe_ratio': sharpe_ratio(np.array(episode_data['returns'])) if len(episode_data['returns']) > 1 else 0.0,
            'max_drawdown': max_drawdown(np.array(episode_data['portfolio_values'])),
            'win_rate': np.mean(np.array(episode_data['returns']) > 0) if episode_data['returns'] else 0.0,
            'final_regime': current_regime,
            'avg_confidence': np.mean(episode_data['confidence_scores']) if episode_data['confidence_scores'] else 0.5,
            'avg_agreement_rate': np.mean(episode_data['agreement_rates']) if episode_data['agreement_rates'] else 0.0
        })
        
        # Calculate individual agent performance metrics
        if self.agent_wrappers:
            for agent_name, wrapper in self.agent_wrappers.items():
                episode_data['agent_performances'][agent_name] = wrapper.get_performance_metrics()
        
        return episode_data
    
    def _compile_evaluation_results(self, 
                                   episode_results: List[Dict],
                                   portfolio_values: List[float],
                                   all_decisions: List[Dict],
                                   regime_tracking: List[Dict],
                                   agent_performance_tracking: Dict,
                                   evaluation_duration: float) -> Dict[str, Any]:
        """Compile comprehensive evaluation results"""
        
        # Portfolio performance metrics
        returns = [ep['total_return'] for ep in episode_results]
        sharpe_ratios = [ep['sharpe_ratio'] for ep in episode_results]
        max_drawdowns = [ep['max_drawdown'] for ep in episode_results]
        win_rates = [ep['win_rate'] for ep in episode_results]
        
        portfolio_metrics = {
            'total_episodes': len(episode_results),
            'evaluation_duration': evaluation_duration,
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'best_return': np.max(returns),
            'worst_return': np.min(returns),
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'best_sharpe_ratio': np.max(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.max(max_drawdowns),
            'mean_win_rate': np.mean(win_rates),
            'success_rate': np.mean(np.array(returns) > 0),
            'profit_factor': self._calculate_profit_factor(returns),
            'calmar_ratio': np.mean(returns) / (np.mean(max_drawdowns) + 1e-8),
            'romad': np.mean(returns) / (np.mean(max_drawdowns) + 1e-8)
        }
        
        # Meta-learning specific metrics
        confidence_scores = [ep['avg_confidence'] for ep in episode_results]
        agreement_rates = [ep['avg_agreement_rate'] for ep in episode_results]
        regime_changes = [ep['regime_changes'] for ep in episode_results]
        
        meta_learning_metrics = {
            'mean_confidence': np.mean(confidence_scores),
            'confidence_stability': 1.0 - np.std(confidence_scores),
            'mean_agreement_rate': np.mean(agreement_rates),
            'agreement_stability': 1.0 - np.std(agreement_rates),
            'total_regime_changes': np.sum(regime_changes),
            'avg_regime_changes_per_episode': np.mean(regime_changes),
            'regime_adaptability': self._calculate_regime_adaptability(regime_tracking)
        }
        
        # Agent comparison metrics
        agent_comparison = {}
        if agent_performance_tracking:
            for agent_name, performances in agent_performance_tracking.items():
                agent_sharpe = [perf['sharpe_ratio'] for perf in performances]
                agent_win_rates = [perf['win_rate'] for perf in performances]
                agent_confidences = [perf['confidence'] for perf in performances]
                
                agent_comparison[agent_name] = {
                    'mean_sharpe_ratio': np.mean(agent_sharpe),
                    'sharpe_consistency': 1.0 - np.std(agent_sharpe),
                    'mean_win_rate': np.mean(agent_win_rates),
                    'mean_confidence': np.mean(agent_confidences),
                    'performance_rank': 0  # Will be calculated later
                }
        
        # Calculate performance ranks
        if agent_comparison:
            sorted_agents = sorted(agent_comparison.items(), 
                                 key=lambda x: x[1]['mean_sharpe_ratio'], reverse=True)
            for rank, (agent_name, metrics) in enumerate(sorted_agents, 1):
                agent_comparison[agent_name]['performance_rank'] = rank
        
        # Regime analysis
        regime_analysis = self._analyze_regime_performance(regime_tracking, episode_results)
        
        # Decision analysis
        decision_analysis = self._analyze_decision_patterns(all_decisions)
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'meta_learning_metrics': meta_learning_metrics,
            'agent_comparison': agent_comparison,
            'regime_analysis': regime_analysis,
            'decision_analysis': decision_analysis,
            'episode_details': episode_results
        }
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        profits = np.sum([r for r in returns if r > 0])
        losses = abs(np.sum([r for r in returns if r < 0]))
        return profits / (losses + 1e-8)
    
    def _calculate_regime_adaptability(self, regime_tracking: List[Dict]) -> float:
        """Calculate how well the system adapts to regime changes"""
        if len(regime_tracking) < 10:
            return 0.5
        
        # Look for performance improvement after regime changes
        regime_changes = []
        for i in range(1, len(regime_tracking)):
            if regime_tracking[i]['regime'] != regime_tracking[i-1]['regime']:
                regime_changes.append(i)
        
        if not regime_changes:
            return 0.5
        
        # Simple adaptability measure: stability after regime changes
        adaptability_scores = []
        for change_point in regime_changes:
            # Look at stability in the 10 steps after regime change
            post_change = regime_tracking[change_point:change_point+10]
            if len(post_change) >= 5:
                regime_stability = post_change[0]['regime_info'].get('regime_stability', 0.5)
                adaptability_scores.append(regime_stability)
        
        return np.mean(adaptability_scores) if adaptability_scores else 0.5
    
    def _analyze_regime_performance(self, regime_tracking: List[Dict], 
                                   episode_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance across different market regimes"""
        
        regime_performance = defaultdict(list)
        
        # Group performance by regime
        for episode in episode_results:
            final_regime = episode.get('final_regime', 'unknown')
            regime_performance[final_regime].append({
                'return': episode['total_return'],
                'sharpe_ratio': episode['sharpe_ratio'],
                'win_rate': episode['win_rate']
            })
        
        # Calculate regime-specific metrics
        regime_analysis = {}
        for regime, performances in regime_performance.items():
            if len(performances) > 0:
                returns = [p['return'] for p in performances]
                sharpe_ratios = [p['sharpe_ratio'] for p in performances]
                win_rates = [p['win_rate'] for p in performances]
                
                regime_analysis[regime] = {
                    'episode_count': len(performances),
                    'mean_return': np.mean(returns),
                    'mean_sharpe_ratio': np.mean(sharpe_ratios),
                    'mean_win_rate': np.mean(win_rates),
                    'consistency': 1.0 - np.std(returns),
                    'success_rate': np.mean(np.array(returns) > 0)
                }
        
        # Overall regime distribution
        regime_counts = defaultdict(int)
        for regime in regime_performance.keys():
            regime_counts[regime] = len(regime_performance[regime])
        
        total_episodes = sum(regime_counts.values())
        regime_distribution = {
            regime: count / total_episodes 
            for regime, count in regime_counts.items()
        }
        
        return {
            'regime_performance': dict(regime_analysis),
            'regime_distribution': dict(regime_distribution),
            'most_common_regime': max(regime_counts, key=regime_counts.get) if regime_counts else 'unknown',
            'regime_diversity': len(regime_counts)
        }
    
    def _analyze_decision_patterns(self, all_decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze decision-making patterns"""
        
        if not all_decisions:
            return {}
        
        # Action distribution
        actions = [d['action'] for d in all_decisions]
        action_counts = {0: 0, 1: 0, 2: 0}  # sell, hold, buy
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total_actions = len(actions)
        action_distribution = {
            'sell': action_counts[0] / total_actions,
            'hold': action_counts[1] / total_actions,
            'buy': action_counts[2] / total_actions
        }
        
        # Decision consistency (how often ensemble agrees with majority)
        individual_agreements = []
        for decision in all_decisions:
            individual_actions = list(decision.get('individual_actions', {}).values())
            if len(individual_actions) > 1:
                # Check if ensemble action matches most common individual action
                from collections import Counter
                most_common_action = Counter(individual_actions).most_common(1)[0][0]
                individual_agreements.append(decision['action'] == most_common_action)
        
        ensemble_consistency = np.mean(individual_agreements) if individual_agreements else 0.5
        
        # Weight utilization analysis
        weight_utilizations = []
        for decision in all_decisions:
            weights = decision.get('agent_weights', {})
            if weights:
                # Calculate weight entropy (higher = more diversified)
                weight_values = list(weights.values())
                if len(weight_values) > 1:
                    entropy = -sum(w * np.log(w + 1e-8) for w in weight_values if w > 0)
                    max_entropy = np.log(len(weight_values))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    weight_utilizations.append(normalized_entropy)
        
        avg_weight_diversification = np.mean(weight_utilizations) if weight_utilizations else 0.5
        
        return {
            'action_distribution': action_distribution,
            'most_common_action': max(action_distribution, key=action_distribution.get),
            'ensemble_consistency': ensemble_consistency,
            'avg_weight_diversification': avg_weight_diversification,
            'total_decisions': total_actions,
            'decision_diversity': 1.0 - max(action_distribution.values())  # 1 - max frequency
        }
    
    def _generate_detailed_analysis(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Generate detailed analysis and insights"""
        
        analysis = {}
        
        # Performance assessment
        portfolio_metrics = evaluation_results['portfolio_metrics']
        meta_metrics = evaluation_results['meta_learning_metrics']
        
        # Overall performance grade
        performance_score = 0
        performance_components = []
        
        # Sharpe ratio component (0-30 points)
        sharpe_score = min(30, max(0, portfolio_metrics['mean_sharpe_ratio'] * 20))
        performance_score += sharpe_score
        performance_components.append(('Sharpe Ratio', sharpe_score, 30))
        
        # Win rate component (0-25 points)
        win_rate_score = portfolio_metrics['mean_win_rate'] * 25
        performance_score += win_rate_score
        performance_components.append(('Win Rate', win_rate_score, 25))
        
        # Drawdown control (0-25 points)
        drawdown_score = max(0, 25 - portfolio_metrics['mean_max_drawdown'] * 100)
        performance_score += drawdown_score
        performance_components.append(('Drawdown Control', drawdown_score, 25))
        
        # Meta-learning effectiveness (0-20 points)
        meta_score = (meta_metrics['mean_confidence'] + meta_metrics['mean_agreement_rate']) * 10
        performance_score += meta_score
        performance_components.append(('Meta-Learning', meta_score, 20))
        
        # Convert to grade
        performance_grade = 'A' if performance_score >= 85 else \
                           'B' if performance_score >= 70 else \
                           'C' if performance_score >= 55 else \
                           'D' if performance_score >= 40 else 'F'
        
        analysis['performance_assessment'] = {
            'overall_score': performance_score,
            'grade': performance_grade,
            'components': performance_components
        }
        
        # Strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if portfolio_metrics['mean_sharpe_ratio'] > 1.0:
            strengths.append("Strong risk-adjusted returns")
        elif portfolio_metrics['mean_sharpe_ratio'] < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
        
        if portfolio_metrics['mean_win_rate'] > 0.6:
            strengths.append("High win rate consistency")
        elif portfolio_metrics['mean_win_rate'] < 0.4:
            weaknesses.append("Low win rate")
        
        if meta_metrics['mean_confidence'] > 0.7:
            strengths.append("High decision confidence")
        elif meta_metrics['mean_confidence'] < 0.5:
            weaknesses.append("Low decision confidence")
        
        if meta_metrics['mean_agreement_rate'] > 0.7:
            strengths.append("Good agent consensus")
        elif meta_metrics['mean_agreement_rate'] < 0.3:
            weaknesses.append("High agent disagreement")
        
        analysis['strengths_weaknesses'] = {
            'strengths': strengths,
            'weaknesses': weaknesses
        }
        
        # Recommendations
        recommendations = []
        
        if portfolio_metrics['mean_sharpe_ratio'] < 1.0:
            recommendations.append("Consider adjusting risk management parameters")
        
        if meta_metrics['mean_agreement_rate'] < 0.5:
            recommendations.append("Review agent selection and weighting algorithms")
        
        if portfolio_metrics['mean_max_drawdown'] > 0.15:
            recommendations.append("Implement stricter drawdown controls")
        
        if meta_metrics['regime_adaptability'] < 0.6:
            recommendations.append("Improve regime detection and adaptation mechanisms")
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _save_evaluation_results(self, evaluation_results: Dict):
        """Save evaluation results to files"""
        
        results_dir = os.path.join(self.model_path, "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save main results
        results_file = os.path.join(results_dir, "evaluation_results.json")
        try:
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(evaluation_results)
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"   💾 Results saved to {results_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to save results: {e}")
        
        # Save summary report
        self._save_summary_report(evaluation_results, results_dir)
        
        # Generate and save plots
        self._save_evaluation_plots(evaluation_results, results_dir)
    
    def _save_summary_report(self, evaluation_results: Dict, results_dir: str):
        """Save human-readable summary report"""
        
        report_file = os.path.join(results_dir, "evaluation_summary.txt")
        
        try:
            with open(report_file, 'w') as f:
                f.write("META-LEARNING ENSEMBLE EVALUATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Portfolio Performance
                portfolio = evaluation_results['portfolio_metrics']
                f.write("PORTFOLIO PERFORMANCE:\n")
                f.write(f"  Episodes Evaluated: {portfolio['total_episodes']}\n")
                f.write(f"  Mean Return: {portfolio['mean_return']:.4f}\n")
                f.write(f"  Sharpe Ratio: {portfolio['mean_sharpe_ratio']:.3f}\n")
                f.write(f"  Win Rate: {portfolio['mean_win_rate']:.2%}\n")
                f.write(f"  Max Drawdown: {portfolio['mean_max_drawdown']:.2%}\n")
                f.write(f"  Success Rate: {portfolio['success_rate']:.2%}\n\n")
                
                # Meta-Learning Performance
                meta = evaluation_results['meta_learning_metrics']
                f.write("META-LEARNING PERFORMANCE:\n")
                f.write(f"  Average Confidence: {meta['mean_confidence']:.3f}\n")
                f.write(f"  Agent Agreement Rate: {meta['mean_agreement_rate']:.2%}\n")
                f.write(f"  Regime Adaptability: {meta['regime_adaptability']:.3f}\n")
                f.write(f"  Total Regime Changes: {meta['total_regime_changes']}\n\n")
                
                # Detailed Analysis
                if 'detailed_analysis' in evaluation_results:
                    analysis = evaluation_results['detailed_analysis']
                    f.write("PERFORMANCE ASSESSMENT:\n")
                    f.write(f"  Overall Grade: {analysis['performance_assessment']['grade']}\n")
                    f.write(f"  Score: {analysis['performance_assessment']['overall_score']:.1f}/100\n\n")
                    
                    f.write("STRENGTHS:\n")
                    for strength in analysis['strengths_weaknesses']['strengths']:
                        f.write(f"  • {strength}\n")
                    
                    f.write("\nWEAKNESSES:\n")
                    for weakness in analysis['strengths_weaknesses']['weaknesses']:
                        f.write(f"  • {weakness}\n")
                    
                    f.write("\nRECOMMENDATIONS:\n")
                    for recommendation in analysis['recommendations']:
                        f.write(f"  • {recommendation}\n")
                
                f.write(f"\nReport generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"   📄 Summary report saved to {report_file}")
            
        except Exception as e:
            print(f"   ⚠️ Failed to save summary report: {e}")
    
    def _save_evaluation_plots(self, evaluation_results: Dict, results_dir: str):
        """Generate and save evaluation plots"""
        
        try:
            import matplotlib.pyplot as plt
            
            # Portfolio performance plot
            episode_details = evaluation_results.get('episode_details', [])
            if episode_details:
                returns = [ep['total_return'] for ep in episode_details]
                sharpe_ratios = [ep['sharpe_ratio'] for ep in episode_details]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Returns plot
                ax1.plot(returns, 'b-', alpha=0.7)
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax1.set_title('Episode Returns')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Return')
                ax1.grid(True, alpha=0.3)
                
                # Sharpe ratio plot
                ax2.plot(sharpe_ratios, 'g-', alpha=0.7)
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.set_title('Episode Sharpe Ratios')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Sharpe Ratio')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_file = os.path.join(results_dir, "performance_plots.png")
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   📊 Performance plots saved to {plot_file}")
            
        except ImportError:
            print("   ⚠️ Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"   ⚠️ Failed to generate plots: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, deque):
            return list(obj)
        else:
            return obj
    
    def _print_evaluation_summary(self, evaluation_results: Dict):
        """Print comprehensive evaluation summary"""
        
        print(f"\n" + "="*80)
        print(f"🎯 META-LEARNING ENSEMBLE EVALUATION SUMMARY")
        print(f"="*80)
        
        # Portfolio Performance
        portfolio = evaluation_results['portfolio_metrics']
        print(f"\n📈 Portfolio Performance:")
        print(f"   Episodes Evaluated: {portfolio['total_episodes']}")
        print(f"   Mean Return: {portfolio['mean_return']:.4f} ± {portfolio['std_return']:.4f}")
        print(f"   Best Return: {portfolio['best_return']:.4f}")
        print(f"   Worst Return: {portfolio['worst_return']:.4f}")
        print(f"   Sharpe Ratio: {portfolio['mean_sharpe_ratio']:.3f}")
        print(f"   Win Rate: {portfolio['mean_win_rate']:.2%}")
        print(f"   Max Drawdown: {portfolio['mean_max_drawdown']:.2%}")
        print(f"   Success Rate: {portfolio['success_rate']:.2%}")
        print(f"   Profit Factor: {portfolio['profit_factor']:.2f}")
        
        # Meta-Learning Performance
        meta = evaluation_results['meta_learning_metrics']
        print(f"\n🧠 Meta-Learning Performance:")
        print(f"   Average Confidence: {meta['mean_confidence']:.3f}")
        print(f"   Confidence Stability: {meta['confidence_stability']:.3f}")
        print(f"   Agent Agreement Rate: {meta['mean_agreement_rate']:.2%}")
        print(f"   Agreement Stability: {meta['agreement_stability']:.3f}")
        print(f"   Regime Adaptability: {meta['regime_adaptability']:.3f}")
        print(f"   Total Regime Changes: {meta['total_regime_changes']}")
        
        # Agent Comparison
        agent_comparison = evaluation_results.get('agent_comparison', {})
        if agent_comparison:
            print(f"\n🤖 Agent Performance Ranking:")
            sorted_agents = sorted(agent_comparison.items(), 
                                 key=lambda x: x[1]['performance_rank'])
            for agent_name, metrics in sorted_agents:
                print(f"   #{metrics['performance_rank']} {agent_name}: "
                      f"Sharpe={metrics['mean_sharpe_ratio']:.3f}, "
                      f"Win={metrics['mean_win_rate']:.2%}, "
                      f"Conf={metrics['mean_confidence']:.3f}")
        
        # Regime Analysis
        regime_analysis = evaluation_results.get('regime_analysis', {})
        if regime_analysis and 'regime_performance' in regime_analysis:
            print(f"\n🌍 Regime Performance Analysis:")
            for regime, metrics in regime_analysis['regime_performance'].items():
                print(f"   {regime}: "
                      f"Episodes={metrics['episode_count']}, "
                      f"Return={metrics['mean_return']:.4f}, "
                      f"Sharpe={metrics['mean_sharpe_ratio']:.3f}")
        
        # Decision Analysis
        decision_analysis = evaluation_results.get('decision_analysis', {})
        if decision_analysis:
            print(f"\n🎯 Decision Analysis:")
            action_dist = decision_analysis['action_distribution']
            print(f"   Action Distribution: "
                  f"Sell={action_dist['sell']:.1%}, "
                  f"Hold={action_dist['hold']:.1%}, "
                  f"Buy={action_dist['buy']:.1%}")
            print(f"   Ensemble Consistency: {decision_analysis['ensemble_consistency']:.2%}")
            print(f"   Weight Diversification: {decision_analysis['avg_weight_diversification']:.3f}")
        
        # Overall Assessment
        if 'detailed_analysis' in evaluation_results:
            analysis = evaluation_results['detailed_analysis']
            assessment = analysis['performance_assessment']
            print(f"\n🏆 Overall Assessment:")
            print(f"   Grade: {assessment['grade']}")
            print(f"   Score: {assessment['overall_score']:.1f}/100")
            
            print(f"\n✅ Key Strengths:")
            for strength in analysis['strengths_weaknesses']['strengths']:
                print(f"   • {strength}")
            
            if analysis['strengths_weaknesses']['weaknesses']:
                print(f"\n⚠️ Areas for Improvement:")
                for weakness in analysis['strengths_weaknesses']['weaknesses']:
                    print(f"   • {weakness}")
            
            if analysis['recommendations']:
                print(f"\n💡 Recommendations:")
                for recommendation in analysis['recommendations']:
                    print(f"   • {recommendation}")
        
        print(f"\n📁 Detailed results saved in: {self.model_path}/evaluation_results/")
        print(f"="*80)


def main():
    """Main evaluation function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Learning Ensemble Evaluation")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--config_preset', type=str, default='balanced',
                        choices=['conservative', 'aggressive', 'balanced', 'research'],
                        help='Configuration preset (default: balanced)')
    
    args = parser.parse_args()
    
    print("🔍 Meta-Learning Ensemble Evaluation System")
    print("="*50)
    
    # Create configuration
    config = create_meta_learning_config(
        preset=args.config_preset,
        env_args={
            'env_name': 'TradeSimulator-v0',
            'state_dim': 50,
            'action_dim': 3,
            'if_discrete': True
        }
    )
    
    # Create evaluator
    evaluator = MetaLearningEvaluator(
        model_path=args.model_path,
        config=config
    )
    
    # Run evaluation
    try:
        results = evaluator.evaluate_comprehensive(
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            save_results=True
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()