import os
import sys

# Fix encoding issues on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import numpy as np
from erl_config import Config, build_env
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN, AgentNoisyDQN, AgentNoisyDuelDQN, AgentRainbowDQN, AgentAdaptiveDQN
from collections import Counter
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


class EnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config, ensemble_method='majority_voting', 
                 performance_window=100, weight_decay=0.95, verbose_logging=False):
        self.save_path = save_path
        self.agent_classes = agent_classes
        self.ensemble_method = ensemble_method
        self.performance_window = performance_window
        self.weight_decay = weight_decay
        self.verbose_logging = verbose_logging

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        # Get state_dim from TradeSimulator (supports both original and enhanced features)
        temp_sim = TradeSimulator(num_sims=1)
        self.state_dim = temp_sim.state_dim
        print(f"Using state_dim: {self.state_dim}")
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)

        # REMOVED: Independent portfolio tracking that caused lookahead bias
        # No longer maintaining self.current_btc, self.cash, self.btc_assets, self.net_assets
        # Will use TradeSimulator's ground truth values exclusively
        self.starting_cash = args.starting_cash
        
        # Track ground truth portfolio values from TradeSimulator
        self.portfolio_history = []  # Will store actual portfolio values from trade_env
        
        # Advanced ensemble tracking
        self._initialize_performance_tracking()

    def _initialize_performance_tracking(self):
        """Initialize comprehensive performance tracking for each agent"""
        self.agent_performance = {
            'returns': [],           # Individual agent returns
            'correct_predictions': [], # Win/loss tracking per agent
            'confidence_scores': [], # Q-value confidence tracking
            'action_history': [],    # Action history per agent
            'weights_history': [],   # Weight evolution over time
        }
        
        self.performance_metrics = {
            'rolling_returns': [],   # Rolling window returns per agent
            'rolling_sharpe': [],    # Rolling Sharpe ratio per agent
            'win_rates': [],         # Win rate per agent
            'profit_factors': [],    # Profit factor per agent
        }
        
        self.ensemble_stats = {
            'total_decisions': 0,
            'agent_contributions': [], # Track which agent's action was selected
            'weight_updates': [],      # Track weight change events
            'regime_detections': [],   # Market regime change events
        }

    def load_agents(self):
        args = self.args
        for i, agent_class in enumerate(self.agent_classes):
            agent = agent_class(
                args.net_dims,
                args.state_dim,
                args.action_dim,
                gpu_id=args.gpu_id,
                args=args,
            )
            agent_name = agent_class.__name__
            cwd = os.path.join(self.save_path, agent_name)
            agent.save_or_load_agent(cwd, if_save=False)  # Load agent
            self.agents.append(agent)
            
            # Initialize tracking for this agent
            self._initialize_agent_tracking(i, agent_name)
            
        # Initialize agent weights (equal initially)
        num_agents = len(self.agents)
        self.agent_weights = np.ones(num_agents) / num_agents
        self.min_weight = 0.05  # Minimum weight to prevent complete exclusion
        self.max_weight = 0.7   # Maximum weight to prevent over-reliance
        
        print(f"🤖 Loaded {num_agents} agents with equal initial weights: {self.agent_weights}")

    def _initialize_agent_tracking(self, agent_idx, agent_name):
        """Initialize tracking arrays for a specific agent"""
        # Ensure lists exist for this agent
        for key in self.agent_performance:
            if len(self.agent_performance[key]) <= agent_idx:
                self.agent_performance[key].extend([[] for _ in range(agent_idx + 1 - len(self.agent_performance[key]))])
        
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) <= agent_idx:
                self.performance_metrics[key].extend([[] for _ in range(agent_idx + 1 - len(self.performance_metrics[key]))])
                
        if self.verbose_logging:
            print(f"📊 Initialized tracking for Agent {agent_idx}: {agent_name}")

    def multi_trade(self):
        """Evaluation loop using ensemble of agents"""

        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()

        last_state = state
        last_price = 0

        positions = []
        action_ints = []
        correct_pred = []
        # REMOVED: current_btcs tracking (was using biased self.current_btc)
        # Position tracking now handled via TradeSimulator ground truth

        for step_idx in range(trade_env.max_step):
            actions = []
            q_values_list = []
            intermediate_state = last_state

            # Collect actions and Q-values from each agent
            for i, agent in enumerate(agents):
                actor = agent.act
                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                tensor_q_values = actor(tensor_state)
                tensor_action = tensor_q_values.argmax(dim=1)
                action = tensor_action.detach().cpu().unsqueeze(1)
                
                actions.append(action)
                q_values_list.append(tensor_q_values.detach().cpu())
                
                # Store action history for this agent (take first element from batch)
                self.agent_performance['action_history'][i].append(action[0].item())

            # Enhanced ensemble action with Q-values
            action, selected_agent_idx = self._ensemble_action(
                actions=actions, 
                q_values=q_values_list,
                method=self.ensemble_method, 
                step_idx=step_idx
            )
            action_int = action[0].item() - 1
            
            # DEBUG: Track actions and positions
            if step_idx < 10 or step_idx % 500 == 0:  # Print first 10 steps and every 500 steps
                print(f"DEBUG Step {step_idx}: Raw action={action[0].item()}, action_int={action_int}")
                if hasattr(trade_env, 'position'):
                    old_position = trade_env.position[0].item() if hasattr(trade_env.position, 'item') else trade_env.position[0]
                    print(f"DEBUG Step {step_idx}: Position before={old_position}")

            state, reward, done, _ = trade_env.step(action=action)
            reward_value = reward[0].item() if torch.is_tensor(reward) else reward
            
            # DEBUG: Track position changes and rewards
            if step_idx < 10 or step_idx % 500 == 0:
                if hasattr(trade_env, 'position'):
                    new_position = trade_env.position[0].item() if hasattr(trade_env.position, 'item') else trade_env.position[0] 
                    print(f"DEBUG Step {step_idx}: Position after={new_position}, reward={reward_value}")
                if hasattr(trade_env, 'asset'):
                    asset_value = trade_env.asset[0].item() if hasattr(trade_env.asset, 'item') else trade_env.asset[0]
                    print(f"DEBUG Step {step_idx}: Asset value={asset_value}")
                    
            # Track action distribution
            if not hasattr(self, 'action_counts'):
                self.action_counts = {0: 0, 1: 0, 2: 0}  # sell, hold, buy
            self.action_counts[action[0].item()] += 1
            
            # Track individual agent performance and update ensemble
            self._update_agent_performance(reward_value, selected_agent_idx, step_idx, actions, q_values_list)
            
            # Update performance tracking for adaptive ensemble methods
            if self.ensemble_method in ['weighted_voting', 'adaptive_meta']:
                self.update_performance_tracking(reward_value)

            action_ints.append(action_int)
            positions.append(trade_env.position)

            # FIXED: Use TradeSimulator's ground truth portfolio values (no lookahead bias)
            # Extract actual portfolio state after trade execution
            true_asset_value = trade_env.asset[0].item()  # Ground truth total asset value
            true_cash_value = trade_env.cash[0].item()    # Ground truth cash balance
            true_position = trade_env.position[0].item()  # Ground truth position
            
            # Store ground truth portfolio value for metrics calculation
            self.portfolio_history.append({
                'step': step_idx,
                'total_asset': true_asset_value,
                'cash': true_cash_value,
                'position': true_position,
                'action': action_int
            })

            last_state = state

            # Log win rate using ground truth values (remove lookahead bias)
            if step_idx > 0:  # Need previous portfolio value for comparison
                prev_asset = self.portfolio_history[-2]['total_asset'] if len(self.portfolio_history) > 1 else self.starting_cash
                asset_change = true_asset_value - prev_asset
                
                if action_int == 1:  # Buy action
                    correct_pred.append(1 if asset_change > 0 else -1 if asset_change < 0 else 0)
                elif action_int == -1:  # Sell action
                    correct_pred.append(-1 if asset_change > 0 else 1 if asset_change < 0 else 0)
                else:  # Hold action
                    correct_pred.append(0)
            else:
                correct_pred.append(0)  # First step, no comparison possible

        # FIXED: Save ground truth results (no more biased portfolio values)
        positions_cpu = [pos.cpu().numpy() if hasattr(pos, 'cpu') else pos for pos in positions]
        np.save("evaluation_positions.npy", np.array(positions_cpu))
        
        # Extract ground truth portfolio values for saving and metrics
        true_net_assets = [entry['total_asset'] for entry in self.portfolio_history]
        true_cash_values = [entry['cash'] for entry in self.portfolio_history]
        true_positions = [entry['position'] for entry in self.portfolio_history]
        
        np.save("evaluation_net_assets.npy", np.array(true_net_assets))
        np.save("evaluation_cash_values.npy", np.array(true_cash_values))
        np.save("evaluation_true_positions.npy", np.array(true_positions))
        np.save("evaluation_correct_predictions.npy", np.array(correct_pred))

        # Print honest action distribution
        if hasattr(self, 'action_counts'):
            total_actions = sum(self.action_counts.values())
            print(f"\n📊 HONEST Action Distribution (Fixed Evaluation):")
            print(f"  Sell (0): {self.action_counts[0]} ({self.action_counts[0]/total_actions*100:.1f}%)")
            print(f"  Hold (1): {self.action_counts[1]} ({self.action_counts[1]/total_actions*100:.1f}%)")
            print(f"  Buy (2): {self.action_counts[2]} ({self.action_counts[2]/total_actions*100:.1f}%)")
            print(f"  Total: {total_actions} actions")

        # FIXED: Compute metrics using ground truth portfolio values
        if len(true_net_assets) > 1:
            returns = np.diff(true_net_assets) / np.array(true_net_assets[:-1])
            
            print(f"\n🔍 HONEST Portfolio Analysis (Corrected for Lookahead Bias):")
            print(f"   Starting portfolio value: ${true_net_assets[0]:,.2f}")
            print(f"   Final portfolio value: ${true_net_assets[-1]:,.2f}")
            print(f"   Total return: {(true_net_assets[-1] / true_net_assets[0] - 1)*100:.2f}%")
            print(f"   Number of returns: {len(returns)}")
            print(f"   Returns mean: {np.mean(returns):.6f}")
            print(f"   Returns std: {np.std(returns):.6f}")
            print(f"   Contains NaN: {np.isnan(returns).any()}")
            print(f"   Contains Inf: {np.isinf(returns).any()}")
            
            # Calculate corrected performance metrics
            final_sharpe_ratio = sharpe_ratio(returns)
            final_max_drawdown = max_drawdown(returns)
            final_roma = return_over_max_drawdown(returns)

            print(f"\n📊 CORRECTED Performance Metrics:")
            print(f"   Sharpe Ratio: {final_sharpe_ratio:.4f}")
            print(f"   Max Drawdown: {final_max_drawdown:.4f}")
            print(f"   Return over Max Drawdown: {final_roma:.4f}")
            
            # Reality check on results
            if final_sharpe_ratio > 2.5:
                print(f"⚠️  WARNING: Sharpe ratio ({final_sharpe_ratio:.2f}) still unusually high - may need further investigation")
            elif final_sharpe_ratio > 0:
                print(f"✅ Sharpe ratio ({final_sharpe_ratio:.2f}) within realistic range")
            else:
                print(f"📉 Negative Sharpe ratio ({final_sharpe_ratio:.2f}) - strategy underperforming")
        else:
            print(f"❌ Error: Insufficient portfolio history for metrics calculation")
            final_sharpe_ratio = final_max_drawdown = final_roma = 0.0
        
        # Print ensemble performance summary
        self._print_ensemble_summary()

    def _print_ensemble_summary(self):
        """Print comprehensive ensemble performance summary"""
        print(f"\n🎯 =============== ENSEMBLE PERFORMANCE SUMMARY ===============")
        print(f"🔧 Ensemble Method: {self.ensemble_method}")
        print(f"📊 Total Decisions: {len(self.ensemble_stats['agent_contributions'])}")
        
        if self.ensemble_stats['agent_contributions']:
            # Agent contribution analysis
            from collections import Counter
            contributions = Counter(self.ensemble_stats['agent_contributions'])
            print(f"\n🤖 Agent Contributions:")
            agent_names = [cls.__name__ for cls in self.agent_classes]
            for i, name in enumerate(agent_names):
                count = contributions.get(i, 0)
                percentage = (count / len(self.ensemble_stats['agent_contributions'])) * 100
                print(f"   {name}: {count} decisions ({percentage:.1f}%)")
        
        # Final weights
        if hasattr(self, 'agent_weights'):
            print(f"\n⚖️  Final Agent Weights:")
            for i, weight in enumerate(self.agent_weights):
                print(f"   {agent_names[i]}: {weight:.3f}")
        
        # Performance metrics summary
        if self.performance_metrics['rolling_returns']:
            print(f"\n📈 Final Performance Metrics:")
            for i, name in enumerate(agent_names):
                if (i < len(self.performance_metrics['rolling_returns']) and 
                    self.performance_metrics['rolling_returns'][i]):
                    last_return = self.performance_metrics['rolling_returns'][i][-1]
                    last_sharpe = (self.performance_metrics['rolling_sharpe'][i][-1] 
                                 if self.performance_metrics['rolling_sharpe'][i] else 0)
                    last_win_rate = (self.performance_metrics['win_rates'][i][-1] 
                                   if self.performance_metrics['win_rates'][i] else 0)
                    print(f"   {name}: Return={last_return:.4f}, Sharpe={last_sharpe:.2f}, WinRate={last_win_rate:.1%}")
        
        print(f"🎯 =========================================================\n")

    def _update_agent_performance(self, reward, selected_agent_idx, step_idx, actions, q_values_list):
        """Update comprehensive agent performance tracking"""
        num_agents = len(self.agents)
        
        # Track which agent's action was selected
        if selected_agent_idx is not None:
            self.ensemble_stats['agent_contributions'].append(selected_agent_idx)
        
        # Update confidence scores (max Q-value for each agent)
        for i, q_values in enumerate(q_values_list):
            max_q_value = torch.max(q_values).item()
            self.agent_performance['confidence_scores'][i].append(max_q_value)
        
        # Estimate individual agent returns (simulate what each would have earned)
        current_price = self.trade_env.price_ary[self.trade_env.step_i, 2].item()
        if hasattr(self, 'last_price') and self.last_price > 0:
            price_change = (current_price - self.last_price) / self.last_price
            
            for i, action in enumerate(actions):
                action_val = action[0].item() - 1  # Convert to -1, 0, 1
                agent_return = action_val * price_change  # Simulate agent-specific return
                self.agent_performance['returns'][i].append(agent_return)
                
                # Track correct predictions
                if action_val != 0:  # Only for buy/sell actions
                    correct = (action_val > 0 and price_change > 0) or (action_val < 0 and price_change < 0)
                    self.agent_performance['correct_predictions'][i].append(1 if correct else 0)
        
        self.last_price = current_price
        
        # Update rolling metrics every N steps
        if step_idx % 50 == 0 and step_idx > 0:
            self._update_rolling_metrics()
            
        # Update weights based on recent performance
        if step_idx % 100 == 0 and step_idx > 0 and self.ensemble_method == 'weighted_voting':
            self._update_agent_weights()

    def _update_rolling_metrics(self):
        """Update rolling performance metrics for each agent"""
        window = min(self.performance_window, len(self.agent_performance['returns'][0]))
        if window < 10:  # Need minimum data
            return
            
        for i in range(len(self.agents)):
            # Rolling returns
            recent_returns = self.agent_performance['returns'][i][-window:]
            if recent_returns:
                rolling_return = np.mean(recent_returns)
                self.performance_metrics['rolling_returns'][i].append(rolling_return)
                
                # Rolling Sharpe ratio
                if len(recent_returns) > 1:
                    returns_std = np.std(recent_returns)
                    if returns_std > 0:
                        sharpe = rolling_return / returns_std * np.sqrt(252)  # Annualized
                        self.performance_metrics['rolling_sharpe'][i].append(sharpe)
                    else:
                        self.performance_metrics['rolling_sharpe'][i].append(0.0)
                
                # Win rate
                recent_predictions = self.agent_performance['correct_predictions'][i][-window:]
                if recent_predictions:
                    win_rate = np.mean(recent_predictions)
                    self.performance_metrics['win_rates'][i].append(win_rate)

    def _update_agent_weights(self):
        """Update agent weights based on recent performance"""
        if not self.performance_metrics['rolling_returns'] or not all(self.performance_metrics['rolling_returns']):
            return
            
        new_weights = np.zeros(len(self.agents))
        
        for i in range(len(self.agents)):
            # Combine multiple metrics
            recent_return = self.performance_metrics['rolling_returns'][i][-1] if self.performance_metrics['rolling_returns'][i] else 0
            recent_sharpe = self.performance_metrics['rolling_sharpe'][i][-1] if self.performance_metrics['rolling_sharpe'][i] else 0
            recent_win_rate = self.performance_metrics['win_rates'][i][-1] if self.performance_metrics['win_rates'][i] else 0.5
            
            # Combined performance score
            performance_score = (0.5 * recent_return + 0.3 * recent_sharpe + 0.2 * recent_win_rate)
            new_weights[i] = max(performance_score, 0.01)  # Ensure positive weights
        
        # Normalize weights
        new_weights = new_weights / np.sum(new_weights)
        
        # Apply min/max constraints
        new_weights = np.clip(new_weights, self.min_weight, self.max_weight)
        new_weights = new_weights / np.sum(new_weights)  # Renormalize
        
        # Apply decay to prevent sudden changes
        self.agent_weights = self.weight_decay * self.agent_weights + (1 - self.weight_decay) * new_weights
        
        if self.verbose_logging:
            print(f"📊 Updated agent weights: {self.agent_weights}")
        
        # Track weight updates
        self.ensemble_stats['weight_updates'].append({
            'step': len(self.ensemble_stats['agent_contributions']),
            'weights': self.agent_weights.copy()
        })

    def _ensemble_action(self, actions, q_values=None, method='majority_voting', step_idx=0):
        """
        Advanced ensemble methods for combining agent actions
        
        Args:
            actions: List of agent actions
            q_values: List of Q-value tensors from each agent
            method: Ensemble method to use
            step_idx: Current step index
                
        Returns:
            tuple: (ensemble_action, selected_agent_index)
        """
        
        if method == 'majority_voting':
            # Original simple majority voting
            count = Counter([a[0].item() for a in actions])
            majority_action, _ = count.most_common(1)[0]
            # Find which agent contributed (first match)
            selected_idx = next((i for i, a in enumerate(actions) if a[0].item() == majority_action), None)
            return torch.tensor([[majority_action]], dtype=torch.int32), selected_idx
            
        elif method == 'weighted_voting':
            # Enhanced weighted voting with performance-based weights
            return self._weighted_voting(actions, q_values)
            
        elif method == 'confidence_weighted':
            # Weight by Q-value confidence
            return self._confidence_weighted_voting(actions, q_values)
            
        elif method == 'adaptive_meta':
            # Adaptive meta-learning ensemble
            return self._adaptive_meta_voting(actions, q_values, step_idx)
            
        else:
            # Fallback to majority voting
            count = Counter([a[0].item() for a in actions])
            majority_action, _ = count.most_common(1)[0]
            selected_idx = next((i for i, a in enumerate(actions) if a[0].item() == majority_action), None)
            return torch.tensor([[majority_action]], dtype=torch.int32), selected_idx
    
    def _weighted_voting(self, actions, q_values=None):
        """Enhanced weighted voting based on agent historical performance"""
        
        # Convert actions to numpy for easier manipulation
        action_values = np.array([a.item() for a in actions])
        
        # Weighted voting using performance-based agent weights
        vote_counts = np.zeros(3)  # Assuming 3 actions: 0, 1, 2
        
        for i, action in enumerate(action_values):
            vote_counts[action] += self.agent_weights[i]
            
        # Select action with highest weighted vote
        final_action = np.argmax(vote_counts)
        
        # Find which agent contributed most to this decision
        selected_idx = None
        max_contribution = 0
        for i, action in enumerate(action_values):
            if action == final_action and self.agent_weights[i] > max_contribution:
                max_contribution = self.agent_weights[i]
                selected_idx = i
        
        if self.verbose_logging and len(self.ensemble_stats['agent_contributions']) % 100 == 0:
            print(f"🎯 Weighted vote: Action {final_action}, Agent {selected_idx} (weight: {max_contribution:.3f})")
            
        return torch.tensor([[final_action]], dtype=torch.int32), selected_idx
    
    def _confidence_weighted_voting(self, actions, q_values):
        """Confidence-weighted voting using Q-value confidence scores"""
        if q_values is None:
            return self._weighted_voting(actions)
        
        # Calculate confidence scores from Q-values
        confidence_scores = []
        for q_vals in q_values:
            # Use the spread between max and second-max Q-values as confidence
            sorted_q = torch.sort(q_vals, descending=True)[0]
            if len(sorted_q[0]) > 1:
                confidence = (sorted_q[0][0] - sorted_q[0][1]).item()  # Q-value spread
            else:
                confidence = sorted_q[0][0].item()  # Single Q-value
            confidence_scores.append(max(confidence, 0.01))  # Ensure positive
        
        # Normalize confidence scores
        confidence_scores = np.array(confidence_scores)
        confidence_weights = confidence_scores / np.sum(confidence_scores)
        
        # Combine with performance weights (50-50 split)
        combined_weights = 0.5 * self.agent_weights + 0.5 * confidence_weights
        
        # Convert actions to numpy
        action_values = np.array([a.item() for a in actions])
        
        # Confidence-weighted voting
        vote_counts = np.zeros(3)
        for i, action in enumerate(action_values):
            vote_counts[action] += combined_weights[i]
            
        final_action = np.argmax(vote_counts)
        
        # Find contributing agent with highest combined weight
        selected_idx = None
        max_contribution = 0
        for i, action in enumerate(action_values):
            if action == final_action and combined_weights[i] > max_contribution:
                max_contribution = combined_weights[i]
                selected_idx = i
        
        if self.verbose_logging and len(self.ensemble_stats['agent_contributions']) % 100 == 0:
            print(f"🔮 Confidence vote: Action {final_action}, Agent {selected_idx} "
                  f"(perf: {self.agent_weights[selected_idx]:.3f}, conf: {confidence_weights[selected_idx]:.3f})")
        
        return torch.tensor([[final_action]], dtype=torch.int32), selected_idx
    
    def _adaptive_meta_voting(self, actions, q_values, step_idx):
        """Adaptive meta-learning ensemble method with market context"""
        
        # Initialize meta-learning parameters if not exists
        if not hasattr(self, 'meta_weights'):
            self.meta_weights = np.ones((len(actions), 3)) / 3  # Action-specific weights
            self.meta_learning_rate = 0.01
            self.recent_performance = []
            
        # Get enhanced market context
        current_context = self._get_enhanced_market_context()
        
        # Adaptive weight adjustment based on recent performance and market context
        if len(self.recent_performance) > 10:  # Need some history
            self._update_meta_weights(current_context)
            
        # Meta-weighted voting with market regime awareness
        action_values = np.array([a.item() for a in actions])
        final_scores = np.zeros(3)
        
        # Apply market regime multipliers
        regime_multipliers = self._get_regime_multipliers(current_context)
        
        for i, action in enumerate(action_values):
            action_score = self.meta_weights[i] * np.eye(3)[action]
            # Apply regime-specific weighting
            final_scores += action_score * regime_multipliers[i]
            
        final_action = np.argmax(final_scores)
        
        # Find contributing agent
        selected_idx = None
        max_score = 0
        for i, action in enumerate(action_values):
            if action == final_action:
                score = self.meta_weights[i][action] * regime_multipliers[i]
                if score > max_score:
                    max_score = score
                    selected_idx = i
        
        if self.verbose_logging and step_idx % 100 == 0:
            print(f"🧠 Meta vote: Action {final_action}, Agent {selected_idx}, "
                  f"Regime: {current_context.get('regime', 'unknown')}")
        
        return torch.tensor([[final_action]], dtype=torch.int32), selected_idx
    
    def _get_enhanced_market_context(self):
        """Enhanced market context detection with multiple indicators"""
        context = {'regime': 'neutral', 'volatility': 0.01, 'trend': 0.0, 'momentum': 0.0}
        
        if not (hasattr(self.trade_env, 'price_ary') and hasattr(self.trade_env, 'step_i')):
            return context
            
        step_i = self.trade_env.step_i
        prices = self.trade_env.price_ary[:, 2]  # Midpoint prices
        
        if step_i < 50:  # Need minimum history
            return context
            
        # Current and recent prices
        current_price = prices[step_i].item()
        
        # Volatility calculation (20-period)
        vol_window = min(20, step_i)
        recent_prices = prices[step_i-vol_window:step_i]
        if len(recent_prices) > 1:
            returns = torch.diff(recent_prices) / recent_prices[:-1]
            volatility = torch.std(returns).item()
            context['volatility'] = max(volatility, 0.001)
        
        # Trend calculation (short vs long MA)
        short_window = min(10, step_i // 2)
        long_window = min(30, step_i)
        
        if step_i >= long_window:
            short_ma = torch.mean(prices[step_i-short_window:step_i]).item()
            long_ma = torch.mean(prices[step_i-long_window:step_i]).item()
            trend = (short_ma - long_ma) / long_ma
            context['trend'] = trend
            
            # Momentum calculation (rate of change)
            momentum_window = min(15, step_i)
            if step_i >= momentum_window:
                past_price = prices[step_i-momentum_window].item()
                momentum = (current_price - past_price) / past_price
                context['momentum'] = momentum
        
        # Market regime classification
        vol_threshold = 0.005  # Volatility threshold
        trend_threshold = 0.002  # Trend threshold
        
        if context['volatility'] > vol_threshold:
            if abs(context['trend']) > trend_threshold:
                context['regime'] = 'volatile_trending'
            else:
                context['regime'] = 'volatile_ranging'
        else:
            if abs(context['trend']) > trend_threshold:
                context['regime'] = 'stable_trending'
            else:
                context['regime'] = 'stable_ranging'
        
        return context
    
    def _get_regime_multipliers(self, context):
        """Get agent multipliers based on market regime"""
        regime = context.get('regime', 'neutral')
        num_agents = len(self.agents)
        
        # Default equal multipliers
        multipliers = np.ones(num_agents)
        
        # Regime-specific agent preferences (can be learned/optimized)
        if regime == 'volatile_trending':
            # Favor trend-following agents (assume D3QN is trend-following)
            multipliers[0] = 1.2  # D3QN
            multipliers[1] = 0.9  # DoubleDQN  
            multipliers[2] = 1.0  # TwinD3QN
        elif regime == 'stable_ranging':
            # Favor mean-reverting agents (assume DoubleDQN is mean-reverting)
            multipliers[0] = 0.9  # D3QN
            multipliers[1] = 1.2  # DoubleDQN
            multipliers[2] = 1.0  # TwinD3QN
        elif regime == 'volatile_ranging':
            # Favor conservative agents
            multipliers[0] = 0.8  # D3QN
            multipliers[1] = 0.8  # DoubleDQN
            multipliers[2] = 1.3  # TwinD3QN (assume more conservative)
        
        return multipliers
    
    def _get_market_context(self):
        """Backward compatibility - calls enhanced version"""
        context = self._get_enhanced_market_context()
        return {'volatility': context['volatility'], 'price': self.trade_env.price_ary[self.trade_env.step_i, 2].item()}
    
    def _update_meta_weights(self, context=None):
        """Update meta-learning weights based on recent performance and market context"""
        if len(self.recent_performance) < 2:
            return
            
        # Performance-based weight update with market context
        recent_returns = np.diff(self.recent_performance[-20:])  # Last 20 steps
        
        if len(recent_returns) > 0:
            avg_return = np.mean(recent_returns)
            
            # Market regime adjustment
            regime_factor = 1.0
            if context:
                regime = context.get('regime', 'neutral')
                volatility = context.get('volatility', 0.01)
                # More aggressive updates in stable markets, conservative in volatile
                regime_factor = 0.5 if 'volatile' in regime else 1.5
            
            adjusted_lr = self.meta_learning_rate * regime_factor
            
            # Reward successful agents, penalize unsuccessful ones
            if avg_return > 0:
                # Boost weights of recently successful strategies  
                self.meta_weights *= (1 + adjusted_lr * avg_return)
            else:
                # Reduce weights and add exploration
                self.meta_weights *= (1 + adjusted_lr * avg_return)
                self.meta_weights += np.random.normal(0, 0.001, self.meta_weights.shape)
                
            # Normalize weights
            self.meta_weights = np.abs(self.meta_weights)
            self.meta_weights /= np.sum(self.meta_weights, axis=1, keepdims=True)
    
    def update_performance_tracking(self, reward):
        """Update performance tracking for adaptive methods"""
        if not hasattr(self, 'recent_performance'):
            self.recent_performance = []
            
        self.recent_performance.append(reward)
        
        # Keep only recent history
        if len(self.recent_performance) > 100:
            self.recent_performance = self.recent_performance[-100:]


def run_evaluation(save_path, agent_list, ensemble_method='majority_voting', ensemble_path=None,
                  performance_window=100, weight_decay=0.95, verbose_logging=False):
    """
    Run ensemble evaluation with configurable ensemble method and advanced options
    
    Args:
        save_path: Path to saved ensemble models (legacy parameter)
        agent_list: List of agent classes to evaluate
        ensemble_method: Ensemble method to use ('majority_voting', 'weighted_voting', 'confidence_weighted', 'adaptive_meta')
        ensemble_path: Alternative path to ensemble models (used by HPO script)
        performance_window: Rolling window size for performance metrics (default: 100)
        weight_decay: Decay factor for weight updates (default: 0.95)
        verbose_logging: Enable detailed logging (default: False)
    """
    import sys

    # Use ensemble_path if provided (for HPO compatibility), otherwise use save_path
    if ensemble_path is not None:
        save_path = ensemble_path
        print(f"🔧 HPO Mode: Using ensemble_path = {ensemble_path}")

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # Get GPU_ID from command line arguments

    num_sims = 1
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7

    max_step = (4800 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": None,  # Will be auto-detected
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "eval_split": 0.8,  # Use last 20% of data for out-of-sample evaluation
    }
    # Detect state dimension first
    temp_sim = TradeSimulator(num_sims=1)
    detected_state_dim = temp_sim.state_dim
    env_args["state_dim"] = detected_state_dim
    
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.state_dim = detected_state_dim
    args.starting_cash = 1e6
    
    # Use optimized architecture for 8-feature models
    if detected_state_dim == 8:
        args.net_dims = (128, 64, 32)
        print(f"Using optimized architecture for 8-feature models: {args.net_dims}")
    else:
        args.net_dims = (128, 128, 128)
        print(f"Using default architecture for {detected_state_dim}-feature models: {args.net_dims}")

    ensemble_evaluator = EnsembleEvaluator(
        save_path,
        agent_list,
        args,
        ensemble_method=ensemble_method,
        performance_window=performance_window,
        weight_decay=weight_decay,
        verbose_logging=verbose_logging
    )
    ensemble_evaluator.load_agents()
    ensemble_evaluator.multi_trade()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='FinRL Contest 2024 - Advanced Ensemble Evaluation')
    parser.add_argument('gpu_id', nargs='?', type=int, default=-1, help='GPU ID to use (default: -1 for CPU)')
    parser.add_argument('--save-path', type=str, help='Path to saved ensemble models')
    parser.add_argument('--ensemble-method', type=str, default='majority_voting',
                       choices=['majority_voting', 'weighted_voting', 'confidence_weighted', 'adaptive_meta'],
                       help='Ensemble method to use')
    parser.add_argument('--performance-window', type=int, default=100, 
                       help='Rolling window size for performance metrics (default: 100)')
    parser.add_argument('--weight-decay', type=float, default=0.95,
                       help='Decay factor for weight updates (default: 0.95)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging for ensemble analysis')
    
    args = parser.parse_args()
    
    # Determine save path
    if args.save_path:
        save_path = args.save_path
        print(f"🎯 Using specified models: {save_path}")
    else:
        # Use optimized models if available, fallback to standard
        save_path = "ensemble_optimized_phase2/ensemble_models"
        if not os.path.exists(save_path):
            save_path = "ensemble_teamname/ensemble_models"
            print(f"⚠️  Optimized models not found, using: {save_path}")
        else:
            print(f"✅ Using optimized models: {save_path}")
    
    print(f"🚀 FinRL Contest 2024 - Advanced Ensemble Evaluation")
    print(f"🔧 Using GPU: {args.gpu_id}")
    print(f"📁 Model path: {save_path}")
    print(f"🎯 Ensemble method: {args.ensemble_method}")
    print(f"📊 Performance window: {args.performance_window}")
    print(f"⚖️  Weight decay: {args.weight_decay}")
    print(f"🔍 Verbose logging: {args.verbose}")
    
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN]
    run_evaluation(
        save_path, 
        agent_list, 
        ensemble_method=args.ensemble_method,
        performance_window=args.performance_window,
        weight_decay=args.weight_decay,
        verbose_logging=args.verbose
    )
