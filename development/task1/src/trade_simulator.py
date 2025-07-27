import sys
import os
import torch as th
import numpy as np
import pandas as pd
from data_config import ConfigData
from reward_functions import create_reward_calculator


class TradeSimulator:
    def __init__(
        self,
        num_sims=64,
        slippage=5e-5,
        max_position=2,
        step_gap=1,
        delay_step=1,
        num_ignore_step=60,
        device=th.device("cpu"),
        gpu_id=-1,
        data_length=None,
    ):
        self.device = th.device(f"cuda:{gpu_id}") if gpu_id >= 0 else device
        self.num_sims = num_sims

        self.slippage = slippage
        self.delay_step = delay_step
        self.max_holding = 60 * 60 // step_gap
        self.max_position = max_position
        self.step_gap = step_gap
        self.sim_ids = th.arange(self.num_sims, device=self.device)

        """config"""
        args = ConfigData()

        """load data"""
        # Priority loading: enhanced_v3 > optimized > enhanced > original
        enhanced_v3_path = args.predict_ary_path.replace('.npy', '_enhanced_v3.npy')
        optimized_path = args.predict_ary_path.replace('.npy', '_optimized.npy')
        enhanced_path = args.predict_ary_path.replace('.npy', '_enhanced.npy')
        
        if os.path.exists(enhanced_v3_path):
            print(f"Loading enhanced features v3 from {enhanced_v3_path}")
            self.factor_ary = np.load(enhanced_v3_path)
            
            # Load metadata for enhanced v3 features
            metadata_path = enhanced_v3_path.replace('.npy', '_metadata.npy')
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True).item()
                self.feature_names = metadata.get('feature_names', [])
                print(f"Enhanced v3 features loaded: {len(self.feature_names)} features")
            else:
                self.feature_names = []
        elif os.path.exists(optimized_path):
            print(f"Loading optimized features from {optimized_path}")
            self.factor_ary = np.load(optimized_path)
            
            # Load metadata for optimized features
            metadata_path = optimized_path.replace('.npy', '_metadata.npy')
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True).item()
                self.feature_names = metadata.get('feature_names', [])
                print(f"Optimized features loaded: {len(self.feature_names)} features")
            else:
                self.feature_names = []
        elif os.path.exists(enhanced_path):
            print(f"Loading enhanced features from {enhanced_path}")
            self.factor_ary = np.load(enhanced_path)
            
            # Load metadata for enhanced features
            metadata_path = enhanced_path.replace('.npy', '_metadata.npy')
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True).item()
                self.feature_names = metadata.get('feature_names', [])
                print(f"Enhanced features loaded: {len(self.feature_names)} features")
            else:
                self.feature_names = []
        else:
            print(f"Loading original features from {args.predict_ary_path}")
            self.factor_ary = np.load(args.predict_ary_path)
            self.feature_names = []
            
        self.factor_ary = th.tensor(self.factor_ary, dtype=th.float32, device=self.device)  # Move to correct device

        data_df = pd.read_csv(args.csv_path)  # CSV READ HERE

        self.price_ary = data_df[["bids_distance_3", "asks_distance_3", "midpoint"]].values
        self.price_ary[:, 0] = self.price_ary[:, 2] * (1 + self.price_ary[:, 0])
        self.price_ary[:, 1] = self.price_ary[:, 2] * (1 + self.price_ary[:, 1])
        self.price_ary = th.tensor(self.price_ary, dtype=th.float32, device=self.device)  # Move to correct device

        """
        DATA ALIGNMENT LOGIC - CRITICAL FOR PROPER FUNCTIONING
        
        This section handles alignment between factor_ary (predictive features) and price_ary (market prices).
        
        Key Details:
        - factor_ary: Contains predictive features/signals (usually shorter than price data)
        - price_ary: Contains market price data (bid, ask, midpoint prices)
        
        Historical Alignment Strategy:
        The system uses "rear alignment" instead of "front alignment" because:
        
        1. REAR ALIGNMENT (Current Implementation):
           - Takes the LAST N rows of price_ary where N = len(factor_ary)
           - Ensures we use the most recent/relevant market data
           - Code: self.price_ary = self.price_ary[-self.factor_ary.shape[0]:, :]
           
        2. FRONT ALIGNMENT (Commented Out):
           - Would take the FIRST N rows of price_ary
           - Code: self.price_ary = self.price_ary[: self.factor_ary.shape[0], :]
           - Less desirable as it uses older market data
        
        Why Rear Alignment Matters:
        - Predictive models often generate features for recent time periods
        - Using corresponding recent price data ensures temporal consistency
        - Avoids train/test data leakage by maintaining chronological order
        - Critical for realistic backtesting and live trading deployment
        
        Data Shape Validation:
        - After alignment: assert self.price_ary.shape[0] == self.factor_ary.shape[0]
        - Ensures perfect temporal alignment between features and prices
        - Both arrays must have identical length for environment to function properly
        
        Potential Issues & Considerations:
        - If factor_ary is longer than price_ary, this will fail (need more price data)
        - If there's a significant time gap between factor and price data, alignment may be incorrect
        - Feature generation and price data collection must be synchronized in time
        - Any changes to this logic should be thoroughly tested across different data periods
        """
        # Original front alignment (commented out for reference)
        # self.price_ary = self.price_ary[: self.factor_ary.shape[0], :]
        
        # Current rear alignment implementation
        self.price_ary = self.price_ary[-self.factor_ary.shape[0] :, :]

        self.price_ary = th.tensor(self.price_ary, dtype=th.float32, device=self.device)  # Move to correct device

        self.seq_len = 3600
        self.full_seq_len = self.price_ary.shape[0]
        
        # Apply data_length limitation if specified
        if data_length is not None and data_length > 0:
            # Ensure data_length doesn't exceed actual data size
            safe_data_length = min(data_length, self.full_seq_len)
            
            # Truncate data to safe_data_length (using rear alignment to keep most recent data)
            if safe_data_length < self.full_seq_len:
                self.factor_ary = self.factor_ary[-safe_data_length:].contiguous()
                self.price_ary = self.price_ary[-safe_data_length:]
                self.full_seq_len = safe_data_length
                print(f"TradeSimulator: Limited data length to {safe_data_length} samples")
            else:
                print(f"TradeSimulator: Using full dataset ({self.full_seq_len} samples)")
        else:
            print(f"TradeSimulator: Using full dataset ({self.full_seq_len} samples)")
        
        
        assert self.price_ary.shape[0] == self.factor_ary.shape[0]

        # reset()
        self.step_i = 0
        self.step_is = th.zeros((num_sims,), dtype=th.long, device=device)
        self.action_int = th.zeros((num_sims,), dtype=th.long, device=device)
        self.rolling_asset = th.zeros((num_sims,), dtype=th.long, device=device)

        self.position = th.zeros((num_sims,), dtype=th.long, device=device)
        self.holding = th.zeros((num_sims,), dtype=th.long, device=device)
        self.empty_count = th.zeros((num_sims,), dtype=th.long, device=device)

        self.cash = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.asset = th.zeros((num_sims,), dtype=th.float32, device=device)

        # environment information
        self.env_name = "TradeSimulator-v0"
        # Dynamic state_dim based on loaded features
        factor_dim = self.factor_ary.shape[1] - 2  # Subtract 2 for position features
        self.state_dim = factor_dim + 2  # factor_dim + (position, holding)
        self.action_dim = 3  # short, nothing, long
        print(f"State dimension: {self.state_dim} (factor_dim: {factor_dim} + 2 position features)")
        self.if_discrete = True
        self.max_step = (self.seq_len - num_ignore_step) // step_gap
        self.target_return = +np.inf
        
        # Enhanced validation with detailed error reporting - moved after state_dim is set
        self._validate_data_alignment()

        """stop-loss"""
        self.best_price = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.stop_loss_thresh = 1e-3
        
        """Enhanced Reward System - addresses profitability issues"""
        # Default to adaptive_multi_objective for best performance, can be overridden
        self.reward_calculator = create_reward_calculator(
            reward_type="adaptive_multi_objective",
            lookback_window=100,
            device=str(self.device),
            reward_weights=None  # Will use defaults, can be overridden
        )
        self.reward_type = "adaptive_multi_objective"  # Track current reward type

    def _validate_data_alignment(self):
        """
        Comprehensive validation of data alignment between factor_ary and price_ary
        
        Provides detailed diagnostics to help debug alignment issues and ensure
        data integrity for training and evaluation.
        """
        factor_shape = self.factor_ary.shape
        price_shape = self.price_ary.shape
        
        print(f"📊 Data Alignment Validation:")
        print(f"   Factor array shape: {factor_shape}")
        print(f"   Price array shape: {price_shape}")
        print(f"   Feature names available: {len(self.feature_names) > 0}")
        
        # Check basic shape compatibility
        if factor_shape[0] != price_shape[0]:
            print(f"❌ ALIGNMENT ERROR:")
            print(f"   Factor length: {factor_shape[0]}")
            print(f"   Price length: {price_shape[0]}")
            print(f"   Difference: {abs(factor_shape[0] - price_shape[0])}")
            
            # Suggest potential fixes
            if factor_shape[0] > price_shape[0]:
                print(f"💡 Suggested fix: Need more price data or reduce factor data")
            else:
                print(f"💡 Suggested fix: Current rear alignment should work")
                
        else:
            print(f"✅ Length alignment successful: {factor_shape[0]} timesteps")
            
        # Validate feature dimensions
        expected_factor_features = factor_shape[1] - 2  # Subtract position features
        actual_state_features = self.state_dim - 2      # Subtract position features
        
        if expected_factor_features != actual_state_features:
            print(f"⚠️  Feature dimension mismatch:")
            print(f"   Expected: {expected_factor_features}")
            print(f"   Actual: {actual_state_features}")
        else:
            print(f"✅ Feature dimension alignment successful: {expected_factor_features} features")
            
        # Check for NaN or infinite values
        if th.isnan(self.factor_ary).any():
            nan_count = th.isnan(self.factor_ary).sum().item()
            print(f"⚠️  Found {nan_count} NaN values in factor_ary")
            
        if th.isnan(self.price_ary).any():
            nan_count = th.isnan(self.price_ary).sum().item()
            print(f"⚠️  Found {nan_count} NaN values in price_ary")
            
        # Data range validation
        price_stats = {
            'bid_min': self.price_ary[:, 0].min().item(),
            'bid_max': self.price_ary[:, 0].max().item(),
            'ask_min': self.price_ary[:, 1].min().item(),
            'ask_max': self.price_ary[:, 1].max().item(),
            'mid_min': self.price_ary[:, 2].min().item(),
            'mid_max': self.price_ary[:, 2].max().item(),
        }
        
        print(f"📈 Price data ranges:")
        print(f"   Bid: [{price_stats['bid_min']:.2f}, {price_stats['bid_max']:.2f}]")
        print(f"   Ask: [{price_stats['ask_min']:.2f}, {price_stats['ask_max']:.2f}]")
        print(f"   Mid: [{price_stats['mid_min']:.2f}, {price_stats['mid_max']:.2f}]")
        
        # Validate bid-ask spread sanity
        spreads = self.price_ary[:, 1] - self.price_ary[:, 0]  # ask - bid
        negative_spreads = (spreads < 0).sum().item()
        
        if negative_spreads > 0:
            print(f"❌ Found {negative_spreads} negative bid-ask spreads!")
            print(f"   This indicates data quality issues")
        else:
            avg_spread = spreads.mean().item()
            print(f"✅ Bid-ask spreads look healthy (avg: {avg_spread:.4f})")
            
        print("📊 Data alignment validation complete\n")

    def _reset(self, slippage=None, _if_random=True):
        self.slippage = slippage if isinstance(slippage, float) else self.slippage

        num_sims = self.num_sims
        device = self.device

        # Calculate safe random range for starting positions
        min_start = self.seq_len
        max_start = self.full_seq_len - self.seq_len * 2
        
        # Handle case where data is too small for normal random range
        if min_start >= max_start:
            # Use a much smaller sequence length for small datasets
            reduced_seq_len = min(self.seq_len, self.full_seq_len // 4)
            min_start = reduced_seq_len
            max_start = self.full_seq_len - reduced_seq_len
            
            if min_start >= max_start:
                # Ultimate fallback - use fixed small positions
                min_start = min(100, self.full_seq_len // 10)
                max_start = max(min_start + 1, self.full_seq_len - min_start)
        
        i0s = np.random.randint(min_start, max_start, size=self.num_sims)
        self.step_i = 0
        self.step_is = th.tensor(i0s, dtype=th.long, device=self.device)
        self.cash = th.zeros((num_sims,), dtype=th.float32, device=device)
        self.asset = th.zeros((num_sims,), dtype=th.float32, device=device)

        self.holding = th.zeros((num_sims,), dtype=th.long, device=device)
        self.position = th.zeros((num_sims,), dtype=th.long, device=device)
        self.empty_count = th.zeros((num_sims,), dtype=th.long, device=device)

        """stop-loss"""
        self.best_price = th.zeros((self.num_sims,), dtype=th.float32, device=self.device)
        
        """reset reward calculator for new episode"""
        self.reward_calculator.reset()

        step_is = self.step_is + self.step_i
        
        # Ensure step_is doesn't exceed data bounds
        max_valid_index = self.full_seq_len - 1
        step_is = th.clamp(step_is, 0, max_valid_index)
        
        state = self.get_state(step_is_cpu=step_is)
        return state

    def _step(self, action, _if_random=True):
        self.step_i += self.step_gap
        step_is = self.step_is + self.step_i
        
        # Ensure step_is doesn't exceed data bounds
        max_valid_index = self.full_seq_len - 1
        step_is = th.clamp(step_is, 0, max_valid_index)
        
        step_is_cpu = step_is

        action = action.squeeze(1).to(self.device)
        action_int = action - 1  # map (0, 1, 2) to (-1, 0, +1), means (sell, nothing, buy)
        # action_int = (action - self.max_position) - self.position
        del action

        old_cash = self.cash
        old_asset = self.asset
        old_position = self.position

        # the data in price_ary is ['bid', 'ask', 'mid']
        # bid_price = self.price_ary[step_is_cpu, 0]
        # ask_price = self.price_ary[step_is_cpu, 1]
        mid_price = self.price_ary[step_is_cpu, 2]

        """get action_int"""
        truncated = self.step_i >= (self.max_step * self.step_gap)
        if truncated:
            action_int = -old_position
        else:
            new_position = (old_position + action_int).clip(
                -self.max_position, self.max_position
            )  # limit the position
            action_int = new_position - old_position  # get the limit action

            done_mask = (new_position * old_position).lt(0) & old_position.ne(0)
            if done_mask.sum() > 0:
                action_int[done_mask] = -old_position[done_mask]

        """holding"""
        self.holding = self.holding + 1
        mask_max_holding = self.holding.gt(self.max_holding)

        if mask_max_holding.sum() > 0:
            action_int[mask_max_holding] = -old_position[mask_max_holding]
        self.holding[old_position == 0] = 0

        # mask_min_holding = th.logical_and(self.holding.le(self.min_holding), old_position.ne(0))
        # if mask_min_holding.sum() > 0:
        #     action_int[mask_min_holding] = 0

        """stop-loss"""
        direction_mask1 = old_position.gt(0)
        if direction_mask1.sum() > 0:
            _best_price = th.max(
                th.stack([self.best_price[direction_mask1], mid_price[direction_mask1]]),
                dim=0,
            )[0]
            self.best_price[direction_mask1] = _best_price

        direction_mask2 = old_position.lt(0)
        if direction_mask2.sum() > 0:
            _best_price = th.min(
                th.stack([self.best_price[direction_mask2], mid_price[direction_mask2]]),
                dim=0,
            )[0]
            self.best_price[direction_mask2] = _best_price

        # stop_loss_thresh = mid_price * self.stop_loss_rate
        stop_loss_mask1 = th.logical_and(direction_mask1, (self.best_price - mid_price).gt(self.stop_loss_thresh))
        stop_loss_mask2 = th.logical_and(direction_mask2, (mid_price - self.best_price).gt(self.stop_loss_thresh))
        stop_loss_mask = th.logical_or(stop_loss_mask1, stop_loss_mask2)
        if stop_loss_mask.sum() > 0:
            action_int[stop_loss_mask] = -old_position[stop_loss_mask]

        """get new_position via action_int"""
        new_position = old_position + action_int

        entry_mask = old_position.eq(0)
        if entry_mask.sum() > 0:
            self.best_price[entry_mask] = mid_price[entry_mask]

        """executing"""
        direction = action_int.gt(0)  # True: buy, False: sell
        cost = action_int * mid_price  # action_int * th.where(direction, ask_price, bid_price)

        new_cash = old_cash - cost * th.where(direction, 1 + self.slippage, 1 - self.slippage)
        new_asset = new_cash + new_position * mid_price

        # Enhanced reward calculation - addresses profitability issues
        reward = self.reward_calculator.calculate_reward(
            old_asset=old_asset,
            new_asset=new_asset, 
            action_int=action_int,
            mid_price=mid_price,
            slippage=self.slippage
        )

        self.cash = new_cash  # update the cash
        self.asset = new_asset  # update the total asset
        self.position = new_position  # update the position
        self.action_int = action_int  # update the action_int

        state = self.get_state(step_is_cpu)
        info_dict = {}
        if truncated:
            terminal = th.ones_like(self.position, dtype=th.bool)
            state = self.reset()
        else:
            # terminal = old_position.ne(0) & new_position.eq(0)
            terminal = th.zeros_like(self.position, dtype=th.bool)

        return state, reward, terminal, info_dict

    def reset(self, slippage=None, date_strs=()):
        return self._reset(slippage=slippage, _if_random=True)

    def step(self, action):
        return self._step(action, _if_random=True)

    def get_state(self, step_is_cpu):
        # Ensure step_is_cpu is a tensor for proper indexing
        if not isinstance(step_is_cpu, th.Tensor):
            step_is_cpu = th.tensor(step_is_cpu, dtype=th.long, device=self.device)
        
        # Index factor_ary - already on correct device
        factor_ary = self.factor_ary[step_is_cpu, :]
        
        # Ensure factor_ary is 2D
        if factor_ary.dim() == 1:
            factor_ary = factor_ary.unsqueeze(0)
        
        # If enhanced features, update position features in-place
        if hasattr(self, 'feature_names') and len(self.feature_names) > 0:
            # Enhanced features: position features are at indices 0 and 1
            factor_ary = factor_ary.clone()  # Clone to avoid in-place modification issues
            factor_ary[:, 0] = self.position.float() / self.max_position
            factor_ary[:, 1] = self.holding.float() / self.max_holding
            return factor_ary
        else:
            # Original features: concatenate position features
            state = th.concat(
                (
                    (self.position.float() / self.max_position)[:, None],
                    (self.holding.float() / self.max_holding)[:, None],
                    factor_ary,
                ),
                dim=1,
            )
            return state
    
    def set_reward_type(self, reward_type: str, reward_weights: Optional[Dict[str, float]] = None):
        """
        Change the reward calculation method
        
        Args:
            reward_type: "simple", "transaction_cost_adjusted", "sharpe_adjusted", 
                        "multi_objective", "adaptive_multi_objective"
            reward_weights: Optional custom weights for multi-objective rewards
        """
        if reward_type != self.reward_type or reward_weights is not None:
            print(f"🎯 Switching reward type from '{self.reward_type}' to '{reward_type}'")
            self.reward_calculator = create_reward_calculator(
                reward_type=reward_type,
                lookback_window=100,
                device=str(self.device),
                reward_weights=reward_weights
            )
            self.reward_type = reward_type
    
    def get_reward_metrics(self) -> dict:
        """Get current reward system performance metrics"""
        metrics = self.reward_calculator.get_performance_metrics()
        metrics["reward_type"] = self.reward_type
        return metrics
    
    def print_reward_performance(self):
        """Print current reward system performance"""
        metrics = self.get_reward_metrics()
        print(f"\n📊 Reward System Performance ({metrics['reward_type']}):")
        print(f"   💰 Total Transaction Costs: ${metrics.get('total_transaction_costs', 0):.2f}")
        print(f"   📉 Current Drawdown: {metrics.get('current_drawdown', 0):.4f}")
        print(f"   📈 Peak Asset Value: ${metrics.get('peak_asset_value', 0):,.2f}")
        
        if 'mean_return' in metrics:
            print(f"   📊 Mean Return: {metrics['mean_return']:.6f}")
            print(f"   📊 Return Volatility: {metrics['return_volatility']:.6f}")
            print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"   📊 Max Drawdown Period: {metrics['max_drawdown_period']:.4f}")


class EvalTradeSimulator(TradeSimulator):
    def __init__(self, eval_split=0.8, **kwargs):
        """
        Evaluation simulator that uses the last portion of data for out-of-sample testing.
        
        Args:
            eval_split: float, proportion of data to use for training (0.8 means use last 20% for eval)
            **kwargs: arguments passed to parent TradeSimulator
        """
        super().__init__(**kwargs)
        
        # Use the last portion of data for evaluation (out-of-sample)
        split_idx = int(len(self.factor_ary) * eval_split)
        self.factor_ary = self.factor_ary[split_idx:].contiguous()
        self.price_ary = self.price_ary[split_idx:]
        
        # CRITICAL: Update full_seq_len after data truncation
        self.full_seq_len = len(self.factor_ary)
        
        print(f"EvalTradeSimulator: Using last {len(self.factor_ary)} samples for evaluation (out-of-sample)")

    def reset(self, slippage=None, date_strs=()):
        self.stop_loss_thresh = 1e-4
        return self._reset(slippage=slippage, _if_random=False)

    def step(self, action):
        return self._step(action, _if_random=False)


def check_simulator():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # 从命令行参数里获得GPU_ID
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    num_sims = 6
    slippage = 0
    step_gap = 2

    sim = TradeSimulator(num_sims=num_sims, step_gap=step_gap, slippage=slippage)
    action_dim = sim.action_dim
    delay_step = sim.delay_step

    reward_ary = th.zeros((num_sims, 4800), dtype=th.float32, device=device)

    state = sim.reset(slippage=slippage)
    for step_i in range(sim.max_step):
        action = th.randint(action_dim, size=(num_sims, 1), device=device)
        state, reward, done, info_dict = sim.step(action=action)

        reward_ary[:, step_i + delay_step] = reward

        print(sim.asset)  #  if step_i + 2 == sim.max_step else None

    print(reward_ary.sum(dim=1))
    print(state.shape, num_sims, sim.state_dim)
    assert state.shape == (num_sims, sim.state_dim)

    print("############")

    reward_ary = th.zeros((num_sims, sim.max_step + delay_step), dtype=th.float32, device=device)

    state = sim.reset(slippage=slippage)
    for step_i in range(sim.max_step):
        if step_i == 0:
            action = th.ones(size=(num_sims, 1), dtype=th.long, device=device) - 1
        else:
            action = th.ones(size=(num_sims, 1), dtype=th.long, device=device)

        state, reward, done, info_dict = sim.step(action=action)

        reward_ary[:, step_i + delay_step] = reward

        print(sim.asset) if step_i + 2 == sim.max_step else None

    print(reward_ary.sum(dim=1))
    print(state.shape, num_sims, sim.state_dim)
    assert state.shape == (num_sims, sim.state_dim)

    print()


if __name__ == "__main__":
    check_simulator()
