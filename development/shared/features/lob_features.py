"""
Limit Order Book (LOB) Features Module

Provides specialized features from LOB data:
- Order flow imbalance
- Book depth and slope
- Trade imbalance
- VWAP deviation
- Microstructure indicators
- Liquidity metrics
"""

import numpy as np
import pandas as pd

class LOBFeatures:
    """Limit Order Book feature calculator"""
    
    def __init__(self, levels=3, lookback_window=20):
        """
        Initialize LOB feature calculator
        
        Args:
            levels: Number of LOB levels to consider (0-levels)
            lookback_window: Window for rolling calculations
        """
        self.levels = levels
        self.lookback_window = lookback_window
        
    def compute_lob_features(self, lob_data):
        """
        Compute all LOB features from raw market data
        
        Args:
            lob_data: DataFrame with LOB columns from BTC_1sec.csv
            
        Returns:
            Dictionary of LOB feature arrays
        """
        features = {}
        
        # Extract key LOB components
        bid_prices = self._extract_bid_prices(lob_data)
        ask_prices = self._extract_ask_prices(lob_data)
        bid_volumes = self._extract_bid_volumes(lob_data)
        ask_volumes = self._extract_ask_volumes(lob_data)
        midpoint = lob_data['midpoint'].values
        spread = lob_data['spread'].values
        
        # Order Flow Imbalance (3 features)
        features.update(self._compute_order_flow_imbalance(bid_volumes, ask_volumes))
        
        # Book Depth and Slope (2 features)
        features.update(self._compute_book_depth_slope(bid_volumes, ask_volumes))
        
        # Trade Imbalance (2 features)
        if 'buys' in lob_data.columns and 'sells' in lob_data.columns:
            features.update(self._compute_trade_imbalance(lob_data))
        else:
            features['trade_imbalance_10'] = np.zeros_like(midpoint)
            features['trade_imbalance_30'] = np.zeros_like(midpoint)
        
        # VWAP Deviation (1 feature)
        features['vwap_deviation'] = self._compute_vwap_deviation(lob_data)
        
        # Spread Dynamics (1 feature)
        features['spread_percentile'] = self._compute_spread_percentile(spread)
        
        # Liquidity Metrics (2 features)
        features.update(self._compute_liquidity_metrics(bid_prices, ask_prices, 
                                                       bid_volumes, ask_volumes, midpoint))
        
        # Order Wall Detection (2 features)
        features.update(self._compute_order_walls(bid_volumes, ask_volumes))
        
        # Flow Toxicity (1 feature)
        features['flow_toxicity'] = self._compute_flow_toxicity(lob_data)
        
        # Microprice (1 feature)
        features['microprice'] = self._compute_microprice(bid_prices, ask_prices, bid_volumes, ask_volumes)
        
        # Price Impact Measures (3 features)
        features.update(self._compute_price_impact_measures(lob_data))
        
        # Order Arrival/Cancellation Rates (4 features)
        features.update(self._compute_order_arrival_cancellation_rates(lob_data))
        
        # Microstructure Alpha (1 feature)  
        features['microstructure_alpha'] = self._compute_microstructure_alpha(lob_data)
        
        return features
    
    def _extract_bid_prices(self, lob_data):
        """Extract bid prices from LOB data"""
        bid_cols = [f'bids_distance_{i}' for i in range(self.levels)]
        bid_distances = lob_data[bid_cols].values
        midpoint = lob_data['midpoint'].values[:, None]
        return midpoint * (1 + bid_distances)
    
    def _extract_ask_prices(self, lob_data):
        """Extract ask prices from LOB data"""
        ask_cols = [f'asks_distance_{i}' for i in range(self.levels)]
        ask_distances = lob_data[ask_cols].values
        midpoint = lob_data['midpoint'].values[:, None]
        return midpoint * (1 + ask_distances)
    
    def _extract_bid_volumes(self, lob_data):
        """Extract bid volumes from LOB data"""
        bid_vol_cols = [f'bids_notional_{i}' for i in range(self.levels)]
        return lob_data[bid_vol_cols].values
    
    def _extract_ask_volumes(self, lob_data):
        """Extract ask volumes from LOB data"""
        ask_vol_cols = [f'asks_notional_{i}' for i in range(self.levels)]
        return lob_data[ask_vol_cols].values
    
    def _compute_order_flow_imbalance(self, bid_volumes, ask_volumes):
        """Compute order flow imbalance at different levels and timeframes"""
        features = {}
        
        # Instantaneous OFI at different levels  
        for level in range(min(3, self.levels)):
            bid_vol = bid_volumes[:, level]
            ask_vol = ask_volumes[:, level]
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
            features[f'order_imbalance_level_{level}'] = imbalance
        
        # Enhanced OFI with multiple timeframes
        features.update(self._compute_enhanced_order_flow_imbalance(bid_volumes, ask_volumes))
            
        return features
    
    def _compute_book_depth_slope(self, bid_volumes, ask_volumes):
        """Compute book depth and volume decay slope"""
        # Total depth in first 5 levels
        total_bid_depth = np.sum(bid_volumes[:, :self.levels], axis=1)
        total_ask_depth = np.sum(ask_volumes[:, :self.levels], axis=1)
        total_depth = total_bid_depth + total_ask_depth
        
        # Volume decay slope (how quickly volume decreases with distance)
        book_slopes = []
        for i in range(len(bid_volumes)):
            combined_vol = np.concatenate([bid_volumes[i, :self.levels][::-1], 
                                         ask_volumes[i, :self.levels]])
            if len(combined_vol) > 1:
                slope = np.polyfit(range(len(combined_vol)), combined_vol, 1)[0]
            else:
                slope = 0
            book_slopes.append(slope)
        
        # Normalize features
        depth_norm = (total_depth - np.mean(total_depth)) / (np.std(total_depth) + 1e-8)
        slope_norm = np.array(book_slopes)
        slope_norm = (slope_norm - np.mean(slope_norm)) / (np.std(slope_norm) + 1e-8)
        
        return {
            'book_depth': depth_norm,
            'book_slope': slope_norm
        }
    
    def _compute_trade_imbalance(self, lob_data):
        """Compute trade imbalance over different windows"""
        buys = lob_data['buys'].values
        sells = lob_data['sells'].values
        
        # Short-term imbalance (10 periods)
        imbalance_10 = self._rolling_calculation(
            lambda x: (np.sum(x['buys']) - np.sum(x['sells'])) / 
                     (np.sum(x['buys']) + np.sum(x['sells']) + 1e-8),
            pd.DataFrame({'buys': buys, 'sells': sells}), 10
        )
        
        # Medium-term imbalance (30 periods)  
        imbalance_30 = self._rolling_calculation(
            lambda x: (np.sum(x['buys']) - np.sum(x['sells'])) / 
                     (np.sum(x['buys']) + np.sum(x['sells']) + 1e-8),
            pd.DataFrame({'buys': buys, 'sells': sells}), 30
        )
        
        return {
            'trade_imbalance_10': imbalance_10,
            'trade_imbalance_30': imbalance_30
        }
    
    def _compute_vwap_deviation(self, lob_data):
        """Compute deviation from Volume-Weighted Average Price"""
        midpoint = lob_data['midpoint'].values
        
        # Simple VWAP approximation using total notional
        total_bid_notional = np.sum([lob_data[f'bids_notional_{i}'].values 
                                   for i in range(self.levels)], axis=0)
        total_ask_notional = np.sum([lob_data[f'asks_notional_{i}'].values 
                                   for i in range(self.levels)], axis=0)
        total_volume = total_bid_notional + total_ask_notional
        
        # Rolling VWAP over 30 periods
        vwap = self._rolling_calculation(
            lambda x: np.sum(x['price'] * x['volume']) / (np.sum(x['volume']) + 1e-8),
            pd.DataFrame({'price': midpoint, 'volume': total_volume}), 30
        )
        
        # Deviation from VWAP
        deviation = (midpoint - vwap) / (vwap + 1e-8)
        
        return deviation
    
    def _compute_spread_percentile(self, spread):
        """Compute rolling percentile of bid-ask spread"""
        spread_percentiles = []
        
        for i in range(len(spread)):
            start_idx = max(0, i - self.lookback_window + 1)
            window_spread = spread[start_idx:i+1]
            
            if len(window_spread) > 1:
                current_percentile = np.mean(window_spread <= spread[i]) * 100
            else:
                current_percentile = 50
                
            spread_percentiles.append(current_percentile)
        
        return np.array(spread_percentiles) / 100.0  # Normalize to [0,1]
    
    def _compute_liquidity_metrics(self, bid_prices, ask_prices, bid_volumes, ask_volumes, midpoint):
        """Compute liquidity at different price levels"""
        features = {}
        
        # Liquidity within 1% of mid price
        liquidity_1pct = []
        liquidity_2pct = []
        
        for i in range(len(midpoint)):
            mid = midpoint[i]
            
            # 1% liquidity
            bid_mask_1 = bid_prices[i] >= mid * 0.99
            ask_mask_1 = ask_prices[i] <= mid * 1.01
            liq_1 = np.sum(bid_volumes[i][bid_mask_1]) + np.sum(ask_volumes[i][ask_mask_1])
            liquidity_1pct.append(liq_1)
            
            # 2% liquidity
            bid_mask_2 = bid_prices[i] >= mid * 0.98
            ask_mask_2 = ask_prices[i] <= mid * 1.02
            liq_2 = np.sum(bid_volumes[i][bid_mask_2]) + np.sum(ask_volumes[i][ask_mask_2])
            liquidity_2pct.append(liq_2)
        
        # Normalize
        liq_1_norm = np.array(liquidity_1pct)
        liq_1_norm = (liq_1_norm - np.mean(liq_1_norm)) / (np.std(liq_1_norm) + 1e-8)
        
        liq_2_norm = np.array(liquidity_2pct)
        liq_2_norm = (liq_2_norm - np.mean(liq_2_norm)) / (np.std(liq_2_norm) + 1e-8)
        
        return {
            'liquidity_1pct': liq_1_norm,
            'liquidity_2pct': liq_2_norm
        }
    
    def _compute_order_walls(self, bid_volumes, ask_volumes):
        """Detect large order walls in the book"""
        # Identify unusually large orders (walls)
        bid_walls = []
        ask_walls = []
        
        for i in range(len(bid_volumes)):
            # Bid wall: significantly larger than average
            bid_vol = bid_volumes[i]
            bid_avg = np.mean(bid_vol)
            bid_std = np.std(bid_vol)
            bid_wall_strength = np.max((bid_vol - bid_avg) / (bid_std + 1e-8))
            bid_walls.append(max(0, bid_wall_strength))
            
            # Ask wall: significantly larger than average
            ask_vol = ask_volumes[i]
            ask_avg = np.mean(ask_vol)
            ask_std = np.std(ask_vol)
            ask_wall_strength = np.max((ask_vol - ask_avg) / (ask_std + 1e-8))
            ask_walls.append(max(0, ask_wall_strength))
        
        return {
            'bid_wall_strength': np.array(bid_walls),
            'ask_wall_strength': np.array(ask_walls)
        }
    
    def _compute_flow_toxicity(self, lob_data):
        """Compute flow toxicity indicator (adverse selection)"""
        midpoint = lob_data['midpoint'].values
        
        # Simple toxicity measure: volatility of midpoint changes
        mid_returns = np.diff(midpoint) / midpoint[:-1]
        mid_returns = np.concatenate([[0], mid_returns])
        
        # Rolling volatility as proxy for toxicity
        toxicity = self._rolling_calculation(
            lambda x: np.std(x),
            pd.Series(mid_returns), self.lookback_window
        )
        
        # Normalize
        toxicity = (toxicity - np.mean(toxicity)) / (np.std(toxicity) + 1e-8)
        
        return toxicity
    
    def _compute_microstructure_alpha(self, lob_data):
        """Compute microstructure alpha score"""
        # Combination of multiple microstructure signals
        midpoint = lob_data['midpoint'].values
        spread = lob_data['spread'].values
        
        # Price momentum
        returns = np.diff(midpoint) / midpoint[:-1]
        returns = np.concatenate([[0], returns])
        momentum_5 = self._rolling_calculation(np.mean, pd.Series(returns), 5)
        momentum_20 = self._rolling_calculation(np.mean, pd.Series(returns), 20)
        
        # Spread mean reversion
        spread_ma = self._rolling_calculation(np.mean, pd.Series(spread), 20)
        spread_signal = (spread - spread_ma) / (spread_ma + 1e-8)
        
        # Combine signals
        alpha = 0.5 * (momentum_5 - momentum_20) - 0.3 * spread_signal
        
        # Normalize
        alpha = (alpha - np.mean(alpha)) / (np.std(alpha) + 1e-8)
        
        return alpha
    
    def _rolling_calculation(self, func, data, window):
        """
        Optimized rolling calculation function for LOB features
        
        Performance improvements for large datasets:
        - Vectorized operations for common statistical functions
        - Efficient DataFrame column-wise processing
        - Fallback support for complex custom functions
        """
        
        # Handle edge cases
        if len(data) == 0:
            return np.array([])
        if window <= 0:
            window = 1
        if window > len(data):
            window = len(data)
        
        try:
            if isinstance(data, pd.DataFrame):
                # For DataFrames, optimize common operations
                func_name = getattr(func, '__name__', str(func))
                
                # Check if it's a simple aggregation function
                if func_name in ['mean', 'std', 'min', 'max', 'sum', 'var']:
                    # Use pandas rolling aggregation - much faster
                    rolling_obj = data.rolling(window=window, min_periods=1)
                    
                    if func_name == 'mean' or func == np.mean:
                        results = rolling_obj.mean().iloc[:, 0].values  # Take first column result
                    elif func_name == 'std' or func == np.std:
                        results = rolling_obj.std().iloc[:, 0].values
                    elif func_name == 'min' or func == np.min:
                        results = rolling_obj.min().iloc[:, 0].values
                    elif func_name == 'max' or func == np.max:
                        results = rolling_obj.max().iloc[:, 0].values
                    elif func_name == 'sum' or func == np.sum:
                        results = rolling_obj.sum().iloc[:, 0].values
                    elif func_name == 'var' or func == np.var:
                        results = rolling_obj.var().iloc[:, 0].values
                    else:
                        # Fallback to original implementation
                        return self._rolling_calculation_fallback_df(func, data, window)
                else:
                    # For complex functions, use optimized apply
                    rolling_obj = data.rolling(window=window, min_periods=1)
                    results = rolling_obj.apply(func, raw=False).iloc[:, 0].values
                
                # Handle NaN/inf values
                nan_mask = np.isnan(results) | np.isinf(results)
                if np.any(nan_mask):
                    results[nan_mask] = 0.0
                    
                return results
                
            else:
                # For Series, use optimized pandas rolling (already efficient)
                func_name = getattr(func, '__name__', str(func))
                
                if func_name == 'mean' or func == np.mean:
                    results = data.rolling(window, min_periods=1).mean()
                elif func_name == 'std' or func == np.std:
                    results = data.rolling(window, min_periods=1).std()
                elif func_name == 'min' or func == np.min:
                    results = data.rolling(window, min_periods=1).min()
                elif func_name == 'max' or func == np.max:
                    results = data.rolling(window, min_periods=1).max()
                elif func_name == 'sum' or func == np.sum:
                    results = data.rolling(window, min_periods=1).sum()
                elif func_name == 'var' or func == np.var:
                    results = data.rolling(window, min_periods=1).var()
                else:
                    results = data.rolling(window, min_periods=1).apply(func, raw=True)
                
                results = results.values
                
                # Handle NaN/inf values
                nan_mask = np.isnan(results) | np.isinf(results)
                if np.any(nan_mask):
                    results[nan_mask] = 0.0
                    
                return results
                
        except Exception as e:
            # Fallback to original implementation for edge cases
            print(f"⚠️ LOB rolling calculation fallback used: {e}")
            if isinstance(data, pd.DataFrame):
                return self._rolling_calculation_fallback_df(func, data, window)
            else:
                return data.rolling(window, min_periods=1).apply(func, raw=False).values
    
    def _rolling_calculation_fallback_df(self, func, data, window):
        """
        Fallback implementation for DataFrame rolling calculations
        
        This is the original loop-based implementation for complex functions
        """
        results = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data.iloc[start_idx:i+1]
            try:
                result = func(window_data)
                if np.isnan(result) or np.isinf(result):
                    result = 0.0
            except:
                result = 0.0
            results.append(result)
        return np.array(results)
    
    def _compute_microprice(self, bid_prices, ask_prices, bid_volumes, ask_volumes):
        """
        Compute microprice - volume-weighted price that's more accurate than midpoint
        
        Microprice = (ask_price * bid_volume + bid_price * ask_volume) / (bid_volume + ask_volume)
        
        Args:
            bid_prices: Array of bid prices (n_steps, n_levels)
            ask_prices: Array of ask prices (n_steps, n_levels)  
            bid_volumes: Array of bid volumes (n_steps, n_levels)
            ask_volumes: Array of ask volumes (n_steps, n_levels)
            
        Returns:
            Array of microprice values
        """
        # Use top level (level 0) for microprice calculation
        best_bid_price = bid_prices[:, 0]
        best_ask_price = ask_prices[:, 0] 
        best_bid_volume = bid_volumes[:, 0]
        best_ask_volume = ask_volumes[:, 0]
        
        # Compute microprice with numerical stability
        total_volume = best_bid_volume + best_ask_volume + 1e-8
        microprice = (best_ask_price * best_bid_volume + best_bid_price * best_ask_volume) / total_volume
        
        # Normalize as deviation from midpoint
        midpoint = (best_bid_price + best_ask_price) / 2
        microprice_deviation = (microprice - midpoint) / (midpoint + 1e-8)
        
        return microprice_deviation
    
    def _compute_enhanced_order_flow_imbalance(self, bid_volumes, ask_volumes):
        """
        Compute enhanced order flow imbalance with multiple timeframes
        
        Args:
            bid_volumes: Array of bid volumes (n_steps, n_levels)
            ask_volumes: Array of ask volumes (n_steps, n_levels)
            
        Returns:
            Dictionary of enhanced OFI features
        """
        features = {}
        
        # Aggregate volumes across top 3 levels for better signal
        agg_bid_vol = np.sum(bid_volumes[:, :3], axis=1)  
        agg_ask_vol = np.sum(ask_volumes[:, :3], axis=1)
        
        # Compute instantaneous aggregate OFI
        agg_ofi = (agg_bid_vol - agg_ask_vol) / (agg_bid_vol + agg_ask_vol + 1e-8)
        
        # Multi-timeframe rolling OFI
        timeframes = [5, 10, 30]
        
        for window in timeframes:
            # Rolling mean OFI
            rolling_ofi = self._rolling_calculation(
                np.mean, 
                pd.Series(agg_ofi), 
                window
            )
            features[f'ofi_rolling_mean_{window}'] = rolling_ofi
            
            # Rolling volatility of OFI (measures OFI instability)
            rolling_ofi_vol = self._rolling_calculation(
                np.std,
                pd.Series(agg_ofi),
                window  
            )
            features[f'ofi_rolling_vol_{window}'] = rolling_ofi_vol
            
            # OFI momentum (current vs rolling mean)
            ofi_momentum = agg_ofi - rolling_ofi
            features[f'ofi_momentum_{window}'] = ofi_momentum
        
        return features
    
    def _compute_price_impact_measures(self, lob_data):
        """
        Compute price impact measures from trade data and price movements
        
        Args:
            lob_data: DataFrame with LOB data including buys, sells, midpoint
            
        Returns:
            Dictionary of price impact features
        """
        features = {}
        
        midpoint = lob_data['midpoint'].values
        buys = lob_data['buys'].values if 'buys' in lob_data.columns else np.zeros_like(midpoint)
        sells = lob_data['sells'].values if 'sells' in lob_data.columns else np.zeros_like(midpoint)
        
        # Compute price returns
        price_changes = np.diff(midpoint)
        price_changes = np.concatenate([[0], price_changes])
        
        # Trade imbalance
        trade_imbalance = buys - sells
        
        # Price impact: correlation between trade imbalance and subsequent price changes
        # Shift price changes forward to measure impact
        future_price_changes = np.concatenate([price_changes[1:], [0]])
        
        # Rolling correlation between trade imbalance and future price changes
        impact_5 = self._rolling_correlation(trade_imbalance, future_price_changes, 5)
        impact_15 = self._rolling_correlation(trade_imbalance, future_price_changes, 15)
        
        features['price_impact_5'] = impact_5
        features['price_impact_15'] = impact_15
        
        # Volatility-adjusted price impact
        price_vol = self._rolling_calculation(np.std, pd.Series(price_changes), 20)
        vol_adjusted_impact = impact_15 / (price_vol + 1e-8)
        features['vol_adjusted_impact'] = vol_adjusted_impact
        
        return features
    
    def _rolling_correlation(self, x, y, window):
        """Compute rolling correlation between two series"""
        correlations = []
        
        for i in range(len(x)):
            start_idx = max(0, i - window + 1)
            x_window = x[start_idx:i+1]
            y_window = y[start_idx:i+1]
            
            if len(x_window) > 1 and np.std(x_window) > 1e-8 and np.std(y_window) > 1e-8:
                corr = np.corrcoef(x_window, y_window)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
                
            correlations.append(corr)
            
        return np.array(correlations)
    
    def _compute_order_arrival_cancellation_rates(self, lob_data):
        """
        Compute order arrival and cancellation rates from LOB data
        
        Args:
            lob_data: DataFrame with LOB data including limit, market, cancel notionals
            
        Returns:
            Dictionary of order flow features
        """
        features = {}
        
        # Extract order flow data from top 3 levels
        bid_limit_total = np.sum([lob_data[f'bids_limit_notional_{i}'].values 
                                for i in range(min(3, self.levels))], axis=0)
        ask_limit_total = np.sum([lob_data[f'asks_limit_notional_{i}'].values 
                                for i in range(min(3, self.levels))], axis=0)
        
        bid_cancel_total = np.sum([lob_data[f'bids_cancel_notional_{i}'].values 
                                 for i in range(min(3, self.levels))], axis=0)
        ask_cancel_total = np.sum([lob_data[f'asks_cancel_notional_{i}'].values 
                                 for i in range(min(3, self.levels))], axis=0)
        
        bid_market_total = np.sum([lob_data[f'bids_market_notional_{i}'].values 
                                 for i in range(min(3, self.levels))], axis=0)
        ask_market_total = np.sum([lob_data[f'asks_market_notional_{i}'].values 
                                 for i in range(min(3, self.levels))], axis=0)
        
        # Order arrival rate (new limit orders)
        total_limit_arrivals = bid_limit_total + ask_limit_total
        limit_imbalance = (bid_limit_total - ask_limit_total) / (total_limit_arrivals + 1e-8)
        
        # Order cancellation rate
        total_cancellations = bid_cancel_total + ask_cancel_total
        cancel_imbalance = (bid_cancel_total - ask_cancel_total) / (total_cancellations + 1e-8)
        
        # Market order arrival rate (immediate execution)
        total_market_orders = bid_market_total + ask_market_total
        market_imbalance = (bid_market_total - ask_market_total) / (total_market_orders + 1e-8)
        
        # Rolling averages to smooth the signals
        features['limit_order_imbalance'] = self._rolling_calculation(
            np.mean, pd.Series(limit_imbalance), 10
        )
        
        features['cancel_order_imbalance'] = self._rolling_calculation(
            np.mean, pd.Series(cancel_imbalance), 10
        )
        
        features['market_order_imbalance'] = self._rolling_calculation(
            np.mean, pd.Series(market_imbalance), 10
        )
        
        # Net order flow (limit arrivals - cancellations)
        net_bid_flow = bid_limit_total - bid_cancel_total
        net_ask_flow = ask_limit_total - ask_cancel_total
        net_flow_imbalance = (net_bid_flow - net_ask_flow) / (
            np.abs(net_bid_flow) + np.abs(net_ask_flow) + 1e-8
        )
        
        features['net_order_flow_imbalance'] = self._rolling_calculation(
            np.mean, pd.Series(net_flow_imbalance), 15
        )
        
        return features