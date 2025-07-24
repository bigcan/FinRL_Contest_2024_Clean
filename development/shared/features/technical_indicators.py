"""
Technical Indicators Module using pandas-ta

Provides wrapper functions for standard technical indicators:
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- OBV (On-Balance Volume)
- EMA/SMA crossovers
- Stochastic oscillators
"""

import numpy as np
import pandas as pd

class TechnicalIndicators:
    """Technical indicators calculator using pandas-ta"""
    
    def __init__(self, lookback_window=100):
        """
        Initialize technical indicators calculator
        
        Args:
            lookback_window: Number of periods to keep for indicator calculation
        """
        self.lookback_window = lookback_window
        
    def compute_indicators(self, price_data, volume_data=None):
        """
        Compute all technical indicators
        
        Args:
            price_data: Array of shape (n_steps, 3) with [bid, ask, mid] prices
            volume_data: Array of shape (n_steps,) with volume data (optional)
            
        Returns:
            Dictionary of indicator arrays
        """
        try:
            import pandas_ta as ta
        except ImportError:
            print("Warning: pandas-ta not installed. Using fallback indicators.")
            return self._compute_fallback_indicators(price_data, volume_data)
            
        # Convert to pandas DataFrame
        mid_prices = price_data[:, 2] if price_data.shape[1] >= 3 else price_data[:, 0]
        df = pd.DataFrame({
            'close': mid_prices,
            'high': mid_prices * 1.001,  # Approximate high
            'low': mid_prices * 0.999,   # Approximate low
            'volume': volume_data if volume_data is not None else np.ones_like(mid_prices)
        })
        
        indicators = {}
        
        # MACD (3 features)
        macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_data is not None:
            indicators['macd'] = macd_data['MACD_12_26_9'].fillna(0).values
            indicators['macd_signal'] = macd_data['MACDs_12_26_9'].fillna(0).values
            indicators['macd_histogram'] = macd_data['MACDh_12_26_9'].fillna(0).values
        else:
            indicators.update(self._fallback_macd(mid_prices))
            
        # RSI (1 feature)
        rsi_data = ta.rsi(df['close'], length=14)
        if rsi_data is not None:
            indicators['rsi'] = rsi_data.fillna(50).values / 100.0  # Normalize to [0,1]
        else:
            indicators['rsi'] = self._fallback_rsi(mid_prices)
            
        # Bollinger Bands (4 features)
        bb_data = ta.bbands(df['close'], length=20, std=2)
        if bb_data is not None:
            mid_price_mean = mid_prices.mean()
            mid_price_std = mid_prices.std()
            indicators['bb_upper'] = (bb_data['BBU_20_2.0'].fillna(mid_price_mean).values - mid_price_mean) / mid_price_std
            indicators['bb_middle'] = (bb_data['BBM_20_2.0'].fillna(mid_price_mean).values - mid_price_mean) / mid_price_std
            indicators['bb_lower'] = (bb_data['BBL_20_2.0'].fillna(mid_price_mean).values - mid_price_mean) / mid_price_std
            indicators['bb_percent'] = bb_data['BBB_20_2.0'].fillna(0.5).values
        else:
            indicators.update(self._fallback_bollinger(mid_prices))
            
        # OBV (2 features)
        if volume_data is not None:
            obv_data = ta.obv(df['close'], df['volume'])
            if obv_data is not None:
                obv_norm = obv_data.fillna(0).values
                obv_norm = (obv_norm - obv_norm.mean()) / (obv_norm.std() + 1e-8)
                indicators['obv'] = obv_norm
                indicators['obv_rate'] = np.gradient(obv_norm)
            else:
                indicators.update(self._fallback_obv(mid_prices, volume_data))
        else:
            indicators['obv'] = np.zeros_like(mid_prices)
            indicators['obv_rate'] = np.zeros_like(mid_prices)
            
        # EMA Crossover (1 feature)
        ema_20 = ta.ema(df['close'], length=20)
        ema_50 = ta.ema(df['close'], length=50)
        if ema_20 is not None and ema_50 is not None:
            crossover = (ema_20.fillna(0) > ema_50.fillna(0)).astype(float).values
            indicators['ema_crossover'] = crossover
        else:
            indicators['ema_crossover'] = self._fallback_ema_crossover(mid_prices)
            
        # Stochastic %K (1 feature)
        stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch_data is not None:
            indicators['stoch_k'] = stoch_data['STOCHk_14_3_3'].fillna(50).values / 100.0
        else:
            indicators['stoch_k'] = self._fallback_stoch(mid_prices)
        
        # Historical Volatility (2 features)
        indicators.update(self._compute_historical_volatility(mid_prices))
        
        # ATR Approximation (1 feature) 
        indicators['atr_approximation'] = self._compute_atr_approximation(df)
        
        # Market Regime Indicators (2 features)
        indicators.update(self._compute_market_regime_indicators(mid_prices))
            
        return indicators
    
    def _compute_fallback_indicators(self, price_data, volume_data):
        """Fallback indicators when pandas-ta is not available"""
        mid_prices = price_data[:, 2] if price_data.shape[1] >= 3 else price_data[:, 0]
        
        indicators = {}
        indicators.update(self._fallback_macd(mid_prices))
        indicators['rsi'] = self._fallback_rsi(mid_prices)
        indicators.update(self._fallback_bollinger(mid_prices))
        indicators.update(self._fallback_obv(mid_prices, volume_data))
        indicators['ema_crossover'] = self._fallback_ema_crossover(mid_prices)
        indicators['stoch_k'] = self._fallback_stoch(mid_prices)
        
        # Additional indicators
        indicators.update(self._compute_historical_volatility(mid_prices))
        indicators['atr_approximation'] = self._compute_atr_approximation_fallback(mid_prices)
        indicators.update(self._compute_market_regime_indicators(mid_prices))
        
        return indicators
    
    def _fallback_macd(self, prices):
        """Simple MACD implementation"""
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd = ema_12 - ema_26
        signal = self._ema(macd, 9)
        histogram = macd - signal
        
        # Normalize
        macd_std = np.std(macd) + 1e-8
        return {
            'macd': macd / macd_std,
            'macd_signal': signal / macd_std,
            'macd_histogram': histogram / macd_std
        }
    
    def _fallback_rsi(self, prices, period=14):
        """Simple RSI implementation"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = self._sma(np.concatenate([[0], gains]), period)
        avg_losses = self._sma(np.concatenate([[0], losses]), period)
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0  # Normalize to [0,1]
    
    def _fallback_bollinger(self, prices, period=20, std_mult=2):
        """Simple Bollinger Bands implementation"""
        sma = self._sma(prices, period)
        std = self._rolling_std(prices, period)
        
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        
        # Normalize relative to price mean and std
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        
        return {
            'bb_upper': (upper - price_mean) / price_std,
            'bb_middle': (sma - price_mean) / price_std,
            'bb_lower': (lower - price_mean) / price_std,
            'bb_percent': (prices - lower) / (upper - lower + 1e-8)
        }
    
    def _fallback_obv(self, prices, volumes):
        """Simple OBV implementation"""
        if volumes is None:
            return {'obv': np.zeros_like(prices), 'obv_rate': np.zeros_like(prices)}
            
        price_changes = np.diff(prices)
        obv = np.zeros_like(prices)
        
        for i in range(1, len(prices)):
            if price_changes[i-1] > 0:
                obv[i] = obv[i-1] + volumes[i]
            elif price_changes[i-1] < 0:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        # Normalize
        obv_norm = (obv - np.mean(obv)) / (np.std(obv) + 1e-8)
        return {
            'obv': obv_norm,
            'obv_rate': np.gradient(obv_norm)
        }
    
    def _fallback_ema_crossover(self, prices):
        """Simple EMA crossover signal"""
        ema_20 = self._ema(prices, 20)
        ema_50 = self._ema(prices, 50)
        return (ema_20 > ema_50).astype(float)
    
    def _fallback_stoch(self, prices, period=14):
        """Simple Stochastic %K"""
        stoch_k = np.zeros_like(prices)
        for i in range(period, len(prices)):
            window = prices[i-period:i+1]
            high = np.max(window)
            low = np.min(window)
            stoch_k[i] = (prices[i] - low) / (high - low + 1e-8)
        return stoch_k
    
    def _ema(self, prices, period):
        """Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _sma(self, prices, period):
        """Simple Moving Average"""
        sma = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[max(0, i-period+1):i+1])
        return sma
    
    def _rolling_std(self, prices, period):
        """Rolling standard deviation"""
        std = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[max(0, i-period+1):i+1])
        return std
    
    def _compute_historical_volatility(self, prices):
        """
        Compute historical volatility measures
        
        Args:
            prices: Array of price values
            
        Returns:
            Dictionary of volatility features
        """
        # Log returns for volatility calculation
        log_returns = np.diff(np.log(prices + 1e-8))
        log_returns = np.concatenate([[0], log_returns])
        
        # Short-term volatility (10 periods)
        vol_10 = self._rolling_std(log_returns, 10)
        
        # Medium-term volatility (20 periods)  
        vol_20 = self._rolling_std(log_returns, 20)
        
        # Normalize volatilities
        vol_10_norm = (vol_10 - np.mean(vol_10)) / (np.std(vol_10) + 1e-8)
        vol_20_norm = (vol_20 - np.mean(vol_20)) / (np.std(vol_20) + 1e-8)
        
        return {
            'historical_vol_10': vol_10_norm,
            'historical_vol_20': vol_20_norm
        }
    
    def _compute_atr_approximation(self, df):
        """
        Compute ATR approximation for crypto data using pandas-ta
        
        Args:
            df: DataFrame with high, low, close prices
            
        Returns:
            Array of normalized ATR values
        """
        try:
            import pandas_ta as ta
            atr_data = ta.atr(df['high'], df['low'], df['close'], length=14)
            if atr_data is not None:
                atr_norm = atr_data.fillna(0).values
                atr_norm = (atr_norm - np.mean(atr_norm)) / (np.std(atr_norm) + 1e-8)
                return atr_norm
        except:
            pass
        
        # Fallback to simple ATR approximation
        return self._compute_atr_approximation_fallback(df['close'].values)
    
    def _compute_atr_approximation_fallback(self, prices):
        """
        Fallback ATR approximation using price ranges
        
        Args:
            prices: Array of price values
            
        Returns:
            Array of normalized ATR approximation
        """
        # Use rolling high-low range as ATR proxy
        period = 14
        atr_values = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            window = prices[max(0, i-period+1):i+1]
            high_low_range = np.max(window) - np.min(window)
            
            # True range approximation using consecutive price differences
            if i > 0:
                price_change = abs(prices[i] - prices[i-1])
                true_range = max(high_low_range, price_change)
            else:
                true_range = high_low_range
                
            atr_values[i] = true_range
        
        # Smooth with exponential moving average
        atr_smoothed = self._ema(atr_values, period)
        
        # Normalize relative to price level
        atr_normalized = atr_smoothed / (prices + 1e-8)
        atr_normalized = (atr_normalized - np.mean(atr_normalized)) / (np.std(atr_normalized) + 1e-8)
        
        return atr_normalized
    
    def _compute_market_regime_indicators(self, prices):
        """
        Compute market regime indicators (trending vs ranging)
        
        Args:
            prices: Array of price values
            
        Returns:
            Dictionary of market regime features
        """
        # ADX approximation using directional movement
        adx_approx = self._compute_adx_approximation(prices)
        
        # Trend strength indicator
        trend_strength = self._compute_trend_strength(prices)
        
        return {
            'adx_approximation': adx_approx,
            'trend_strength': trend_strength
        }
    
    def _compute_adx_approximation(self, prices, period=14):
        """Approximate ADX using price movements"""
        adx_values = np.zeros_like(prices)
        
        for i in range(1, len(prices)):
            # Directional movement
            up_move = prices[i] - prices[i-1] if prices[i] > prices[i-1] else 0
            down_move = prices[i-1] - prices[i] if prices[i] < prices[i-1] else 0
            
            # Smooth directional movements
            if i >= period:
                window_up = []
                window_down = []
                
                for j in range(max(0, i-period+1), i+1):
                    if j > 0:
                        up_j = prices[j] - prices[j-1] if prices[j] > prices[j-1] else 0
                        down_j = prices[j-1] - prices[j] if prices[j] < prices[j-1] else 0
                        window_up.append(up_j)
                        window_down.append(down_j)
                
                avg_up = np.mean(window_up)
                avg_down = np.mean(window_down)
                
                # ADX approximation
                if avg_up + avg_down > 1e-8:
                    dx = abs(avg_up - avg_down) / (avg_up + avg_down)
                    adx_values[i] = dx
        
        # Normalize
        adx_norm = (adx_values - np.mean(adx_values)) / (np.std(adx_values) + 1e-8)
        return adx_norm
    
    def _compute_trend_strength(self, prices, period=20):
        """Compute trend strength using linear regression slope"""
        trend_values = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            window = prices[max(0, i-period+1):i+1]
            x = np.arange(len(window))
            
            # Linear regression slope
            if len(window) > 1:
                slope = np.polyfit(x, window, 1)[0]
                trend_values[i] = slope / (np.mean(window) + 1e-8)  # Normalize by price level
        
        # Normalize trend strength
        trend_norm = (trend_values - np.mean(trend_values)) / (np.std(trend_values) + 1e-8)
        return trend_norm