#!/usr/bin/env python3
"""
Advanced Microstructure Features Module
Extracts sophisticated trading signals from Bitcoin LOB (Limit Order Book) data

This module implements cutting-edge microstructure features that provide unique alpha
in cryptocurrency markets by analyzing order flow, liquidity dynamics, and price discovery.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LOBMicrostructureFeatures:
    """
    Advanced microstructure feature calculator for Bitcoin LOB data
    
    Implements state-of-the-art order book analytics including:
    - Multi-level Order Book Imbalance (OBI)
    - Volume-weighted Microprice
    - Spread dynamics and liquidity measures
    - Order flow momentum and regime detection
    """
    
    def __init__(self, max_levels: int = 5):
        """
        Initialize the microstructure feature calculator
        
        Args:
            max_levels: Maximum LOB levels to use (default 5 for optimal signal/noise)
        """
        self.max_levels = max_levels
        self.feature_names = []
        
    def load_lob_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate LOB data structure"""
        
        print(f"📊 Loading LOB data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Validate required columns exist
        required_patterns = ['midpoint', 'spread', 'buys', 'sells', 'bids_distance_', 'asks_distance_', 
                           'bids_notional_', 'asks_notional_']
        
        missing_patterns = []
        for pattern in required_patterns:
            matching_cols = [col for col in df.columns if pattern in col]
            if not matching_cols:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            raise ValueError(f"Missing required LOB data patterns: {missing_patterns}")
        
        print(f"✅ LOB data loaded successfully: {df.shape}")
        print(f"   Available levels: {self.max_levels}")
        print(f"   Core columns: midpoint, spread, buys, sells")
        print(f"   LOB levels: 0-{self.max_levels-1} (bid/ask distances and notionals)")
        
        return df
    
    def calculate_order_book_imbalance_suite(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate multi-level Order Book Imbalance (OBI) features
        
        OBI measures buy vs sell pressure in the order book, providing early signals
        of price direction changes based on supply/demand dynamics.
        """
        
        print("📈 Calculating Order Book Imbalance (OBI) suite...")
        features = {}
        
        # Extract bid and ask notionals for multiple levels
        bid_notionals = []
        ask_notionals = []
        
        for level in range(self.max_levels):
            bid_col = f'bids_notional_{level}'
            ask_col = f'asks_notional_{level}'
            
            if bid_col in df.columns and ask_col in df.columns:
                bid_notionals.append(df[bid_col].values)
                ask_notionals.append(df[ask_col].values)
        
        if not bid_notionals:
            print("   ⚠️  No notional data found, using buys/sells as proxy")
            # Fallback to buy/sell flow data
            features['obi_flow'] = self._calculate_flow_imbalance(df['buys'].values, df['sells'].values)
            return features
        
        bid_notionals = np.array(bid_notionals).T  # Shape: (timesteps, levels)
        ask_notionals = np.array(ask_notionals).T
        
        # 1. Multi-level OBI calculations
        for levels in [1, 3, 5]:
            if levels <= len(bid_notionals[0]):
                bid_sum = np.sum(bid_notionals[:, :levels], axis=1)
                ask_sum = np.sum(ask_notionals[:, :levels], axis=1)
                
                # Standard OBI: (bids - asks) / (bids + asks)
                total_volume = bid_sum + ask_sum
                obi = np.where(total_volume > 0, (bid_sum - ask_sum) / total_volume, 0)
                
                features[f'obi_{levels}_level'] = obi
        
        # 2. Weighted OBI (closer levels have higher weights)
        if len(bid_notionals[0]) >= 3:
            num_levels = min(len(bid_notionals[0]), 5)
            weights = np.array([1.0, 0.7, 0.5, 0.3, 0.1])[:num_levels]
            weighted_bids = np.sum(bid_notionals[:, :num_levels] * weights, axis=1)
            weighted_asks = np.sum(ask_notionals[:, :num_levels] * weights, axis=1)
            
            total_weighted = weighted_bids + weighted_asks
            weighted_obi = np.where(total_weighted > 0, 
                                  (weighted_bids - weighted_asks) / total_weighted, 0)
            features['obi_weighted'] = weighted_obi
        
        # 3. OBI Momentum (rate of change)
        if 'obi_3_level' in features:
            obi_momentum = np.diff(features['obi_3_level'])
            obi_momentum = np.concatenate([[0], obi_momentum])  # Pad for length
            features['obi_momentum'] = obi_momentum
        
        # 4. OBI Volatility (measure of imbalance stability)
        if 'obi_3_level' in features:
            obi_vol = pd.Series(features['obi_3_level']).rolling(window=20).std().fillna(0).values
            features['obi_volatility'] = obi_vol
        
        print(f"   ✅ Created {len(features)} OBI features")
        return features
    
    def calculate_microprice_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate volume-weighted microprice features
        
        Microprice provides a more accurate measure of the "true" price by incorporating
        volume information, leading to better entry/exit timing than simple midpoint.
        """
        
        print("💰 Calculating microprice features...")
        features = {}
        
        # Get best bid/ask data
        if 'bids_notional_0' in df.columns and 'asks_notional_0' in df.columns:
            # Method 1: Volume-weighted microprice using best levels
            bid_vol = df['bids_notional_0'].values
            ask_vol = df['asks_notional_0'].values
            midpoint = df['midpoint'].values
            spread = df['spread'].values
            
            # Calculate bid/ask prices from midpoint and spread
            bid_price = midpoint - spread / 2
            ask_price = midpoint + spread / 2
            
            # Volume-weighted microprice
            total_vol = bid_vol + ask_vol
            microprice = np.where(total_vol > 0,
                                (ask_vol * bid_price + bid_vol * ask_price) / total_vol,
                                midpoint)
            
            features['microprice'] = microprice
            
            # Microprice vs midpoint deviation
            microprice_deviation = (microprice - midpoint) / midpoint * 10000  # bps
            features['microprice_deviation'] = microprice_deviation
            
            # Microprice momentum
            microprice_momentum = np.diff(microprice)
            microprice_momentum = np.concatenate([[0], microprice_momentum])
            features['microprice_momentum'] = microprice_momentum
            
        else:
            print("   ⚠️  Using midpoint as microprice proxy")
            features['microprice'] = df['midpoint'].values
            features['microprice_deviation'] = np.zeros(len(df))
            features['microprice_momentum'] = np.zeros(len(df))
        
        # Enhanced microprice stability measure
        if 'microprice' in features:
            microprice_stability = pd.Series(features['microprice']).rolling(window=10).std().fillna(0).values
            features['microprice_stability'] = microprice_stability
        
        print(f"   ✅ Created {len(features)} microprice features")
        return features
    
    def calculate_spread_dynamics_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate spread dynamics and liquidity measures
        
        Spread behavior provides insights into market liquidity conditions and
        can predict volatility regime changes and market stress periods.
        """
        
        print("📊 Calculating spread dynamics features...")
        features = {}
        
        spread = df['spread'].values
        midpoint = df['midpoint'].values
        
        # 1. Spread Volatility (14-period rolling std)
        spread_volatility = pd.Series(spread).rolling(window=14).std().fillna(spread.std()).values
        features['spread_volatility'] = spread_volatility
        
        # 2. Relative Spread (normalized by price level)
        relative_spread = spread / midpoint * 10000  # bps
        features['relative_spread'] = relative_spread
        
        # 3. Spread Momentum (rate of change)
        spread_momentum = np.diff(spread)
        spread_momentum = np.concatenate([[0], spread_momentum])
        features['spread_momentum'] = spread_momentum
        
        # 4. Spread Regime Indicator (high/normal/low spread periods)
        spread_percentiles = np.percentile(spread, [25, 75])
        spread_regime = np.where(spread > spread_percentiles[1], 1,  # High spread
                               np.where(spread < spread_percentiles[0], -1, 0))  # Low spread
        features['spread_regime'] = spread_regime
        
        # 5. Spread Mean Reversion Indicator
        spread_ma = pd.Series(spread).rolling(window=20).mean().fillna(spread.mean()).values
        spread_mean_reversion = (spread - spread_ma) / spread_ma
        features['spread_mean_reversion'] = spread_mean_reversion
        
        # 6. Liquidity Stress Indicator (high spread + low volume)
        if 'bids_notional_0' in df.columns and 'asks_notional_0' in df.columns:
            total_top_volume = df['bids_notional_0'].values + df['asks_notional_0'].values
            volume_percentile_20 = np.percentile(total_top_volume, 20)
            spread_percentile_80 = np.percentile(spread, 80)
            
            liquidity_stress = ((spread > spread_percentile_80) & 
                              (total_top_volume < volume_percentile_20)).astype(int)
            features['liquidity_stress'] = liquidity_stress
        
        print(f"   ✅ Created {len(features)} spread dynamics features")
        return features
    
    def calculate_order_flow_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate advanced order flow features
        
        Order flow captures the directional pressure from market participants,
        providing early signals of institutional activity and momentum shifts.
        """
        
        print("🌊 Calculating order flow features...")
        features = {}
        
        buys = df['buys'].values
        sells = df['sells'].values
        
        # 1. Net Order Flow
        net_flow = buys - sells
        features['net_order_flow'] = net_flow
        
        # 2. Order Flow Ratio
        total_flow = buys + sells
        flow_ratio = np.where(total_flow > 0, net_flow / total_flow, 0)
        features['order_flow_ratio'] = flow_ratio
        
        # 3. Order Flow Momentum (different windows)
        for window in [5, 10, 20]:
            flow_momentum = pd.Series(net_flow).rolling(window=window).mean().fillna(0).values
            features[f'order_flow_momentum_{window}'] = flow_momentum
        
        # 4. Order Flow Volatility
        flow_volatility = pd.Series(net_flow).rolling(window=20).std().fillna(0).values
        features['order_flow_volatility'] = flow_volatility
        
        # 5. Buy/Sell Pressure Asymmetry
        buy_pressure = pd.Series(buys).rolling(window=10).mean().fillna(0).values
        sell_pressure = pd.Series(sells).rolling(window=10).mean().fillna(0).values
        
        pressure_asymmetry = np.where((buy_pressure + sell_pressure) > 0,
                                    (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure),
                                    0)
        features['pressure_asymmetry'] = pressure_asymmetry
        
        print(f"   ✅ Created {len(features)} order flow features")
        return features
    
    def _calculate_flow_imbalance(self, buys: np.ndarray, sells: np.ndarray) -> np.ndarray:
        """Helper function for flow-based imbalance calculation"""
        total_flow = buys + sells
        return np.where(total_flow > 0, (buys - sells) / total_flow, 0)
    
    def generate_all_microstructure_features(self, csv_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Generate complete suite of microstructure features
        
        Returns:
            features_dict: Dictionary of feature name -> numpy array
            feature_names: List of feature names in order
        """
        
        print("🚀 GENERATING ADVANCED MICROSTRUCTURE FEATURES")
        print("=" * 70)
        
        # Load data
        df = self.load_lob_data(csv_path)
        
        # Generate all feature categories
        all_features = {}
        
        # 1. Order Book Imbalance suite
        obi_features = self.calculate_order_book_imbalance_suite(df)
        all_features.update(obi_features)
        
        # 2. Microprice features
        microprice_features = self.calculate_microprice_features(df)
        all_features.update(microprice_features)
        
        # 3. Spread dynamics features
        spread_features = self.calculate_spread_dynamics_features(df)
        all_features.update(spread_features)
        
        # 4. Order flow features
        flow_features = self.calculate_order_flow_features(df)
        all_features.update(flow_features)
        
        # Compile feature names
        feature_names = list(all_features.keys())
        self.feature_names = feature_names
        
        print(f"\n📊 MICROSTRUCTURE FEATURE GENERATION COMPLETE")
        print(f"   Total features generated: {len(feature_names)}")
        print(f"   Feature categories: OBI, Microprice, Spread Dynamics, Order Flow")
        print(f"   Data shape: ({len(df)}, {len(feature_names)})")
        
        print(f"\n🎯 Generated Features:")
        for i, name in enumerate(feature_names, 1):
            print(f"   {i:2d}. {name}")
        
        return all_features, feature_names

def main():
    """Test the microstructure feature generator"""
    
    print("🧪 TESTING ADVANCED MICROSTRUCTURE FEATURES")
    print("=" * 50)
    
    try:
        # Initialize feature calculator
        lob_features = LOBMicrostructureFeatures(max_levels=5)
        
        # Generate features
        csv_path = "../../../data/raw/task1/BTC_1sec.csv"
        features_dict, feature_names = lob_features.generate_all_microstructure_features(csv_path)
        
        # Display feature statistics
        print(f"\n📈 FEATURE STATISTICS:")
        for name in feature_names[:10]:  # Show first 10 features
            feature_data = features_dict[name]
            print(f"   {name}:")
            print(f"     Range: [{feature_data.min():.6f}, {feature_data.max():.6f}]")
            print(f"     Mean: {feature_data.mean():.6f}, Std: {feature_data.std():.6f}")
        
        if len(feature_names) > 10:
            print(f"   ... and {len(feature_names) - 10} more features")
        
        print(f"\n✅ Microstructure feature generation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in microstructure feature generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Advanced microstructure features module is ready!")
    else:
        print("\n💥 Microstructure features module test failed!")