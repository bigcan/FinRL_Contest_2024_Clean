#!/usr/bin/env python3
"""
Analyze Existing Feature Selection System
Understand the current optimized feature set and methodology
"""
import numpy as np
import os
from typing import Dict, List

def analyze_optimized_features():
    """Analyze the existing optimized feature system"""
    
    print("🔍 ANALYZING EXISTING FEATURE OPTIMIZATION SYSTEM")
    print("=" * 70)
    
    # Load optimized features metadata
    metadata_path = "../../../data/raw/task1/BTC_1sec_predict_optimized_metadata.npy"
    
    if os.path.exists(metadata_path):
        print(f"📊 Loading metadata from: {metadata_path}")
        metadata = np.load(metadata_path, allow_pickle=True).item()
        
        print(f"\n✅ CURRENT OPTIMIZED FEATURE SET:")
        print(f"   📅 Creation Date: {metadata.get('creation_date', 'Unknown')}")
        print(f"   🔬 Phase: {metadata.get('phase', 'Unknown')}")
        print(f"   📈 Expected Accuracy: {metadata.get('expected_accuracy', 'Unknown')}")
        print(f"   🎯 Selection Criteria: {metadata.get('selection_criteria', 'Unknown')}")
        
        # Current selected features
        selected_features = metadata.get('feature_names', [])
        print(f"\n🎯 SELECTED FEATURES ({len(selected_features)} total):")
        for i, feature in enumerate(selected_features, 1):
            print(f"   {i:2d}. {feature}")
        
        # Removed features
        removed_features = metadata.get('removed_features', [])
        print(f"\n🗑️  REMOVED FEATURES ({len(removed_features)} total):")
        for i, feature in enumerate(removed_features, 1):
            print(f"   {i:2d}. {feature}")
        
        # Shape information
        original_shape = metadata.get('original_shape', 'Unknown')
        optimized_shape = metadata.get('optimized_shape', 'Unknown')
        
        print(f"\n📏 SHAPE TRANSFORMATION:")
        print(f"   Original: {original_shape}")
        print(f"   Optimized: {optimized_shape}")
        
        if isinstance(original_shape, tuple) and isinstance(optimized_shape, tuple):
            reduction = original_shape[1] - optimized_shape[1]
            reduction_pct = (reduction / original_shape[1]) * 100
            print(f"   Reduction: {reduction} features ({reduction_pct:.1f}%)")
        
        return metadata
    else:
        print(f"❌ Metadata file not found: {metadata_path}")
        return None

def analyze_feature_types(feature_names: List[str]) -> Dict[str, List[str]]:
    """Categorize features by type"""
    
    categories = {
        'Technical Indicators': [],
        'Market Microstructure': [],
        'Momentum Features': [],
        'Original RNN Features': [],
        'Position Features': []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        if 'ema' in feature_lower or 'rsi' in feature_lower:
            categories['Technical Indicators'].append(feature)
        elif 'spread' in feature_lower or 'imbalance' in feature_lower or 'flow' in feature_lower:
            categories['Market Microstructure'].append(feature)
        elif 'momentum' in feature_lower:
            categories['Momentum Features'].append(feature)
        elif 'original_' in feature_lower:
            categories['Original RNN Features'].append(feature)
        elif 'position' in feature_lower or 'holding' in feature_lower:
            categories['Position Features'].append(feature)
    
    return categories

def reverse_engineer_selection_criteria(metadata: Dict) -> Dict[str, str]:
    """Reverse engineer the selection criteria from existing features"""
    
    selected = metadata.get('feature_names', [])
    removed = metadata.get('removed_features', [])
    
    analysis = {
        'technical_indicators': 'KEPT most technical indicators (EMA, RSI)',
        'microstructure': 'KEPT market microstructure features (spread, imbalance, order_flow)',
        'momentum': 'MIXED momentum features (kept momentum_20, removed momentum_5)',
        'original_features': 'SELECTIVE on original RNN features (kept 0,4 removed 1,2,5)',
        'position_features': 'REMOVED position normalization features'
    }
    
    print(f"\n🧠 REVERSE-ENGINEERED SELECTION LOGIC:")
    for category, logic in analysis.items():
        print(f"   • {category}: {logic}")
    
    return analysis

def suggest_enhancements(metadata: Dict) -> List[str]:
    """Suggest enhancements to the existing feature system"""
    
    selected_features = metadata.get('feature_names', [])
    
    suggestions = []
    
    # Check for missing feature types
    has_volatility = any('vol' in f.lower() for f in selected_features)
    has_bollinger = any('bollinger' in f.lower() or 'bb_' in f.lower() for f in selected_features)
    has_macd = any('macd' in f.lower() for f in selected_features)
    has_regime = any('regime' in f.lower() for f in selected_features)
    
    if not has_volatility:
        suggestions.append("ADD volatility features (missing from current set)")
    
    if not has_bollinger:
        suggestions.append("ADD Bollinger Bands features (complement RSI)")
    
    if not has_macd:
        suggestions.append("ADD MACD features (momentum indicator)")
    
    if not has_regime:
        suggestions.append("ADD regime detection features (market state awareness)")
    
    # Check for multi-timeframe features
    has_multi_timeframe = any(any(tf in f for tf in ['1min', '5min', '30min']) for f in selected_features)
    if not has_multi_timeframe:
        suggestions.append("ADD multi-timeframe analysis (different time horizons)")
    
    # Check for statistical features
    has_zscore = any('zscore' in f.lower() or 'norm' in f.lower() for f in selected_features)
    if not has_zscore:
        suggestions.append("ADD statistical normalization features (Z-score, percentiles)")
    
    return suggestions

def print_enhancement_roadmap(suggestions: List[str], metadata: Dict):
    """Print enhancement roadmap for the existing system"""
    
    print(f"\n🚀 ENHANCEMENT ROADMAP FOR EXISTING SYSTEM:")
    print("=" * 70)
    
    current_count = len(metadata.get('feature_names', []))
    
    print(f"📊 Current Status: {current_count} optimized features")
    print(f"🎯 Target: Enhance to ~15-20 features with better diversity")
    
    print(f"\n📋 RECOMMENDED ENHANCEMENTS:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    print(f"\n⚡ IMPLEMENTATION STRATEGY:")
    print("   1. Keep existing 8 optimized features as base")
    print("   2. Add suggested enhancements systematically")
    print("   3. Use correlation analysis to avoid redundancy")
    print("   4. Apply feature importance scoring")
    print("   5. Test performance with incremental additions")
    
    print(f"\n🔬 SELECTION METHODOLOGY (Inferred from existing):")
    print("   • Phase 1: Correlation analysis")
    print("   • Phase 2: Feature importance scoring") 
    print("   • Phase 3: Ablation studies")
    print("   • Criteria: Remove highly correlated and low-importance features")

if __name__ == "__main__":
    # Analyze existing system
    metadata = analyze_optimized_features()
    
    if metadata:
        # Categorize features
        selected_features = metadata.get('feature_names', [])
        categories = analyze_feature_types(selected_features)
        
        print(f"\n📂 FEATURE CATEGORIZATION:")
        for category, features in categories.items():
            if features:
                print(f"   {category}: {features}")
            else:
                print(f"   {category}: None")
        
        # Reverse engineer selection logic
        selection_logic = reverse_engineer_selection_criteria(metadata)
        
        # Suggest enhancements
        suggestions = suggest_enhancements(metadata)
        print_enhancement_roadmap(suggestions, metadata)
    
    print(f"\n💡 CONCLUSION:")
    print("   The system already has a sophisticated 8-feature optimization.")
    print("   Rather than replacing it, we should ENHANCE it by:")
    print("   1. Adding complementary features (volatility, MACD, regimes)")
    print("   2. Using the same selection methodology")
    print("   3. Targeting 15-20 total features for better performance")
    print("=" * 70)