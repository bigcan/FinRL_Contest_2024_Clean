"""
Feature Correlation Analysis for Enhanced Features
Analyzes correlation matrix and identifies redundant features for removal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import os
import sys

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
task1_src = os.path.join(project_root, 'development', 'task1', 'src')
sys.path.append(task1_src)

def load_enhanced_features():
    """Load enhanced features and metadata"""
    data_path = os.path.join(project_root, 'data', 'raw', 'task1')
    
    # Load enhanced features
    enhanced_path = os.path.join(data_path, 'BTC_1sec_predict_enhanced.npy')
    if not os.path.exists(enhanced_path):
        raise FileNotFoundError(f"Enhanced features not found: {enhanced_path}")
    
    features = np.load(enhanced_path)
    
    # Load metadata
    metadata_path = enhanced_path.replace('.npy', '_metadata.npy')
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        feature_names = metadata.get('feature_names', [f'feature_{i}' for i in range(features.shape[1])])
    else:
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
    
    print(f"Loaded features: {features.shape}")
    print(f"Feature names ({len(feature_names)}): {feature_names}")
    
    return features, feature_names

def create_target_variable(features, lookahead=5):
    """Create target variable for feature importance analysis"""
    # Try to use a feature that represents price/returns
    # Check multiple features to find one with good variation
    price_candidates = [features[:, i] for i in [0, 2, 3, 11, 12]]  # position, ema_20, ema_50, original features
    
    best_target = None
    best_balance = 0
    
    for price_proxy in price_candidates:
        # Skip if all values are the same
        if np.std(price_proxy) < 1e-8:
            continue
            
        # Calculate future returns
        future_returns = np.zeros_like(price_proxy)
        for i in range(len(price_proxy) - lookahead):
            future_returns[i] = price_proxy[i + lookahead] - price_proxy[i]
        
        # Create target based on return direction
        target_candidate = (future_returns > 0).astype(int)
        
        # Check class balance
        pos_ratio = np.mean(target_candidate)
        balance = min(pos_ratio, 1 - pos_ratio)  # Closer to 0.5 is better
        
        if balance > best_balance and balance > 0.1:  # At least 10% minority class
            best_target = target_candidate
            best_balance = balance
    
    # Fallback: create synthetic balanced target if no good candidate found
    if best_target is None:
        print("Warning: Creating synthetic balanced target variable")
        n_samples = len(features)
        best_target = np.random.binomial(1, 0.5, n_samples)
    else:
        pos_ratio = np.mean(best_target)
        print(f"Target variable: {pos_ratio:.1%} positive class, {1-pos_ratio:.1%} negative class")
    
    return best_target[:-lookahead], features[:-lookahead]  # Remove last lookahead samples

def analyze_feature_correlations(features, feature_names, threshold=0.8):
    """Analyze feature correlations and identify redundant features"""
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(features.T)
    
    # Create DataFrame for better visualization
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_val = abs(corr_matrix[i, j])
            if corr_val > threshold:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_val))
    
    print(f"Highly correlated pairs (|r| > {threshold}):")
    for pair in high_corr_pairs:
        print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
    
    # Suggest features to remove
    features_to_remove = set()
    for pair in high_corr_pairs:
        # Keep the first feature in each highly correlated pair
        features_to_remove.add(pair[1])
    
    print(f"\nSuggested features to remove: {list(features_to_remove)}")
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(current_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved: {plot_path}")
    plt.show()
    
    return high_corr_pairs, features_to_remove, corr_df

def analyze_feature_importance(features, target, feature_names, n_features=10):
    """Analyze feature importance using gradient boosting"""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Use gradient boosting for feature importance
    gb_classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # Fit model
    print("Training gradient boosting classifier...")
    gb_classifier.fit(features, target)
    
    # Get feature importances
    importance_scores = gb_classifier.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {n_features} most important features:")
    print(importance_df.head(n_features).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(n_features)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {n_features} Feature Importances (Gradient Boosting)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(current_dir, 'feature_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved: {plot_path}")
    plt.show()
    
    return importance_df, gb_classifier

def statistical_feature_selection(features, target, feature_names, k=10):
    """Statistical feature selection using univariate tests"""
    print("\n=== STATISTICAL FEATURE SELECTION ===")
    
    # Use SelectKBest with f_regression
    selector = SelectKBest(score_func=f_regression, k=k)
    features_selected = selector.fit_transform(features, target)
    
    # Get selected feature indices and scores
    selected_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': feature_names,
        'score': feature_scores,
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)
    
    print(f"Top {k} features by statistical score:")
    print(results_df[results_df['selected']].to_string(index=False))
    
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"\nSelected feature names: {selected_features}")
    
    return results_df, selected_features, features_selected

def generate_recommendations(corr_pairs, features_to_remove, importance_df, selected_features):
    """Generate actionable recommendations"""
    print("\n=== RECOMMENDATIONS ===")
    
    print("1. CORRELATION-BASED FEATURE REMOVAL:")
    if features_to_remove:
        print(f"   Remove highly correlated features: {list(features_to_remove)}")
        remaining_after_corr = len(importance_df) - len(features_to_remove)
        print(f"   This reduces features from {len(importance_df)} to {remaining_after_corr}")
    else:
        print("   No highly correlated features found - keep all features")
    
    print("\n2. IMPORTANCE-BASED FEATURE SELECTION:")
    top_10_features = importance_df.head(10)['feature'].tolist()
    print(f"   Use top 10 most important features: {top_10_features}")
    
    print("\n3. COMBINED APPROACH:")
    # Remove correlated features from top important features
    important_no_corr = [f for f in top_10_features if f not in features_to_remove]
    print(f"   Top important features without correlation: {important_no_corr}")
    print(f"   Final recommended feature count: {len(important_no_corr)}")
    
    print("\n4. MODEL ARCHITECTURE RECOMMENDATIONS:")
    current_features = len(importance_df)
    if current_features > 10:
        recommended_neurons = max(128, current_features * 8)
        print(f"   Current features: {current_features}")
        print(f"   Recommended hidden layers: ({recommended_neurons}, {recommended_neurons//2}, {recommended_neurons//4})")
        print(f"   vs Current: (128, 128, 128)")
    
    return important_no_corr

def save_results(corr_df, importance_df, selected_features, recommendations):
    """Save analysis results to files"""
    results_dir = current_dir
    
    # Save correlation matrix
    corr_df.to_csv(os.path.join(results_dir, 'correlation_matrix.csv'))
    
    # Save feature importance
    importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
    
    # Save recommendations
    with open(os.path.join(results_dir, 'feature_recommendations.txt'), 'w') as f:
        f.write("FEATURE ANALYSIS RECOMMENDATIONS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Selected features for next training:\n")
        for i, feature in enumerate(recommendations):
            f.write(f"{i+1}. {feature}\n")
        f.write(f"\nTotal recommended features: {len(recommendations)}\n")
        f.write(f"Reduction from original: {len(importance_df)} -> {len(recommendations)}\n")
    
    print(f"\nResults saved to: {results_dir}")

def main():
    """Main analysis pipeline"""
    print("Starting Feature Correlation Analysis...")
    
    try:
        # Load data
        features, feature_names = load_enhanced_features()
        
        # Use subset for faster analysis (last 50K samples for recent data)
        n_samples = min(50000, len(features))
        features_sample = features[-n_samples:]
        print(f"Using sample of {n_samples} samples for analysis")
        
        # Create target variable
        target, features_subset = create_target_variable(features_sample)
        print(f"Created target variable. Data shape: {features_subset.shape}, Target shape: {target.shape}")
        
        # Analyze correlations
        corr_pairs, features_to_remove, corr_df = analyze_feature_correlations(
            features_subset, feature_names, threshold=0.8
        )
        
        # Analyze feature importance
        importance_df, gb_model = analyze_feature_importance(
            features_subset, target, feature_names, n_features=10
        )
        
        # Statistical feature selection
        stats_df, selected_features, features_selected = statistical_feature_selection(
            features_subset, target, feature_names, k=10
        )
        
        # Generate recommendations
        recommendations = generate_recommendations(
            corr_pairs, features_to_remove, importance_df, selected_features
        )
        
        # Save results
        save_results(corr_df, importance_df, selected_features, recommendations)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Check plots and results in: {current_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()