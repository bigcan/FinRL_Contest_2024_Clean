"""
Optimized Hyperparameters for Profitability Improvements
Based on analysis of baseline issues and targeted improvements
"""

from enhanced_training_config import EnhancedConfig


def get_optimized_hyperparameters(reward_type="multi_objective"):
    """
    Get optimized hyperparameters based on profitability analysis
    
    Key improvements over baseline:
    - Learning rate: 2e-6 → 1e-4 (50x increase)
    - Exploration: 0.005 → 0.1 (20x increase) 
    - Architecture: Optimized for 8 features
    - Training steps: 8-16 → 200 (12x increase)
    """
    
    optimized_params = {
        # CRITICAL IMPROVEMENTS for profitability
        "learning_rate": 1e-4,        # 50x higher than baseline (2e-6)
        "initial_exploration": 0.1,    # 20x higher than baseline (0.005)
        "final_exploration": 0.001,    # Decay to reasonable level
        "break_step": 200,             # 12x longer training
        
        # ARCHITECTURE optimized for 8-feature state space
        "net_dims": (128, 64, 32),     # Optimized for 8 features vs (128,128,128)
        "batch_size": 512,             # Good balance of stability/speed
        "buffer_size_multiplier": 8,   # Larger buffer for better experience replay
        
        # REWARD AND RISK SETTINGS
        "gamma": 0.995,                # Slightly higher discount for crypto
        "soft_update_tau": 0.005,      # Good target network update rate
        "clip_grad_norm": 3.0,         # Gradient clipping for stability
        
        # EVALUATION AND EARLY STOPPING
        "eval_per_step": 10,           # Frequent evaluation for early stopping
        "eval_times": 3,               # Multiple evaluation episodes
        "early_stopping_patience": 50, # Stop if no improvement for 50 steps
        "early_stopping_min_delta": 0.001,  # Minimum improvement threshold
        
        # LEARNING RATE SCHEDULING
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine_annealing",
        "lr_min": 1e-7,
        
        # TIME LIMITS
        "max_training_time": 1800,     # 30 minutes max per agent
    }
    
    # Reward-specific adjustments
    if reward_type == "simple":
        # Simple reward needs higher learning for signal detection
        optimized_params["learning_rate"] = 2e-4
        optimized_params["initial_exploration"] = 0.15
    elif reward_type == "transaction_cost_adjusted":
        # Transaction cost reward needs conservative exploration
        optimized_params["initial_exploration"] = 0.05
        optimized_params["final_exploration"] = 0.0005
    elif reward_type == "multi_objective":
        # Multi-objective reward works well with balanced settings
        pass  # Use defaults
    
    print(f"🎯 Optimized Hyperparameters for {reward_type} reward:")
    print(f"   📚 Learning Rate: {optimized_params['learning_rate']:.2e} (vs 2e-6 baseline)")
    print(f"   🔍 Exploration: {optimized_params['initial_exploration']:.3f} → {optimized_params['final_exploration']:.3f}")
    print(f"   📈 Training Steps: {optimized_params['break_step']} (vs 8-16 baseline)")
    print(f"   🧠 Architecture: {optimized_params['net_dims']}")
    print(f"   💾 Batch Size: {optimized_params['batch_size']}")
    
    return optimized_params


def apply_optimized_hyperparameters(config: EnhancedConfig, 
                                   optimized_params: dict,
                                   env_args: dict) -> EnhancedConfig:
    """
    Apply optimized hyperparameters to configuration
    """
    
    # Core training parameters
    config.learning_rate = optimized_params["learning_rate"]
    config.break_step = optimized_params["break_step"]
    config.net_dims = optimized_params["net_dims"]
    config.batch_size = optimized_params["batch_size"]
    config.gamma = optimized_params["gamma"]
    config.soft_update_tau = optimized_params["soft_update_tau"]
    config.clip_grad_norm = optimized_params["clip_grad_norm"]
    
    # Exploration settings
    config.initial_exploration = optimized_params["initial_exploration"]
    config.final_exploration = optimized_params["final_exploration"]
    config.explore_rate = optimized_params["initial_exploration"]
    config.exploration_decay_steps = optimized_params["break_step"]
    
    # Buffer size based on environment
    max_step = env_args.get("max_step", 2370)
    config.buffer_size = max_step * optimized_params["buffer_size_multiplier"]
    config.horizon_len = max_step
    
    # Evaluation settings
    config.eval_per_step = optimized_params["eval_per_step"]
    config.eval_times = optimized_params["eval_times"]
    config.early_stopping_patience = optimized_params["early_stopping_patience"]
    config.early_stopping_min_delta = optimized_params["early_stopping_min_delta"]
    
    # Learning rate scheduling
    config.use_lr_scheduler = optimized_params["use_lr_scheduler"]
    config.lr_scheduler_type = optimized_params["lr_scheduler_type"]
    config.lr_min = optimized_params["lr_min"]
    
    # Time limits
    config.max_training_time = optimized_params["max_training_time"]
    
    print(f"✅ Optimized hyperparameters applied to configuration")
    print(f"   🔧 Buffer size: {config.buffer_size}")
    print(f"   🔧 Horizon length: {config.horizon_len}")
    
    return config


def get_baseline_comparison():
    """Get comparison between baseline and optimized hyperparameters"""
    
    baseline = {
        "learning_rate": 2e-6,
        "exploration_rate": 0.005,
        "training_steps": 16,  # Upper end of 8-16 range
        "net_dims": "(128, 128, 128)",
        "batch_size": 256,
    }
    
    optimized = get_optimized_hyperparameters("multi_objective")
    
    comparison = {
        "learning_rate": {
            "baseline": baseline["learning_rate"],
            "optimized": optimized["learning_rate"],
            "improvement": f"{optimized['learning_rate'] / baseline['learning_rate']:.0f}x increase"
        },
        "exploration_rate": {
            "baseline": baseline["exploration_rate"],
            "optimized": optimized["initial_exploration"],
            "improvement": f"{optimized['initial_exploration'] / baseline['exploration_rate']:.0f}x increase"
        },
        "training_steps": {
            "baseline": baseline["training_steps"],
            "optimized": optimized["break_step"],
            "improvement": f"{optimized['break_step'] / baseline['training_steps']:.0f}x increase"
        },
        "architecture": {
            "baseline": baseline["net_dims"],
            "optimized": str(optimized["net_dims"]),
            "improvement": "Optimized for 8-feature space"
        },
        "batch_size": {
            "baseline": baseline["batch_size"],
            "optimized": optimized["batch_size"],
            "improvement": f"{optimized['batch_size'] / baseline['batch_size']:.1f}x increase"
        }
    }
    
    print(f"\n📊 BASELINE vs OPTIMIZED COMPARISON:")
    print("=" * 60)
    for param, values in comparison.items():
        print(f"{param:15} | {str(values['baseline']):15} → {str(values['optimized']):15} | {values['improvement']}")
    
    return comparison


def validate_hyperparameters():
    """Validate that optimized hyperparameters are reasonable"""
    
    print(f"\n🔍 Hyperparameter Validation:")
    print("-" * 40)
    
    params = get_optimized_hyperparameters("multi_objective")
    
    validations = []
    
    # Learning rate validation
    if 1e-5 <= params["learning_rate"] <= 1e-3:
        validations.append("✅ Learning rate in reasonable range")
    else:
        validations.append("⚠️ Learning rate may be too extreme")
    
    # Exploration validation
    if 0.01 <= params["initial_exploration"] <= 0.3:
        validations.append("✅ Exploration rate in reasonable range")
    else:
        validations.append("⚠️ Exploration rate may be too extreme")
    
    # Training steps validation
    if 50 <= params["break_step"] <= 500:
        validations.append("✅ Training steps in reasonable range")
    else:
        validations.append("⚠️ Training steps may be too extreme")
    
    # Architecture validation
    total_params = sum(params["net_dims"])
    if 100 <= total_params <= 1000:
        validations.append("✅ Network architecture reasonably sized")
    else:
        validations.append("⚠️ Network architecture may be extreme")
    
    for validation in validations:
        print(f"   {validation}")
    
    all_good = all("✅" in v for v in validations)
    
    if all_good:
        print(f"\n🎉 All hyperparameters validated successfully!")
    else:
        print(f"\n⚠️ Some hyperparameters may need adjustment")
    
    return all_good


if __name__ == "__main__":
    print("🎯 Optimized Hyperparameters for Profitability")
    print("=" * 60)
    
    # Test each reward type
    reward_types = ["simple", "transaction_cost_adjusted", "multi_objective"]
    
    for reward_type in reward_types:
        print(f"\n🧪 {reward_type.upper()} REWARD:")
        params = get_optimized_hyperparameters(reward_type)
    
    # Show comparison
    get_baseline_comparison()
    
    # Validate
    validate_hyperparameters()
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Use these optimized parameters in extended training")
    print(f"   2. Run: python3 task1_ensemble_extended.py 0 multi_objective")
    print(f"   3. Compare results with baseline (-0.19% return)")
    print(f"   4. Target: +0.5-2% returns with 55-60% win rate")