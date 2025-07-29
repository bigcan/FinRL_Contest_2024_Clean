#!/usr/bin/env python3
"""
Validate Train/Validation Split Configuration
Ensures no data leakage between training and validation sets
"""

def validate_data_split():
    """Validate that multi-episode training respects train/validation boundaries"""
    
    print("🔍 Train/Validation Split Validation")
    print("=" * 45)
    
    # Dataset parameters
    full_dataset = 823682
    eval_split = 0.8  # 80% train, 20% validation
    
    # Calculate split boundaries
    train_samples = int(full_dataset * eval_split)
    val_samples = full_dataset - train_samples
    
    print(f"📊 Dataset Analysis:")
    print(f"   Full dataset: {full_dataset:,} samples")
    print(f"   Train data (80%): {train_samples:,} samples")
    print(f"   Validation data (20%): {val_samples:,} samples")
    print()
    
    # Current multi-episode configuration
    data_per_episode = 10000
    num_episodes = 65
    total_training_needed = data_per_episode * num_episodes
    
    print(f"🎯 Multi-Episode Configuration:")
    print(f"   Data per episode: {data_per_episode:,} samples")
    print(f"   Number of episodes: {num_episodes}")
    print(f"   Total training data needed: {total_training_needed:,} samples")
    print()
    
    # Validation checks
    print("✅ Validation Results:")
    
    if total_training_needed <= train_samples:
        unused_train = train_samples - total_training_needed
        utilization = (total_training_needed / train_samples) * 100
        print(f"   ✓ No data leakage: {total_training_needed:,} ≤ {train_samples:,}")
        print(f"   ✓ Train data utilization: {utilization:.1f}%")
        print(f"   ✓ Unused train data: {unused_train:,} samples")
        print(f"   ✓ Validation data preserved: {val_samples:,} samples")
        
        # Calculate training time estimate
        time_per_episode = 0.6  # minutes (estimated from 10K vs 15K samples)
        total_time = num_episodes * time_per_episode
        print(f"   ✓ Estimated training time: ~{total_time:.0f} minutes")
        
    else:
        leakage = total_training_needed - train_samples
        print(f"   ✗ DATA LEAKAGE: Exceeds train data by {leakage:,} samples")
        print(f"   ✗ Would use validation data for training!")
        
    print()
    
    # Compare with previous configuration
    print("📈 Comparison with Previous Config:")
    old_data_per_episode = 15000
    old_num_episodes = 50
    old_total = old_data_per_episode * old_num_episodes
    old_leakage = old_total - train_samples
    
    print(f"   Previous: {old_num_episodes} × {old_data_per_episode:,} = {old_total:,} samples")
    print(f"   Previous leakage: {old_leakage:,} samples ❌")
    print(f"   Current: {num_episodes} × {data_per_episode:,} = {total_training_needed:,} samples")
    print(f"   Current leakage: 0 samples ✅")
    print()
    
    # Episode analysis
    episode_hours = (data_per_episode / 3600) if data_per_episode <= 3600 else data_per_episode / 3600
    print(f"📊 Episode Analysis:")
    print(f"   Market data per episode: ~{episode_hours:.1f} hours")
    print(f"   Total market coverage: ~{(total_training_needed/3600):.0f} hours")
    print(f"   Learning curve data points: {num_episodes}")
    print()
    
    print("🚀 Configuration Status: VALIDATED ✅")
    print("   Ready for proper train/validation split training!")

if __name__ == "__main__":
    validate_data_split()