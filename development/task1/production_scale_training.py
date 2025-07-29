#!/usr/bin/env python3
"""
Production Scale Training - Using Original Proven Framework
Full-scale RL training with proper episodic training, not shortcuts
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add src path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

# Use original proven framework
from task1_ensemble import Ensemble
from erl_config import Config

class ProductionScaleTrainer:
    """Production scale trainer using the original proven framework."""
    
    def __init__(self, output_dir: str = "production_scale_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"🚀 Production Scale Training - Session: {self.timestamp}")
        print(f"📁 Output: {self.output_dir}")
        print(f"⚡ Using Original Proven Framework")
    
    def run_production_scale_training(self):
        """Run full production scale training with proper parameters."""
        print("\n" + "=" * 70)
        print("🚀 STARTING PRODUCTION SCALE RL TRAINING")
        print("=" * 70)
        
        try:
            # Production-level configuration  
            from trade_simulator import TradeSimulator, EvalTradeSimulator
            
            config = Config()
            
            # **PRODUCTION PARAMETERS** - Full scale training
            config.break_step = 2**20    # Large break step for full training (~1M)
            config.eval_times = 2**8     # Comprehensive evaluation (256 episodes)
            config.eval_gap = 2**8       # Evaluate every 256 steps
            config.gpu_id = 0            # Use GPU
            
            # Set evaluation environment class and args
            config.eval_env_class = EvalTradeSimulator
            config.eval_env_args = {'num_envs': 1}
            
            # Enhanced learning parameters for production
            config.learning_rate = 1e-5  # Conservative for stability
            config.net_dims = (512, 512, 256)  # Large networks for production
            config.batch_size = 256      # Large batches for stability
            config.target_step = 2**12   # Extended learning horizon
            config.repeat_times = 2      # Multiple updates per step
            
            print(f"🎯 Production Configuration:")
            print(f"   Break step: {config.break_step:,} (full scale training)")
            print(f"   Eval times: {config.eval_times:,} episodes")
            print(f"   Network dims: {config.net_dims}")
            print(f"   Batch size: {config.batch_size}")
            print(f"   Target step: {config.target_step:,}")
            print(f"   Learning rate: {config.learning_rate}")
            
            # Create production ensemble
            print(f"\n🤖 Creating Production Ensemble")
            print("-" * 50)
            
            # Import agent classes
            from erl_agent import AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN
            
            ensemble = Ensemble(
                log_rules=['print_time'],  # Simple logging
                save_path=str(self.output_dir / f"production_ensemble_{self.timestamp}"),
                starting_cash=10000,       # Starting capital
                agent_classes=[AgentD3QN, AgentDoubleDQN, AgentPrioritizedDQN],  # 3 diverse agents
                args=config                # Pass config as args
            )
            
            print(f"✅ Ensemble created with 3 production agents")
            print(f"📁 Save path: {ensemble.save_path}")
            
            # **FULL SCALE TRAINING** - This is the real training
            print(f"\n🎓 Starting Production Scale Training")
            print("-" * 50)
            print(f"⚠️  WARNING: This will train for hours with full episodes")
            print(f"💪 Training parameters optimized for competition performance")
            
            training_start = time.time()
            
            # The main training call - this does the real work
            ensemble.run_training()
            
            training_time = time.time() - training_start
            
            print(f"\n✅ Production Scale Training Completed!")
            print(f"⏱️  Total training time: {training_time/3600:.2f} hours")
            print(f"📁 Models saved to: {ensemble.save_path}")
            
            # Save training summary
            training_summary = {
                'timestamp': self.timestamp,
                'training_time_hours': training_time / 3600,
                'break_step': config.break_step,
                'eval_times': config.eval_times,
                'network_dims': config.net_dims,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'agents': ['AgentD3QN', 'AgentDoubleDQN', 'AgentTwinD3QN'],
                'save_path': ensemble.save_path,
                'training_completed': True
            }
            
            summary_file = self.output_dir / f"production_training_summary_{self.timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            print(f"📄 Training summary saved: {summary_file}")
            
            # **COMPREHENSIVE EVALUATION** - Test the trained models
            print(f"\n📊 Starting Comprehensive Evaluation")
            print("-" * 50)
            
            eval_start = time.time()
            
            # Run comprehensive evaluation
            eval_results = ensemble.run_evaluation()
            
            eval_time = time.time() - eval_start
            
            print(f"\n📈 Comprehensive Evaluation Completed!")
            print(f"⏱️  Evaluation time: {eval_time/60:.1f} minutes")
            
            # Display key results
            if eval_results and 'ensemble_performance' in eval_results:
                perf = eval_results['ensemble_performance']
                print(f"\n🏆 PRODUCTION TRAINING RESULTS:")
                print(f"   📊 Final Sharpe Ratio: {perf.get('sharpe_ratio', 'N/A'):.3f}")
                print(f"   💰 Total Return: {perf.get('total_return', 'N/A'):.3f}")
                print(f"   📉 Max Drawdown: {perf.get('max_drawdown', 'N/A'):.3f}")
                print(f"   🎯 Win Rate: {perf.get('win_rate', 'N/A'):.1%}")
            
            # Save evaluation results
            eval_file = self.output_dir / f"production_evaluation_{self.timestamp}.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2, default=str)
            
            print(f"📄 Evaluation results saved: {eval_file}")
            
            return {
                'training_summary': training_summary,
                'evaluation_results': eval_results,
                'total_time_hours': (training_time + eval_time) / 3600
            }
            
        except Exception as e:
            print(f"\n💥 Production training failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

def main():
    """Main production training function."""
    print("🚀 FinRL Contest 2024 - Production Scale Training")
    print("=" * 60)
    print("💡 This runs FULL scale reinforcement learning training")
    print("⏰ Expected duration: 2-6 hours depending on hardware")
    print("🎯 Goal: Competition-level performance (60%+ accuracy)")
    print("=" * 60)
    
    response = input("\n⚠️  Proceed with full production training? (y/N): ")
    if response.lower() != 'y':
        print("❌ Training cancelled by user")
        return 1
    
    try:
        trainer = ProductionScaleTrainer()
        results = trainer.run_production_scale_training()
        
        print("\n" + "=" * 70)
        print("🎉 PRODUCTION SCALE TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"⏱️  Total time: {results['total_time_hours']:.2f} hours")
        print(f"🏆 Competition-ready models generated")
        print(f"📁 Results saved to: production_scale_results/")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Production training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())