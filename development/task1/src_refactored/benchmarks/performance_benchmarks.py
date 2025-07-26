"""
Performance benchmarking suite for the FinRL Contest 2024 refactored framework.

This module provides comprehensive performance analysis including timing,
memory usage, and scalability testing for all framework components.
"""

import time
import torch
import numpy as np
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import framework components for benchmarking
from ..agents import create_agent, create_ensemble_agents, AGENT_REGISTRY
from ..ensemble import create_voting_ensemble, create_stacking_ensemble, EnsembleStrategy
from ..tests.utils.mock_environment import MockEnvironment
from ..tests.utils.test_helpers import create_test_batch, set_random_seeds


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: float = 0.0
    throughput: float = 0.0  # Operations per second
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'gpu_memory_mb': self.gpu_memory_mb,
            'throughput': self.throughput,
            'additional_metrics': self.additional_metrics
        }


class PerformanceProfiler:
    """Context manager for performance profiling."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.start_gpu_memory = None
        
    def __enter__(self):
        # Clear caches and collect garbage
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Record starting metrics
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = psutil.cpu_percent()
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            self.start_gpu_memory = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            self.start_gpu_memory = 0.0
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Synchronize GPU operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def get_metrics(self) -> Tuple[float, float, float, float]:
        """Get performance metrics."""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            end_gpu_memory = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            end_gpu_memory = 0.0
        
        execution_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        cpu_usage = (self.start_cpu + end_cpu) / 2  # Average
        gpu_memory_delta = end_gpu_memory - self.start_gpu_memory
        
        return execution_time, memory_delta, cpu_usage, gpu_memory_delta


class AgentBenchmark:
    """Benchmark suite for agent performance."""
    
    def __init__(self, device: Optional[torch.device] = None, seed: int = 42):
        self.device = device or torch.device('cpu')
        self.seed = seed
        set_random_seeds(seed)
        
        # Test parameters
        self.state_dim = 100
        self.action_dim = 3
        self.batch_size = 64
        self.num_iterations = 100
        
    def benchmark_agent_creation(self, agent_type: str) -> BenchmarkResult:
        """Benchmark agent creation time and memory."""
        with PerformanceProfiler(self.device) as profiler:
            agents = []
            for _ in range(10):  # Create 10 agents
                agent = create_agent(
                    agent_type=agent_type,
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    device=self.device
                )
                agents.append(agent)
        
        exec_time, memory_usage, cpu_usage, gpu_memory = profiler.get_metrics()
        
        return BenchmarkResult(
            test_name=f"{agent_type}_creation",
            execution_time=exec_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            throughput=10 / exec_time if exec_time > 0 else 0,
            additional_metrics={'agents_created': 10}
        )
    
    def benchmark_action_selection(self, agent_type: str) -> BenchmarkResult:
        """Benchmark action selection performance."""
        agent = create_agent(
            agent_type=agent_type,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        state = torch.randn(self.state_dim, device=self.device)
        
        with PerformanceProfiler(self.device) as profiler:
            actions = []
            for _ in range(self.num_iterations):
                action = agent.select_action(state, deterministic=True)
                actions.append(action)
        
        exec_time, memory_usage, cpu_usage, gpu_memory = profiler.get_metrics()
        
        return BenchmarkResult(
            test_name=f"{agent_type}_action_selection",
            execution_time=exec_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            throughput=self.num_iterations / exec_time if exec_time > 0 else 0,
            additional_metrics={'actions_per_second': self.num_iterations / exec_time if exec_time > 0 else 0}
        )
    
    def benchmark_batch_action_selection(self, agent_type: str) -> BenchmarkResult:
        """Benchmark batch action selection performance."""
        agent = create_agent(
            agent_type=agent_type,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        states = torch.randn(self.batch_size, self.state_dim, device=self.device)
        
        with PerformanceProfiler(self.device) as profiler:
            for _ in range(self.num_iterations):
                actions = agent.select_action(states, deterministic=True)
        
        exec_time, memory_usage, cpu_usage, gpu_memory = profiler.get_metrics()
        total_actions = self.num_iterations * self.batch_size
        
        return BenchmarkResult(
            test_name=f"{agent_type}_batch_action_selection",
            execution_time=exec_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            throughput=total_actions / exec_time if exec_time > 0 else 0,
            additional_metrics={
                'batch_size': self.batch_size,
                'total_actions': total_actions
            }
        )
    
    def benchmark_agent_update(self, agent_type: str) -> BenchmarkResult:
        """Benchmark agent update performance."""
        agent = create_agent(
            agent_type=agent_type,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        batch_data = create_test_batch(self.state_dim, self.action_dim, self.batch_size)
        # Move to device
        batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
        
        with PerformanceProfiler(self.device) as profiler:
            for _ in range(self.num_iterations):
                result = agent.update(batch_data)
        
        exec_time, memory_usage, cpu_usage, gpu_memory = profiler.get_metrics()
        
        return BenchmarkResult(
            test_name=f"{agent_type}_update",
            execution_time=exec_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            throughput=self.num_iterations / exec_time if exec_time > 0 else 0,
            additional_metrics={
                'updates_per_second': self.num_iterations / exec_time if exec_time > 0 else 0,
                'batch_size': self.batch_size
            }
        )
    
    def benchmark_all_agents(self) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark all available agent types."""
        results = {}
        
        # Test subset of agents for performance (to avoid long execution)
        agents_to_test = ['AgentDoubleDQN', 'AgentD3QN', 'AgentPrioritizedDQN', 'AgentRainbowDQN']
        
        for agent_type in agents_to_test:
            if agent_type in AGENT_REGISTRY:
                print(f"Benchmarking {agent_type}...")
                agent_results = []
                
                try:
                    # Creation benchmark
                    creation_result = self.benchmark_agent_creation(agent_type)
                    agent_results.append(creation_result)
                    
                    # Action selection benchmark
                    action_result = self.benchmark_action_selection(agent_type)
                    agent_results.append(action_result)
                    
                    # Batch action selection benchmark
                    batch_action_result = self.benchmark_batch_action_selection(agent_type)
                    agent_results.append(batch_action_result)
                    
                    # Update benchmark
                    update_result = self.benchmark_agent_update(agent_type)
                    agent_results.append(update_result)
                    
                    results[agent_type] = agent_results
                    print(f"  ✅ {agent_type} benchmarked successfully")
                    
                except Exception as e:
                    print(f"  ❌ {agent_type} benchmark failed: {e}")
                    results[agent_type] = []
        
        return results


class EnsembleBenchmark:
    """Benchmark suite for ensemble performance."""
    
    def __init__(self, device: Optional[torch.device] = None, seed: int = 42):
        self.device = device or torch.device('cpu')
        self.seed = seed
        set_random_seeds(seed)
        
        # Test parameters
        self.state_dim = 100
        self.action_dim = 3
        self.num_iterations = 50
        
    def benchmark_ensemble_creation(self, ensemble_type: str, num_agents: int = 3) -> BenchmarkResult:
        """Benchmark ensemble creation performance."""
        # Create agents for ensemble
        agent_configs = {
            f"agent_{i}": {"agent_type": "AgentDoubleDQN"}
            for i in range(num_agents)
        }
        
        with PerformanceProfiler(self.device) as profiler:
            agents = create_ensemble_agents(
                agent_configs,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                device=self.device
            )
            
            if ensemble_type == "voting":
                ensemble = create_voting_ensemble(
                    agents=agents,
                    strategy=EnsembleStrategy.MAJORITY_VOTE,
                    device=self.device
                )
            elif ensemble_type == "stacking":
                ensemble = create_stacking_ensemble(
                    agents=agents,
                    action_dim=self.action_dim,
                    device=self.device
                )
        
        exec_time, memory_usage, cpu_usage, gpu_memory = profiler.get_metrics()
        
        return BenchmarkResult(
            test_name=f"{ensemble_type}_ensemble_creation",
            execution_time=exec_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            additional_metrics={'num_agents': num_agents}
        )
    
    def benchmark_ensemble_action_selection(self, ensemble_type: str, num_agents: int = 3) -> BenchmarkResult:
        """Benchmark ensemble action selection performance."""
        # Create ensemble
        agent_configs = {
            f"agent_{i}": {"agent_type": "AgentDoubleDQN"}
            for i in range(num_agents)
        }
        
        agents = create_ensemble_agents(
            agent_configs,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        if ensemble_type == "voting":
            ensemble = create_voting_ensemble(
                agents=agents,
                strategy=EnsembleStrategy.MAJORITY_VOTE,
                device=self.device
            )
        elif ensemble_type == "stacking":
            ensemble = create_stacking_ensemble(
                agents=agents,
                action_dim=self.action_dim,
                device=self.device
            )
        
        state = torch.randn(self.state_dim, device=self.device)
        
        with PerformanceProfiler(self.device) as profiler:
            for _ in range(self.num_iterations):
                action = ensemble.select_action(state, deterministic=True)
        
        exec_time, memory_usage, cpu_usage, gpu_memory = profiler.get_metrics()
        
        return BenchmarkResult(
            test_name=f"{ensemble_type}_ensemble_action_selection",
            execution_time=exec_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            throughput=self.num_iterations / exec_time if exec_time > 0 else 0,
            additional_metrics={
                'num_agents': num_agents,
                'actions_per_second': self.num_iterations / exec_time if exec_time > 0 else 0
            }
        )
    
    def benchmark_ensemble_update(self, ensemble_type: str, num_agents: int = 3) -> BenchmarkResult:
        """Benchmark ensemble update performance."""
        # Create ensemble
        agent_configs = {
            f"agent_{i}": {"agent_type": "AgentDoubleDQN"}
            for i in range(num_agents)
        }
        
        agents = create_ensemble_agents(
            agent_configs,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        if ensemble_type == "voting":
            ensemble = create_voting_ensemble(
                agents=agents,
                strategy=EnsembleStrategy.MAJORITY_VOTE,
                device=self.device
            )
        elif ensemble_type == "stacking":
            ensemble = create_stacking_ensemble(
                agents=agents,
                action_dim=self.action_dim,
                device=self.device
            )
        
        batch_data = create_test_batch(self.state_dim, self.action_dim, 32)
        batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
        
        with PerformanceProfiler(self.device) as profiler:
            for _ in range(self.num_iterations):
                result = ensemble.update(batch_data)
        
        exec_time, memory_usage, cpu_usage, gpu_memory = profiler.get_metrics()
        
        return BenchmarkResult(
            test_name=f"{ensemble_type}_ensemble_update",
            execution_time=exec_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            throughput=self.num_iterations / exec_time if exec_time > 0 else 0,
            additional_metrics={
                'num_agents': num_agents,
                'updates_per_second': self.num_iterations / exec_time if exec_time > 0 else 0
            }
        )


class BenchmarkSuite:
    """Complete benchmarking suite for the framework."""
    
    def __init__(self, device: Optional[torch.device] = None, seed: int = 42):
        self.device = device or torch.device('cpu')
        self.seed = seed
        
        self.agent_benchmark = AgentBenchmark(device, seed)
        self.ensemble_benchmark = EnsembleBenchmark(device, seed)
        
        self.results = {}
    
    def run_full_benchmark(self, save_results: bool = True) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 Starting Full Performance Benchmark Suite")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Seed: {self.seed}")
        print()
        
        start_time = time.time()
        
        # Agent benchmarks
        print("📊 Running Agent Benchmarks...")
        agent_results = self.agent_benchmark.benchmark_all_agents()
        self.results['agents'] = agent_results
        
        # Ensemble benchmarks
        print("\n📊 Running Ensemble Benchmarks...")
        ensemble_results = {}
        
        for ensemble_type in ['voting', 'stacking']:
            print(f"  Testing {ensemble_type} ensemble...")
            ensemble_type_results = []
            
            try:
                # Creation benchmark
                creation_result = self.ensemble_benchmark.benchmark_ensemble_creation(ensemble_type)
                ensemble_type_results.append(creation_result)
                
                # Action selection benchmark
                action_result = self.ensemble_benchmark.benchmark_ensemble_action_selection(ensemble_type)
                ensemble_type_results.append(action_result)
                
                # Update benchmark
                update_result = self.ensemble_benchmark.benchmark_ensemble_update(ensemble_type)
                ensemble_type_results.append(update_result)
                
                ensemble_results[ensemble_type] = ensemble_type_results
                print(f"    ✅ {ensemble_type} ensemble benchmarked")
                
            except Exception as e:
                print(f"    ❌ {ensemble_type} ensemble failed: {e}")
                ensemble_results[ensemble_type] = []
        
        self.results['ensembles'] = ensemble_results
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(total_time)
        self.results['summary'] = summary
        
        # Save results
        if save_results:
            self._save_results()
        
        # Print summary
        self._print_summary(summary)
        
        return self.results
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate benchmark summary statistics."""
        summary = {
            'total_benchmark_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'agent_performance': {},
            'ensemble_performance': {},
            'recommendations': []
        }
        
        # Agent performance summary
        if 'agents' in self.results:
            for agent_type, results in self.results['agents'].items():
                if results:
                    agent_summary = {
                        'creation_time': next((r.execution_time for r in results if 'creation' in r.test_name), 0),
                        'action_selection_throughput': next((r.throughput for r in results if 'action_selection' in r.test_name and 'batch' not in r.test_name), 0),
                        'update_throughput': next((r.throughput for r in results if 'update' in r.test_name), 0),
                        'avg_memory_usage': sum(r.memory_usage_mb for r in results) / len(results)
                    }
                    summary['agent_performance'][agent_type] = agent_summary
        
        # Ensemble performance summary
        if 'ensembles' in self.results:
            for ensemble_type, results in self.results['ensembles'].items():
                if results:
                    ensemble_summary = {
                        'creation_time': next((r.execution_time for r in results if 'creation' in r.test_name), 0),
                        'action_selection_throughput': next((r.throughput for r in results if 'action_selection' in r.test_name), 0),
                        'update_throughput': next((r.throughput for r in results if 'update' in r.test_name), 0),
                        'avg_memory_usage': sum(r.memory_usage_mb for r in results) / len(results)
                    }
                    summary['ensemble_performance'][ensemble_type] = ensemble_summary
        
        # Generate recommendations
        recommendations = []
        
        # Performance recommendations
        if summary['agent_performance']:
            fastest_agent = max(summary['agent_performance'].items(), 
                              key=lambda x: x[1]['action_selection_throughput'])
            recommendations.append(f"Fastest agent for action selection: {fastest_agent[0]} ({fastest_agent[1]['action_selection_throughput']:.1f} actions/sec)")
        
        if summary['ensemble_performance']:
            fastest_ensemble = max(summary['ensemble_performance'].items(),
                                 key=lambda x: x[1]['action_selection_throughput'])
            recommendations.append(f"Fastest ensemble for action selection: {fastest_ensemble[0]} ({fastest_ensemble[1]['action_selection_throughput']:.1f} actions/sec)")
        
        # Memory recommendations
        if total_time > 60:  # If benchmark took more than 1 minute
            recommendations.append("Consider running benchmarks on GPU for better performance")
        
        summary['recommendations'] = recommendations
        
        return summary
    
    def _save_results(self):
        """Save benchmark results to file."""
        # Convert results to serializable format
        serializable_results = {}
        
        for category, category_results in self.results.items():
            if category == 'summary':
                serializable_results[category] = category_results
            else:
                serializable_results[category] = {}
                for sub_category, sub_results in category_results.items():
                    if isinstance(sub_results, list):
                        serializable_results[category][sub_category] = [
                            r.to_dict() for r in sub_results
                        ]
                    else:
                        serializable_results[category][sub_category] = sub_results
        
        # Save to file
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 Benchmark results saved to {filename}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"Total benchmark time: {summary['total_benchmark_time']:.2f} seconds")
        print(f"Device: {summary['device']}")
        print(f"Timestamp: {summary['timestamp']}")
        
        # Agent performance
        if summary['agent_performance']:
            print(f"\n📊 Agent Performance Summary:")
            print("-" * 40)
            for agent_type, perf in summary['agent_performance'].items():
                print(f"{agent_type:20} | "
                      f"Actions/sec: {perf['action_selection_throughput']:6.1f} | "
                      f"Updates/sec: {perf['update_throughput']:5.1f} | "
                      f"Memory: {perf['avg_memory_usage']:5.1f}MB")
        
        # Ensemble performance
        if summary['ensemble_performance']:
            print(f"\n🤝 Ensemble Performance Summary:")
            print("-" * 40)
            for ensemble_type, perf in summary['ensemble_performance'].items():
                print(f"{ensemble_type:20} | "
                      f"Actions/sec: {perf['action_selection_throughput']:6.1f} | "
                      f"Updates/sec: {perf['update_throughput']:5.1f} | "
                      f"Memory: {perf['avg_memory_usage']:5.1f}MB")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("=" * 60)


def run_full_benchmark(device: Optional[torch.device] = None, 
                      save_results: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run the full benchmark suite.
    
    Args:
        device: Computing device for benchmarks
        save_results: Whether to save results to file
        
    Returns:
        Complete benchmark results
    """
    suite = BenchmarkSuite(device=device)
    return suite.run_full_benchmark(save_results=save_results)


if __name__ == '__main__':
    # Run benchmark suite
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = run_full_benchmark(device=device)