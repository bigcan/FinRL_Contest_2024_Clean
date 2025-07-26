"""
Scalability testing suite for the FinRL Contest 2024 refactored framework.

This module tests framework performance across different scales of operation,
including varying numbers of agents, state dimensions, and training episodes.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import concurrent.futures
from threading import Lock
import gc

# Import framework components
from ..agents import create_agent, create_ensemble_agents, AGENT_REGISTRY
from ..ensemble import create_voting_ensemble, create_stacking_ensemble, EnsembleStrategy
from ..tests.utils.mock_environment import MockEnvironment
from ..tests.utils.test_helpers import create_test_batch, set_random_seeds


@dataclass
class ScalabilityTestResult:
    """Container for scalability test results."""
    test_name: str
    scale_parameter: str
    scale_values: List[Any]
    execution_times: List[float]
    throughputs: List[float]
    memory_usage: List[float]
    success_rates: List[float]
    error_messages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'scale_parameter': self.scale_parameter,
            'scale_values': self.scale_values,
            'execution_times': self.execution_times,
            'throughputs': self.throughputs,
            'memory_usage': self.memory_usage,
            'success_rates': self.success_rates,
            'error_messages': self.error_messages
        }


class ScalabilityTester:
    """Comprehensive scalability testing framework."""
    
    def __init__(self, device: Optional[torch.device] = None, seed: int = 42):
        self.device = device or torch.device('cpu')
        self.seed = seed
        set_random_seeds(seed)
        
        self.results: List[ScalabilityTestResult] = []
        self._lock = Lock()
        
        # Base test parameters
        self.base_state_dim = 50
        self.base_action_dim = 3
        self.base_episodes = 10
        self.base_agents = 2
        
    def test_agent_state_dimension_scaling(self, agent_type: str = "AgentDoubleDQN") -> ScalabilityTestResult:
        """Test agent performance scaling with state dimension."""
        test_name = f"{agent_type}_state_dimension_scaling"
        state_dims = [10, 25, 50, 100, 200, 500]
        
        print(f"🔍 Testing {agent_type} state dimension scaling...")
        
        execution_times = []
        throughputs = []
        memory_usage = []
        success_rates = []
        error_messages = []
        
        for state_dim in state_dims:
            print(f"  Testing state_dim={state_dim}...")
            
            try:
                # Create agent
                agent = create_agent(
                    agent_type=agent_type,
                    state_dim=state_dim,
                    action_dim=self.base_action_dim,
                    device=self.device
                )
                
                # Prepare test data
                states = torch.randn(100, state_dim, device=self.device)
                batch_data = create_test_batch(state_dim, self.base_action_dim, 32)
                batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                
                # Memory tracking
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
                
                # Performance test
                start_time = time.time()
                
                successful_operations = 0
                total_operations = 100  # 50 actions + 50 updates
                
                # Action selection test
                for i in range(50):
                    try:
                        action = agent.select_action(states[i], deterministic=True)
                        successful_operations += 1
                    except Exception:
                        pass
                
                # Update test
                for i in range(50):
                    try:
                        result = agent.update(batch_data)
                        successful_operations += 1
                    except Exception:
                        pass
                
                end_time = time.time()
                
                # Memory measurement
                end_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
                memory_mb = (end_memory - start_memory) / 1024 / 1024
                
                # Calculate metrics
                execution_time = end_time - start_time
                throughput = total_operations / execution_time if execution_time > 0 else 0
                success_rate = successful_operations / total_operations
                
                execution_times.append(execution_time)
                throughputs.append(throughput)
                memory_usage.append(memory_mb)
                success_rates.append(success_rate)
                error_messages.append("")
                
                print(f"    ✅ Success rate: {success_rate:.2%}, Throughput: {throughput:.1f} ops/sec")
                
                # Cleanup
                del agent
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                execution_times.append(float('inf'))
                throughputs.append(0.0)
                memory_usage.append(0.0)
                success_rates.append(0.0)
                error_messages.append(str(e))
        
        result = ScalabilityTestResult(
            test_name=test_name,
            scale_parameter="state_dimension",
            scale_values=state_dims,
            execution_times=execution_times,
            throughputs=throughputs,
            memory_usage=memory_usage,
            success_rates=success_rates,
            error_messages=error_messages
        )
        
        self.results.append(result)
        return result
    
    def test_ensemble_agent_count_scaling(self, ensemble_type: str = "voting") -> ScalabilityTestResult:
        """Test ensemble performance scaling with number of agents."""
        test_name = f"{ensemble_type}_ensemble_agent_count_scaling"
        agent_counts = [1, 2, 3, 5, 8, 10]
        
        print(f"🤝 Testing {ensemble_type} ensemble agent count scaling...")
        
        execution_times = []
        throughputs = []
        memory_usage = []
        success_rates = []
        error_messages = []
        
        for num_agents in agent_counts:
            print(f"  Testing num_agents={num_agents}...")
            
            try:
                # Create agents
                agent_configs = {
                    f"agent_{i}": {"agent_type": "AgentDoubleDQN"}
                    for i in range(num_agents)
                }
                
                agents = create_ensemble_agents(
                    agent_configs,
                    state_dim=self.base_state_dim,
                    action_dim=self.base_action_dim,
                    device=self.device
                )
                
                # Create ensemble
                if ensemble_type == "voting":
                    ensemble = create_voting_ensemble(
                        agents=agents,
                        strategy=EnsembleStrategy.MAJORITY_VOTE,
                        device=self.device
                    )
                elif ensemble_type == "stacking":
                    ensemble = create_stacking_ensemble(
                        agents=agents,
                        action_dim=self.base_action_dim,
                        device=self.device
                    )
                
                # Prepare test data
                states = torch.randn(50, self.base_state_dim, device=self.device)
                batch_data = create_test_batch(self.base_state_dim, self.base_action_dim, 16)
                batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                
                # Memory tracking
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
                
                # Performance test
                start_time = time.time()
                
                successful_operations = 0
                total_operations = 75  # 50 actions + 25 updates
                
                # Action selection test
                for i in range(50):
                    try:
                        action = ensemble.select_action(states[i], deterministic=True)
                        successful_operations += 1
                    except Exception:
                        pass
                
                # Update test
                for i in range(25):
                    try:
                        result = ensemble.update(batch_data)
                        successful_operations += 1
                    except Exception:
                        pass
                
                end_time = time.time()
                
                # Memory measurement
                end_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
                memory_mb = (end_memory - start_memory) / 1024 / 1024
                
                # Calculate metrics
                execution_time = end_time - start_time
                throughput = total_operations / execution_time if execution_time > 0 else 0
                success_rate = successful_operations / total_operations
                
                execution_times.append(execution_time)
                throughputs.append(throughput)
                memory_usage.append(memory_mb)
                success_rates.append(success_rate)
                error_messages.append("")
                
                print(f"    ✅ Success rate: {success_rate:.2%}, Throughput: {throughput:.1f} ops/sec")
                
                # Cleanup
                del ensemble
                del agents
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                execution_times.append(float('inf'))
                throughputs.append(0.0)
                memory_usage.append(0.0)
                success_rates.append(0.0)
                error_messages.append(str(e))
        
        result = ScalabilityTestResult(
            test_name=test_name,
            scale_parameter="agent_count",
            scale_values=agent_counts,
            execution_times=execution_times,
            throughputs=throughputs,
            memory_usage=memory_usage,
            success_rates=success_rates,
            error_messages=error_messages
        )
        
        self.results.append(result)
        return result
    
    def test_batch_size_scaling(self, agent_type: str = "AgentDoubleDQN") -> ScalabilityTestResult:
        """Test agent performance scaling with batch size."""
        test_name = f"{agent_type}_batch_size_scaling"
        batch_sizes = [8, 16, 32, 64, 128, 256]
        
        print(f"📦 Testing {agent_type} batch size scaling...")
        
        execution_times = []
        throughputs = []
        memory_usage = []
        success_rates = []
        error_messages = []
        
        for batch_size in batch_sizes:
            print(f"  Testing batch_size={batch_size}...")
            
            try:
                # Create agent
                agent = create_agent(
                    agent_type=agent_type,
                    state_dim=self.base_state_dim,
                    action_dim=self.base_action_dim,
                    device=self.device
                )
                
                # Prepare test data
                batch_data = create_test_batch(self.base_state_dim, self.base_action_dim, batch_size)
                batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                
                # Memory tracking
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
                
                # Performance test
                start_time = time.time()
                
                successful_operations = 0
                total_operations = 50
                
                # Update test with different batch sizes
                for i in range(50):
                    try:
                        result = agent.update(batch_data)
                        successful_operations += 1
                    except Exception:
                        pass
                
                end_time = time.time()
                
                # Memory measurement
                end_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
                memory_mb = (end_memory - start_memory) / 1024 / 1024
                
                # Calculate metrics (throughput in samples/sec)
                execution_time = end_time - start_time
                throughput = (successful_operations * batch_size) / execution_time if execution_time > 0 else 0
                success_rate = successful_operations / total_operations
                
                execution_times.append(execution_time)
                throughputs.append(throughput)
                memory_usage.append(memory_mb)
                success_rates.append(success_rate)
                error_messages.append("")
                
                print(f"    ✅ Success rate: {success_rate:.2%}, Throughput: {throughput:.1f} samples/sec")
                
                # Cleanup
                del agent
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                execution_times.append(float('inf'))
                throughputs.append(0.0)
                memory_usage.append(0.0)
                success_rates.append(0.0)
                error_messages.append(str(e))
        
        result = ScalabilityTestResult(
            test_name=test_name,
            scale_parameter="batch_size",
            scale_values=batch_sizes,
            execution_times=execution_times,
            throughputs=throughputs,
            memory_usage=memory_usage,
            success_rates=success_rates,
            error_messages=error_messages
        )
        
        self.results.append(result)
        return result
    
    def test_concurrent_agent_operations(self, agent_type: str = "AgentDoubleDQN") -> ScalabilityTestResult:
        """Test agent performance under concurrent operations."""
        test_name = f"{agent_type}_concurrent_operations"
        thread_counts = [1, 2, 4, 8]
        
        print(f"🔄 Testing {agent_type} concurrent operations...")
        
        execution_times = []
        throughputs = []
        memory_usage = []
        success_rates = []
        error_messages = []
        
        for num_threads in thread_counts:
            print(f"  Testing num_threads={num_threads}...")
            
            try:
                # Create multiple agents for concurrent testing
                agents = []
                for i in range(num_threads):
                    agent = create_agent(
                        agent_type=agent_type,
                        state_dim=self.base_state_dim,
                        action_dim=self.base_action_dim,
                        device=self.device
                    )
                    agents.append(agent)
                
                # Prepare test data
                states = [torch.randn(self.base_state_dim, device=self.device) for _ in range(num_threads)]
                batch_data = create_test_batch(self.base_state_dim, self.base_action_dim, 16)
                batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                
                # Memory tracking
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
                
                # Concurrent operations test
                def worker_function(agent_idx):
                    """Worker function for concurrent testing."""
                    agent = agents[agent_idx]
                    state = states[agent_idx]
                    successful_ops = 0
                    
                    try:
                        # Perform multiple operations
                        for _ in range(25):
                            # Action selection
                            action = agent.select_action(state, deterministic=True)
                            successful_ops += 1
                            
                            # Update
                            result = agent.update(batch_data)
                            successful_ops += 1
                    except Exception:
                        pass
                    
                    return successful_ops
                
                # Run concurrent operations
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    future_to_idx = {
                        executor.submit(worker_function, i): i 
                        for i in range(num_threads)
                    }
                    
                    total_successful_ops = 0
                    for future in concurrent.futures.as_completed(future_to_idx):
                        total_successful_ops += future.result()
                
                end_time = time.time()
                
                # Memory measurement
                end_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
                memory_mb = (end_memory - start_memory) / 1024 / 1024
                
                # Calculate metrics
                execution_time = end_time - start_time
                total_operations = num_threads * 50  # 25 actions + 25 updates per thread
                throughput = total_successful_ops / execution_time if execution_time > 0 else 0
                success_rate = total_successful_ops / total_operations
                
                execution_times.append(execution_time)
                throughputs.append(throughput)
                memory_usage.append(memory_mb)
                success_rates.append(success_rate)
                error_messages.append("")
                
                print(f"    ✅ Success rate: {success_rate:.2%}, Throughput: {throughput:.1f} ops/sec")
                
                # Cleanup
                del agents
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                execution_times.append(float('inf'))
                throughputs.append(0.0)
                memory_usage.append(0.0)
                success_rates.append(0.0)
                error_messages.append(str(e))
        
        result = ScalabilityTestResult(
            test_name=test_name,
            scale_parameter="thread_count",
            scale_values=thread_counts,
            execution_times=execution_times,
            throughputs=throughputs,
            memory_usage=memory_usage,
            success_rates=success_rates,
            error_messages=error_messages
        )
        
        self.results.append(result)
        return result
    
    def run_all_scalability_tests(self) -> Dict[str, ScalabilityTestResult]:
        """Run all scalability tests."""
        print("🚀 Running Comprehensive Scalability Tests")
        print("=" * 60)
        
        results = {}
        
        # Test agent state dimension scaling
        results['state_dimension'] = self.test_agent_state_dimension_scaling()
        
        # Test ensemble agent count scaling
        results['ensemble_agents'] = self.test_ensemble_agent_count_scaling()
        
        # Test batch size scaling
        results['batch_size'] = self.test_batch_size_scaling()
        
        # Test concurrent operations
        results['concurrent_ops'] = self.test_concurrent_agent_operations()
        
        return results
    
    def generate_scalability_report(self) -> Dict[str, Any]:
        """Generate comprehensive scalability analysis report."""
        if not self.results:
            return {'error': 'No scalability test results available'}
        
        # Analyze scaling patterns
        scaling_analysis = {}
        
        for result in self.results:
            # Calculate scaling efficiency
            scale_values = np.array(result.scale_values)
            throughputs = np.array(result.throughputs)
            
            # Filter out failed tests (inf execution times)
            valid_indices = np.isfinite(result.execution_times)
            if not any(valid_indices):
                continue
                
            valid_scales = scale_values[valid_indices]
            valid_throughputs = throughputs[valid_indices]
            
            # Calculate scaling coefficient (how throughput changes with scale)
            if len(valid_scales) >= 2:
                # Linear regression to find scaling trend
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_scales, valid_throughputs)
                
                scaling_analysis[result.test_name] = {
                    'scaling_slope': slope,
                    'correlation': r_value,
                    'p_value': p_value,
                    'max_scale_tested': max(valid_scales),
                    'throughput_at_max_scale': valid_throughputs[np.argmax(valid_scales)],
                    'scaling_efficiency': 'linear' if r_value > 0.8 else 'sublinear' if r_value > 0.5 else 'poor'
                }
        
        # Generate recommendations
        recommendations = []
        
        for test_name, analysis in scaling_analysis.items():
            if analysis['scaling_efficiency'] == 'poor':
                recommendations.append(f"⚠️ {test_name} shows poor scaling (correlation: {analysis['correlation']:.2f})")
            elif analysis['scaling_efficiency'] == 'linear':
                recommendations.append(f"✅ {test_name} shows good linear scaling")
        
        # Performance bottlenecks
        bottlenecks = []
        for result in self.results:
            failed_tests = sum(1 for rate in result.success_rates if rate < 0.9)
            if failed_tests > 0:
                bottlenecks.append(f"{result.test_name}: {failed_tests} scale points with <90% success rate")
        
        if bottlenecks:
            recommendations.extend([f"🔧 Performance bottlenecks detected:"] + bottlenecks)
        
        # Memory usage analysis
        memory_concerns = []
        for result in self.results:
            max_memory = max(result.memory_usage) if result.memory_usage else 0
            if max_memory > 100:  # More than 100MB
                memory_concerns.append(f"{result.test_name}: Peak memory usage {max_memory:.1f}MB")
        
        if memory_concerns:
            recommendations.extend([f"🧠 High memory usage detected:"] + memory_concerns)
        
        report = {
            'summary': {
                'total_tests': len(self.results),
                'scaling_analysis': scaling_analysis,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device)
            },
            'recommendations': recommendations,
            'detailed_results': [r.to_dict() for r in self.results]
        }
        
        return report
    
    def save_report(self, filename: Optional[str] = None):
        """Save scalability report to file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"scalability_report_{timestamp}.json"
        
        report = self.generate_scalability_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Scalability report saved to {filename}")
    
    def plot_scaling_results(self, save_plots: bool = True):
        """Generate and save scaling visualization plots."""
        if not self.results:
            print("No results to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Framework Scalability Analysis', fontsize=16)
        
        for i, result in enumerate(self.results[:4]):  # Plot first 4 results
            if i >= 4:
                break
                
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Filter valid data points
            valid_indices = np.isfinite(result.execution_times)
            if not any(valid_indices):
                ax.text(0.5, 0.5, f'{result.test_name}\nNo valid data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            scale_values = np.array(result.scale_values)[valid_indices]
            throughputs = np.array(result.throughputs)[valid_indices]
            
            # Plot throughput vs scale
            ax.plot(scale_values, throughputs, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel(result.scale_parameter.replace('_', ' ').title())
            ax.set_ylabel('Throughput (ops/sec)')
            ax.set_title(result.test_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            plot_filename = f"scalability_plots_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📈 Scalability plots saved to {plot_filename}")
        
        plt.show()


def run_scalability_analysis(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive scalability analysis.
    
    Args:
        device: Computing device for analysis
        
    Returns:
        Scalability analysis results
    """
    print("🚀 Starting Comprehensive Scalability Analysis")
    print("=" * 60)
    
    tester = ScalabilityTester(device=device)
    
    # Run all scalability tests
    test_results = tester.run_all_scalability_tests()
    
    # Generate and save report
    report = tester.generate_scalability_report()
    tester.save_report()
    
    # Generate plots
    tester.plot_scaling_results()
    
    # Print summary
    print(f"\n📊 Scalability Analysis Summary:")
    print(f"  Tests completed: {len(test_results)}")
    
    if 'scaling_analysis' in report['summary']:
        analysis = report['summary']['scaling_analysis']
        linear_scaling = sum(1 for a in analysis.values() if a.get('scaling_efficiency') == 'linear')
        print(f"  Linear scaling tests: {linear_scaling}/{len(analysis)}")
    
    if report['recommendations']:
        print(f"\n💡 Key Findings:")
        for rec in report['recommendations'][:5]:  # Show first 5 recommendations
            print(f"  - {rec}")
    
    return report


if __name__ == '__main__':
    # Run scalability analysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = run_scalability_analysis(device=device)