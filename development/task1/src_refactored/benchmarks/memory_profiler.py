"""
Memory profiling and analysis for the FinRL Contest 2024 refactored framework.

This module provides detailed memory usage analysis, leak detection,
and memory optimization recommendations for all framework components.
"""

import gc
import tracemalloc
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, NamedTuple, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import time
import threading
from collections import defaultdict

# Import framework components
from ..agents import create_agent, AGENT_REGISTRY
from ..ensemble import create_voting_ensemble, EnsembleStrategy
from ..tests.utils.mock_environment import MockEnvironment
from ..tests.utils.test_helpers import create_test_batch, set_random_seeds


@dataclass
class MemorySnapshot:
    """Container for memory usage data at a specific point in time."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    shared_mb: float
    percent: float
    available_mb: float
    torch_allocated_mb: float = 0.0
    torch_cached_mb: float = 0.0
    tracemalloc_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'rss_mb': self.rss_mb,
            'vms_mb': self.vms_mb,
            'shared_mb': self.shared_mb,
            'percent': self.percent,
            'available_mb': self.available_mb,
            'torch_allocated_mb': self.torch_allocated_mb,
            'torch_cached_mb': self.torch_cached_mb,
            'tracemalloc_mb': self.tracemalloc_mb
        }


@dataclass
class MemoryProfileResult:
    """Container for memory profiling results."""
    test_name: str
    start_snapshot: MemorySnapshot
    end_snapshot: MemorySnapshot
    peak_snapshot: MemorySnapshot
    memory_delta_mb: float
    torch_delta_mb: float
    leak_detected: bool
    gc_collections: int
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'start_snapshot': self.start_snapshot.to_dict(),
            'end_snapshot': self.end_snapshot.to_dict(),
            'peak_snapshot': self.peak_snapshot.to_dict(),
            'memory_delta_mb': self.memory_delta_mb,
            'torch_delta_mb': self.torch_delta_mb,
            'leak_detected': self.leak_detected,
            'gc_collections': self.gc_collections,
            'duration_seconds': self.duration_seconds
        }


class MemoryTracker:
    """Real-time memory usage tracker."""
    
    def __init__(self, interval: float = 0.1, device: Optional[torch.device] = None):
        self.interval = interval
        self.device = device or torch.device('cpu')
        self.snapshots: List[MemorySnapshot] = []
        self.running = False
        self.thread = None
        self._peak_snapshot = None
        
    def start(self):
        """Start memory tracking."""
        if self.running:
            return
        
        self.running = True
        self.snapshots = []
        self._peak_snapshot = None
        self.thread = threading.Thread(target=self._track_memory)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self) -> MemorySnapshot:
        """Stop memory tracking and return final snapshot."""
        if not self.running:
            return self._get_current_snapshot()
        
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        return self._get_current_snapshot()
    
    def _track_memory(self):
        """Background thread for memory tracking."""
        while self.running:
            snapshot = self._get_current_snapshot()
            self.snapshots.append(snapshot)
            
            # Track peak memory usage
            if (self._peak_snapshot is None or 
                snapshot.rss_mb > self._peak_snapshot.rss_mb):
                self._peak_snapshot = snapshot
            
            time.sleep(self.interval)
    
    def _get_current_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # System memory
        virtual_memory = psutil.virtual_memory()
        
        # PyTorch memory (if CUDA available)
        torch_allocated = 0.0
        torch_cached = 0.0
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch_allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            torch_cached = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        
        # Tracemalloc memory (if enabled)
        tracemalloc_mb = 0.0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_mb = current / 1024 / 1024
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            shared_mb=getattr(memory_info, 'shared', 0) / 1024 / 1024,
            percent=memory_percent,
            available_mb=virtual_memory.available / 1024 / 1024,
            torch_allocated_mb=torch_allocated,
            torch_cached_mb=torch_cached,
            tracemalloc_mb=tracemalloc_mb
        )
    
    def get_peak_snapshot(self) -> Optional[MemorySnapshot]:
        """Get peak memory snapshot."""
        return self._peak_snapshot or (self.snapshots[-1] if self.snapshots else None)


class MemoryProfiler:
    """Comprehensive memory profiler for framework components."""
    
    def __init__(self, device: Optional[torch.device] = None, enable_tracemalloc: bool = True):
        self.device = device or torch.device('cpu')
        self.enable_tracemalloc = enable_tracemalloc
        self.results: List[MemoryProfileResult] = []
        
        # Memory leak detection threshold (MB)
        self.leak_threshold = 10.0
        
        # Test parameters
        self.state_dim = 100
        self.action_dim = 3
        self.test_iterations = 50
        
    def profile_agent_memory(self, agent_type: str) -> MemoryProfileResult:
        """Profile memory usage for agent operations."""
        test_name = f"{agent_type}_memory_profile"
        
        # Start profiling
        if self.enable_tracemalloc:
            tracemalloc.start()
        
        tracker = MemoryTracker(device=self.device)
        tracker.start()
        
        # Initial memory state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_snapshot = tracker._get_current_snapshot()
        start_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))
        start_time = time.time()
        
        try:
            # Create and use agent multiple times
            for i in range(self.test_iterations):
                agent = create_agent(
                    agent_type=agent_type,
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    device=self.device
                )
                
                # Perform operations
                state = torch.randn(self.state_dim, device=self.device)
                batch_data = create_test_batch(self.state_dim, self.action_dim, 32)
                batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                
                # Action selection and updates
                for _ in range(10):
                    action = agent.select_action(state, deterministic=True)
                    result = agent.update(batch_data)
                
                # Force cleanup
                del agent
                if i % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        finally:
            # Final memory state
            end_time = time.time()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            end_snapshot = tracker.stop()
            end_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))
            peak_snapshot = tracker.get_peak_snapshot()
            
            if self.enable_tracemalloc:
                tracemalloc.stop()
        
        # Calculate metrics
        memory_delta = end_snapshot.rss_mb - start_snapshot.rss_mb
        torch_delta = end_snapshot.torch_allocated_mb - start_snapshot.torch_allocated_mb
        leak_detected = memory_delta > self.leak_threshold
        
        result = MemoryProfileResult(
            test_name=test_name,
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            peak_snapshot=peak_snapshot or end_snapshot,
            memory_delta_mb=memory_delta,
            torch_delta_mb=torch_delta,
            leak_detected=leak_detected,
            gc_collections=end_collections - start_collections,
            duration_seconds=end_time - start_time
        )
        
        self.results.append(result)
        return result
    
    def profile_ensemble_memory(self, ensemble_type: str, num_agents: int = 3) -> MemoryProfileResult:
        """Profile memory usage for ensemble operations."""
        test_name = f"{ensemble_type}_ensemble_{num_agents}agents_memory"
        
        # Start profiling
        if self.enable_tracemalloc:
            tracemalloc.start()
        
        tracker = MemoryTracker(device=self.device)
        tracker.start()
        
        # Initial memory state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_snapshot = tracker._get_current_snapshot()
        start_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))
        start_time = time.time()
        
        try:
            # Create and use ensemble multiple times
            for i in range(self.test_iterations // 2):  # Fewer iterations for ensembles
                # Create agents
                agents = {}
                for j in range(num_agents):
                    agents[f"agent_{j}"] = create_agent(
                        agent_type="AgentDoubleDQN",
                        state_dim=self.state_dim,
                        action_dim=self.action_dim,
                        device=self.device
                    )
                
                # Create ensemble
                ensemble = create_voting_ensemble(
                    agents=agents,
                    strategy=EnsembleStrategy.MAJORITY_VOTE,
                    device=self.device
                )
                
                # Perform operations
                state = torch.randn(self.state_dim, device=self.device)
                batch_data = create_test_batch(self.state_dim, self.action_dim, 16)
                batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                
                # Action selection and updates
                for _ in range(5):
                    action = ensemble.select_action(state, deterministic=True)
                    result = ensemble.update(batch_data)
                
                # Force cleanup
                del ensemble
                del agents
                if i % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        finally:
            # Final memory state
            end_time = time.time()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            end_snapshot = tracker.stop()
            end_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))
            peak_snapshot = tracker.get_peak_snapshot()
            
            if self.enable_tracemalloc:
                tracemalloc.stop()
        
        # Calculate metrics
        memory_delta = end_snapshot.rss_mb - start_snapshot.rss_mb
        torch_delta = end_snapshot.torch_allocated_mb - start_snapshot.torch_allocated_mb
        leak_detected = memory_delta > self.leak_threshold * num_agents  # Scale threshold
        
        result = MemoryProfileResult(
            test_name=test_name,
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            peak_snapshot=peak_snapshot or end_snapshot,
            memory_delta_mb=memory_delta,
            torch_delta_mb=torch_delta,
            leak_detected=leak_detected,
            gc_collections=end_collections - start_collections,
            duration_seconds=end_time - start_time
        )
        
        self.results.append(result)
        return result
    
    def stress_test_memory(self, test_name: str, test_function: Callable) -> MemoryProfileResult:
        """Run stress test to detect memory leaks."""
        # Start profiling
        if self.enable_tracemalloc:
            tracemalloc.start()
        
        tracker = MemoryTracker(device=self.device)
        tracker.start()
        
        # Initial memory state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_snapshot = tracker._get_current_snapshot()
        start_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))
        start_time = time.time()
        
        try:
            # Run stress test
            for i in range(100):  # High iteration count for stress testing
                test_function()
                
                # Periodic cleanup
                if i % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        finally:
            # Final memory state
            end_time = time.time()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            end_snapshot = tracker.stop()
            end_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))
            peak_snapshot = tracker.get_peak_snapshot()
            
            if self.enable_tracemalloc:
                tracemalloc.stop()
        
        # Calculate metrics
        memory_delta = end_snapshot.rss_mb - start_snapshot.rss_mb
        torch_delta = end_snapshot.torch_allocated_mb - start_snapshot.torch_allocated_mb
        leak_detected = memory_delta > self.leak_threshold
        
        result = MemoryProfileResult(
            test_name=f"stress_{test_name}",
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            peak_snapshot=peak_snapshot or end_snapshot,
            memory_delta_mb=memory_delta,
            torch_delta_mb=torch_delta,
            leak_detected=leak_detected,
            gc_collections=end_collections - start_collections,
            duration_seconds=end_time - start_time
        )
        
        self.results.append(result)
        return result
    
    def profile_all_agents(self) -> Dict[str, MemoryProfileResult]:
        """Profile memory usage for all available agents."""
        print("🧠 Running Agent Memory Profiling...")
        
        results = {}
        agents_to_test = ['AgentDoubleDQN', 'AgentD3QN']  # Test subset for efficiency
        
        for agent_type in agents_to_test:
            if agent_type in AGENT_REGISTRY:
                print(f"  Profiling {agent_type}...")
                try:
                    result = self.profile_agent_memory(agent_type)
                    results[agent_type] = result
                    
                    status = "🟡 LEAK DETECTED" if result.leak_detected else "✅ OK"
                    print(f"    {status} - Memory delta: {result.memory_delta_mb:.1f}MB")
                    
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
        
        return results
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory analysis report."""
        if not self.results:
            return {'error': 'No profiling results available'}
        
        # Analyze results
        total_tests = len(self.results)
        leaks_detected = sum(1 for r in self.results if r.leak_detected)
        avg_memory_delta = sum(r.memory_delta_mb for r in self.results) / total_tests
        max_memory_delta = max(r.memory_delta_mb for r in self.results)
        
        # Find most problematic test
        worst_test = max(self.results, key=lambda r: r.memory_delta_mb)
        
        # Memory usage patterns
        memory_patterns = defaultdict(list)
        for result in self.results:
            test_category = result.test_name.split('_')[0]
            memory_patterns[test_category].append(result.memory_delta_mb)
        
        # Generate recommendations
        recommendations = []
        
        if leaks_detected > 0:
            recommendations.append(f"🚨 {leaks_detected} potential memory leaks detected")
            recommendations.append("Consider reviewing object lifecycle management")
        
        if avg_memory_delta > 5.0:
            recommendations.append(f"Average memory increase of {avg_memory_delta:.1f}MB per test")
            recommendations.append("Consider implementing more aggressive garbage collection")
        
        if any(r.torch_delta_mb > 50.0 for r in self.results):
            recommendations.append("High GPU memory usage detected")
            recommendations.append("Consider using torch.cuda.empty_cache() more frequently")
        
        # Peak memory analysis
        peak_memories = [r.peak_snapshot.rss_mb for r in self.results]
        max_peak = max(peak_memories) if peak_memories else 0
        
        if max_peak > 1000:  # 1GB
            recommendations.append(f"Peak memory usage: {max_peak:.1f}MB")
            recommendations.append("Consider batch size optimization for large models")
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'leaks_detected': leaks_detected,
                'avg_memory_delta_mb': avg_memory_delta,
                'max_memory_delta_mb': max_memory_delta,
                'max_peak_memory_mb': max_peak,
                'worst_test': worst_test.test_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'memory_patterns': dict(memory_patterns),
            'recommendations': recommendations,
            'detailed_results': [r.to_dict() for r in self.results]
        }
        
        return report
    
    def save_report(self, filename: Optional[str] = None):
        """Save memory profiling report to file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"memory_profile_report_{timestamp}.json"
        
        report = self.generate_memory_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Memory profiling report saved to {filename}")


def run_memory_analysis(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive memory analysis.
    
    Args:
        device: Computing device for analysis
        
    Returns:
        Memory analysis results
    """
    print("🧠 Starting Comprehensive Memory Analysis")
    print("=" * 60)
    
    profiler = MemoryProfiler(device=device)
    
    # Profile all agents
    agent_results = profiler.profile_all_agents()
    
    # Profile ensembles
    print("\n🤝 Running Ensemble Memory Profiling...")
    ensemble_results = {}
    
    for ensemble_type in ['voting']:  # Test voting ensemble
        try:
            result = profiler.profile_ensemble_memory(ensemble_type, num_agents=2)
            ensemble_results[ensemble_type] = result
            
            status = "🟡 LEAK DETECTED" if result.leak_detected else "✅ OK"
            print(f"  {ensemble_type}: {status} - Memory delta: {result.memory_delta_mb:.1f}MB")
            
        except Exception as e:
            print(f"  {ensemble_type}: ❌ Failed: {e}")
    
    # Generate and save report
    report = profiler.generate_memory_report()
    profiler.save_report()
    
    # Print summary
    print(f"\n📊 Memory Analysis Summary:")
    print(f"  Tests run: {report['summary']['total_tests']}")
    print(f"  Leaks detected: {report['summary']['leaks_detected']}")
    print(f"  Average memory delta: {report['summary']['avg_memory_delta_mb']:.1f}MB")
    print(f"  Peak memory usage: {report['summary']['max_peak_memory_mb']:.1f}MB")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    return report


if __name__ == '__main__':
    # Run memory analysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = run_memory_analysis(device=device)