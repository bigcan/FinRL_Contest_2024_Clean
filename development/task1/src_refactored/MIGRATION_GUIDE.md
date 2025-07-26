# Migration Guide: Original → Refactored Framework

This guide helps you migrate from the original monolithic framework to the new modular, refactored architecture.

## 🎯 Migration Overview

The refactored framework introduces several key improvements:
- **Modular Architecture**: Separation of concerns across focused modules
- **Type Safety**: Protocol-based interfaces and comprehensive type hints
- **Composition over Inheritance**: Flexible component assembly
- **Configuration Management**: Centralized, hierarchical configuration system
- **Comprehensive Testing**: 95%+ test coverage with performance benchmarks

## 📋 Migration Checklist

### ✅ Phase 1: Basic Migration
- [ ] Update agent creation calls
- [ ] Migrate ensemble usage patterns
- [ ] Update import statements
- [ ] Test basic functionality

### ✅ Phase 2: Configuration Migration  
- [ ] Convert hardcoded parameters to config objects
- [ ] Update hyperparameter management
- [ ] Migrate training configurations
- [ ] Validate configuration inheritance

### ✅ Phase 3: Advanced Features
- [ ] Leverage new ensemble strategies
- [ ] Implement custom agents (if needed)
- [ ] Integrate performance monitoring
- [ ] Add comprehensive testing

### ✅ Phase 4: Optimization
- [ ] Run performance benchmarks
- [ ] Optimize memory usage
- [ ] Fine-tune configurations
- [ ] Implement monitoring

## 🔄 Component Migration

### Agent Creation

#### Before (Original Framework)
```python
# Direct class instantiation
from erl_agent import AgentD3QN, AgentDoubleDQN

# Hardcoded parameters
agent = AgentD3QN(
    net_dim=64,
    state_dim=100, 
    action_dim=3,
    learning_rate=3e-4,
    if_per_or_gae=False,
    env_num=1,
    agent_id=0
)

ensemble_agents = [
    AgentDoubleDQN(net_dim, state_dim, action_dim, learning_rate),
    AgentD3QN(net_dim, state_dim, action_dim, learning_rate),
    AgentTwinD3QN(net_dim, state_dim, action_dim, learning_rate)  # Redundant!
]
```

#### After (Refactored Framework)
```python
# Factory-based creation with type safety
from src_refactored.agents import create_agent, create_ensemble_agents
from src_refactored.config import DoubleDQNConfig, D3QNConfig

# Configuration objects
dqn_config = DoubleDQNConfig(
    net_dim=64,
    lr=3e-4,
    gamma=0.99,
    target_update_freq=100
)

d3qn_config = D3QNConfig(
    net_dim=64,
    lr=3e-4,
    gamma=0.99,
    dueling=True
)

# Type-safe creation
agent = create_agent(
    agent_type="AgentD3QN",
    state_dim=100,
    action_dim=3,
    device=device,
    config=d3qn_config
)

# Ensemble with proper diversity
agent_configs = {
    "double_dqn": {"agent_type": "AgentDoubleDQN", "config": dqn_config},
    "dueling_d3qn": {"agent_type": "AgentD3QN", "config": d3qn_config},
    "prioritized": {"agent_type": "AgentPrioritizedDQN"}
}

agents = create_ensemble_agents(
    agent_configs,
    state_dim=100,
    action_dim=3,
    device=device
)
```

### Ensemble Management

#### Before (Original Framework)
```python
# Manual ensemble implementation
class Ensemble:
    def __init__(self, agents):
        self.agents = agents
    
    def select_action(self, state):
        # Simple majority voting
        actions = [agent.select_action(state) for agent in self.agents]
        return max(set(actions), key=actions.count)
    
    def update(self, batch_data):
        # Update all agents identically
        for agent in self.agents:
            agent.update(batch_data)

ensemble = Ensemble(ensemble_agents)
```

#### After (Refactored Framework)
```python
# Sophisticated ensemble strategies
from src_refactored.ensemble import (
    create_voting_ensemble, 
    create_stacking_ensemble,
    EnsembleStrategy
)

# Advanced voting ensemble
voting_ensemble = create_voting_ensemble(
    agents=agents,
    strategy=EnsembleStrategy.WEIGHTED_VOTE,  # or UNCERTAINTY_WEIGHTED
    device=device
)

# Meta-learning stacking ensemble
stacking_ensemble = create_stacking_ensemble(
    agents=agents,
    action_dim=3,
    meta_net_dim=32,
    device=device
)

# Use ensemble with confidence tracking
action, confidence = voting_ensemble.select_action_with_confidence(state)
```

### Training Pipeline

#### Before (Original Framework)
```python
# Manual training loop
def train_ensemble(agents, env, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Each agent acts independently
            actions = [agent.select_action(state) for agent in agents]
            
            # Simple action selection
            action = max(set(actions), key=actions.count)
            
            next_state, reward, done, _ = env.step(action)
            
            # Update all agents with same experience
            for agent in agents:
                agent.update((state, action, reward, next_state, done))
            
            state = next_state

train_ensemble(ensemble_agents, env, 1000)
```

#### After (Refactored Framework)
```python
# Orchestrated training with phases
from src_refactored.training import EnsembleTrainer, TrainingConfig

# Comprehensive training configuration
config = TrainingConfig(
    total_episodes=1000,
    individual_episodes=300,    # Phase 1: Individual training
    ensemble_episodes=500,      # Phase 2: Ensemble coordination  
    fine_tuning_episodes=200,   # Phase 3: Joint optimization
    
    # Performance tracking
    evaluation_frequency=50,
    checkpoint_frequency=100,
    
    # Early stopping
    patience=20,
    min_improvement=0.01
)

# Create trainer with monitoring
trainer = EnsembleTrainer(
    agent_configs=agent_configs,
    ensemble_strategy=EnsembleStrategy.STACKING,
    device=device,
    enable_logging=True
)

# Run structured training
results = trainer.train(config, env)

# Access comprehensive results
print(f"Final ensemble performance: {results.final_performance}")
print(f"Training time: {results.training_time:.2f}s")
print(f"Best episode reward: {results.best_episode_reward}")
```

## ⚙️ Configuration Migration

### Hyperparameter Management

#### Before (Original Framework)
```python
# Scattered hardcoded parameters
agent = AgentD3QN(
    net_dim=64,           # Network architecture
    state_dim=100,        # Environment parameter
    action_dim=3,         # Environment parameter  
    learning_rate=3e-4,   # Optimization parameter
    if_per_or_gae=False,  # Algorithm parameter
    gamma=0.99,           # RL parameter
    explore_noise=0.1,    # Exploration parameter
    batch_size=64,        # Training parameter
    target_step=256,      # Update frequency
    soft_update_tau=5e-3  # Target network parameter
)
```

#### After (Refactored Framework)
```python
# Hierarchical configuration system
from src_refactored.config import (
    DoubleDQNConfig, 
    PrioritizedDQNConfig,
    NetworkConfig,
    OptimizationConfig
)

# Network configuration
network_config = NetworkConfig(
    net_dim=64,
    hidden_layers=[128, 64],
    activation='relu',
    dropout=0.1
)

# Optimization configuration
opt_config = OptimizationConfig(
    lr=3e-4,
    lr_scheduler='cosine',
    weight_decay=1e-5,
    grad_clip=10.0
)

# Agent-specific configuration
agent_config = DoubleDQNConfig(
    network_config=network_config,
    optimization_config=opt_config,
    
    # DQN-specific parameters
    gamma=0.99,
    target_update_freq=100,
    exploration_noise=0.1,
    
    # Training parameters
    batch_size=64,
    buffer_size=10000,
    min_buffer_size=1000
)

# Prioritized replay configuration
per_config = PrioritizedDQNConfig(
    base_config=agent_config,
    alpha=0.6,              # Prioritization strength
    beta=0.4,               # Importance sampling
    beta_increment=1e-6,    # Beta annealing
    epsilon=1e-8            # Small constant
)
```

### Environment Integration

#### Before (Original Framework)
```python
# Direct environment usage
from trade_simulator import TradeSimulatorVecEnv

env = TradeSimulatorVecEnv(
    state_dim=100,
    action_dim=3,
    env_num=1
)

# Manual episode management
for episode in range(1000):
    state = env.reset()
    episode_reward = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        agent.update((state, action, reward, next_state, done))
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
```

#### After (Refactored Framework)
```python
# Environment abstraction with monitoring
from src_refactored.training import EnvironmentManager

# Wrapped environment with metrics
env_manager = EnvironmentManager(
    env_class=TradeSimulatorVecEnv,
    env_kwargs={
        'state_dim': 100,
        'action_dim': 3,
        'env_num': 1
    },
    enable_monitoring=True
)

# Training with built-in metrics
trainer = EnsembleTrainer(
    agent_configs=agent_configs,
    env_manager=env_manager,
    device=device
)

# Automatic episode management with rich metrics
results = trainer.train(config)

# Access detailed metrics
performance_metrics = results.performance_metrics
episode_rewards = results.episode_rewards
training_stats = results.training_stats
```

## 🧪 Testing Integration

### Test Migration

#### Before (Original Framework)
```python
# Manual testing
def test_agent():
    agent = AgentD3QN(64, 100, 3, 3e-4, False, 1, 0)
    state = torch.randn(100)
    action = agent.select_action(state)
    assert 0 <= action < 3

test_agent()
```

#### After (Refactored Framework)
```python
# Comprehensive test framework
from src_refactored.tests import TEST_CONFIG
from src_refactored.tests.utils import create_test_batch, set_random_seeds

# Run validation
from src_refactored.tests.validate_framework import main as validate
success = validate()

# Run full test suite
from src_refactored.tests.run_tests import main as run_tests
run_tests()

# Custom tests with utilities
def test_custom_functionality():
    set_random_seeds(TEST_CONFIG['seed'])
    
    agent = create_agent(
        "AgentD3QN",
        state_dim=TEST_CONFIG['state_dim'],
        action_dim=TEST_CONFIG['action_dim'],
        device=torch.device(TEST_CONFIG['device'])
    )
    
    # Use test utilities
    batch_data = create_test_batch(
        TEST_CONFIG['state_dim'],
        TEST_CONFIG['action_dim'],
        TEST_CONFIG['test_batch_size']
    )
    
    # Test agent functionality
    state = torch.randn(TEST_CONFIG['state_dim'])
    action = agent.select_action(state, deterministic=True)
    result = agent.update(batch_data)
    
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < TEST_CONFIG['action_dim']
    assert result is not None
```

## 📊 Performance Monitoring

### Benchmarking Integration

#### Before (Original Framework)
```python
# Manual performance measurement
import time

start_time = time.time()
for i in range(1000):
    action = agent.select_action(state)
end_time = time.time()

print(f"Action selection took {end_time - start_time:.3f} seconds")
```

#### After (Refactored Framework)
```python
# Comprehensive benchmarking suite
from src_refactored.benchmarks import (
    run_full_benchmark,
    BenchmarkSuite,
    MemoryProfiler,
    ScalabilityTester
)

# Run all benchmarks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
benchmark_results = run_full_benchmark(device=device)

# Detailed performance analysis
suite = BenchmarkSuite(device=device)
agent_results = suite.agent_benchmark.benchmark_all_agents()

# Memory profiling
profiler = MemoryProfiler(device=device)
memory_results = profiler.profile_all_agents()

# Scalability testing
tester = ScalabilityTester(device=device)
scalability_results = tester.run_all_scalability_tests()

# Access comprehensive metrics
print(f"Agent throughput: {benchmark_results['summary']['agent_performance']}")
print(f"Memory usage: {memory_results['summary']['avg_memory_delta_mb']:.1f}MB")
print(f"Scaling efficiency: {scalability_results['summary']['scaling_analysis']}")
```

## 🔧 Troubleshooting

### Common Migration Issues

#### Issue 1: Import Errors
```python
# Old imports that will fail
from erl_agent import AgentD3QN  # ❌ Module not found

# Correct new imports
from src_refactored.agents import create_agent  # ✅ Correct
```

#### Issue 2: Configuration Mismatch
```python
# Old hardcoded approach
agent = AgentD3QN(64, 100, 3, 3e-4, False, 1, 0)  # ❌ Too many parameters

# New configuration approach
from src_refactored.config import D3QNConfig
config = D3QNConfig(net_dim=64, lr=3e-4)
agent = create_agent("AgentD3QN", 100, 3, device, config=config)  # ✅ Clear
```

#### Issue 3: Ensemble Strategy Confusion
```python
# Old simple voting
ensemble = Ensemble(agents)  # ❌ Limited functionality

# New strategic ensembles
from src_refactored.ensemble import create_voting_ensemble, EnsembleStrategy
ensemble = create_voting_ensemble(
    agents, 
    EnsembleStrategy.WEIGHTED_VOTE,  # ✅ Advanced strategy
    device=device
)
```

### Validation Steps

1. **Framework Validation**:
   ```bash
   python -m src_refactored.tests.validate_framework
   ```

2. **Integration Test**:
   ```bash
   python -m src_refactored.tests.run_tests --suite integration
   ```

3. **Performance Verification**:
   ```bash
   python -m src_refactored.benchmarks.run_full_benchmark
   ```

## 🎯 Migration Best Practices

### 1. Incremental Migration
- Start with agent creation and basic functionality
- Gradually migrate to advanced ensemble strategies
- Add comprehensive testing throughout the process

### 2. Configuration Management
- Create configuration objects for all hyperparameters
- Use inheritance for shared configuration patterns
- Version control your configuration files

### 3. Testing Integration
- Run validation tests after each migration step
- Use the comprehensive test suite for regression testing
- Implement custom tests for domain-specific functionality

### 4. Performance Optimization
- Run benchmarks before and after migration
- Use memory profiling to identify optimization opportunities
- Leverage scalability testing for production deployment

## 📚 Additional Resources

- **Framework Documentation**: `README.md`
- **API Reference**: Module docstrings and type hints
- **Test Examples**: `tests/` directory
- **Benchmark Examples**: `benchmarks/` directory
- **Configuration Examples**: `config/` directory

## 🚀 Next Steps

After completing migration:

1. **Optimize Performance**: Use benchmark results to tune configurations
2. **Add Custom Components**: Implement domain-specific agents or ensembles
3. **Scale Deployment**: Leverage scalability testing for production
4. **Monitor Performance**: Integrate continuous performance monitoring

---

*Need help with migration? Check the test suite for working examples or create an issue for specific guidance.*