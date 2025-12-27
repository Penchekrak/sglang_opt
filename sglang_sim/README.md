# SGLang LLM Inference Simulator

A discrete-event simulator for LLM inference with symbolic tracing, designed to model SGLang-style workloads with prefill/decode disaggregation, prefix caching, and parallel execution.

## Features

- **Symbolic Metric Tracing**: All latency computations produce both numeric values and SymPy expressions for optimization
- **P/D Disaggregation**: Separate prefill and decode worker pools with explicit KV transfer modeling
- **Prefix Cache (RadixAttention)**: LRU-based radix cache with cache-aware routing
- **Chunked Prefill**: Configurable chunk sizes for TTFT/throughput tradeoffs
- **Parallelism Support**: Tensor Parallelism (TP), Data Parallelism (DP), Expert Parallelism (EP)
- **MoE Models**: Full support for Mixture-of-Experts with all-to-all communication
- **Configurable Interconnects**: NVLink, InfiniBand, Ethernet with proper latency/bandwidth modeling
- **Multi-Objective Optimization**: Pareto-optimal search with weighted-sum and epsilon-constraint methods
- **Constraint Handling**: Memory limits, SLO targets, cost budgets, GPU constraints
- **Compute Graph Visualization**: DOT and Mermaid diagrams for expression trees and metric dependencies

## Symbolic Metrics Available

| Metric | Description |
|--------|-------------|
| `throughput` | Tokens per second |
| `ttft` | Time to First Token |
| `tpot` | Time per Output Token |
| `e2e_latency` | End-to-end request latency |
| `peak_memory` | Peak GPU memory usage |
| `compute_utilization` | GPU compute utilization |
| `memory_bandwidth_utilization` | Memory bandwidth utilization |
| `network_utilization` | Network bandwidth utilization |
| `cost_per_token` | Cost efficiency metric |
| `power_consumption` | Power consumption estimate |

## Installation

```bash
cd sglang_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# For visualization support
pip install -e ".[viz]"
```

## Quick Start

```python
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution

# Configure simulation
model = ModelConfig.llama_7b()
cluster = ClusterConfig.single_node_8gpu()
scheduler = SchedulerConfig.default()
cache = CacheConfig.h100_default()

# Create engine and run
engine = SimulationEngine(model, cluster, scheduler, cache)
# ... initialize workers, router, etc.
result = engine.run_until_idle()

print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
```

## Symbolic Throughput Optimization

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.optimizer.interface import OptimizerInterface

symbols = ConfigSymbols()

# Build symbolic throughput expression
throughput_expr = ...  # Your throughput formula

# Optimize over configuration space
optimizer = OptimizerInterface(objective=throughput_expr, maximize=True)
result = optimizer.grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
})

print(f"Optimal config: {result.optimal_values}")
```

## Project Structure

```
sim/
├── config/          # Configuration dataclasses
├── core/            # Simulation engine, events, state
├── symbolic/        # SymVal and symbolic combinators
├── workers/         # Prefill and decode workers
├── schedulers/      # Continuous batching scheduler
├── cache/           # Radix cache implementation
├── routing/         # Cache-aware router
├── parallel/        # TP/DP/EP group abstractions
├── network/         # Interconnect and KV transfer
├── kernels/         # Attention, MLP, MoE kernels
├── models/          # Operator graph templates
├── metrics/         # Metric collection and symbolic throughput
├── optimizer/       # Scipy/grid search interface
└── workload/        # Synthetic and trace-based workloads
```

## Multi-Objective Optimization with Constraints

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer.interface import (
    MultiObjectiveOptimizer,
    ObjectiveSpec,
    ObjectiveDirection,
)

symbols = ConfigSymbols()
metric_builder = SymbolicMetricBuilder(symbols)
metrics = metric_builder.build_all_expressions()

# Define constraints
constraints = ConstraintBuilder(symbols)
constraints.add_memory_constraint(80 * 1024**3)  # 80GB H100
constraints.add_ttft_slo(0.5)  # 500ms TTFT
constraints.add_tpot_slo(0.05)  # 50ms TPOT
constraints.add_total_gpu_constraint(16)

# Multi-objective optimization
optimizer = MultiObjectiveOptimizer(
    objectives=[
        ObjectiveSpec("throughput", metrics.throughput, ObjectiveDirection.MAXIMIZE),
        ObjectiveSpec("cost", metrics.cost_per_token, ObjectiveDirection.MINIMIZE),
        ObjectiveSpec("latency", metrics.e2e_latency, ObjectiveDirection.MINIMIZE),
    ],
    constraints=constraints.get_all_constraints(),
)

# Find Pareto-optimal configurations
result = optimizer.pareto_grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
    symbols.batch_cap_requests: [64, 128, 256],
})

print(f"Found {len(result.pareto_front)} Pareto-optimal solutions")
```

## Compute Graph Visualization

```python
from sim.visualization import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)

# Visualize expression tree
viz = ExpressionGraphVisualizer()
viz.build_graph(metrics.throughput, "Throughput")
dot_code = viz.to_dot("Throughput Expression")  # For Graphviz
mermaid = viz.to_mermaid("Throughput Expression")  # For Markdown

# Visualize metric dependencies
dep_graph = MetricDependencyGraph()
dep_graph.add_metrics(metrics.as_dict())
print(dep_graph.find_shared_dependencies())  # Variables affecting multiple metrics
```

## Memory Analysis

```python
from sim.metrics.memory import SymbolicMemoryModel

mem_model = SymbolicMemoryModel(symbols)

# Get symbolic expression for peak memory
peak_mem = mem_model.symbolic_peak_memory()

# Evaluate for specific configuration
config = {symbols.batch_cap_requests: 256, symbols.tp_size: 4, ...}
memory_gb = float(peak_mem.subs(list(config.items()))) / 1e9
```

## Examples

See the `examples/` directory for:
- `basic_simulation.py` - Full simulation workflow
- `optimize_pd_ratio.py` - P/D ratio optimization with symbolic expressions
- `sweep_chunk_size.py` - Chunk size analysis and cache effects
- `multi_objective_optimization.py` - Pareto optimization with constraints and visualization

## Running Tests

```bash
python -m pytest tests/ -v
```

