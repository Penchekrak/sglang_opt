#!/usr/bin/env python3
"""Example demonstrating multi-objective optimization with constraints and visualization."""

import sys
sys.path.insert(0, "..")

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder, MetricExpressions
from sim.metrics.memory import SymbolicMemoryModel
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer.interface import (
    MultiObjectiveOptimizer,
    ObjectiveSpec,
    ObjectiveDirection,
)
from sim.visualization.graph import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)


def build_optimization_problem():
    symbols = ConfigSymbols()
    metric_builder = SymbolicMetricBuilder(symbols)
    
    metrics = metric_builder.build_all_expressions()
    
    print("=== Available Metrics ===")
    for name, expr in metrics.as_dict().items():
        print(f"  {name}")
    
    return symbols, metric_builder, metrics


def run_constrained_optimization():
    symbols, metric_builder, metrics = build_optimization_problem()
    
    constraint_builder = ConstraintBuilder(symbols)
    
    constraint_builder.add_memory_constraint(
        gpu_memory_bytes=80 * 1024**3,
        safety_margin=0.9,
    )
    constraint_builder.add_ttft_slo(max_ttft_seconds=0.5)
    constraint_builder.add_tpot_slo(max_tpot_seconds=0.05)
    constraint_builder.add_total_gpu_constraint(max_gpus=16)
    constraint_builder.add_min_throughput(min_tokens_per_sec=100)
    
    print("\n=== Constraints ===")
    print(constraint_builder.summarize())
    
    objectives = [
        ObjectiveSpec(
            name="throughput",
            expr=metrics.throughput,
            direction=ObjectiveDirection.MAXIMIZE,
            weight=1.0,
        ),
        ObjectiveSpec(
            name="cost_per_token",
            expr=metrics.cost_per_token,
            direction=ObjectiveDirection.MINIMIZE,
            weight=0.5,
        ),
        ObjectiveSpec(
            name="e2e_latency",
            expr=metrics.e2e_latency,
            direction=ObjectiveDirection.MINIMIZE,
            weight=0.3,
        ),
    ]
    
    optimizer = MultiObjectiveOptimizer(
        objectives=objectives,
        constraints=constraint_builder.get_all_constraints(),
        symbols=symbols,
    )
    
    frozen_params = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.network_bw: 100e9,
        symbols.network_latency: 10e-6,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.head_dim: 128,
        symbols.num_heads: 32,
        symbols.num_experts: 1,
        symbols.kv_bytes_per_token: 256,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.cache_hit_rate: 0.3,
        symbols.avg_prefix_match: 0.5,
    }
    optimizer.set_frozen_params(frozen_params)
    
    param_grid = {
        symbols.N_p: [1, 2, 4, 8],
        symbols.N_d: [1, 2, 4, 8],
        symbols.chunk_size: [4096, 8192, 16384],
        symbols.batch_cap_requests: [64, 128, 256],
        symbols.tp_size: [1, 2, 4],
    }
    
    print("\n=== Running Pareto Optimization ===")
    result = optimizer.pareto_grid_search(param_grid)
    
    print(f"\nFound {len(result.pareto_front)} Pareto-optimal solutions")
    print(f"Total feasible solutions evaluated: {len(result.all_evaluations)}")
    
    print("\n=== Pareto Front (Top 5) ===")
    sorted_pareto = sorted(
        zip(result.pareto_front, result.pareto_objectives),
        key=lambda x: x[1].get("throughput", 0),
        reverse=True,
    )[:5]
    
    for i, (params, objs) in enumerate(sorted_pareto, 1):
        print(f"\n{i}. Configuration:")
        print(f"   N_p={params.get('N_p')}, N_d={params.get('N_d')}, "
              f"TP={params.get('TP')}, chunk={params.get('c')}")
        print(f"   Objectives:")
        for name, val in objs.items():
            if "latency" in name or "ttft" in name or "tpot" in name:
                print(f"     {name}: {val*1000:.2f}ms")
            elif "cost" in name:
                print(f"     {name}: ${val*1000:.4f}/1K tokens")
            else:
                print(f"     {name}: {val:.2f}")
    
    return result


def visualize_metrics():
    symbols = ConfigSymbols()
    metric_builder = SymbolicMetricBuilder(symbols)
    metrics = metric_builder.build_all_expressions()
    
    print("\n=== Generating Metric Dependency Graph ===")
    dep_graph = MetricDependencyGraph()
    dep_graph.add_metrics({
        "throughput": metrics.throughput,
        "ttft": metrics.ttft,
        "tpot": metrics.tpot,
        "peak_memory": metrics.peak_memory,
        "cost_per_token": metrics.cost_per_token,
    })
    
    shared = dep_graph.find_shared_dependencies()
    print(f"\nShared dependencies affecting multiple metrics:")
    for sym, affected_metrics in list(shared.items())[:10]:
        print(f"  {sym}: affects {affected_metrics}")
    
    mermaid = dep_graph.to_mermaid("Metric Dependencies")
    print(f"\n=== Mermaid Diagram ===")
    print(mermaid[:500] + "..." if len(mermaid) > 500 else mermaid)
    
    print("\n=== Throughput Expression Tree ===")
    viz = ExpressionGraphVisualizer()
    viz.build_graph(metrics.throughput, "Throughput")
    dot = viz.to_dot("Throughput Expression")
    print(f"DOT graph generated ({len(dot)} chars)")
    print("First 300 chars:")
    print(dot[:300] + "...")
    
    return dep_graph, viz


def analyze_memory_scaling():
    symbols = ConfigSymbols()
    mem_model = SymbolicMemoryModel(symbols)
    
    print("\n=== Memory Scaling Analysis ===")
    
    peak_mem_expr = mem_model.symbolic_peak_memory()
    
    base_params = {
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.num_heads: 32,
        symbols.head_dim: 128,
        symbols.num_experts: 1,
        symbols.chunk_size: 8192,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.tp_size: 1,
    }
    
    print("\nPeak Memory vs Batch Size:")
    for batch in [32, 64, 128, 256, 512]:
        params = base_params.copy()
        params[symbols.batch_cap_requests] = batch
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  Batch {batch:3d}: {mem_gb:.1f} GB")
    
    print("\nPeak Memory vs TP Size (batch=256):")
    base_params[symbols.batch_cap_requests] = 256
    for tp in [1, 2, 4, 8]:
        params = base_params.copy()
        params[symbols.tp_size] = tp
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  TP={tp}: {mem_gb:.1f} GB per GPU")
    
    print("\nPeak Memory vs Sequence Length (batch=128):")
    base_params[symbols.batch_cap_requests] = 128
    base_params[symbols.tp_size] = 1
    for seq_len in [256, 512, 1024, 2048, 4096]:
        params = base_params.copy()
        params[symbols.avg_prompt_len] = seq_len
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  Seq len {seq_len:4d}: {mem_gb:.1f} GB")


if __name__ == "__main__":
    run_constrained_optimization()
    visualize_metrics()
    analyze_memory_scaling()

