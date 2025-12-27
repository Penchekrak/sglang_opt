#!/usr/bin/env python3
"""Example demonstrating P/D ratio optimization using symbolic throughput expressions."""

import sys
sys.path.insert(0, "..")

import sympy
from sympy import Symbol, ceiling

from sim.symbolic.symbols import ConfigSymbols
from sim.symbolic.expr import SymVal, sym_const
from sim.metrics.symbolic import SymbolicThroughputBuilder
from sim.optimizer.interface import OptimizerInterface, Constraint
from sim.config.model import ModelConfig
from sim.config.cluster import GPUSpec


def build_throughput_model():
    symbols = ConfigSymbols()

    N_p = symbols.N_p
    N_d = symbols.N_d
    chunk_size = symbols.chunk_size
    batch_cap = symbols.batch_cap_tokens

    gpu_flops = symbols.gpu_flops
    gpu_mem_bw = symbols.gpu_mem_bw

    avg_prompt_len = symbols.avg_prompt_len
    avg_output_len = symbols.avg_output_len

    num_layers = symbols.num_layers
    hidden_dim = symbols.hidden_dim
    kv_bytes = symbols.kv_bytes_per_token

    prefill_flops_per_chunk = 4 * num_layers * hidden_dim * hidden_dim * chunk_size
    prefill_time_per_chunk = prefill_flops_per_chunk / gpu_flops

    num_prefill_chunks = ceiling(avg_prompt_len / chunk_size)
    total_prefill_time = num_prefill_chunks * prefill_time_per_chunk

    prefill_capacity = N_p / total_prefill_time

    avg_kv_len = avg_prompt_len + avg_output_len / 2
    kv_read_bytes = avg_kv_len * kv_bytes
    decode_time_per_token = kv_read_bytes / gpu_mem_bw

    decode_capacity = N_d * batch_cap / decode_time_per_token

    arrival_rate = Symbol("lambda", positive=True)

    input_rate = arrival_rate * avg_prompt_len
    output_rate = arrival_rate * avg_output_len

    prefill_util = input_rate / prefill_capacity
    decode_util = output_rate / decode_capacity

    max_throughput = sympy.Min(
        prefill_capacity * avg_output_len / avg_prompt_len,
        decode_capacity,
    )

    return max_throughput, symbols


def optimize_pd_ratio():
    throughput_expr, symbols = build_throughput_model()

    frozen_params = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.kv_bytes_per_token: 256,
        symbols.chunk_size: 8192,
        symbols.batch_cap_tokens: 256,
    }

    optimizer = OptimizerInterface(
        objective=throughput_expr,
        symbols=symbols,
        maximize=True,
    )
    optimizer.set_frozen_params(frozen_params)

    result = optimizer.grid_search(
        param_grid={
            symbols.N_p: [1, 2, 3, 4, 5, 6, 7, 8],
            symbols.N_d: [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    print("\n=== P/D Ratio Optimization Results ===")
    print(f"Optimal N_p: {result.optimal_values.get('N_p', 'N/A')}")
    print(f"Optimal N_d: {result.optimal_values.get('N_d', 'N/A')}")
    print(f"Optimal throughput: {result.optimal_objective:.2f} tokens/sec")

    print("\n=== Top 5 Configurations ===")
    sorted_evals = sorted(result.all_evaluations, key=lambda x: x[1], reverse=True)[:5]
    for i, (config, throughput) in enumerate(sorted_evals, 1):
        print(f"{i}. N_p={config['N_p']}, N_d={config['N_d']}: {throughput:.2f} tok/s")

    print("\n=== Symbolic Throughput Expression ===")
    print(f"TP = {throughput_expr}")

    return result


def analyze_scaling():
    throughput_expr, symbols = build_throughput_model()

    frozen_base = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.kv_bytes_per_token: 256,
        symbols.chunk_size: 8192,
        symbols.batch_cap_tokens: 256,
        symbols.N_p: 4,
        symbols.N_d: 4,
    }

    print("\n=== Throughput vs Prompt Length ===")
    for prompt_len in [128, 256, 512, 1024, 2048, 4096]:
        params = frozen_base.copy()
        params[symbols.avg_prompt_len] = prompt_len
        params[symbols.avg_output_len] = 128

        tp = float(throughput_expr.subs(list(params.items())))
        print(f"Prompt len {prompt_len:4d}: {tp:10.2f} tok/s")

    print("\n=== Throughput vs Output Length ===")
    for output_len in [32, 64, 128, 256, 512, 1024]:
        params = frozen_base.copy()
        params[symbols.avg_prompt_len] = 512
        params[symbols.avg_output_len] = output_len

        tp = float(throughput_expr.subs(list(params.items())))
        print(f"Output len {output_len:4d}: {tp:10.2f} tok/s")


if __name__ == "__main__":
    optimize_pd_ratio()
    analyze_scaling()

