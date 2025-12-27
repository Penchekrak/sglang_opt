#!/usr/bin/env python3
"""Example demonstrating chunk size parameter sweep and its effect on TTFT and throughput."""

import sys
sys.path.insert(0, "..")

import sympy
from sympy import Symbol, ceiling

from sim.symbolic.symbols import ConfigSymbols
from sim.symbolic.expr import SymVal
from sim.config.model import ModelConfig
from sim.config.cluster import GPUSpec
from sim.models.operator_graph import OperatorGraph


def analyze_chunk_size_tradeoff():
    symbols = ConfigSymbols()
    model = ModelConfig.llama_7b()
    gpu = GPUSpec.h100_sxm()

    chunk_size = symbols.chunk_size
    prompt_len = symbols.avg_prompt_len

    prefill_flops_per_chunk = (
        4 * model.num_heads * chunk_size * chunk_size * model.head_dim +
        4 * chunk_size * model.hidden_dim * model.actual_intermediate_dim
    ) * model.num_layers

    prefill_time_per_chunk = prefill_flops_per_chunk / gpu.flops_fp16

    num_chunks = ceiling(prompt_len / chunk_size)
    total_prefill_time = num_chunks * prefill_time_per_chunk

    ttft = prefill_time_per_chunk

    print("=== Chunk Size Analysis ===")
    print(f"Model: {model.name}")
    print(f"GPU: {gpu.name}")
    print(f"GPU FLOPS: {gpu.flops_fp16/1e12:.1f} TFLOPS")
    print()

    print("=== TTFT vs Chunk Size (for 1024 token prompt) ===")
    prompt_val = 1024
    for cs in [512, 1024, 2048, 4096, 8192, 16384]:
        ttft_val = float(ttft.subs({chunk_size: cs}))
        num_chunks_val = int(ceiling(prompt_val / cs).evalf())
        total_time = float(total_prefill_time.subs({chunk_size: cs, prompt_len: prompt_val}))

        print(f"Chunk size {cs:5d}: TTFT={ttft_val*1000:7.2f}ms, "
              f"Chunks={num_chunks_val}, Total Prefill={total_time*1000:.2f}ms")

    print()
    print("=== Prefill Time vs Prompt Length (chunk_size=8192) ===")
    for pl in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        total_time = float(total_prefill_time.subs({chunk_size: 8192, prompt_len: pl}))
        num_chunks_val = int(ceiling(pl / 8192).evalf())
        print(f"Prompt len {pl:5d}: {total_time*1000:8.2f}ms ({num_chunks_val} chunks)")


def sweep_with_cache_effects():
    symbols = ConfigSymbols()
    model = ModelConfig.llama_7b()
    gpu = GPUSpec.h100_sxm()

    chunk_size = symbols.chunk_size
    cache_hit_rate = symbols.cache_hit_rate
    prompt_len = symbols.avg_prompt_len

    effective_prompt = prompt_len * (1 - cache_hit_rate)

    prefill_flops_per_token = (
        4 * model.num_heads * model.head_dim +
        4 * model.hidden_dim * model.actual_intermediate_dim
    ) * model.num_layers

    effective_flops = effective_prompt * prefill_flops_per_token * effective_prompt
    prefill_time = effective_flops / gpu.flops_fp16

    print("\n=== Cache Hit Rate Impact on TTFT ===")
    print("(Prompt len = 1024, Chunk size = 8192)")

    for hit_rate in [0.0, 0.25, 0.5, 0.75, 0.9]:
        time_val = float(prefill_time.subs({
            prompt_len: 1024,
            cache_hit_rate: hit_rate,
        }))
        speedup = 1.0 / (1.0 - hit_rate) if hit_rate < 1.0 else float('inf')
        print(f"Cache hit rate {hit_rate:.0%}: {time_val*1000:8.4f}ms (speedup: {speedup:.2f}x)")


def run_operator_graph_analysis():
    model = ModelConfig.llama_7b()
    gpu = GPUSpec.h100_sxm()
    op_graph = OperatorGraph()

    print("\n=== Operator-Level Latency Analysis ===")
    print(f"Model: {model.name}")

    for batch_tokens in [128, 256, 512, 1024, 2048, 4096]:
        prefill_cost = op_graph.prefill_iteration(
            batch_tokens=batch_tokens,
            model=model,
            gpu=gpu,
        )
        print(f"Prefill {batch_tokens:4d} tokens: {prefill_cost.val*1000:8.3f}ms")

    print()
    for batch_size in [1, 4, 16, 64, 128, 256]:
        kv_lengths = [512] * batch_size
        decode_cost = op_graph.decode_iteration(
            batch_size=batch_size,
            kv_lengths=kv_lengths,
            model=model,
            gpu=gpu,
        )
        print(f"Decode batch {batch_size:3d} (kv=512): {decode_cost.val*1000:8.3f}ms, "
              f"{batch_size/(decode_cost.val*1000):.2f} tok/ms")


if __name__ == "__main__":
    analyze_chunk_size_tradeoff()
    sweep_with_cache_effects()
    run_operator_graph_analysis()

