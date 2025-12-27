from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import sympy
from sympy import Symbol, Piecewise, Sum, ceiling

from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_piecewise
from sim.symbolic.symbols import ConfigSymbols

if TYPE_CHECKING:
    from sim.config.model import ModelConfig
    from sim.config.cluster import GPUSpec


class SymbolicThroughputBuilder:
    def __init__(self, symbols: ConfigSymbols | None = None):
        self.symbols = symbols or ConfigSymbols()
        self.accumulated_time: sympy.Expr = sympy.Integer(0)
        self.accumulated_tokens: sympy.Expr = sympy.Integer(0)

    def build_ttft_expression(
        self,
        prompt_len: int | sympy.Symbol,
        chunk_size: sympy.Symbol,
        prefill_time_per_chunk: sympy.Expr,
        cache_hit_ratio: sympy.Symbol,
        cache_speedup: float = 0.5,
    ) -> sympy.Expr:
        effective_prompt = prompt_len * (1 - cache_hit_ratio * cache_speedup)
        num_chunks = ceiling(effective_prompt / chunk_size)
        ttft = num_chunks * prefill_time_per_chunk
        return ttft

    def build_tpot_expression(
        self,
        avg_kv_len: sympy.Symbol,
        kv_bytes_per_token: sympy.Symbol,
        memory_bandwidth: sympy.Symbol,
        batch_size: sympy.Symbol,
    ) -> sympy.Expr:
        kv_read_bytes = avg_kv_len * kv_bytes_per_token
        memory_time = kv_read_bytes / memory_bandwidth
        tpot = memory_time / batch_size
        return tpot

    def build_request_latency(
        self,
        prompt_len: int,
        output_len: int,
        model_params: dict,
        gpu_params: dict,
    ) -> SymVal:
        s = self.symbols

        prefill_flops = (
            4 * s.num_heads * prompt_len * prompt_len * s.head_dim +
            4 * prompt_len * s.hidden_dim * s.hidden_dim * 4
        ) * s.num_layers

        prefill_time = prefill_flops / s.gpu_flops

        decode_kv_read_per_token = (prompt_len + output_len / 2) * s.kv_bytes_per_token
        decode_time_per_token = decode_kv_read_per_token / s.gpu_mem_bw
        total_decode_time = output_len * decode_time_per_token

        total_time = prefill_time + total_decode_time

        numeric_prefill = float(prefill_time.subs([
            (s.num_heads, model_params.get("num_heads", 32)),
            (s.head_dim, model_params.get("head_dim", 128)),
            (s.hidden_dim, model_params.get("hidden_dim", 4096)),
            (s.num_layers, model_params.get("num_layers", 32)),
            (s.gpu_flops, gpu_params.get("flops", 1e15)),
            (s.kv_bytes_per_token, model_params.get("kv_bytes", 256)),
            (s.gpu_mem_bw, gpu_params.get("bandwidth", 2e12)),
        ]))

        numeric_decode = float(total_decode_time.subs([
            (s.kv_bytes_per_token, model_params.get("kv_bytes", 256)),
            (s.gpu_mem_bw, gpu_params.get("bandwidth", 2e12)),
        ]))

        return SymVal(
            expr=total_time,
            val=numeric_prefill + numeric_decode,
            meta={"type": "request_latency", "prompt_len": prompt_len, "output_len": output_len},
        )

    def build_throughput(
        self,
        total_tokens: sympy.Expr,
        total_time: sympy.Expr,
    ) -> sympy.Expr:
        return total_tokens / total_time

    def build_system_throughput(
        self,
        arrival_rate: sympy.Symbol,
        avg_prompt_len: sympy.Symbol,
        avg_output_len: sympy.Symbol,
        num_prefill_workers: sympy.Symbol,
        num_decode_workers: sympy.Symbol,
        prefill_capacity: sympy.Expr,
        decode_capacity: sympy.Expr,
    ) -> sympy.Expr:
        input_token_rate = arrival_rate * avg_prompt_len
        output_token_rate = arrival_rate * avg_output_len

        prefill_utilization = input_token_rate / (num_prefill_workers * prefill_capacity)
        decode_utilization = output_token_rate / (num_decode_workers * decode_capacity)

        prefill_bottleneck = prefill_utilization >= 1
        decode_bottleneck = decode_utilization >= 1

        max_throughput_prefill = num_prefill_workers * prefill_capacity / avg_prompt_len
        max_throughput_decode = num_decode_workers * decode_capacity

        return sympy.Min(
            max_throughput_prefill * avg_output_len,
            max_throughput_decode,
        )

    def export_lambdified(
        self,
        expr: sympy.Expr,
        symbols: list[Symbol] | None = None,
    ) -> Callable[..., float]:
        if symbols is None:
            symbols = self.symbols.all_symbols()
        return sympy.lambdify(symbols, expr, modules=["numpy"])

    def add_request_to_trace(
        self,
        prompt_len: int,
        output_len: int,
        ttft: sympy.Expr,
        itl: sympy.Expr,
    ) -> None:
        request_time = ttft + itl * (output_len - 1)
        request_tokens = output_len

        self.accumulated_time = self.accumulated_time + request_time
        self.accumulated_tokens = self.accumulated_tokens + request_tokens

    def get_accumulated_throughput(self) -> sympy.Expr:
        if self.accumulated_time == 0:
            return sympy.Integer(0)
        return self.accumulated_tokens / self.accumulated_time

    def reset(self) -> None:
        self.accumulated_time = sympy.Integer(0)
        self.accumulated_tokens = sympy.Integer(0)

