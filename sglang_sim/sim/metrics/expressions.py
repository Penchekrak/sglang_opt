from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sympy
from sympy import Symbol, Max, Min, ceiling, Piecewise, sqrt

from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_max, sym_min, sym_div
from sim.symbolic.symbols import ConfigSymbols

if TYPE_CHECKING:
    pass


@dataclass
class MetricExpressions:
    ttft: sympy.Expr
    tpot: sympy.Expr
    e2e_latency: sympy.Expr
    throughput: sympy.Expr
    peak_memory: sympy.Expr
    compute_utilization: sympy.Expr
    memory_bandwidth_utilization: sympy.Expr
    network_utilization: sympy.Expr
    cost_per_token: sympy.Expr
    power_consumption: sympy.Expr
    
    def as_dict(self) -> dict[str, sympy.Expr]:
        return {
            "ttft": self.ttft,
            "tpot": self.tpot,
            "e2e_latency": self.e2e_latency,
            "throughput": self.throughput,
            "peak_memory": self.peak_memory,
            "compute_utilization": self.compute_utilization,
            "memory_bandwidth_utilization": self.memory_bandwidth_utilization,
            "network_utilization": self.network_utilization,
            "cost_per_token": self.cost_per_token,
            "power_consumption": self.power_consumption,
        }


class SymbolicMetricBuilder:
    def __init__(self, symbols: ConfigSymbols | None = None):
        self.symbols = symbols or ConfigSymbols()
        self._cache: dict[str, sympy.Expr] = {}

    def ttft_expression(self) -> sympy.Expr:
        if "ttft" in self._cache:
            return self._cache["ttft"]
        
        s = self.symbols
        
        effective_prompt = s.avg_prompt_len * (1 - s.cache_hit_rate * s.avg_prefix_match)
        
        prefill_flops_per_token = (
            4 * s.num_heads * s.head_dim * s.avg_prompt_len +
            8 * s.hidden_dim * s.hidden_dim
        ) * s.num_layers
        
        chunk_tokens = Min(s.chunk_size, effective_prompt)
        chunk_flops = chunk_tokens * prefill_flops_per_token
        
        compute_time = chunk_flops / (s.gpu_flops * s.tp_size)
        
        kv_write_bytes = chunk_tokens * 2 * s.num_layers * s.num_heads * s.head_dim * 2
        memory_time = kv_write_bytes / s.gpu_mem_bw
        
        ttft = Max(compute_time, memory_time)
        
        queue_wait = s.avg_prompt_len / (s.gpu_flops / (8 * s.hidden_dim * s.hidden_dim * s.num_layers))
        
        self._cache["ttft"] = ttft
        return ttft

    def tpot_expression(self) -> sympy.Expr:
        if "tpot" in self._cache:
            return self._cache["tpot"]
        
        s = self.symbols
        
        avg_kv_len = s.avg_prompt_len + s.avg_output_len / 2
        
        kv_read_bytes = avg_kv_len * 2 * s.num_layers * s.num_heads * s.head_dim * 2
        memory_time = kv_read_bytes / s.gpu_mem_bw
        
        decode_flops = (
            4 * s.num_heads * s.head_dim * avg_kv_len +
            8 * s.hidden_dim * s.hidden_dim
        ) * s.num_layers
        compute_time = decode_flops / (s.gpu_flops * s.tp_size)
        
        tpot = Max(memory_time, compute_time) / s.batch_cap_requests
        
        self._cache["tpot"] = tpot
        return tpot

    def e2e_latency_expression(self) -> sympy.Expr:
        if "e2e_latency" in self._cache:
            return self._cache["e2e_latency"]
        
        s = self.symbols
        
        num_prefill_chunks = ceiling(s.avg_prompt_len / s.chunk_size)
        
        prefill_time_per_chunk = self.ttft_expression()
        total_prefill = num_prefill_chunks * prefill_time_per_chunk
        
        kv_transfer_bytes = s.avg_prompt_len * 2 * s.num_layers * s.num_heads * s.head_dim * 2
        transfer_time = s.network_latency + kv_transfer_bytes / s.network_bw
        
        tpot = self.tpot_expression()
        total_decode = s.avg_output_len * tpot
        
        e2e = total_prefill + transfer_time + total_decode
        
        self._cache["e2e_latency"] = e2e
        return e2e

    def throughput_expression(self) -> sympy.Expr:
        if "throughput" in self._cache:
            return self._cache["throughput"]
        
        s = self.symbols
        
        prefill_flops = (
            4 * s.num_heads * s.head_dim * s.avg_prompt_len +
            8 * s.hidden_dim * s.hidden_dim
        ) * s.num_layers * s.avg_prompt_len
        
        prefill_capacity = s.N_p * s.gpu_flops * s.tp_size / prefill_flops
        
        avg_kv_len = s.avg_prompt_len + s.avg_output_len / 2
        kv_bytes = avg_kv_len * 2 * s.num_layers * s.num_heads * s.head_dim * 2
        decode_time_per_batch = kv_bytes / s.gpu_mem_bw
        
        decode_capacity = s.N_d * s.batch_cap_requests / decode_time_per_batch
        
        throughput = Min(
            prefill_capacity * s.avg_output_len,
            decode_capacity,
        )
        
        self._cache["throughput"] = throughput
        return throughput

    def peak_memory_expression(self) -> sympy.Expr:
        if "peak_memory" in self._cache:
            return self._cache["peak_memory"]
        
        s = self.symbols
        
        vocab_size = 32000
        weights = (
            2 * vocab_size * s.hidden_dim +
            s.num_layers * (
                4 * s.hidden_dim * s.hidden_dim +
                s.num_experts * 3 * s.hidden_dim * s.hidden_dim * 4
            )
        ) * 2 / s.tp_size
        
        max_seq = s.avg_prompt_len + s.avg_output_len
        kv_cache = (
            s.batch_cap_requests * max_seq *
            2 * s.num_layers * s.num_heads * s.head_dim * 2 / s.tp_size
        )
        
        activations = 4 * s.chunk_size * s.hidden_dim * 2
        
        peak_memory = weights + kv_cache + activations
        
        self._cache["peak_memory"] = peak_memory
        return peak_memory

    def compute_utilization_expression(self) -> sympy.Expr:
        if "compute_util" in self._cache:
            return self._cache["compute_util"]
        
        s = self.symbols
        
        achieved_throughput = self.throughput_expression()
        
        flops_per_token = (
            4 * s.num_heads * s.head_dim * (s.avg_prompt_len + s.avg_output_len) / 2 +
            8 * s.hidden_dim * s.hidden_dim
        ) * s.num_layers
        
        achieved_flops = achieved_throughput * flops_per_token
        
        total_gpu_flops = (s.N_p + s.N_d) * s.gpu_flops * s.tp_size
        
        utilization = achieved_flops / total_gpu_flops
        
        self._cache["compute_util"] = utilization
        return utilization

    def memory_bandwidth_utilization_expression(self) -> sympy.Expr:
        if "mem_bw_util" in self._cache:
            return self._cache["mem_bw_util"]
        
        s = self.symbols
        
        achieved_throughput = self.throughput_expression()
        
        avg_kv_len = s.avg_prompt_len + s.avg_output_len / 2
        bytes_per_token = avg_kv_len * 2 * s.num_layers * s.num_heads * s.head_dim * 2
        
        achieved_bandwidth = achieved_throughput * bytes_per_token
        
        total_bandwidth = (s.N_p + s.N_d) * s.gpu_mem_bw
        
        utilization = achieved_bandwidth / total_bandwidth
        
        self._cache["mem_bw_util"] = utilization
        return utilization

    def network_utilization_expression(self) -> sympy.Expr:
        if "net_util" in self._cache:
            return self._cache["net_util"]
        
        s = self.symbols
        
        achieved_throughput = self.throughput_expression()
        requests_per_sec = achieved_throughput / s.avg_output_len
        
        kv_bytes_per_request = s.avg_prompt_len * 2 * s.num_layers * s.num_heads * s.head_dim * 2
        
        achieved_network = requests_per_sec * kv_bytes_per_request
        
        available_network = Min(s.N_p, s.N_d) * s.network_bw
        
        utilization = achieved_network / available_network
        
        self._cache["net_util"] = utilization
        return utilization

    def cost_per_token_expression(
        self,
        gpu_cost_per_hour: float = 2.0,
    ) -> sympy.Expr:
        if "cost_per_token" in self._cache:
            return self._cache["cost_per_token"]
        
        s = self.symbols
        
        total_gpus = (s.N_p + s.N_d) * s.tp_size
        
        cost_per_second = total_gpus * gpu_cost_per_hour / 3600
        
        throughput = self.throughput_expression()
        
        cost_per_token = cost_per_second / throughput
        
        self._cache["cost_per_token"] = cost_per_token
        return cost_per_token

    def power_consumption_expression(
        self,
        gpu_tdp_watts: float = 700,
    ) -> sympy.Expr:
        if "power" in self._cache:
            return self._cache["power"]
        
        s = self.symbols
        
        total_gpus = (s.N_p + s.N_d) * s.tp_size
        
        compute_util = self.compute_utilization_expression()
        
        idle_power_fraction = 0.3
        active_power = total_gpus * gpu_tdp_watts * (
            idle_power_fraction + (1 - idle_power_fraction) * compute_util
        )
        
        self._cache["power"] = active_power
        return active_power

    def goodput_expression(
        self,
        slo_ttft_seconds: float = 0.5,
        slo_tpot_seconds: float = 0.05,
    ) -> sympy.Expr:
        s = self.symbols
        
        ttft = self.ttft_expression()
        tpot = self.tpot_expression()
        throughput = self.throughput_expression()
        
        ttft_ok = Piecewise((1, ttft <= slo_ttft_seconds), (0, True))
        tpot_ok = Piecewise((1, tpot <= slo_tpot_seconds), (0, True))
        
        goodput = throughput * ttft_ok * tpot_ok
        
        return goodput

    def build_all_expressions(self) -> MetricExpressions:
        return MetricExpressions(
            ttft=self.ttft_expression(),
            tpot=self.tpot_expression(),
            e2e_latency=self.e2e_latency_expression(),
            throughput=self.throughput_expression(),
            peak_memory=self.peak_memory_expression(),
            compute_utilization=self.compute_utilization_expression(),
            memory_bandwidth_utilization=self.memory_bandwidth_utilization_expression(),
            network_utilization=self.network_utilization_expression(),
            cost_per_token=self.cost_per_token_expression(),
            power_consumption=self.power_consumption_expression(),
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_expression(self, metric_name: str) -> sympy.Expr:
        builders = {
            "ttft": self.ttft_expression,
            "tpot": self.tpot_expression,
            "e2e_latency": self.e2e_latency_expression,
            "throughput": self.throughput_expression,
            "peak_memory": self.peak_memory_expression,
            "compute_utilization": self.compute_utilization_expression,
            "memory_bandwidth_utilization": self.memory_bandwidth_utilization_expression,
            "network_utilization": self.network_utilization_expression,
            "cost_per_token": self.cost_per_token_expression,
            "power_consumption": self.power_consumption_expression,
        }
        
        if metric_name not in builders:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(builders.keys())}")
        
        return builders[metric_name]()

