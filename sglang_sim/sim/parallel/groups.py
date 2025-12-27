from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_max
from sim.parallel.collectives import CollectiveOp, collective_cost

if TYPE_CHECKING:
    from sim.config.cluster import InterconnectConfig, GPUSpec
    from sim.config.model import ModelConfig


@dataclass
class TPGroup:
    size: int
    interconnect_bandwidth: float  # bytes/sec
    interconnect_latency: float  # seconds

    def compute_cost(
        self,
        flops: int,
        gpu_flops: float,
    ) -> SymVal:
        compute_time = flops / (gpu_flops * self.size)
        return sym_const(compute_time, "t_compute_tp")

    def all_reduce_cost(self, message_bytes: int) -> SymVal:
        return collective_cost(
            CollectiveOp.ALL_REDUCE,
            message_bytes,
            self.size,
            self.interconnect_bandwidth,
            self.interconnect_latency,
        )

    def all_gather_cost(self, message_bytes: int) -> SymVal:
        return collective_cost(
            CollectiveOp.ALL_GATHER,
            message_bytes,
            self.size,
            self.interconnect_bandwidth,
            self.interconnect_latency,
        )

    def reduce_scatter_cost(self, message_bytes: int) -> SymVal:
        return collective_cost(
            CollectiveOp.REDUCE_SCATTER,
            message_bytes,
            self.size,
            self.interconnect_bandwidth,
            self.interconnect_latency,
        )

    def layer_cost(
        self,
        batch_tokens: int,
        model: ModelConfig,
        gpu: GPUSpec,
        is_prefill: bool = True,
    ) -> SymVal:
        qkv_flops = 3 * batch_tokens * model.hidden_dim * model.hidden_dim
        proj_flops = batch_tokens * model.hidden_dim * model.hidden_dim

        if is_prefill:
            attn_flops = 2 * batch_tokens * batch_tokens * model.hidden_dim
        else:
            avg_kv_len = batch_tokens * 512
            attn_flops = 2 * batch_tokens * avg_kv_len * model.hidden_dim / batch_tokens

        mlp_flops = 2 * batch_tokens * model.hidden_dim * model.actual_intermediate_dim

        total_flops = qkv_flops + proj_flops + attn_flops + mlp_flops
        compute_time = self.compute_cost(int(total_flops), gpu.flops_fp16)

        ar_bytes_attn = batch_tokens * model.hidden_dim * 2
        ar_bytes_mlp = batch_tokens * model.hidden_dim * 2
        ar_cost = sym_add(
            self.all_reduce_cost(ar_bytes_attn),
            self.all_reduce_cost(ar_bytes_mlp),
        )

        return sym_add(compute_time, ar_cost)

    @classmethod
    def from_config(cls, size: int, interconnect: InterconnectConfig) -> TPGroup:
        return cls(
            size=size,
            interconnect_bandwidth=interconnect.bandwidth_bytes_per_sec,
            interconnect_latency=interconnect.latency_seconds,
        )


@dataclass
class DPGroup:
    size: int
    interconnect_bandwidth: float
    interconnect_latency: float

    def attention_all_gather_cost(self, hidden_dim: int, batch_tokens: int) -> SymVal:
        message_bytes = batch_tokens * hidden_dim * 2
        return collective_cost(
            CollectiveOp.ALL_GATHER,
            message_bytes,
            self.size,
            self.interconnect_bandwidth,
            self.interconnect_latency,
        )

    def dp_attention_cost(
        self,
        batch_tokens: int,
        hidden_dim: int,
    ) -> SymVal:
        ag_cost = self.attention_all_gather_cost(hidden_dim, batch_tokens)
        return ag_cost

    @classmethod
    def from_config(cls, size: int, interconnect: InterconnectConfig) -> DPGroup:
        return cls(
            size=size,
            interconnect_bandwidth=interconnect.bandwidth_bytes_per_sec,
            interconnect_latency=interconnect.latency_seconds,
        )


@dataclass
class EPGroup:
    size: int
    interconnect_bandwidth: float
    interconnect_latency: float
    num_experts: int
    top_k: int

    def dispatch_cost(self, tokens: int, hidden_dim: int) -> SymVal:
        message_bytes = tokens * hidden_dim * 2 * self.top_k / self.size
        return collective_cost(
            CollectiveOp.ALL_TO_ALL,
            int(message_bytes),
            self.size,
            self.interconnect_bandwidth,
            self.interconnect_latency,
        )

    def combine_cost(self, tokens: int, hidden_dim: int) -> SymVal:
        message_bytes = tokens * hidden_dim * 2 * self.top_k / self.size
        return collective_cost(
            CollectiveOp.ALL_TO_ALL,
            int(message_bytes),
            self.size,
            self.interconnect_bandwidth,
            self.interconnect_latency,
        )

    def moe_layer_cost(
        self,
        tokens: int,
        hidden_dim: int,
        intermediate_dim: int,
        gpu: GPUSpec,
    ) -> SymVal:
        dispatch = self.dispatch_cost(tokens, hidden_dim)

        experts_per_rank = self.num_experts // self.size
        tokens_per_expert = tokens * self.top_k / self.num_experts
        expert_flops = 2 * tokens_per_expert * hidden_dim * intermediate_dim * 2
        total_expert_flops = expert_flops * experts_per_rank
        compute_time = sym_const(total_expert_flops / gpu.flops_fp16, "t_expert_compute")

        combine = self.combine_cost(tokens, hidden_dim)

        return sym_add(sym_add(dispatch, compute_time), combine)

    @classmethod
    def from_config(
        cls,
        size: int,
        interconnect: InterconnectConfig,
        num_experts: int,
        top_k: int,
    ) -> EPGroup:
        return cls(
            size=size,
            interconnect_bandwidth=interconnect.bandwidth_bytes_per_sec,
            interconnect_latency=interconnect.latency_seconds,
            num_experts=num_experts,
            top_k=top_k,
        )

