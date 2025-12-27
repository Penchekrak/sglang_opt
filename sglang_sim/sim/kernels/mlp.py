from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_max

if TYPE_CHECKING:
    from sim.config.cluster import GPUSpec
    from sim.config.model import ModelConfig


class MLPKernel:
    def forward(
        self,
        batch_tokens: int,
        hidden_dim: int,
        intermediate_dim: int,
        gpu: GPUSpec,
        tp_size: int = 1,
    ) -> SymVal:
        up_proj_flops = 2 * batch_tokens * hidden_dim * intermediate_dim
        down_proj_flops = 2 * batch_tokens * intermediate_dim * hidden_dim

        if intermediate_dim > hidden_dim * 2:
            gate_proj_flops = 2 * batch_tokens * hidden_dim * intermediate_dim
            total_flops = up_proj_flops + gate_proj_flops + down_proj_flops
        else:
            total_flops = up_proj_flops + down_proj_flops

        effective_flops = gpu.flops_fp16 * tp_size
        compute_time = total_flops / effective_flops

        weight_bytes = hidden_dim * intermediate_dim * 2 * 2
        activation_bytes = batch_tokens * (hidden_dim + intermediate_dim) * 2
        total_bytes = weight_bytes + activation_bytes

        memory_time = total_bytes / gpu.memory_bandwidth

        return sym_max(
            sym_const(compute_time, "t_mlp_compute"),
            sym_const(memory_time, "t_mlp_memory"),
        )

    def gated_mlp_forward(
        self,
        batch_tokens: int,
        hidden_dim: int,
        intermediate_dim: int,
        gpu: GPUSpec,
        tp_size: int = 1,
    ) -> SymVal:
        gate_proj_flops = 2 * batch_tokens * hidden_dim * intermediate_dim
        up_proj_flops = 2 * batch_tokens * hidden_dim * intermediate_dim
        down_proj_flops = 2 * batch_tokens * intermediate_dim * hidden_dim

        total_flops = gate_proj_flops + up_proj_flops + down_proj_flops
        effective_flops = gpu.flops_fp16 * tp_size
        compute_time = total_flops / effective_flops

        weight_bytes = 3 * hidden_dim * intermediate_dim * 2
        activation_bytes = batch_tokens * (hidden_dim + 2 * intermediate_dim) * 2
        total_bytes = weight_bytes + activation_bytes

        memory_time = total_bytes / gpu.memory_bandwidth

        return sym_max(
            sym_const(compute_time, "t_gated_mlp_compute"),
            sym_const(memory_time, "t_gated_mlp_memory"),
        )

    def layer_forward(
        self,
        batch_tokens: int,
        model: ModelConfig,
        gpu: GPUSpec,
        tp_size: int = 1,
    ) -> SymVal:
        return self.gated_mlp_forward(
            batch_tokens=batch_tokens,
            hidden_dim=model.hidden_dim,
            intermediate_dim=model.actual_intermediate_dim,
            gpu=gpu,
            tp_size=tp_size,
        )

