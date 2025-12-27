from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_max
from sim.parallel.groups import EPGroup

if TYPE_CHECKING:
    from sim.config.cluster import GPUSpec
    from sim.config.model import ModelConfig


class MoEKernel:
    def gating(
        self,
        batch_tokens: int,
        hidden_dim: int,
        num_experts: int,
        gpu: GPUSpec,
    ) -> SymVal:
        gating_flops = 2 * batch_tokens * hidden_dim * num_experts
        compute_time = gating_flops / gpu.flops_fp16
        return sym_const(compute_time, "t_gating")

    def expert_forward(
        self,
        tokens_per_expert: float,
        hidden_dim: int,
        intermediate_dim: int,
        num_local_experts: int,
        gpu: GPUSpec,
    ) -> SymVal:
        flops_per_expert = 4 * tokens_per_expert * hidden_dim * intermediate_dim
        total_flops = flops_per_expert * num_local_experts
        compute_time = total_flops / gpu.flops_fp16
        return sym_const(compute_time, "t_expert_compute")

    def moe_layer(
        self,
        batch_tokens: int,
        model: ModelConfig,
        gpu: GPUSpec,
        ep_group: EPGroup | None = None,
    ) -> SymVal:
        gating_cost = self.gating(
            batch_tokens,
            model.hidden_dim,
            model.num_experts,
            gpu,
        )

        if ep_group is not None and ep_group.size > 1:
            dispatch_cost = ep_group.dispatch_cost(batch_tokens, model.hidden_dim)

            experts_per_rank = model.num_experts // ep_group.size
            tokens_per_expert = batch_tokens * model.top_k_experts / model.num_experts

            expert_cost = self.expert_forward(
                tokens_per_expert,
                model.hidden_dim,
                model.actual_intermediate_dim,
                experts_per_rank,
                gpu,
            )

            combine_cost = ep_group.combine_cost(batch_tokens, model.hidden_dim)

            return sym_add(
                gating_cost,
                sym_add(dispatch_cost, sym_add(expert_cost, combine_cost)),
            )
        else:
            tokens_per_expert = batch_tokens * model.top_k_experts / model.num_experts

            expert_cost = self.expert_forward(
                tokens_per_expert,
                model.hidden_dim,
                model.actual_intermediate_dim,
                model.num_experts,
                gpu,
            )

            return sym_add(gating_cost, expert_cost)

    def full_moe_iteration(
        self,
        batch_tokens: int,
        model: ModelConfig,
        gpu: GPUSpec,
        ep_group: EPGroup | None,
        dp_group_size: int = 1,
    ) -> SymVal:
        moe_cost = self.moe_layer(batch_tokens, model, gpu, ep_group)

        if dp_group_size > 1 and ep_group is not None:
            ag_before_moe = sym_const(
                batch_tokens * model.hidden_dim * 2 * (dp_group_size - 1) /
                (ep_group.interconnect_bandwidth * dp_group_size),
                "t_ag_before_moe",
            )
            scatter_after_moe = sym_const(
                batch_tokens * model.hidden_dim * 2 * (dp_group_size - 1) /
                (ep_group.interconnect_bandwidth * dp_group_size),
                "t_scatter_after_moe",
            )
            return sym_add(ag_before_moe, sym_add(moe_cost, scatter_after_moe))

        return moe_cost

