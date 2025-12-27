from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_sum
from sim.kernels.attention import AttentionKernel
from sim.kernels.mlp import MLPKernel
from sim.kernels.moe import MoEKernel
import sympy

if TYPE_CHECKING:
    from sim.config.model import ModelConfig
    from sim.config.cluster import GPUSpec
    from sim.parallel.groups import TPGroup, DPGroup, EPGroup


class OperatorGraph:
    def __init__(self):
        self.attention_kernel = AttentionKernel()
        self.mlp_kernel = MLPKernel()
        self.moe_kernel = MoEKernel()

    def prefill_iteration(
        self,
        batch_tokens: int,
        model: ModelConfig,
        gpu: GPUSpec,
        tp_group: TPGroup | None = None,
        ep_group: EPGroup | None = None,
    ) -> SymVal:
        tp_size = tp_group.size if tp_group else 1

        attn_cost = self.attention_kernel.prefill_attention(
            seq_len=batch_tokens,
            batch_size=1,
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            gpu=gpu,
        )

        if model.is_moe and model.num_experts > 1:
            ffn_cost = self.moe_kernel.moe_layer(
                batch_tokens=batch_tokens,
                model=model,
                gpu=gpu,
                ep_group=ep_group,
            )
        else:
            ffn_cost = self.mlp_kernel.layer_forward(
                batch_tokens=batch_tokens,
                model=model,
                gpu=gpu,
                tp_size=tp_size,
            )

        layer_cost = sym_add(attn_cost, ffn_cost)

        if tp_group is not None and tp_group.size > 1:
            ar_bytes = batch_tokens * model.hidden_dim * 2
            ar_cost = tp_group.all_reduce_cost(ar_bytes)
            layer_cost = sym_add(layer_cost, ar_cost)

        total_cost = layer_cost * model.num_layers

        return SymVal(
            expr=total_cost.expr,
            val=total_cost.val,
            meta={"phase": "prefill", "layers": model.num_layers},
        )

    def decode_iteration(
        self,
        batch_size: int,
        kv_lengths: list[int],
        model: ModelConfig,
        gpu: GPUSpec,
        tp_group: TPGroup | None = None,
        ep_group: EPGroup | None = None,
    ) -> SymVal:
        tp_size = tp_group.size if tp_group else 1
        avg_kv_len = sum(kv_lengths) / len(kv_lengths) if kv_lengths else 512

        attn_cost = self.attention_kernel.decode_attention(
            batch_size=batch_size,
            kv_lengths=kv_lengths,
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            gpu=gpu,
        )

        if model.is_moe and model.num_experts > 1:
            ffn_cost = self.moe_kernel.moe_layer(
                batch_tokens=batch_size,
                model=model,
                gpu=gpu,
                ep_group=ep_group,
            )
        else:
            ffn_cost = self.mlp_kernel.layer_forward(
                batch_tokens=batch_size,
                model=model,
                gpu=gpu,
                tp_size=tp_size,
            )

        layer_cost = sym_add(attn_cost, ffn_cost)

        if tp_group is not None and tp_group.size > 1:
            ar_bytes = batch_size * model.hidden_dim * 2
            ar_cost = tp_group.all_reduce_cost(ar_bytes)
            layer_cost = sym_add(layer_cost, ar_cost)

        total_cost = layer_cost * model.num_layers

        return SymVal(
            expr=total_cost.expr,
            val=total_cost.val,
            meta={"phase": "decode", "layers": model.num_layers, "batch_size": batch_size},
        )

    def full_forward_symbolic(
        self,
        prompt_len: sympy.Symbol,
        output_len: sympy.Symbol,
        model: ModelConfig,
        gpu: GPUSpec,
        chunk_size: sympy.Symbol,
        batch_cap: sympy.Symbol,
    ) -> SymVal:
        prefill_iters = sympy.ceiling(prompt_len / chunk_size)

        prefill_flops_per_chunk = (
            4 * model.num_heads * chunk_size * chunk_size * model.head_dim +
            4 * chunk_size * model.hidden_dim * model.actual_intermediate_dim
        ) * model.num_layers

        prefill_time_per_chunk = prefill_flops_per_chunk / gpu.flops_fp16
        total_prefill_time = prefill_iters * prefill_time_per_chunk

        decode_kv_bytes_per_iter = (prompt_len + output_len / 2) * model.kv_bytes_per_token
        decode_time_per_token = decode_kv_bytes_per_iter / gpu.memory_bandwidth
        total_decode_time = output_len * decode_time_per_token

        total_time = total_prefill_time + total_decode_time
        total_tokens = prompt_len + output_len

        return SymVal(
            expr=total_time,
            val=0.0,
            meta={"type": "full_forward_symbolic"},
        )

    def e2e_latency_expression(
        self,
        prompt_len: int,
        output_len: int,
        model: ModelConfig,
        gpu: GPUSpec,
        tp_group: TPGroup | None = None,
        ep_group: EPGroup | None = None,
    ) -> SymVal:
        prefill_cost = self.prefill_iteration(
            batch_tokens=prompt_len,
            model=model,
            gpu=gpu,
            tp_group=tp_group,
            ep_group=ep_group,
        )

        kv_lengths = [prompt_len + i for i in range(output_len)]
        total_decode_cost = sym_const(0.0)

        for i in range(output_len):
            decode_cost = self.decode_iteration(
                batch_size=1,
                kv_lengths=[prompt_len + i],
                model=model,
                gpu=gpu,
                tp_group=tp_group,
                ep_group=ep_group,
            )
            total_decode_cost = sym_add(total_decode_cost, decode_cost)

        return sym_add(prefill_cost, total_decode_cost)

