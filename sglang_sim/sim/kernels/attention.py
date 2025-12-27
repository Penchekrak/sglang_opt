from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_min, sym_max, sym_piecewise
import sympy

if TYPE_CHECKING:
    from sim.config.cluster import GPUSpec


@dataclass
class AttentionKernelConfig:
    max_seq_len_flash: int = 16384
    supported_head_dims: tuple[int, ...] = (64, 128, 256)


class AttentionKernel:
    def __init__(self, config: AttentionKernelConfig | None = None):
        self.config = config or AttentionKernelConfig()

    def flash_attention_feasible(self, seq_len: int, head_dim: int) -> bool:
        return (
            seq_len <= self.config.max_seq_len_flash
            and head_dim in self.config.supported_head_dims
        )

    def flash_attention(
        self,
        seq_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        gpu: GPUSpec,
    ) -> SymVal:
        flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
        compute_time = flops / gpu.flops_fp16

        io_bytes = 2 * batch_size * num_heads * seq_len * head_dim * 2  # Q, K, V, O
        io_time = io_bytes / gpu.memory_bandwidth

        return sym_max(
            sym_const(compute_time, "t_flash_compute"),
            sym_const(io_time, "t_flash_io"),
        )

    def paged_attention(
        self,
        num_tokens: int,
        kv_len: int,
        num_heads: int,
        head_dim: int,
        gpu: GPUSpec,
    ) -> SymVal:
        kv_bytes = 2 * kv_len * num_heads * head_dim * 2
        memory_time = kv_bytes / gpu.memory_bandwidth

        flops = 2 * num_tokens * kv_len * num_heads * head_dim
        compute_time = flops / gpu.flops_fp16

        return sym_max(
            sym_const(memory_time, "t_paged_mem"),
            sym_const(compute_time, "t_paged_compute"),
        )

    def prefill_attention(
        self,
        seq_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        gpu: GPUSpec,
    ) -> SymVal:
        if self.flash_attention_feasible(seq_len, head_dim):
            flash_cost = self.flash_attention(seq_len, batch_size, num_heads, head_dim, gpu)
            return flash_cost

        flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
        compute_time = flops / gpu.flops_fp16
        return sym_const(compute_time, "t_prefill_attn_fallback")

    def decode_attention(
        self,
        batch_size: int,
        kv_lengths: list[int],
        num_heads: int,
        head_dim: int,
        gpu: GPUSpec,
    ) -> SymVal:
        total_kv_len = sum(kv_lengths)
        avg_kv_len = total_kv_len / batch_size if batch_size > 0 else 0

        return self.paged_attention(
            num_tokens=batch_size,
            kv_len=int(avg_kv_len),
            num_heads=num_heads,
            head_dim=head_dim,
            gpu=gpu,
        )

    def select_kernel(
        self,
        seq_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        gpu: GPUSpec,
        is_prefill: bool = True,
    ) -> SymVal:
        if is_prefill:
            flash_feasible = self.flash_attention_feasible(seq_len, head_dim)

            if flash_feasible:
                flash_cost = self.flash_attention(seq_len, batch_size, num_heads, head_dim, gpu)
                paged_cost = self.paged_attention(seq_len, seq_len, num_heads, head_dim, gpu)
                return sym_min(flash_cost, paged_cost)
            else:
                return self.paged_attention(seq_len, seq_len, num_heads, head_dim, gpu)
        else:
            return self.paged_attention(batch_size, seq_len, num_heads, head_dim, gpu)

    def select_kernel_symbolic(
        self,
        seq_len_sym: sympy.Symbol,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        gpu: GPUSpec,
    ) -> SymVal:
        max_flash_seq = self.config.max_seq_len_flash

        flash_expr = self._flash_attention_expr(seq_len_sym, batch_size, num_heads, head_dim, gpu)
        paged_expr = self._paged_attention_expr(seq_len_sym, batch_size, num_heads, head_dim, gpu)

        condition = seq_len_sym <= max_flash_seq

        result_expr = sympy.Piecewise(
            (sympy.Min(flash_expr.expr, paged_expr.expr), condition),
            (paged_expr.expr, True),
        )

        return SymVal(
            expr=result_expr,
            val=min(flash_expr.val, paged_expr.val),
            meta={"op": "kernel_select", "kernels": ["flash", "paged"]},
        )

    def _flash_attention_expr(
        self,
        seq_len: sympy.Expr | int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        gpu: GPUSpec,
    ) -> SymVal:
        flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
        compute_time = flops / gpu.flops_fp16

        if isinstance(seq_len, int):
            return sym_const(float(compute_time), "t_flash")

        return SymVal(
            expr=compute_time,
            val=0.0,  # Placeholder, needs numeric seq_len
            meta={"kernel": "flash"},
        )

    def _paged_attention_expr(
        self,
        kv_len: sympy.Expr | int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        gpu: GPUSpec,
    ) -> SymVal:
        kv_bytes = 2 * kv_len * num_heads * head_dim * 2
        memory_time = kv_bytes / gpu.memory_bandwidth

        if isinstance(kv_len, int):
            return sym_const(float(memory_time), "t_paged")

        return SymVal(
            expr=memory_time,
            val=0.0,
            meta={"kernel": "paged"},
        )

