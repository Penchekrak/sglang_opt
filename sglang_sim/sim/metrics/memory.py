from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy
from sympy import Symbol, Max, ceiling

from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_max, sym_mul
from sim.symbolic.symbols import ConfigSymbols

if TYPE_CHECKING:
    from sim.config.model import ModelConfig
    from sim.config.cluster import GPUSpec
    from sim.config.scheduler import SchedulerConfig


@dataclass
class MemoryBreakdown:
    model_weights: SymVal
    kv_cache: SymVal
    activations: SymVal
    workspace: SymVal
    total: SymVal


class SymbolicMemoryModel:
    def __init__(self, symbols: ConfigSymbols | None = None):
        self.symbols = symbols or ConfigSymbols()

    def model_weights_memory(
        self,
        num_layers: int | sympy.Expr,
        hidden_dim: int | sympy.Expr,
        intermediate_dim: int | sympy.Expr,
        vocab_size: int | sympy.Expr,
        num_experts: int | sympy.Expr = 1,
        dtype_bytes: int = 2,
        tp_size: int | sympy.Expr = 1,
    ) -> SymVal:
        embedding_params = vocab_size * hidden_dim
        
        attn_params_per_layer = 4 * hidden_dim * hidden_dim
        
        if isinstance(num_experts, int) and num_experts > 1:
            mlp_params_per_layer = num_experts * 3 * hidden_dim * intermediate_dim
        else:
            mlp_params_per_layer = 3 * hidden_dim * intermediate_dim
        
        layer_norm_params = 4 * hidden_dim
        
        total_params = (
            2 * embedding_params +
            num_layers * (attn_params_per_layer + mlp_params_per_layer + layer_norm_params)
        )
        
        total_bytes = total_params * dtype_bytes / tp_size
        
        if isinstance(total_bytes, (int, float)):
            return sym_const(float(total_bytes), "M_weights")
        
        return SymVal(
            expr=total_bytes,
            val=0.0,
            meta={"type": "model_weights"},
        )

    def kv_cache_memory_per_token(
        self,
        num_layers: int | sympy.Expr,
        num_heads: int | sympy.Expr,
        head_dim: int | sympy.Expr,
        dtype_bytes: int = 2,
        tp_size: int | sympy.Expr = 1,
    ) -> SymVal:
        kv_bytes = 2 * num_layers * num_heads * head_dim * dtype_bytes / tp_size
        
        if isinstance(kv_bytes, (int, float)):
            return sym_const(float(kv_bytes), "kv_bytes_per_token")
        
        return SymVal(
            expr=kv_bytes,
            val=0.0,
            meta={"type": "kv_per_token"},
        )

    def peak_kv_cache_memory(
        self,
        max_batch_requests: int | sympy.Expr,
        max_seq_len: int | sympy.Expr,
        num_layers: int | sympy.Expr,
        num_heads: int | sympy.Expr,
        head_dim: int | sympy.Expr,
        dtype_bytes: int = 2,
        tp_size: int | sympy.Expr = 1,
    ) -> SymVal:
        kv_per_token = self.kv_cache_memory_per_token(
            num_layers, num_heads, head_dim, dtype_bytes, tp_size
        )
        
        max_tokens = max_batch_requests * max_seq_len
        
        if isinstance(max_tokens, (int, float)) and isinstance(kv_per_token.val, (int, float)):
            total_bytes = float(max_tokens) * kv_per_token.val
            return sym_const(total_bytes, "M_kv_peak")
        
        return SymVal(
            expr=max_tokens * kv_per_token.expr,
            val=0.0,
            meta={"type": "kv_cache_peak"},
        )

    def activation_memory(
        self,
        batch_tokens: int | sympy.Expr,
        hidden_dim: int | sympy.Expr,
        intermediate_dim: int | sympy.Expr,
        num_heads: int | sympy.Expr,
        dtype_bytes: int = 2,
    ) -> SymVal:
        qkv_activations = 3 * batch_tokens * hidden_dim * dtype_bytes
        
        attn_scores = batch_tokens * batch_tokens * num_heads * dtype_bytes
        
        mlp_activations = 2 * batch_tokens * intermediate_dim * dtype_bytes
        
        residual = 2 * batch_tokens * hidden_dim * dtype_bytes
        
        total = qkv_activations + attn_scores + mlp_activations + residual
        
        if isinstance(total, (int, float)):
            return sym_const(float(total), "M_activations")
        
        return SymVal(
            expr=total,
            val=0.0,
            meta={"type": "activations"},
        )

    def workspace_memory(
        self,
        batch_tokens: int | sympy.Expr,
        hidden_dim: int | sympy.Expr,
        dtype_bytes: int = 2,
    ) -> SymVal:
        workspace = 4 * batch_tokens * hidden_dim * dtype_bytes
        
        if isinstance(workspace, (int, float)):
            return sym_const(float(workspace), "M_workspace")
        
        return SymVal(
            expr=workspace,
            val=0.0,
            meta={"type": "workspace"},
        )

    def peak_memory_prefill(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        dtype_bytes: int = 2,
    ) -> MemoryBreakdown:
        weights = self.model_weights_memory(
            num_layers=model_config.num_layers,
            hidden_dim=model_config.hidden_dim,
            intermediate_dim=model_config.actual_intermediate_dim,
            vocab_size=model_config.vocab_size,
            num_experts=model_config.num_experts,
            dtype_bytes=dtype_bytes,
            tp_size=scheduler_config.tp_size,
        )
        
        chunk_size = scheduler_config.chunk_size
        max_requests = scheduler_config.max_batch_requests
        max_seq = scheduler_config.max_batch_tokens
        
        kv_cache = self.peak_kv_cache_memory(
            max_batch_requests=max_requests,
            max_seq_len=max_seq,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            dtype_bytes=dtype_bytes,
            tp_size=scheduler_config.tp_size,
        )
        
        activations = self.activation_memory(
            batch_tokens=chunk_size,
            hidden_dim=model_config.hidden_dim,
            intermediate_dim=model_config.actual_intermediate_dim,
            num_heads=model_config.num_heads,
            dtype_bytes=dtype_bytes,
        )
        
        workspace = self.workspace_memory(
            batch_tokens=chunk_size,
            hidden_dim=model_config.hidden_dim,
            dtype_bytes=dtype_bytes,
        )
        
        total = sym_add(sym_add(weights, kv_cache), sym_add(activations, workspace))
        
        return MemoryBreakdown(
            model_weights=weights,
            kv_cache=kv_cache,
            activations=activations,
            workspace=workspace,
            total=total,
        )

    def peak_memory_decode(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        avg_kv_len: int = 512,
        dtype_bytes: int = 2,
    ) -> MemoryBreakdown:
        weights = self.model_weights_memory(
            num_layers=model_config.num_layers,
            hidden_dim=model_config.hidden_dim,
            intermediate_dim=model_config.actual_intermediate_dim,
            vocab_size=model_config.vocab_size,
            num_experts=model_config.num_experts,
            dtype_bytes=dtype_bytes,
            tp_size=scheduler_config.tp_size,
        )
        
        batch_size = scheduler_config.max_batch_requests
        
        kv_cache = self.peak_kv_cache_memory(
            max_batch_requests=batch_size,
            max_seq_len=avg_kv_len,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            dtype_bytes=dtype_bytes,
            tp_size=scheduler_config.tp_size,
        )
        
        activations = self.activation_memory(
            batch_tokens=batch_size,
            hidden_dim=model_config.hidden_dim,
            intermediate_dim=model_config.actual_intermediate_dim,
            num_heads=model_config.num_heads,
            dtype_bytes=dtype_bytes,
        )
        
        workspace = self.workspace_memory(
            batch_tokens=batch_size,
            hidden_dim=model_config.hidden_dim,
            dtype_bytes=dtype_bytes,
        )
        
        total = sym_add(sym_add(weights, kv_cache), sym_add(activations, workspace))
        
        return MemoryBreakdown(
            model_weights=weights,
            kv_cache=kv_cache,
            activations=activations,
            workspace=workspace,
            total=total,
        )

    def symbolic_peak_memory(
        self,
        symbols: ConfigSymbols | None = None,
    ) -> sympy.Expr:
        s = symbols or self.symbols
        
        embedding_params = s.hidden_dim * 32000
        layer_params = (
            4 * s.hidden_dim * s.hidden_dim +
            3 * s.hidden_dim * s.hidden_dim * 4
        )
        weights_bytes = (embedding_params * 2 + s.num_layers * layer_params) * 2 / s.tp_size
        
        kv_bytes_per_token = 2 * s.num_layers * s.num_heads * s.head_dim * 2 / s.tp_size
        kv_cache_bytes = s.batch_cap_requests * (s.avg_prompt_len + s.avg_output_len) * kv_bytes_per_token
        
        activation_bytes = 4 * s.chunk_size * s.hidden_dim * 2
        
        total = weights_bytes + kv_cache_bytes + activation_bytes
        
        return total

    def memory_constraint(
        self,
        gpu_memory_bytes: int | sympy.Expr,
        safety_margin: float = 0.9,
    ) -> sympy.Expr:
        peak_mem = self.symbolic_peak_memory()
        available = gpu_memory_bytes * safety_margin
        return available - peak_mem

