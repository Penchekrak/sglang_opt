from dataclasses import dataclass
import sympy
from sympy import Symbol


@dataclass
class ConfigSymbols:
    # Worker pool configuration
    N_p: Symbol = Symbol("N_p", positive=True, integer=True)  # num prefill workers
    N_d: Symbol = Symbol("N_d", positive=True, integer=True)  # num decode workers

    # Scheduler configuration
    chunk_size: Symbol = Symbol("c", positive=True, integer=True)
    batch_cap_tokens: Symbol = Symbol("B_tok", positive=True, integer=True)
    batch_cap_requests: Symbol = Symbol("B_req", positive=True, integer=True)

    # Parallelism
    tp_size: Symbol = Symbol("TP", positive=True, integer=True)
    dp_size: Symbol = Symbol("DP", positive=True, integer=True)
    ep_size: Symbol = Symbol("EP", positive=True, integer=True)

    # Hardware
    gpu_flops: Symbol = Symbol("F_gpu", positive=True)  # FLOP/s
    gpu_mem_bw: Symbol = Symbol("B_mem", positive=True)  # bytes/s
    gpu_mem_cap: Symbol = Symbol("M_gpu", positive=True)  # bytes
    nvlink_bw: Symbol = Symbol("B_nvlink", positive=True)  # bytes/s
    network_bw: Symbol = Symbol("B_net", positive=True)  # bytes/s
    network_latency: Symbol = Symbol("L_net", positive=True)  # seconds

    # Model
    num_layers: Symbol = Symbol("L", positive=True, integer=True)
    hidden_dim: Symbol = Symbol("H", positive=True, integer=True)
    head_dim: Symbol = Symbol("d", positive=True, integer=True)
    num_heads: Symbol = Symbol("n_h", positive=True, integer=True)
    num_experts: Symbol = Symbol("E", positive=True, integer=True)
    top_k_experts: Symbol = Symbol("k", positive=True, integer=True)
    kv_bytes_per_token: Symbol = Symbol("kv_b", positive=True)

    # Cache
    cache_capacity: Symbol = Symbol("C_kv", positive=True)
    block_size: Symbol = Symbol("B_blk", positive=True, integer=True)

    # Frozen params (measured during simulation)
    cache_hit_rate: Symbol = Symbol("pi_hit", positive=True)
    avg_prompt_len: Symbol = Symbol("n_in_avg", positive=True)
    avg_output_len: Symbol = Symbol("n_out_avg", positive=True)
    avg_prefix_match: Symbol = Symbol("prefix_match", positive=True)

    def decision_vars(self) -> list[Symbol]:
        return [
            self.N_p,
            self.N_d,
            self.chunk_size,
            self.batch_cap_tokens,
            self.batch_cap_requests,
            self.tp_size,
            self.dp_size,
            self.ep_size,
        ]

    def frozen_params(self) -> list[Symbol]:
        return [
            self.cache_hit_rate,
            self.avg_prompt_len,
            self.avg_output_len,
            self.avg_prefix_match,
        ]

    def hardware_params(self) -> list[Symbol]:
        return [
            self.gpu_flops,
            self.gpu_mem_bw,
            self.gpu_mem_cap,
            self.nvlink_bw,
            self.network_bw,
            self.network_latency,
        ]

    def model_params(self) -> list[Symbol]:
        return [
            self.num_layers,
            self.hidden_dim,
            self.head_dim,
            self.num_heads,
            self.num_experts,
            self.top_k_experts,
            self.kv_bytes_per_token,
        ]

    def all_symbols(self) -> list[Symbol]:
        return (
            self.decision_vars()
            + self.frozen_params()
            + self.hardware_params()
            + self.model_params()
            + [self.cache_capacity, self.block_size]
        )

