# Large-scale LLM inference simulator plan

_Created: 27/12/2025, 21:04:19_

---

### User

based on @implementation_discussion.md  suggest a concrete and detailed plan on how we implement a large scale llm inference simulator with support of symbolic tracing for each metric and that will closely resemble the workflow of sglang. 

---

### AI

Looking at the detailed discussion in your `implementation_discussion.md`, I have a good understanding of the requirements. Before I create a concrete implementation plan, I need to clarify a few critical architectural decisions:

---

### AI

Excellent! Now I have all the context needed. Let me create a comprehensive implementation plan.

---

### User

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

### AI

I'll implement the LLM Inference Simulator according to the plan. Let me start by marking the first todos as in progress and begin implementing.

---

### AI

Let me start by creating the project structure and implementing the symbolic expression system:

---

### AI



```
[project]
name = "sglang-sim"
version = "0.1.0"
description = "LLM inference simulator with symbolic tracing for SGLang-style workloads"
requires-python = ">=3.10"
dependencies = [
    "sympy>=1.12",
    "numpy>=1.24",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]

```

---

### AI



```
from sim.symbolic.expr import SymVal
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig, GPUSpec, InterconnectConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig

__all__ = [
    "SymVal",
    "ModelConfig",
    "ClusterConfig",
    "GPUSpec",
    "InterconnectConfig",
    "SchedulerConfig",
    "CacheConfig",
]

```

---

### AI



```
from sim.symbolic.expr import (
    SymVal,
    sym_add,
    sym_sub,
    sym_mul,
    sym_div,
    sym_max,
    sym_min,
    sym_piecewise,
    sym_sum,
    sym_const,
)
from sim.symbolic.symbols import ConfigSymbols

__all__ = [
    "SymVal",
    "sym_add",
    "sym_sub",
    "sym_mul",
    "sym_div",
    "sym_max",
    "sym_min",
    "sym_piecewise",
    "sym_sum",
    "sym_const",
    "ConfigSymbols",
]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import sympy
from sympy import Max, Min, Piecewise, Sum, Symbol, ceiling, floor


@dataclass
class SymVal:
    expr: sympy.Expr
    val: float
    meta: dict = field(default_factory=dict)

    def __add__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr + other.expr,
                val=self.val + other.val,
                meta={"op": "add", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr + other,
            val=self.val + float(other),
            meta={"op": "add_scalar", "operands": [self.meta, other]},
        )

    def __radd__(self, other: float | int) -> SymVal:
        return self.__add__(other)

    def __sub__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr - other.expr,
                val=self.val - other.val,
                meta={"op": "sub", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr - other,
            val=self.val - float(other),
            meta={"op": "sub_scalar", "operands": [self.meta, other]},
        )

    def __rsub__(self, other: float | int) -> SymVal:
        return SymVal(
            expr=other - self.expr,
            val=float(other) - self.val,
            meta={"op": "rsub_scalar", "operands": [other, self.meta]},
        )

    def __mul__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr * other.expr,
                val=self.val * other.val,
                meta={"op": "mul", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr * other,
            val=self.val * float(other),
            meta={"op": "mul_scalar", "operands": [self.meta, other]},
        )

    def __rmul__(self, other: float | int) -> SymVal:
        return self.__mul__(other)

    def __truediv__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr / other.expr,
                val=self.val / other.val if other.val != 0 else float("inf"),
                meta={"op": "div", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr / other,
            val=self.val / float(other) if other != 0 else float("inf"),
            meta={"op": "div_scalar", "operands": [self.meta, other]},
        )

    def __rtruediv__(self, other: float | int) -> SymVal:
        return SymVal(
            expr=other / self.expr,
            val=float(other) / self.val if self.val != 0 else float("inf"),
            meta={"op": "rdiv_scalar", "operands": [other, self.meta]},
        )

    def __neg__(self) -> SymVal:
        return SymVal(
            expr=-self.expr,
            val=-self.val,
            meta={"op": "neg", "operands": [self.meta]},
        )

    def __repr__(self) -> str:
        return f"SymVal(expr={self.expr}, val={self.val:.6g})"

    def lambdify(self, symbols: list[Symbol]) -> Callable[..., float]:
        return sympy.lambdify(symbols, self.expr, modules=["numpy"])

    def simplify(self) -> SymVal:
        return SymVal(
            expr=sympy.simplify(self.expr),
            val=self.val,
            meta=self.meta,
        )


def sym_const(value: float, name: str | None = None) -> SymVal:
    if name:
        return SymVal(expr=Symbol(name), val=value, meta={"type": "symbol", "name": name})
    return SymVal(expr=sympy.Float(value), val=value, meta={"type": "const", "value": value})


def sym_add(a: SymVal, b: SymVal) -> SymVal:
    return a + b


def sym_sub(a: SymVal, b: SymVal) -> SymVal:
    return a - b


def sym_mul(a: SymVal, b: SymVal) -> SymVal:
    return a * b


def sym_div(a: SymVal, b: SymVal) -> SymVal:
    return a / b


def sym_max(a: SymVal, b: SymVal) -> SymVal:
    return SymVal(
        expr=Max(a.expr, b.expr),
        val=max(a.val, b.val),
        meta={"op": "max", "operands": [a.meta, b.meta]},
    )


def sym_min(a: SymVal, b: SymVal) -> SymVal:
    return SymVal(
        expr=Min(a.expr, b.expr),
        val=min(a.val, b.val),
        meta={"op": "min", "operands": [a.meta, b.meta]},
    )


def sym_piecewise(
    cases: list[tuple[sympy.Expr | bool, SymVal]],
    numeric_condition_results: list[bool],
) -> SymVal:
    pw_args = [(case[1].expr, case[0]) for case in cases]
    pw_expr = Piecewise(*pw_args)

    result_val = cases[-1][1].val
    for i, cond_result in enumerate(numeric_condition_results):
        if cond_result:
            result_val = cases[i][1].val
            break

    return SymVal(
        expr=pw_expr,
        val=result_val,
        meta={
            "op": "piecewise",
            "cases": [(str(c[0]), c[1].meta) for c in cases],
        },
    )


def sym_sum(
    body_fn: Callable[[Symbol], SymVal],
    var: Symbol,
    start: int,
    end: int,
) -> SymVal:
    symbolic_body = body_fn(var)
    sum_expr = Sum(symbolic_body.expr, (var, start, end))

    numeric_sum = 0.0
    for i in range(start, end + 1):
        step_val = body_fn(sympy.Integer(i))
        numeric_sum += step_val.val

    return SymVal(
        expr=sum_expr,
        val=numeric_sum,
        meta={
            "op": "sum",
            "var": str(var),
            "range": (start, end),
            "body_meta": symbolic_body.meta,
        },
    )


def sym_ceiling(a: SymVal) -> SymVal:
    return SymVal(
        expr=ceiling(a.expr),
        val=float(int(a.val) + (1 if a.val > int(a.val) else 0)),
        meta={"op": "ceiling", "operands": [a.meta]},
    )


def sym_floor(a: SymVal) -> SymVal:
    return SymVal(
        expr=floor(a.expr),
        val=float(int(a.val)),
        meta={"op": "floor", "operands": [a.meta]},
    )

```

---

### AI



```
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

```

---

### AI

Now let me create the configuration classes:

---

### AI



```
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig, GPUSpec, InterconnectConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig

__all__ = [
    "ModelConfig",
    "ClusterConfig",
    "GPUSpec",
    "InterconnectConfig",
    "SchedulerConfig",
    "CacheConfig",
]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    vocab_size: int
    is_moe: bool = False
    num_experts: int = 1
    top_k_experts: int = 1
    intermediate_dim: int | None = None
    dtype_bytes: int = 2  # fp16 by default

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.head_dim * self.num_heads * self.dtype_bytes

    @property
    def actual_intermediate_dim(self) -> int:
        return self.intermediate_dim or 4 * self.hidden_dim

    @classmethod
    def llama_7b(cls) -> ModelConfig:
        return cls(
            name="llama-7b",
            num_layers=32,
            hidden_dim=4096,
            num_heads=32,
            head_dim=128,
            vocab_size=32000,
            intermediate_dim=11008,
        )

    @classmethod
    def llama_70b(cls) -> ModelConfig:
        return cls(
            name="llama-70b",
            num_layers=80,
            hidden_dim=8192,
            num_heads=64,
            head_dim=128,
            vocab_size=32000,
            intermediate_dim=28672,
        )

    @classmethod
    def mixtral_8x7b(cls) -> ModelConfig:
        return cls(
            name="mixtral-8x7b",
            num_layers=32,
            hidden_dim=4096,
            num_heads=32,
            head_dim=128,
            vocab_size=32000,
            is_moe=True,
            num_experts=8,
            top_k_experts=2,
            intermediate_dim=14336,
        )

    @classmethod
    def deepseek_v3(cls) -> ModelConfig:
        return cls(
            name="deepseek-v3",
            num_layers=61,
            hidden_dim=7168,
            num_heads=128,
            head_dim=128,
            vocab_size=129280,
            is_moe=True,
            num_experts=256,
            top_k_experts=8,
            intermediate_dim=18432,
        )

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class InterconnectType(Enum):
    NVLINK = "nvlink"
    ETHERNET = "ethernet"
    INFINIBAND = "infiniband"


@dataclass
class InterconnectConfig:
    type: InterconnectType
    bandwidth_gbps: float
    latency_us: float

    @property
    def bandwidth_bytes_per_sec(self) -> float:
        return self.bandwidth_gbps * 1e9 / 8

    @property
    def latency_seconds(self) -> float:
        return self.latency_us * 1e-6

    def transfer_time_seconds(self, bytes_to_transfer: int) -> float:
        return self.latency_seconds + bytes_to_transfer / self.bandwidth_bytes_per_sec

    @classmethod
    def nvlink_4(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.NVLINK,
            bandwidth_gbps=900,  # NVLink 4.0: ~900 Gbps bidirectional
            latency_us=1.0,
        )

    @classmethod
    def infiniband_hdr(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.INFINIBAND,
            bandwidth_gbps=200,  # HDR InfiniBand
            latency_us=2.0,
        )

    @classmethod
    def ethernet_100g(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.ETHERNET,
            bandwidth_gbps=100,
            latency_us=10.0,
        )


@dataclass
class GPUSpec:
    name: str
    flops_fp16: float  # FLOP/s
    memory_bandwidth: float  # bytes/s
    memory_capacity: int  # bytes

    @classmethod
    def h100_sxm(cls) -> GPUSpec:
        return cls(
            name="H100-SXM",
            flops_fp16=1979e12,  # ~2 PFLOP/s FP16
            memory_bandwidth=3.35e12,  # 3.35 TB/s
            memory_capacity=80 * 1024**3,  # 80 GB
        )

    @classmethod
    def a100_80g(cls) -> GPUSpec:
        return cls(
            name="A100-80G",
            flops_fp16=312e12,  # 312 TFLOP/s FP16
            memory_bandwidth=2.0e12,  # 2 TB/s
            memory_capacity=80 * 1024**3,
        )

    @classmethod
    def h200(cls) -> GPUSpec:
        return cls(
            name="H200",
            flops_fp16=1979e12,
            memory_bandwidth=4.8e12,  # 4.8 TB/s with HBM3e
            memory_capacity=141 * 1024**3,  # 141 GB
        )


@dataclass
class ClusterConfig:
    num_prefill_workers: int
    num_decode_workers: int
    gpus_per_worker: int
    gpu_spec: GPUSpec
    intra_node_interconnect: InterconnectConfig
    inter_node_interconnect: InterconnectConfig

    @property
    def total_workers(self) -> int:
        return self.num_prefill_workers + self.num_decode_workers

    @property
    def total_gpus(self) -> int:
        return self.total_workers * self.gpus_per_worker

    @classmethod
    def single_node_8gpu(cls) -> ClusterConfig:
        return cls(
            num_prefill_workers=4,
            num_decode_workers=4,
            gpus_per_worker=1,
            gpu_spec=GPUSpec.h100_sxm(),
            intra_node_interconnect=InterconnectConfig.nvlink_4(),
            inter_node_interconnect=InterconnectConfig.nvlink_4(),
        )

    @classmethod
    def multi_node_16gpu(cls) -> ClusterConfig:
        return cls(
            num_prefill_workers=8,
            num_decode_workers=8,
            gpus_per_worker=1,
            gpu_spec=GPUSpec.h100_sxm(),
            intra_node_interconnect=InterconnectConfig.nvlink_4(),
            inter_node_interconnect=InterconnectConfig.infiniband_hdr(),
        )

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    chunk_size: int
    max_batch_tokens: int
    max_batch_requests: int
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    max_prefill_tokens_per_iter: int | None = None
    enable_chunked_prefill: bool = True

    @property
    def effective_max_prefill_tokens(self) -> int:
        return self.max_prefill_tokens_per_iter or self.chunk_size

    @classmethod
    def default(cls) -> SchedulerConfig:
        return cls(
            chunk_size=8192,
            max_batch_tokens=32768,
            max_batch_requests=256,
            tp_size=1,
            dp_size=1,
            ep_size=1,
        )

    @classmethod
    def high_throughput(cls) -> SchedulerConfig:
        return cls(
            chunk_size=16384,
            max_batch_tokens=65536,
            max_batch_requests=512,
            tp_size=8,
            dp_size=1,
            ep_size=1,
        )

    @classmethod
    def moe_optimized(cls) -> SchedulerConfig:
        return cls(
            chunk_size=8192,
            max_batch_tokens=32768,
            max_batch_requests=256,
            tp_size=1,
            dp_size=8,
            ep_size=8,
        )

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class EvictionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"


@dataclass
class CacheConfig:
    capacity_bytes: int
    block_size_tokens: int
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    max_tree_depth: int = 256

    @property
    def capacity_blocks(self) -> int:
        return self.capacity_bytes // self.block_size_tokens

    @classmethod
    def default_for_gpu(cls, gpu_memory_bytes: int, cache_fraction: float = 0.8) -> CacheConfig:
        return cls(
            capacity_bytes=int(gpu_memory_bytes * cache_fraction),
            block_size_tokens=16,
            eviction_policy=EvictionPolicy.LRU,
        )

    @classmethod
    def h100_default(cls) -> CacheConfig:
        return cls(
            capacity_bytes=64 * 1024**3,  # 64 GB for KV cache
            block_size_tokens=16,
            eviction_policy=EvictionPolicy.LRU,
        )

```

---

### AI

Now let me implement the core simulation engine and events:

---

### AI



```
from sim.core.events import (
    Event,
    RequestArrival,
    RouterDispatch,
    PrefillIterationStart,
    PrefillChunkComplete,
    KVTransferStart,
    KVTransferComplete,
    DecodeIterationStart,
    TokenEmit,
    RequestComplete,
)
from sim.core.engine import SimulationEngine, SimulationResult
from sim.core.state import ClusterState, WorkerState
from sim.core.request import Request, PrefillTask, DecodeTask, KVHandle, RequestPhase

__all__ = [
    "Event",
    "RequestArrival",
    "RouterDispatch",
    "PrefillIterationStart",
    "PrefillChunkComplete",
    "KVTransferStart",
    "KVTransferComplete",
    "DecodeIterationStart",
    "TokenEmit",
    "RequestComplete",
    "SimulationEngine",
    "SimulationResult",
    "ClusterState",
    "WorkerState",
    "Request",
    "PrefillTask",
    "DecodeTask",
    "KVHandle",
    "RequestPhase",
]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestPhase(Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    TRANSFERRING = "transferring"
    DECODING = "decoding"
    COMPLETE = "complete"


@dataclass
class Request:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_time: float
    stream: bool = False
    prefix_group_id: int | None = None
    sampling_params: dict[str, Any] = field(default_factory=dict)

    phase: RequestPhase = field(default=RequestPhase.QUEUED)
    prefill_worker_id: int | None = None
    decode_worker_id: int | None = None

    first_token_time: float | None = None
    complete_time: float | None = None
    tokens_generated: int = 0
    token_times: list[float] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def e2e_latency(self) -> float | None:
        if self.complete_time is None:
            return None
        return self.complete_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        if len(self.token_times) < 2:
            return None
        inter_token_delays = [
            self.token_times[i] - self.token_times[i - 1]
            for i in range(1, len(self.token_times))
        ]
        return sum(inter_token_delays) / len(inter_token_delays) if inter_token_delays else None


@dataclass
class KVHandle:
    request_id: int
    kv_bytes: int
    source_worker_id: int
    dest_worker_id: int | None = None
    transfer_started: float | None = None
    transfer_complete: float | None = None

    @property
    def is_transferred(self) -> bool:
        return self.transfer_complete is not None


@dataclass
class PrefillTask:
    request: Request
    remaining_prompt_tokens: int
    chunk_size: int
    kv_bytes_produced: int = 0
    prefix_match_len: int = 0
    chunks_completed: int = 0
    current_chunk_start: float | None = None

    @property
    def is_complete(self) -> bool:
        return self.remaining_prompt_tokens <= 0

    @property
    def tokens_to_prefill_this_chunk(self) -> int:
        return min(self.chunk_size, self.remaining_prompt_tokens)

    def complete_chunk(self, tokens: int, kv_bytes: int) -> None:
        self.remaining_prompt_tokens -= tokens
        self.kv_bytes_produced += kv_bytes
        self.chunks_completed += 1


@dataclass
class DecodeTask:
    request: Request
    kv_handle: KVHandle
    remaining_tokens: int
    current_kv_len: int
    tokens_emitted: int = 0
    iteration_start: float | None = None

    @property
    def is_complete(self) -> bool:
        return self.remaining_tokens <= 0

    def emit_token(self, kv_bytes_per_token: int) -> None:
        self.remaining_tokens -= 1
        self.tokens_emitted += 1
        self.current_kv_len += 1
        self.kv_handle.kv_bytes += kv_bytes_per_token

```

---

### AI



```
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, PrefillTask, DecodeTask, KVHandle
    from sim.core.engine import SimulationEngine


@dataclass(order=True)
class Event(ABC):
    time: float
    priority: int = field(default=0, compare=True)

    @abstractmethod
    def process(self, engine: SimulationEngine) -> list[Event]:
        pass


@dataclass(order=True)
class RequestArrival(Event):
    request: Request = field(compare=False)
    priority: int = field(default=10, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        engine.state.pending_requests.append(self.request)
        worker_id = engine.router.route_to_prefill(self.request)
        return [RouterDispatch(time=self.time, request=self.request, worker_id=worker_id)]


@dataclass(order=True)
class RouterDispatch(Event):
    request: Request = field(compare=False)
    worker_id: int = field(compare=False)
    priority: int = field(default=20, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.request.phase = RequestPhase.PREFILLING
        self.request.prefill_worker_id = self.worker_id
        worker = engine.prefill_workers[self.worker_id]
        worker.enqueue_request(self.request, engine.scheduler_config)

        if not worker.is_iteration_scheduled:
            worker.is_iteration_scheduled = True
            return [PrefillIterationStart(time=self.time, worker_id=self.worker_id)]
        return []


@dataclass(order=True)
class PrefillIterationStart(Event):
    worker_id: int = field(compare=False)
    priority: int = field(default=30, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        worker = engine.prefill_workers[self.worker_id]
        iteration_result = worker.run_iteration(self.time, engine)

        events: list[Event] = []
        for task in iteration_result.completed_tasks:
            events.append(
                PrefillChunkComplete(
                    time=self.time + iteration_result.iteration_time.val,
                    task=task,
                    worker_id=self.worker_id,
                )
            )

        if worker.has_pending_work():
            events.append(
                PrefillIterationStart(
                    time=self.time + iteration_result.iteration_time.val,
                    worker_id=self.worker_id,
                )
            )
        else:
            worker.is_iteration_scheduled = False

        return events


@dataclass(order=True)
class PrefillChunkComplete(Event):
    task: PrefillTask = field(compare=False)
    worker_id: int = field(compare=False)
    priority: int = field(default=40, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase, KVHandle

        if self.task.is_complete:
            self.task.request.phase = RequestPhase.TRANSFERRING
            kv_handle = KVHandle(
                request_id=self.task.request.id,
                kv_bytes=self.task.kv_bytes_produced,
                source_worker_id=self.worker_id,
            )
            dest_worker = engine.router.route_to_decode(kv_handle)
            kv_handle.dest_worker_id = dest_worker
            return [
                KVTransferStart(
                    time=self.time,
                    kv_handle=kv_handle,
                    request=self.task.request,
                )
            ]
        return []


@dataclass(order=True)
class KVTransferStart(Event):
    kv_handle: KVHandle = field(compare=False)
    request: Request = field(compare=False)
    priority: int = field(default=50, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        self.kv_handle.transfer_started = self.time
        transfer_time = engine.kv_transfer_manager.initiate_transfer(
            self.kv_handle, engine.cluster_config
        )
        return [
            KVTransferComplete(
                time=self.time + transfer_time.val,
                kv_handle=self.kv_handle,
                request=self.request,
            )
        ]


@dataclass(order=True)
class KVTransferComplete(Event):
    kv_handle: KVHandle = field(compare=False)
    request: Request = field(compare=False)
    priority: int = field(default=60, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.kv_handle.transfer_complete = self.time
        self.request.phase = RequestPhase.DECODING
        self.request.decode_worker_id = self.kv_handle.dest_worker_id

        worker = engine.decode_workers[self.kv_handle.dest_worker_id]
        worker.enqueue_request(self.request, self.kv_handle, engine.scheduler_config)

        if not worker.is_iteration_scheduled:
            worker.is_iteration_scheduled = True
            return [
                DecodeIterationStart(
                    time=self.time, worker_id=self.kv_handle.dest_worker_id
                )
            ]
        return []


@dataclass(order=True)
class DecodeIterationStart(Event):
    worker_id: int = field(compare=False)
    priority: int = field(default=70, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        worker = engine.decode_workers[self.worker_id]
        iteration_result = worker.run_iteration(self.time, engine)

        events: list[Event] = []
        for task in iteration_result.token_emitted_tasks:
            if task.request.tokens_generated == 1:
                task.request.first_token_time = self.time + iteration_result.iteration_time.val

            events.append(
                TokenEmit(
                    time=self.time + iteration_result.iteration_time.val,
                    task=task,
                    worker_id=self.worker_id,
                )
            )

        if worker.has_pending_work():
            events.append(
                DecodeIterationStart(
                    time=self.time + iteration_result.iteration_time.val,
                    worker_id=self.worker_id,
                )
            )
        else:
            worker.is_iteration_scheduled = False

        return events


@dataclass(order=True)
class TokenEmit(Event):
    task: DecodeTask = field(compare=False)
    worker_id: int = field(compare=False)
    priority: int = field(default=80, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        self.task.request.token_times.append(self.time)

        if self.task.is_complete:
            return [
                RequestComplete(
                    time=self.time,
                    request=self.task.request,
                )
            ]
        return []


@dataclass(order=True)
class RequestComplete(Event):
    request: Request = field(compare=False)
    priority: int = field(default=90, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.request.phase = RequestPhase.COMPLETE
        self.request.complete_time = self.time
        engine.metrics.record_request_complete(self.request)
        return []

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, PrefillTask, DecodeTask


@dataclass
class WorkerState:
    worker_id: int
    is_prefill: bool
    waiting_queue: list[Request] = field(default_factory=list)
    running_tasks: list[PrefillTask | DecodeTask] = field(default_factory=list)
    total_tokens_processed: int = 0
    total_iterations: int = 0

    @property
    def queue_length(self) -> int:
        return len(self.waiting_queue) + len(self.running_tasks)

    @property
    def is_idle(self) -> bool:
        return len(self.waiting_queue) == 0 and len(self.running_tasks) == 0


@dataclass
class ClusterState:
    prefill_worker_states: list[WorkerState] = field(default_factory=list)
    decode_worker_states: list[WorkerState] = field(default_factory=list)
    pending_requests: list[Request] = field(default_factory=list)
    completed_requests: list[Request] = field(default_factory=list)
    total_tokens_generated: int = 0

    @classmethod
    def initialize(cls, num_prefill: int, num_decode: int) -> ClusterState:
        return cls(
            prefill_worker_states=[
                WorkerState(worker_id=i, is_prefill=True) for i in range(num_prefill)
            ],
            decode_worker_states=[
                WorkerState(worker_id=i, is_prefill=False) for i in range(num_decode)
            ],
        )

    def get_prefill_queue_lengths(self) -> list[int]:
        return [w.queue_length for w in self.prefill_worker_states]

    def get_decode_queue_lengths(self) -> list[int]:
        return [w.queue_length for w in self.decode_worker_states]

    def is_balanced(self, threshold: float = 2.0) -> bool:
        prefill_lengths = self.get_prefill_queue_lengths()
        if not prefill_lengths or max(prefill_lengths) == 0:
            return True
        return max(prefill_lengths) / max(1, min(prefill_lengths)) <= threshold

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import TYPE_CHECKING

from sim.core.state import ClusterState
from sim.core.events import Event

if TYPE_CHECKING:
    from sim.config.model import ModelConfig
    from sim.config.cluster import ClusterConfig
    from sim.config.scheduler import SchedulerConfig
    from sim.config.cache import CacheConfig
    from sim.workers.prefill_worker import PrefillWorker
    from sim.workers.decode_worker import DecodeWorker
    from sim.routing.router import Router
    from sim.network.kv_transfer import KVTransferManager
    from sim.metrics.definitions import MetricCollector


@dataclass
class SimulationResult:
    total_requests: int
    completed_requests: int
    total_tokens_generated: int
    simulation_time: float
    throughput_tokens_per_sec: float
    avg_ttft: float
    avg_tpot: float
    avg_e2e_latency: float
    p50_ttft: float
    p99_ttft: float
    p50_tpot: float
    p99_tpot: float


class SimulationEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        cluster_config: ClusterConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ):
        self.model_config = model_config
        self.cluster_config = cluster_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.clock: float = 0.0
        self.event_queue: list[tuple[float, int, Event]] = []
        self.event_counter: int = 0

        self.state = ClusterState.initialize(
            cluster_config.num_prefill_workers,
            cluster_config.num_decode_workers,
        )

        self.prefill_workers: list[PrefillWorker] = []
        self.decode_workers: list[DecodeWorker] = []
        self.router: Router | None = None
        self.kv_transfer_manager: KVTransferManager | None = None
        self.metrics: MetricCollector | None = None

    def initialize_components(
        self,
        prefill_workers: list[PrefillWorker],
        decode_workers: list[DecodeWorker],
        router: Router,
        kv_transfer_manager: KVTransferManager,
        metrics: MetricCollector,
    ) -> None:
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.router = router
        self.kv_transfer_manager = kv_transfer_manager
        self.metrics = metrics

    def schedule_event(self, event: Event) -> None:
        self.event_counter += 1
        heappush(self.event_queue, (event.time, event.priority, self.event_counter, event))

    def schedule_events(self, events: list[Event]) -> None:
        for event in events:
            self.schedule_event(event)

    def run(self, until: float | None = None) -> SimulationResult:
        while self.event_queue:
            time, priority, _, event = heappop(self.event_queue)

            if until is not None and time > until:
                break

            self.clock = time
            new_events = event.process(self)
            self.schedule_events(new_events)

        return self._compute_results()

    def run_until_idle(self) -> SimulationResult:
        while self.event_queue:
            time, priority, _, event = heappop(self.event_queue)
            self.clock = time
            new_events = event.process(self)
            self.schedule_events(new_events)

        return self._compute_results()

    def _compute_results(self) -> SimulationResult:
        if self.metrics is None:
            raise RuntimeError("MetricCollector not initialized")

        return SimulationResult(
            total_requests=self.metrics.total_requests,
            completed_requests=self.metrics.completed_requests,
            total_tokens_generated=self.metrics.total_tokens,
            simulation_time=self.clock,
            throughput_tokens_per_sec=self.metrics.throughput(self.clock),
            avg_ttft=self.metrics.avg_ttft(),
            avg_tpot=self.metrics.avg_tpot(),
            avg_e2e_latency=self.metrics.avg_e2e_latency(),
            p50_ttft=self.metrics.percentile_ttft(0.5),
            p99_ttft=self.metrics.percentile_ttft(0.99),
            p50_tpot=self.metrics.percentile_tpot(0.5),
            p99_tpot=self.metrics.percentile_tpot(0.99),
        )

```

---

### AI

Now let me implement the schedulers and workers:

---

### AI



```
from sim.schedulers.batch_scheduler import ContinuousBatchScheduler, BatchSelectionResult

__all__ = ["ContinuousBatchScheduler", "BatchSelectionResult"]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_add, sym_const

if TYPE_CHECKING:
    from sim.core.request import PrefillTask, DecodeTask
    from sim.config.scheduler import SchedulerConfig


@dataclass
class BatchSelectionResult:
    selected_tasks: list[PrefillTask | DecodeTask]
    total_tokens: int
    estimated_cost: SymVal


class ContinuousBatchScheduler:
    def select_prefill_batch(
        self,
        waiting_tasks: list[PrefillTask],
        running_tasks: list[PrefillTask],
        config: SchedulerConfig,
    ) -> BatchSelectionResult:
        selected: list[PrefillTask] = []
        total_tokens = 0
        max_tokens = config.max_batch_tokens
        max_requests = config.max_batch_requests

        running_tokens = sum(t.tokens_to_prefill_this_chunk for t in running_tasks)
        current_tokens = running_tokens

        for task in waiting_tasks:
            chunk_tokens = task.tokens_to_prefill_this_chunk

            if len(selected) + len(running_tasks) >= max_requests:
                break
            if current_tokens + chunk_tokens > max_tokens:
                break

            selected.append(task)
            current_tokens += chunk_tokens
            total_tokens += chunk_tokens

        return BatchSelectionResult(
            selected_tasks=selected,
            total_tokens=total_tokens,
            estimated_cost=sym_const(0.0),  # Will be computed by worker
        )

    def select_decode_batch(
        self,
        waiting_tasks: list[DecodeTask],
        running_tasks: list[DecodeTask],
        config: SchedulerConfig,
    ) -> BatchSelectionResult:
        selected: list[DecodeTask] = []
        max_requests = config.max_batch_requests

        running_count = len(running_tasks)

        for task in waiting_tasks:
            if len(selected) + running_count >= max_requests:
                break
            selected.append(task)

        total_tokens = len(selected) + running_count

        return BatchSelectionResult(
            selected_tasks=selected,
            total_tokens=total_tokens,
            estimated_cost=sym_const(0.0),
        )

```

---

### AI



```
from sim.workers.prefill_worker import PrefillWorker, PrefillIterationResult
from sim.workers.decode_worker import DecodeWorker, DecodeIterationResult

__all__ = [
    "PrefillWorker",
    "PrefillIterationResult",
    "DecodeWorker",
    "DecodeIterationResult",
]

```

---

### AI



```
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.core.request import Request, PrefillTask
from sim.schedulers.batch_scheduler import ContinuousBatchScheduler
from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_max

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.config.scheduler import SchedulerConfig
    from sim.cache.radix_cache import RadixCache
    from sim.parallel.groups import TPGroup, EPGroup


@dataclass
class PrefillIterationResult:
    iteration_time: SymVal
    completed_tasks: list[PrefillTask]
    tokens_processed: int


class PrefillWorker:
    def __init__(
        self,
        worker_id: int,
        cache: RadixCache | None = None,
        tp_group: TPGroup | None = None,
        ep_group: EPGroup | None = None,
    ):
        self.worker_id = worker_id
        self.cache = cache
        self.tp_group = tp_group
        self.ep_group = ep_group

        self.waiting_queue: deque[PrefillTask] = deque()
        self.running_batch: list[PrefillTask] = []
        self.scheduler = ContinuousBatchScheduler()
        self.is_iteration_scheduled: bool = False
        self.total_tokens_processed: int = 0
        self.total_iterations: int = 0

    def enqueue_request(self, request: Request, config: SchedulerConfig) -> PrefillTask:
        prefix_match_len = 0
        if self.cache is not None:
            prefix_match_len, _ = self.cache.match_prefix(request.prompt_tokens)

        task = PrefillTask(
            request=request,
            remaining_prompt_tokens=request.prompt_len - prefix_match_len,
            chunk_size=config.chunk_size,
            prefix_match_len=prefix_match_len,
        )
        self.waiting_queue.append(task)
        return task

    def has_pending_work(self) -> bool:
        return len(self.waiting_queue) > 0 or len(self.running_batch) > 0

    def run_iteration(
        self,
        current_time: float,
        engine: SimulationEngine,
    ) -> PrefillIterationResult:
        config = engine.scheduler_config

        selection = self.scheduler.select_prefill_batch(
            waiting_tasks=list(self.waiting_queue),
            running_tasks=self.running_batch,
            config=config,
        )

        for task in selection.selected_tasks:
            self.waiting_queue.remove(task)
            self.running_batch.append(task)
            task.current_chunk_start = current_time

        all_active = self.running_batch.copy()
        total_tokens = sum(t.tokens_to_prefill_this_chunk for t in all_active)

        iteration_time = self._compute_iteration_time(
            total_tokens, all_active, engine
        )

        completed_tasks: list[PrefillTask] = []
        for task in all_active:
            chunk_tokens = task.tokens_to_prefill_this_chunk
            kv_bytes = chunk_tokens * engine.model_config.kv_bytes_per_token
            task.complete_chunk(chunk_tokens, kv_bytes)

            if task.is_complete:
                completed_tasks.append(task)
                self.running_batch.remove(task)

                if self.cache is not None:
                    self.cache.insert(
                        task.request.prompt_tokens,
                        task.kv_bytes_produced,
                    )

        self.total_tokens_processed += total_tokens
        self.total_iterations += 1

        return PrefillIterationResult(
            iteration_time=iteration_time,
            completed_tasks=completed_tasks,
            tokens_processed=total_tokens,
        )

    def _compute_iteration_time(
        self,
        total_tokens: int,
        tasks: list[PrefillTask],
        engine: SimulationEngine,
    ) -> SymVal:
        if total_tokens == 0:
            return sym_const(0.0)

        model = engine.model_config
        gpu = engine.cluster_config.gpu_spec

        flops_per_token = (
            2 * model.num_layers * (
                4 * model.hidden_dim * model.hidden_dim +
                2 * model.hidden_dim * model.actual_intermediate_dim
            )
        )
        total_flops = flops_per_token * total_tokens

        attention_flops_per_layer = 2 * total_tokens * total_tokens * model.hidden_dim
        total_attention_flops = attention_flops_per_layer * model.num_layers
        total_flops += total_attention_flops

        effective_flops = gpu.flops_fp16
        if self.tp_group is not None:
            effective_flops *= self.tp_group.size

        compute_time = total_flops / effective_flops

        kv_bytes = total_tokens * model.kv_bytes_per_token
        memory_time = kv_bytes / gpu.memory_bandwidth

        compute_sym = sym_const(compute_time, "t_compute_prefill")
        memory_sym = sym_const(memory_time, "t_memory_prefill")

        return sym_max(compute_sym, memory_sym)

    @property
    def queue_length(self) -> int:
        return len(self.waiting_queue) + len(self.running_batch)

```

---

### AI



```
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.core.request import Request, DecodeTask, KVHandle
from sim.schedulers.batch_scheduler import ContinuousBatchScheduler
from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_max

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.config.scheduler import SchedulerConfig
    from sim.parallel.groups import TPGroup, EPGroup


@dataclass
class DecodeIterationResult:
    iteration_time: SymVal
    token_emitted_tasks: list[DecodeTask]
    tokens_generated: int


class DecodeWorker:
    def __init__(
        self,
        worker_id: int,
        tp_group: TPGroup | None = None,
        ep_group: EPGroup | None = None,
    ):
        self.worker_id = worker_id
        self.tp_group = tp_group
        self.ep_group = ep_group

        self.waiting_queue: deque[DecodeTask] = deque()
        self.running_batch: list[DecodeTask] = []
        self.scheduler = ContinuousBatchScheduler()
        self.is_iteration_scheduled: bool = False
        self.total_tokens_generated: int = 0
        self.total_iterations: int = 0

    def enqueue_request(
        self,
        request: Request,
        kv_handle: KVHandle,
        config: SchedulerConfig,
    ) -> DecodeTask:
        task = DecodeTask(
            request=request,
            kv_handle=kv_handle,
            remaining_tokens=request.max_new_tokens,
            current_kv_len=request.prompt_len,
        )
        self.waiting_queue.append(task)
        return task

    def has_pending_work(self) -> bool:
        return len(self.waiting_queue) > 0 or len(self.running_batch) > 0

    def run_iteration(
        self,
        current_time: float,
        engine: SimulationEngine,
    ) -> DecodeIterationResult:
        config = engine.scheduler_config

        selection = self.scheduler.select_decode_batch(
            waiting_tasks=list(self.waiting_queue),
            running_tasks=self.running_batch,
            config=config,
        )

        for task in selection.selected_tasks:
            self.waiting_queue.remove(task)
            self.running_batch.append(task)
            task.iteration_start = current_time

        all_active = self.running_batch.copy()
        batch_size = len(all_active)

        iteration_time = self._compute_iteration_time(
            all_active, engine
        )

        completed_tasks: list[DecodeTask] = []
        for task in all_active:
            task.emit_token(engine.model_config.kv_bytes_per_token)
            task.request.tokens_generated += 1

            if task.is_complete:
                completed_tasks.append(task)
                self.running_batch.remove(task)

        self.total_tokens_generated += batch_size
        self.total_iterations += 1

        return DecodeIterationResult(
            iteration_time=iteration_time,
            token_emitted_tasks=all_active,
            tokens_generated=batch_size,
        )

    def _compute_iteration_time(
        self,
        tasks: list[DecodeTask],
        engine: SimulationEngine,
    ) -> SymVal:
        if not tasks:
            return sym_const(0.0)

        model = engine.model_config
        gpu = engine.cluster_config.gpu_spec
        batch_size = len(tasks)

        total_kv_len = sum(t.current_kv_len for t in tasks)
        avg_kv_len = total_kv_len / batch_size if batch_size > 0 else 0

        kv_bytes_to_read = total_kv_len * model.kv_bytes_per_token
        memory_time = kv_bytes_to_read / gpu.memory_bandwidth

        flops_per_token = (
            2 * model.num_layers * (
                4 * model.hidden_dim * model.hidden_dim +
                2 * model.hidden_dim * model.actual_intermediate_dim
            )
        )
        total_flops = flops_per_token * batch_size

        attention_flops = 2 * batch_size * avg_kv_len * model.hidden_dim * model.num_layers
        total_flops += attention_flops

        effective_flops = gpu.flops_fp16
        if self.tp_group is not None:
            effective_flops *= self.tp_group.size

        compute_time = total_flops / effective_flops

        memory_sym = sym_const(memory_time, "t_memory_decode")
        compute_sym = sym_const(compute_time, "t_compute_decode")

        return sym_max(memory_sym, compute_sym)

    @property
    def queue_length(self) -> int:
        return len(self.waiting_queue) + len(self.running_batch)

```

---

### AI



```
from sim.network.interconnect import InterconnectModel
from sim.network.kv_transfer import KVTransferManager

__all__ = ["InterconnectModel", "KVTransferManager"]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add
from sim.config.cluster import InterconnectConfig, InterconnectType

if TYPE_CHECKING:
    pass


class InterconnectModel:
    def __init__(self, config: InterconnectConfig):
        self.config = config

    def transfer_time(self, bytes_to_transfer: int) -> SymVal:
        latency = self.config.latency_seconds
        bandwidth = self.config.bandwidth_bytes_per_sec

        transfer_time = latency + bytes_to_transfer / bandwidth

        latency_sym = sym_const(latency, "L_net")
        transfer_sym = sym_const(bytes_to_transfer / bandwidth, "t_transfer")

        return sym_add(latency_sym, transfer_sym)

    def collective_time(
        self,
        message_bytes: int,
        num_participants: int,
        op: str,
    ) -> SymVal:
        bandwidth = self.config.bandwidth_bytes_per_sec
        latency = self.config.latency_seconds

        if op == "all_reduce":
            effective_bytes = 2 * message_bytes * (num_participants - 1) / num_participants
        elif op == "all_gather":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        elif op == "reduce_scatter":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        elif op == "all_to_all":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        else:
            effective_bytes = message_bytes

        alpha_beta_time = latency * num_participants + effective_bytes / bandwidth

        return sym_const(alpha_beta_time, f"t_{op}")

    @classmethod
    def for_intra_node(cls, config: InterconnectConfig) -> InterconnectModel:
        return cls(config)

    @classmethod
    def for_inter_node(cls, config: InterconnectConfig) -> InterconnectModel:
        return cls(config)

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add
from sim.network.interconnect import InterconnectModel

if TYPE_CHECKING:
    from sim.core.request import KVHandle
    from sim.config.cluster import ClusterConfig


@dataclass
class KVTransfer:
    request_id: int
    kv_bytes: int
    source_worker: int
    dest_worker: int
    start_time: float
    estimated_end_time: float


class KVTransferManager:
    def __init__(
        self,
        intra_node_interconnect: InterconnectModel,
        inter_node_interconnect: InterconnectModel,
        gpus_per_node: int = 8,
    ):
        self.intra_node_interconnect = intra_node_interconnect
        self.inter_node_interconnect = inter_node_interconnect
        self.gpus_per_node = gpus_per_node

        self.pending_transfers: dict[int, KVTransfer] = {}
        self.completed_transfers: dict[int, KVTransfer] = {}

    def initiate_transfer(
        self,
        kv_handle: KVHandle,
        cluster_config: ClusterConfig,
    ) -> SymVal:
        source_node = kv_handle.source_worker_id // self.gpus_per_node
        dest_node = kv_handle.dest_worker_id // self.gpus_per_node

        is_intra_node = source_node == dest_node

        serialization_overhead = 0.0001  # 100 microseconds
        deserialization_overhead = 0.0001

        if is_intra_node:
            transfer_time = self.intra_node_interconnect.transfer_time(kv_handle.kv_bytes)
        else:
            transfer_time = self.inter_node_interconnect.transfer_time(kv_handle.kv_bytes)

        overhead_sym = sym_const(serialization_overhead + deserialization_overhead, "t_serde")
        total_time = sym_add(transfer_time, overhead_sym)

        return total_time

    def is_same_node(self, worker_a: int, worker_b: int) -> bool:
        return worker_a // self.gpus_per_node == worker_b // self.gpus_per_node

    @classmethod
    def from_cluster_config(cls, cluster_config: ClusterConfig) -> KVTransferManager:
        intra = InterconnectModel(cluster_config.intra_node_interconnect)
        inter = InterconnectModel(cluster_config.inter_node_interconnect)
        return cls(
            intra_node_interconnect=intra,
            inter_node_interconnect=inter,
            gpus_per_node=cluster_config.gpus_per_worker,
        )

```

---

### AI



```
from sim.cache.radix_cache import RadixCache, RadixNode

__all__ = ["RadixCache", "RadixNode"]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from sim.config.cache import CacheConfig


@dataclass
class RadixNode:
    token: int | None = None
    kv_bytes: int = 0
    last_access: float = 0.0
    children: dict[int, RadixNode] = field(default_factory=dict)
    parent: RadixNode | None = None
    depth: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_descendant_bytes(self) -> int:
        total = self.kv_bytes
        for child in self.children.values():
            total += child.total_descendant_bytes()
        return total


class RadixCache:
    def __init__(self, config: CacheConfig, current_time: float = 0.0):
        self.config = config
        self.root = RadixNode(depth=0)
        self.capacity = config.capacity_bytes
        self.used_bytes = 0
        self.current_time = current_time

        self.access_order: OrderedDict[int, RadixNode] = OrderedDict()
        self._node_id_counter = 0

    def match_prefix(self, tokens: list[int]) -> tuple[int, int]:
        matched_tokens = 0
        matched_bytes = 0
        current = self.root

        for token in tokens:
            if token in current.children:
                current = current.children[token]
                matched_tokens += 1
                matched_bytes += current.kv_bytes
                self._update_access(current)
            else:
                break

        return matched_tokens, matched_bytes

    def insert(self, tokens: list[int], total_kv_bytes: int) -> bool:
        if not tokens:
            return True

        bytes_per_token = total_kv_bytes // len(tokens) if tokens else 0
        current = self.root

        for i, token in enumerate(tokens):
            if token in current.children:
                current = current.children[token]
                self._update_access(current)
            else:
                new_node = RadixNode(
                    token=token,
                    kv_bytes=bytes_per_token,
                    last_access=self.current_time,
                    parent=current,
                    depth=current.depth + 1,
                )

                while self.used_bytes + bytes_per_token > self.capacity:
                    if not self._evict_lru():
                        return False

                current.children[token] = new_node
                self.used_bytes += bytes_per_token
                self._register_node(new_node)
                current = new_node

        return True

    def _update_access(self, node: RadixNode) -> None:
        node.last_access = self.current_time
        node_id = id(node)
        if node_id in self.access_order:
            self.access_order.move_to_end(node_id)

    def _register_node(self, node: RadixNode) -> None:
        self.access_order[id(node)] = node

    def _evict_lru(self) -> bool:
        if not self.access_order:
            return False

        candidates = [
            node for node in self.access_order.values()
            if node.is_leaf() and node.parent is not None
        ]

        if not candidates:
            return False

        oldest_node = min(candidates, key=lambda n: n.last_access)
        return self._remove_node(oldest_node)

    def _remove_node(self, node: RadixNode) -> bool:
        if node.parent is None:
            return False

        if not node.is_leaf():
            return False

        parent = node.parent
        if node.token is not None and node.token in parent.children:
            del parent.children[node.token]

        self.used_bytes -= node.kv_bytes
        node_id = id(node)
        if node_id in self.access_order:
            del self.access_order[node_id]

        return True

    def update_time(self, new_time: float) -> None:
        self.current_time = new_time

    def get_hit_rate_estimate(self) -> float:
        if self.capacity == 0:
            return 0.0
        return min(1.0, self.used_bytes / self.capacity)

    @property
    def utilization(self) -> float:
        return self.used_bytes / self.capacity if self.capacity > 0 else 0.0

    def get_all_prefixes(self) -> list[tuple[list[int], int]]:
        prefixes: list[tuple[list[int], int]] = []
        self._collect_prefixes(self.root, [], prefixes)
        return prefixes

    def _collect_prefixes(
        self,
        node: RadixNode,
        current_path: list[int],
        prefixes: list[tuple[list[int], int]],
    ) -> None:
        if node.is_leaf() and current_path:
            total_bytes = sum(
                self._get_path_bytes(current_path[:i+1])
                for i in range(len(current_path))
            )
            prefixes.append((current_path.copy(), node.kv_bytes))

        for token, child in node.children.items():
            current_path.append(token)
            self._collect_prefixes(child, current_path, prefixes)
            current_path.pop()

    def _get_path_bytes(self, path: list[int]) -> int:
        current = self.root
        for token in path:
            if token in current.children:
                current = current.children[token]
            else:
                return 0
        return current.kv_bytes

```

---

### AI



```
from sim.routing.router import Router, RoutingPolicy

__all__ = ["Router", "RoutingPolicy"]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, KVHandle
    from sim.workers.prefill_worker import PrefillWorker
    from sim.workers.decode_worker import DecodeWorker
    from sim.cache.radix_cache import RadixCache


class RoutingPolicy(Enum):
    ROUND_ROBIN = "round_robin"
    SHORTEST_QUEUE = "shortest_queue"
    CACHE_AWARE = "cache_aware"


@dataclass
class ApproxRadixTree:
    worker_id: int
    known_prefixes: list[tuple[list[int], int]] = field(default_factory=list)
    last_updated: float = 0.0

    def estimate_prefix_match(self, tokens: list[int]) -> tuple[int, int]:
        best_match_len = 0
        best_match_bytes = 0

        for prefix, kv_bytes in self.known_prefixes:
            match_len = 0
            for i, (a, b) in enumerate(zip(tokens, prefix)):
                if a == b:
                    match_len = i + 1
                else:
                    break

            if match_len > best_match_len:
                best_match_len = match_len
                # Estimate bytes proportionally
                if len(prefix) > 0:
                    best_match_bytes = int(kv_bytes * match_len / len(prefix))

        return best_match_len, best_match_bytes


class Router:
    def __init__(
        self,
        prefill_workers: list[PrefillWorker],
        decode_workers: list[DecodeWorker],
        prefill_policy: RoutingPolicy = RoutingPolicy.CACHE_AWARE,
        decode_policy: RoutingPolicy = RoutingPolicy.ROUND_ROBIN,
        cache_threshold: float = 0.5,
        balance_threshold: float = 2.0,
    ):
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.prefill_policy = prefill_policy
        self.decode_policy = decode_policy
        self.cache_threshold = cache_threshold
        self.balance_threshold = balance_threshold

        self.approx_trees: dict[int, ApproxRadixTree] = {
            w.worker_id: ApproxRadixTree(worker_id=w.worker_id)
            for w in prefill_workers
        }

        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

    def route_to_prefill(self, request: Request) -> int:
        if self.prefill_policy == RoutingPolicy.ROUND_ROBIN:
            return self._round_robin_prefill()
        elif self.prefill_policy == RoutingPolicy.SHORTEST_QUEUE:
            return self._shortest_queue_prefill()
        elif self.prefill_policy == RoutingPolicy.CACHE_AWARE:
            return self._cache_aware_prefill(request)
        return self._round_robin_prefill()

    def route_to_decode(self, kv_handle: KVHandle) -> int:
        if self.decode_policy == RoutingPolicy.ROUND_ROBIN:
            return self._round_robin_decode()
        elif self.decode_policy == RoutingPolicy.SHORTEST_QUEUE:
            return self._shortest_queue_decode()
        return self._round_robin_decode()

    def _round_robin_prefill(self) -> int:
        worker_id = self.prefill_rr_counter % len(self.prefill_workers)
        self.prefill_rr_counter += 1
        return worker_id

    def _round_robin_decode(self) -> int:
        worker_id = self.decode_rr_counter % len(self.decode_workers)
        self.decode_rr_counter += 1
        return worker_id

    def _shortest_queue_prefill(self) -> int:
        return min(
            range(len(self.prefill_workers)),
            key=lambda i: self.prefill_workers[i].queue_length,
        )

    def _shortest_queue_decode(self) -> int:
        return min(
            range(len(self.decode_workers)),
            key=lambda i: self.decode_workers[i].queue_length,
        )

    def _cache_aware_prefill(self, request: Request) -> int:
        queue_lengths = [w.queue_length for w in self.prefill_workers]
        min_queue = min(queue_lengths) if queue_lengths else 0
        max_queue = max(queue_lengths) if queue_lengths else 0

        if max_queue > 0 and max_queue / max(1, min_queue) > self.balance_threshold:
            return self._shortest_queue_prefill()

        best_worker = 0
        best_match_ratio = 0.0

        for worker_id, worker in enumerate(self.prefill_workers):
            if worker.cache is not None:
                matched_tokens, _ = worker.cache.match_prefix(request.prompt_tokens)
                match_ratio = matched_tokens / len(request.prompt_tokens) if request.prompt_tokens else 0
            else:
                approx_tree = self.approx_trees.get(worker_id)
                if approx_tree:
                    matched_tokens, _ = approx_tree.estimate_prefix_match(request.prompt_tokens)
                    match_ratio = matched_tokens / len(request.prompt_tokens) if request.prompt_tokens else 0
                else:
                    match_ratio = 0.0

            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_worker = worker_id

        if best_match_ratio >= self.cache_threshold:
            return best_worker

        return self._worker_with_most_capacity()

    def _worker_with_most_capacity(self) -> int:
        best_worker = 0
        best_capacity = 0

        for worker_id, worker in enumerate(self.prefill_workers):
            if worker.cache is not None:
                remaining = worker.cache.capacity - worker.cache.used_bytes
            else:
                remaining = float('inf')

            if remaining > best_capacity:
                best_capacity = remaining
                best_worker = worker_id

        return best_worker

    def update_approx_tree(self, worker_id: int, prefixes: list[tuple[list[int], int]], time: float) -> None:
        if worker_id in self.approx_trees:
            self.approx_trees[worker_id].known_prefixes = prefixes
            self.approx_trees[worker_id].last_updated = time

    def is_balanced(self) -> bool:
        queue_lengths = [w.queue_length for w in self.prefill_workers]
        if not queue_lengths or max(queue_lengths) == 0:
            return True
        return max(queue_lengths) / max(1, min(queue_lengths)) <= self.balance_threshold

```

---

### AI



```
from sim.parallel.groups import TPGroup, DPGroup, EPGroup
from sim.parallel.collectives import CollectiveOp, collective_cost

__all__ = ["TPGroup", "DPGroup", "EPGroup", "CollectiveOp", "collective_cost"]

```

---

### AI



```
from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add

if TYPE_CHECKING:
    from sim.config.cluster import InterconnectConfig


class CollectiveOp(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


def collective_cost(
    op: CollectiveOp,
    message_bytes: int,
    num_participants: int,
    bandwidth_bytes_per_sec: float,
    latency_seconds: float,
) -> SymVal:
    if num_participants <= 1:
        return sym_const(0.0, f"t_{op.value}_trivial")

    if op == CollectiveOp.ALL_REDUCE:
        effective_bytes = 2 * message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.ALL_GATHER:
        effective_bytes = message_bytes * (num_participants - 1)
    elif op == CollectiveOp.REDUCE_SCATTER:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.ALL_TO_ALL:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.BROADCAST:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    else:
        effective_bytes = message_bytes

    alpha = latency_seconds
    beta_inv = bandwidth_bytes_per_sec

    log_p = 1
    p = num_participants
    while p > 1:
        p //= 2
        log_p += 1

    total_time = log_p * alpha + effective_bytes / beta_inv

    return sym_const(total_time, f"t_{op.value}")


def ring_all_reduce_cost(
    message_bytes: int,
    num_participants: int,
    bandwidth_bytes_per_sec: float,
    latency_seconds: float,
) -> SymVal:
    if num_participants <= 1:
        return sym_const(0.0, "t_ring_ar_trivial")

    n = num_participants
    effective_bytes = 2 * message_bytes * (n - 1) / n
    total_time = 2 * (n - 1) * latency_seconds + effective_bytes / bandwidth_bytes_per_sec

    return sym_const(total_time, "t_ring_all_reduce")

```

---

### AI



```
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

```

---

### AI



```
from sim.kernels.attention import AttentionKernel
from sim.kernels.mlp import MLPKernel
from sim.kernels.moe import MoEKernel

__all__ = ["AttentionKernel", "MLPKernel", "MoEKernel"]

```

---

### AI



```
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

```

---

### AI



```
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

```

---

### AI



```
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

```

---

### AI



```
from sim.models.operator_graph import OperatorGraph

__all__ = ["OperatorGraph"]

```

---

### AI



```
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

```

---

### AI



```
from sim.metrics.definitions import MetricCollector, RequestMetrics
from sim.metrics.symbolic import SymbolicThroughputBuilder

__all__ = ["MetricCollector", "RequestMetrics", "SymbolicThroughputBuilder"]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import statistics

if TYPE_CHECKING:
    from sim.core.request import Request


@dataclass
class RequestMetrics:
    request_id: int
    ttft: float
    tpot: float
    e2e_latency: float
    queue_time: float
    prefill_time: float
    transfer_time: float
    decode_time: float
    prompt_len: int
    output_len: int


class MetricCollector:
    def __init__(self):
        self.request_metrics: list[RequestMetrics] = []
        self.total_requests: int = 0
        self.completed_requests: int = 0
        self.total_tokens: int = 0

        self._ttfts: list[float] = []
        self._tpots: list[float] = []
        self._e2e_latencies: list[float] = []
        self._queue_times: list[float] = []

    def record_request_complete(self, request: Request) -> None:
        self.completed_requests += 1
        self.total_tokens += request.tokens_generated

        if request.ttft is not None:
            self._ttfts.append(request.ttft)

        if request.tpot is not None:
            self._tpots.append(request.tpot)

        if request.e2e_latency is not None:
            self._e2e_latencies.append(request.e2e_latency)

        metrics = RequestMetrics(
            request_id=request.id,
            ttft=request.ttft or 0.0,
            tpot=request.tpot or 0.0,
            e2e_latency=request.e2e_latency or 0.0,
            queue_time=0.0,
            prefill_time=0.0,
            transfer_time=0.0,
            decode_time=0.0,
            prompt_len=request.prompt_len,
            output_len=request.tokens_generated,
        )
        self.request_metrics.append(metrics)

    def record_request_arrival(self, request: Request) -> None:
        self.total_requests += 1

    def throughput(self, elapsed_time: float) -> float:
        if elapsed_time <= 0:
            return 0.0
        return self.total_tokens / elapsed_time

    def avg_ttft(self) -> float:
        if not self._ttfts:
            return 0.0
        return statistics.mean(self._ttfts)

    def avg_tpot(self) -> float:
        if not self._tpots:
            return 0.0
        return statistics.mean(self._tpots)

    def avg_e2e_latency(self) -> float:
        if not self._e2e_latencies:
            return 0.0
        return statistics.mean(self._e2e_latencies)

    def percentile_ttft(self, p: float) -> float:
        if not self._ttfts:
            return 0.0
        return self._percentile(self._ttfts, p)

    def percentile_tpot(self, p: float) -> float:
        if not self._tpots:
            return 0.0
        return self._percentile(self._tpots, p)

    def percentile_e2e(self, p: float) -> float:
        if not self._e2e_latencies:
            return 0.0
        return self._percentile(self._e2e_latencies, p)

    def _percentile(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        if f == c:
            return sorted_data[f]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "total_tokens": self.total_tokens,
            "avg_ttft": self.avg_ttft(),
            "avg_tpot": self.avg_tpot(),
            "avg_e2e_latency": self.avg_e2e_latency(),
            "p50_ttft": self.percentile_ttft(0.5),
            "p99_ttft": self.percentile_ttft(0.99),
            "p50_tpot": self.percentile_tpot(0.5),
            "p99_tpot": self.percentile_tpot(0.99),
        }

```

---

### AI



```
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

```

---

### AI



```
from sim.optimizer.interface import OptimizerInterface, OptimizationResult

__all__ = ["OptimizerInterface", "OptimizationResult"]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from itertools import product

import sympy
from sympy import Symbol
import numpy as np

from sim.symbolic.symbols import ConfigSymbols


@dataclass
class OptimizationResult:
    optimal_values: dict[str, float]
    optimal_objective: float
    success: bool
    message: str
    all_evaluations: list[tuple[dict, float]] = field(default_factory=list)


@dataclass
class Constraint:
    expr: sympy.Expr
    type: str  # "eq" for equality, "ineq" for inequality (>= 0)
    name: str = ""


class OptimizerInterface:
    def __init__(
        self,
        objective: sympy.Expr,
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
        maximize: bool = True,
    ):
        self.objective = objective
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self.maximize = maximize

        self._decision_vars: list[Symbol] = []
        self._frozen_params: dict[Symbol, float] = {}

    def set_decision_vars(self, vars: list[Symbol]) -> None:
        self._decision_vars = vars

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def get_decision_vars(self) -> list[Symbol]:
        if self._decision_vars:
            return self._decision_vars
        return self.symbols.decision_vars()

    def get_frozen_params(self) -> list[Symbol]:
        return self.symbols.frozen_params()

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def to_scipy_minimize(
        self,
        bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        decision_vars = self.get_decision_vars()

        substituted_obj = self.substitute_frozen(self.objective)

        if self.maximize:
            obj_to_minimize = -substituted_obj
        else:
            obj_to_minimize = substituted_obj

        obj_func = sympy.lambdify(decision_vars, obj_to_minimize, modules=["numpy"])

        def objective_wrapper(x: np.ndarray) -> float:
            return float(obj_func(*x))

        scipy_constraints = []
        for constraint in self.constraints:
            substituted = self.substitute_frozen(constraint.expr)
            constraint_func = sympy.lambdify(decision_vars, substituted, modules=["numpy"])

            scipy_constraints.append({
                "type": constraint.type,
                "fun": lambda x, f=constraint_func: float(f(*x)),
            })

        scipy_bounds = None
        if bounds:
            scipy_bounds = [
                bounds.get(var, (None, None)) for var in decision_vars
            ]

        return {
            "fun": objective_wrapper,
            "x0": np.ones(len(decision_vars)),
            "bounds": scipy_bounds,
            "constraints": scipy_constraints,
            "method": "SLSQP",
        }

    def enumerate_discrete_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> list[dict[Symbol, int]]:
        regimes = []
        var_names = list(discrete_vars.keys())
        var_values = list(discrete_vars.values())

        for combo in product(*var_values):
            regime = dict(zip(var_names, combo))
            regimes.append(regime)

        return regimes

    def optimize_over_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> OptimizationResult:
        from scipy.optimize import minimize

        regimes = self.enumerate_discrete_regimes(discrete_vars, continuous_bounds)
        best_result: OptimizationResult | None = None
        all_evaluations: list[tuple[dict, float]] = []

        decision_vars = self.get_decision_vars()
        continuous_vars = [v for v in decision_vars if v not in discrete_vars]

        for regime in regimes:
            regime_objective = self.objective.subs(list(regime.items()))
            regime_objective = self.substitute_frozen(regime_objective)

            if not continuous_vars:
                if self.maximize:
                    obj_val = -float(regime_objective)
                else:
                    obj_val = float(regime_objective)

                result_dict = {str(k): v for k, v in regime.items()}
                all_evaluations.append((result_dict, -obj_val if self.maximize else obj_val))

                if best_result is None or (self.maximize and -obj_val > best_result.optimal_objective) or \
                   (not self.maximize and obj_val < best_result.optimal_objective):
                    best_result = OptimizationResult(
                        optimal_values=result_dict,
                        optimal_objective=-obj_val if self.maximize else obj_val,
                        success=True,
                        message=f"Regime: {regime}",
                    )
            else:
                obj_func = sympy.lambdify(continuous_vars, regime_objective, modules=["numpy"])

                if self.maximize:
                    def objective_wrapper(x):
                        return -float(obj_func(*x))
                else:
                    def objective_wrapper(x):
                        return float(obj_func(*x))

                bounds_list = [
                    continuous_bounds.get(v, (1, 100)) if continuous_bounds else (1, 100)
                    for v in continuous_vars
                ]

                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_list])

                try:
                    result = minimize(
                        objective_wrapper,
                        x0,
                        bounds=bounds_list,
                        method="L-BFGS-B",
                    )

                    result_dict = {str(k): v for k, v in regime.items()}
                    for var, val in zip(continuous_vars, result.x):
                        result_dict[str(var)] = float(val)

                    obj_val = -result.fun if self.maximize else result.fun
                    all_evaluations.append((result_dict, obj_val))

                    if best_result is None or \
                       (self.maximize and obj_val > best_result.optimal_objective) or \
                       (not self.maximize and obj_val < best_result.optimal_objective):
                        best_result = OptimizationResult(
                            optimal_values=result_dict,
                            optimal_objective=obj_val,
                            success=result.success,
                            message=result.message,
                        )
                except Exception as e:
                    continue

        if best_result is None:
            return OptimizationResult(
                optimal_values={},
                optimal_objective=float("inf") if not self.maximize else float("-inf"),
                success=False,
                message="No valid regime found",
                all_evaluations=all_evaluations,
            )

        best_result.all_evaluations = all_evaluations
        return best_result

    def grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())

        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []

        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])

        for combo in product(*var_values):
            try:
                obj_val = float(obj_func(*combo))
                if self.maximize:
                    obj_val = obj_val
                else:
                    obj_val = obj_val

                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))

                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue

        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Grid search complete",
            all_evaluations=all_evaluations,
        )

```

---

### AI



```
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution
from sim.workload.trace_loader import TraceWorkloadLoader
from sim.workload.prefix_groups import PrefixGroup

__all__ = [
    "SyntheticWorkloadGenerator",
    "Distribution",
    "TraceWorkloadLoader",
    "PrefixGroup",
]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
import random


@dataclass
class PrefixGroup:
    prefix_tokens: list[int]
    share_ratio: float
    group_id: int = 0

    @property
    def prefix_length(self) -> int:
        return len(self.prefix_tokens)

    def sample_tokens(self, additional_length: int, vocab_size: int = 32000) -> list[int]:
        suffix = [random.randint(0, vocab_size - 1) for _ in range(additional_length)]
        return self.prefix_tokens + suffix


@dataclass
class PrefixGroupManager:
    groups: list[PrefixGroup] = field(default_factory=list)
    vocab_size: int = 32000

    def add_group(
        self,
        prefix_length: int,
        share_ratio: float,
        group_id: int | None = None,
    ) -> PrefixGroup:
        if group_id is None:
            group_id = len(self.groups)

        prefix_tokens = [random.randint(0, self.vocab_size - 1) for _ in range(prefix_length)]
        group = PrefixGroup(
            prefix_tokens=prefix_tokens,
            share_ratio=share_ratio,
            group_id=group_id,
        )
        self.groups.append(group)
        return group

    def sample_group(self) -> PrefixGroup | None:
        if not self.groups:
            return None

        total_ratio = sum(g.share_ratio for g in self.groups)
        if total_ratio == 0:
            return None

        r = random.random() * total_ratio
        cumulative = 0.0
        for group in self.groups:
            cumulative += group.share_ratio
            if r <= cumulative:
                return group

        return self.groups[-1] if self.groups else None

    def generate_prompt(self, total_length: int) -> tuple[list[int], int | None]:
        group = self.sample_group()
        if group is None:
            tokens = [random.randint(0, self.vocab_size - 1) for _ in range(total_length)]
            return tokens, None

        if group.prefix_length >= total_length:
            return group.prefix_tokens[:total_length], group.group_id

        additional = total_length - group.prefix_length
        tokens = group.sample_tokens(additional, self.vocab_size)
        return tokens, group.group_id

    @classmethod
    def create_uniform_groups(
        cls,
        num_groups: int,
        prefix_length: int,
        vocab_size: int = 32000,
    ) -> PrefixGroupManager:
        manager = cls(vocab_size=vocab_size)
        share_per_group = 1.0 / num_groups
        for i in range(num_groups):
            manager.add_group(prefix_length, share_per_group, group_id=i)
        return manager

    @classmethod
    def create_zipf_groups(
        cls,
        num_groups: int,
        prefix_length: int,
        zipf_param: float = 1.0,
        vocab_size: int = 32000,
    ) -> PrefixGroupManager:
        manager = cls(vocab_size=vocab_size)

        weights = [1.0 / (i + 1) ** zipf_param for i in range(num_groups)]
        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]

        for i, ratio in enumerate(normalized):
            manager.add_group(prefix_length, ratio, group_id=i)

        return manager

```

---

### AI



```
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from typing import TYPE_CHECKING

from sim.core.request import Request
from sim.workload.prefix_groups import PrefixGroupManager

if TYPE_CHECKING:
    pass


class DistributionType(Enum):
    CONSTANT = "constant"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    ZIPF = "zipf"


@dataclass
class Distribution:
    type: DistributionType
    params: dict = field(default_factory=dict)

    def sample(self) -> float:
        if self.type == DistributionType.CONSTANT:
            return self.params.get("value", 0.0)

        elif self.type == DistributionType.UNIFORM:
            low = self.params.get("low", 0.0)
            high = self.params.get("high", 1.0)
            return random.uniform(low, high)

        elif self.type == DistributionType.EXPONENTIAL:
            rate = self.params.get("rate", 1.0)
            return random.expovariate(rate)

        elif self.type == DistributionType.NORMAL:
            mean = self.params.get("mean", 0.0)
            std = self.params.get("std", 1.0)
            return random.gauss(mean, std)

        elif self.type == DistributionType.LOGNORMAL:
            mean = self.params.get("mean", 0.0)
            sigma = self.params.get("sigma", 1.0)
            return random.lognormvariate(mean, sigma)

        elif self.type == DistributionType.ZIPF:
            alpha = self.params.get("alpha", 1.5)
            n = self.params.get("n", 1000)
            return self._zipf_sample(alpha, n)

        return 0.0

    def sample_int(self, min_val: int = 1, max_val: int | None = None) -> int:
        value = int(self.sample())
        value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
        return value

    def _zipf_sample(self, alpha: float, n: int) -> float:
        harmonic = sum(1.0 / (i ** alpha) for i in range(1, n + 1))
        u = random.random()
        cumulative = 0.0
        for i in range(1, n + 1):
            cumulative += 1.0 / (i ** alpha) / harmonic
            if u <= cumulative:
                return float(i)
        return float(n)

    @classmethod
    def constant(cls, value: float) -> Distribution:
        return cls(DistributionType.CONSTANT, {"value": value})

    @classmethod
    def uniform(cls, low: float, high: float) -> Distribution:
        return cls(DistributionType.UNIFORM, {"low": low, "high": high})

    @classmethod
    def exponential(cls, rate: float) -> Distribution:
        return cls(DistributionType.EXPONENTIAL, {"rate": rate})

    @classmethod
    def normal(cls, mean: float, std: float) -> Distribution:
        return cls(DistributionType.NORMAL, {"mean": mean, "std": std})

    @classmethod
    def lognormal(cls, mean: float, sigma: float) -> Distribution:
        return cls(DistributionType.LOGNORMAL, {"mean": mean, "sigma": sigma})


class SyntheticWorkloadGenerator:
    def __init__(
        self,
        arrival_rate: float,
        prompt_len_dist: Distribution,
        output_len_dist: Distribution,
        prefix_groups: PrefixGroupManager | None = None,
        vocab_size: int = 32000,
        seed: int | None = None,
    ):
        self.arrival_rate = arrival_rate
        self.prompt_len_dist = prompt_len_dist
        self.output_len_dist = output_len_dist
        self.prefix_groups = prefix_groups
        self.vocab_size = vocab_size

        if seed is not None:
            random.seed(seed)

        self._request_counter = 0

    def generate(self, duration: float) -> list[Request]:
        requests: list[Request] = []
        current_time = 0.0

        while current_time < duration:
            inter_arrival = random.expovariate(self.arrival_rate)
            current_time += inter_arrival

            if current_time >= duration:
                break

            prompt_len = self.prompt_len_dist.sample_int(min_val=1, max_val=32768)
            output_len = self.output_len_dist.sample_int(min_val=1, max_val=4096)

            if self.prefix_groups is not None:
                prompt_tokens, prefix_group_id = self.prefix_groups.generate_prompt(prompt_len)
            else:
                prompt_tokens = [
                    random.randint(0, self.vocab_size - 1) for _ in range(prompt_len)
                ]
                prefix_group_id = None

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
                prefix_group_id=prefix_group_id,
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    def generate_batch(self, num_requests: int, start_time: float = 0.0) -> list[Request]:
        requests: list[Request] = []
        current_time = start_time

        for _ in range(num_requests):
            inter_arrival = random.expovariate(self.arrival_rate)
            current_time += inter_arrival

            prompt_len = self.prompt_len_dist.sample_int(min_val=1, max_val=32768)
            output_len = self.output_len_dist.sample_int(min_val=1, max_val=4096)

            if self.prefix_groups is not None:
                prompt_tokens, prefix_group_id = self.prefix_groups.generate_prompt(prompt_len)
            else:
                prompt_tokens = [
                    random.randint(0, self.vocab_size - 1) for _ in range(prompt_len)
                ]
                prefix_group_id = None

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
                prefix_group_id=prefix_group_id,
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    @classmethod
    def default(cls, arrival_rate: float = 10.0) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=arrival_rate,
            prompt_len_dist=Distribution.uniform(256, 2048),
            output_len_dist=Distribution.uniform(64, 512),
        )

    @classmethod
    def chat_workload(cls, arrival_rate: float = 5.0) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=arrival_rate,
            prompt_len_dist=Distribution.lognormal(mean=6.0, sigma=1.0),  # ~400 avg
            output_len_dist=Distribution.lognormal(mean=5.0, sigma=1.0),  # ~150 avg
        )

    @classmethod
    def batch_inference(cls, num_requests: int = 1000) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=float("inf"),
            prompt_len_dist=Distribution.uniform(512, 4096),
            output_len_dist=Distribution.uniform(128, 1024),
        )

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass
import json
import csv
from pathlib import Path
from typing import TYPE_CHECKING

from sim.core.request import Request

if TYPE_CHECKING:
    pass


@dataclass
class TraceFormat:
    VIDUR = "vidur"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class TraceWorkloadLoader:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._request_counter = 0

    def load(self, path: str, format: str = "auto") -> list[Request]:
        file_path = Path(path)

        if format == "auto":
            format = self._detect_format(file_path)

        if format == TraceFormat.VIDUR:
            return self._load_vidur(file_path)
        elif format == TraceFormat.SHAREGPT:
            return self._load_sharegpt(file_path)
        elif format == TraceFormat.CUSTOM:
            return self._load_custom(file_path)
        else:
            raise ValueError(f"Unknown trace format: {format}")

    def _detect_format(self, path: Path) -> str:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return TraceFormat.VIDUR
        elif suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if "conversations" in data[0]:
                        return TraceFormat.SHAREGPT
            return TraceFormat.CUSTOM
        return TraceFormat.CUSTOM

    def _load_vidur(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt_len = int(row.get("prompt_len", row.get("input_tokens", 512)))
                output_len = int(row.get("output_len", row.get("output_tokens", 128)))
                arrival_time = float(row.get("arrival_time", 0.0))

                prompt_tokens = self._generate_dummy_tokens(prompt_len)

                request = Request(
                    id=self._request_counter,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=output_len,
                    arrival_time=arrival_time,
                )
                requests.append(request)
                self._request_counter += 1

        return requests

    def _load_sharegpt(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            data = json.load(f)

        current_time = 0.0
        inter_arrival = 0.1

        for item in data:
            conversations = item.get("conversations", [])

            prompt_len = 0
            output_len = 0

            for conv in conversations:
                if conv.get("from") == "human":
                    prompt_len += len(conv.get("value", "").split())
                elif conv.get("from") == "gpt":
                    output_len += len(conv.get("value", "").split())

            prompt_len = max(1, int(prompt_len * 1.3))
            output_len = max(1, int(output_len * 1.3))

            prompt_tokens = self._generate_dummy_tokens(prompt_len)

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
            )
            requests.append(request)
            self._request_counter += 1
            current_time += inter_arrival

        return requests

    def _load_custom(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("requests", [])

        for item in data:
            prompt_tokens = item.get("prompt_tokens")
            if prompt_tokens is None:
                prompt_len = item.get("prompt_len", 512)
                prompt_tokens = self._generate_dummy_tokens(prompt_len)

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=item.get("max_new_tokens", item.get("output_len", 128)),
                arrival_time=item.get("arrival_time", 0.0),
                stream=item.get("stream", False),
                prefix_group_id=item.get("prefix_group_id"),
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    def _generate_dummy_tokens(self, length: int) -> list[int]:
        import random
        return [random.randint(0, self.vocab_size - 1) for _ in range(length)]

    def save_requests(self, requests: list[Request], path: str) -> None:
        data = {
            "requests": [
                {
                    "id": r.id,
                    "prompt_len": r.prompt_len,
                    "max_new_tokens": r.max_new_tokens,
                    "arrival_time": r.arrival_time,
                    "stream": r.stream,
                    "prefix_group_id": r.prefix_group_id,
                }
                for r in requests
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

```

---

### AI

Now let me create the example files to demonstrate usage:

---

### AI



```
#!/usr/bin/env python3
"""Basic simulation example showing the core workflow of the LLM inference simulator."""

import sys
sys.path.insert(0, "..")

from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig, GPUSpec, InterconnectConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.core.events import RequestArrival
from sim.workers.prefill_worker import PrefillWorker
from sim.workers.decode_worker import DecodeWorker
from sim.routing.router import Router, RoutingPolicy
from sim.network.kv_transfer import KVTransferManager
from sim.cache.radix_cache import RadixCache
from sim.metrics.definitions import MetricCollector
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution


def run_basic_simulation():
    model_config = ModelConfig.llama_7b()
    cluster_config = ClusterConfig.single_node_8gpu()
    scheduler_config = SchedulerConfig.default()
    cache_config = CacheConfig.h100_default()

    engine = SimulationEngine(
        model_config=model_config,
        cluster_config=cluster_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )

    prefill_workers = []
    for i in range(cluster_config.num_prefill_workers):
        cache = RadixCache(cache_config)
        worker = PrefillWorker(worker_id=i, cache=cache)
        prefill_workers.append(worker)

    decode_workers = []
    for i in range(cluster_config.num_decode_workers):
        worker = DecodeWorker(worker_id=i)
        decode_workers.append(worker)

    router = Router(
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        prefill_policy=RoutingPolicy.CACHE_AWARE,
        decode_policy=RoutingPolicy.ROUND_ROBIN,
    )

    kv_transfer = KVTransferManager.from_cluster_config(cluster_config)
    metrics = MetricCollector()

    engine.initialize_components(
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        router=router,
        kv_transfer_manager=kv_transfer,
        metrics=metrics,
    )

    workload_gen = SyntheticWorkloadGenerator(
        arrival_rate=5.0,
        prompt_len_dist=Distribution.uniform(256, 1024),
        output_len_dist=Distribution.uniform(64, 256),
        seed=42,
    )

    requests = workload_gen.generate(duration=10.0)
    print(f"Generated {len(requests)} requests")

    for request in requests:
        engine.schedule_event(RequestArrival(time=request.arrival_time, request=request))
        metrics.record_request_arrival(request)

    result = engine.run_until_idle()

    print("\n=== Simulation Results ===")
    print(f"Total requests: {result.total_requests}")
    print(f"Completed requests: {result.completed_requests}")
    print(f"Total tokens generated: {result.total_tokens_generated}")
    print(f"Simulation time: {result.simulation_time:.4f}s")
    print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"Avg TTFT: {result.avg_ttft * 1000:.2f}ms")
    print(f"Avg TPOT: {result.avg_tpot * 1000:.4f}ms")
    print(f"Avg E2E Latency: {result.avg_e2e_latency * 1000:.2f}ms")
    print(f"P50 TTFT: {result.p50_ttft * 1000:.2f}ms")
    print(f"P99 TTFT: {result.p99_ttft * 1000:.2f}ms")

    return result


if __name__ == "__main__":
    run_basic_simulation()

```

---

### AI



```
#!/usr/bin/env python3
"""Example demonstrating P/D ratio optimization using symbolic throughput expressions."""

import sys
sys.path.insert(0, "..")

import sympy
from sympy import Symbol, ceiling

from sim.symbolic.symbols import ConfigSymbols
from sim.symbolic.expr import SymVal, sym_const
from sim.metrics.symbolic import SymbolicThroughputBuilder
from sim.optimizer.interface import OptimizerInterface, Constraint
from sim.config.model import ModelConfig
from sim.config.cluster import GPUSpec


def build_throughput_model():
    symbols = ConfigSymbols()

    N_p = symbols.N_p
    N_d = symbols.N_d
    chunk_size = symbols.chunk_size
    batch_cap = symbols.batch_cap_tokens

    gpu_flops = symbols.gpu_flops
    gpu_mem_bw = symbols.gpu_mem_bw

    avg_prompt_len = symbols.avg_prompt_len
    avg_output_len = symbols.avg_output_len

    num_layers = symbols.num_layers
    hidden_dim = symbols.hidden_dim
    kv_bytes = symbols.kv_bytes_per_token

    prefill_flops_per_chunk = 4 * num_layers * hidden_dim * hidden_dim * chunk_size
    prefill_time_per_chunk = prefill_flops_per_chunk / gpu_flops

    num_prefill_chunks = ceiling(avg_prompt_len / chunk_size)
    total_prefill_time = num_prefill_chunks * prefill_time_per_chunk

    prefill_capacity = N_p / total_prefill_time

    avg_kv_len = avg_prompt_len + avg_output_len / 2
    kv_read_bytes = avg_kv_len * kv_bytes
    decode_time_per_token = kv_read_bytes / gpu_mem_bw

    decode_capacity = N_d * batch_cap / decode_time_per_token

    arrival_rate = Symbol("lambda", positive=True)

    input_rate = arrival_rate * avg_prompt_len
    output_rate = arrival_rate * avg_output_len

    prefill_util = input_rate / prefill_capacity
    decode_util = output_rate / decode_capacity

    max_throughput = sympy.Min(
        prefill_capacity * avg_output_len / avg_prompt_len,
        decode_capacity,
    )

    return max_throughput, symbols


def optimize_pd_ratio():
    throughput_expr, symbols = build_throughput_model()

    frozen_params = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.kv_bytes_per_token: 256,
        symbols.chunk_size: 8192,
        symbols.batch_cap_tokens: 256,
    }

    optimizer = OptimizerInterface(
        objective=throughput_expr,
        symbols=symbols,
        maximize=True,
    )
    optimizer.set_frozen_params(frozen_params)

    result = optimizer.grid_search(
        param_grid={
            symbols.N_p: [1, 2, 3, 4, 5, 6, 7, 8],
            symbols.N_d: [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    print("\n=== P/D Ratio Optimization Results ===")
    print(f"Optimal N_p: {result.optimal_values.get('N_p', 'N/A')}")
    print(f"Optimal N_d: {result.optimal_values.get('N_d', 'N/A')}")
    print(f"Optimal throughput: {result.optimal_objective:.2f} tokens/sec")

    print("\n=== Top 5 Configurations ===")
    sorted_evals = sorted(result.all_evaluations, key=lambda x: x[1], reverse=True)[:5]
    for i, (config, throughput) in enumerate(sorted_evals, 1):
        print(f"{i}. N_p={config['N_p']}, N_d={config['N_d']}: {throughput:.2f} tok/s")

    print("\n=== Symbolic Throughput Expression ===")
    print(f"TP = {throughput_expr}")

    return result


def analyze_scaling():
    throughput_expr, symbols = build_throughput_model()

    frozen_base = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.kv_bytes_per_token: 256,
        symbols.chunk_size: 8192,
        symbols.batch_cap_tokens: 256,
        symbols.N_p: 4,
        symbols.N_d: 4,
    }

    print("\n=== Throughput vs Prompt Length ===")
    for prompt_len in [128, 256, 512, 1024, 2048, 4096]:
        params = frozen_base.copy()
        params[symbols.avg_prompt_len] = prompt_len
        params[symbols.avg_output_len] = 128

        tp = float(throughput_expr.subs(list(params.items())))
        print(f"Prompt len {prompt_len:4d}: {tp:10.2f} tok/s")

    print("\n=== Throughput vs Output Length ===")
    for output_len in [32, 64, 128, 256, 512, 1024]:
        params = frozen_base.copy()
        params[symbols.avg_prompt_len] = 512
        params[symbols.avg_output_len] = output_len

        tp = float(throughput_expr.subs(list(params.items())))
        print(f"Output len {output_len:4d}: {tp:10.2f} tok/s")


if __name__ == "__main__":
    optimize_pd_ratio()
    analyze_scaling()

```

---

### AI



```
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

```

---

### AI



```
# Tests for SGLang LLM Inference Simulator

```

---

### AI



```
import pytest
import sympy
from sympy import Symbol

from sim.symbolic.expr import (
    SymVal,
    sym_add,
    sym_sub,
    sym_mul,
    sym_div,
    sym_max,
    sym_min,
    sym_const,
    sym_piecewise,
    sym_sum,
)
from sim.symbolic.symbols import ConfigSymbols


class TestSymVal:
    def test_creation(self):
        sv = SymVal(expr=sympy.Float(1.5), val=1.5)
        assert sv.val == 1.5
        assert sv.expr == sympy.Float(1.5)

    def test_add(self):
        a = sym_const(2.0)
        b = sym_const(3.0)
        result = sym_add(a, b)
        assert result.val == 5.0

    def test_sub(self):
        a = sym_const(5.0)
        b = sym_const(3.0)
        result = sym_sub(a, b)
        assert result.val == 2.0

    def test_mul(self):
        a = sym_const(2.0)
        b = sym_const(3.0)
        result = sym_mul(a, b)
        assert result.val == 6.0

    def test_div(self):
        a = sym_const(6.0)
        b = sym_const(2.0)
        result = sym_div(a, b)
        assert result.val == 3.0

    def test_max(self):
        a = sym_const(2.0)
        b = sym_const(5.0)
        result = sym_max(a, b)
        assert result.val == 5.0

    def test_min(self):
        a = sym_const(2.0)
        b = sym_const(5.0)
        result = sym_min(a, b)
        assert result.val == 2.0

    def test_symbolic_expression(self):
        x = Symbol("x")
        sv = SymVal(expr=x * 2, val=4.0, meta={"name": "test"})
        assert sv.expr == x * 2
        assert sv.val == 4.0

    def test_piecewise(self):
        x = Symbol("x")
        condition = x > 5
        sv_true = sym_const(10.0)
        sv_false = sym_const(1.0)

        result = sym_piecewise(
            [(condition, sv_true), (True, sv_false)],
            numeric_condition_results=[True],
        )
        assert result.val == 10.0

    def test_lambdify(self):
        x = Symbol("x")
        sv = SymVal(expr=x**2, val=4.0)
        func = sv.lambdify([x])
        assert func(3) == 9


class TestConfigSymbols:
    def test_decision_vars(self):
        symbols = ConfigSymbols()
        dvars = symbols.decision_vars()
        assert len(dvars) > 0
        assert symbols.N_p in dvars
        assert symbols.N_d in dvars

    def test_frozen_params(self):
        symbols = ConfigSymbols()
        fparams = symbols.frozen_params()
        assert symbols.cache_hit_rate in fparams

    def test_all_symbols(self):
        symbols = ConfigSymbols()
        all_syms = symbols.all_symbols()
        assert len(all_syms) > 10

```

---

### AI

Let me verify the code syntax and run a basic import test:

---

### AI

I need to fix the dataclass ordering issue in the events file:

---

### AI



```
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, PrefillTask, DecodeTask, KVHandle
    from sim.core.engine import SimulationEngine


@dataclass(order=True)
class Event(ABC):
    time: float
    priority: int = field(default=0, compare=True)

    @abstractmethod
    def process(self, engine: SimulationEngine) -> list[Event]:
        pass


@dataclass(order=True)
class RequestArrival(Event):
    time: float = field(compare=True)
    priority: int = field(default=10, compare=True)
    request: Request = field(default=None, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        engine.state.pending_requests.append(self.request)
        worker_id = engine.router.route_to_prefill(self.request)
        return [RouterDispatch(time=self.time, request=self.request, worker_id=worker_id)]


@dataclass(order=True)
class RouterDispatch(Event):
    time: float = field(compare=True)
    priority: int = field(default=20, compare=True)
    request: Request = field(default=None, compare=False)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.request.phase = RequestPhase.PREFILLING
        self.request.prefill_worker_id = self.worker_id
        worker = engine.prefill_workers[self.worker_id]
        worker.enqueue_request(self.request, engine.scheduler_config)

        if not worker.is_iteration_scheduled:
            worker.is_iteration_scheduled = True
            return [PrefillIterationStart(time=self.time, worker_id=self.worker_id)]
        return []


@dataclass(order=True)
class PrefillIterationStart(Event):
    time: float = field(compare=True)
    priority: int = field(default=30, compare=True)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        worker = engine.prefill_workers[self.worker_id]
        iteration_result = worker.run_iteration(self.time, engine)

        events: list[Event] = []
        for task in iteration_result.completed_tasks:
            events.append(
                PrefillChunkComplete(
                    time=self.time + iteration_result.iteration_time.val,
                    task=task,
                    worker_id=self.worker_id,
                )
            )

        if worker.has_pending_work():
            events.append(
                PrefillIterationStart(
                    time=self.time + iteration_result.iteration_time.val,
                    worker_id=self.worker_id,
                )
            )
        else:
            worker.is_iteration_scheduled = False

        return events


@dataclass(order=True)
class PrefillChunkComplete(Event):
    time: float = field(compare=True)
    priority: int = field(default=40, compare=True)
    task: PrefillTask = field(default=None, compare=False)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase, KVHandle

        if self.task.is_complete:
            self.task.request.phase = RequestPhase.TRANSFERRING
            kv_handle = KVHandle(
                request_id=self.task.request.id,
                kv_bytes=self.task.kv_bytes_produced,
                source_worker_id=self.worker_id,
            )
            dest_worker = engine.router.route_to_decode(kv_handle)
            kv_handle.dest_worker_id = dest_worker
            return [
                KVTransferStart(
                    time=self.time,
                    kv_handle=kv_handle,
                    request=self.task.request,
                )
            ]
        return []


@dataclass(order=True)
class KVTransferStart(Event):
    time: float = field(compare=True)
    priority: int = field(default=50, compare=True)
    kv_handle: KVHandle = field(default=None, compare=False)
    request: Request = field(default=None, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        self.kv_handle.transfer_started = self.time
        transfer_time = engine.kv_transfer_manager.initiate_transfer(
            self.kv_handle, engine.cluster_config
        )
        return [
            KVTransferComplete(
                time=self.time + transfer_time.val,
                kv_handle=self.kv_handle,
                request=self.request,
            )
        ]


@dataclass(order=True)
class KVTransferComplete(Event):
    time: float = field(compare=True)
    priority: int = field(default=60, compare=True)
    kv_handle: KVHandle = field(default=None, compare=False)
    request: Request = field(default=None, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.kv_handle.transfer_complete = self.time
        self.request.phase = RequestPhase.DECODING
        self.request.decode_worker_id = self.kv_handle.dest_worker_id

        worker = engine.decode_workers[self.kv_handle.dest_worker_id]
        worker.enqueue_request(self.request, self.kv_handle, engine.scheduler_config)

        if not worker.is_iteration_scheduled:
            worker.is_iteration_scheduled = True
            return [
                DecodeIterationStart(
                    time=self.time, worker_id=self.kv_handle.dest_worker_id
                )
            ]
        return []


@dataclass(order=True)
class DecodeIterationStart(Event):
    time: float = field(compare=True)
    priority: int = field(default=70, compare=True)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        worker = engine.decode_workers[self.worker_id]
        iteration_result = worker.run_iteration(self.time, engine)

        events: list[Event] = []
        for task in iteration_result.token_emitted_tasks:
            if task.request.tokens_generated == 1:
                task.request.first_token_time = self.time + iteration_result.iteration_time.val

            events.append(
                TokenEmit(
                    time=self.time + iteration_result.iteration_time.val,
                    task=task,
                    worker_id=self.worker_id,
                )
            )

        if worker.has_pending_work():
            events.append(
                DecodeIterationStart(
                    time=self.time + iteration_result.iteration_time.val,
                    worker_id=self.worker_id,
                )
            )
        else:
            worker.is_iteration_scheduled = False

        return events


@dataclass(order=True)
class TokenEmit(Event):
    time: float = field(compare=True)
    priority: int = field(default=80, compare=True)
    task: DecodeTask = field(default=None, compare=False)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        self.task.request.token_times.append(self.time)

        if self.task.is_complete:
            return [
                RequestComplete(
                    time=self.time,
                    request=self.task.request,
                )
            ]
        return []


@dataclass(order=True)
class RequestComplete(Event):
    time: float = field(compare=True)
    priority: int = field(default=90, compare=True)
    request: Request = field(default=None, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.request.phase = RequestPhase.COMPLETE
        self.request.complete_time = self.time
        engine.metrics.record_request_complete(self.request)
        return []

```

---

### AI

Let me run the tests to verify everything is working:

---

### AI

Excellent! All tests pass. Let me run one of the examples to verify the full simulation workflow:

---

### AI

All examples run successfully! Let me add the README file to complete the project:

---

### AI



```
# SGLang LLM Inference Simulator

A discrete-event simulator for LLM inference with symbolic tracing, designed to model SGLang-style workloads with prefill/decode disaggregation, prefix caching, and parallel execution.

## Features

- **Symbolic Metric Tracing**: All latency computations produce both numeric values and SymPy expressions for optimization
- **P/D Disaggregation**: Separate prefill and decode worker pools with explicit KV transfer modeling
- **Prefix Cache (RadixAttention)**: LRU-based radix cache with cache-aware routing
- **Chunked Prefill**: Configurable chunk sizes for TTFT/throughput tradeoffs
- **Parallelism Support**: Tensor Parallelism (TP), Data Parallelism (DP), Expert Parallelism (EP)
- **MoE Models**: Full support for Mixture-of-Experts with all-to-all communication
- **Configurable Interconnects**: NVLink, InfiniBand, Ethernet with proper latency/bandwidth modeling

## Installation

```bash
cd sglang_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```python
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution

# Configure simulation
model = ModelConfig.llama_7b()
cluster = ClusterConfig.single_node_8gpu()
scheduler = SchedulerConfig.default()
cache = CacheConfig.h100_default()

# Create engine and run
engine = SimulationEngine(model, cluster, scheduler, cache)
# ... initialize workers, router, etc.
result = engine.run_until_idle()

print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
```

## Symbolic Throughput Optimization

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.optimizer.interface import OptimizerInterface

symbols = ConfigSymbols()

# Build symbolic throughput expression
throughput_expr = ...  # Your throughput formula

# Optimize over configuration space
optimizer = OptimizerInterface(objective=throughput_expr, maximize=True)
result = optimizer.grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
})

print(f"Optimal config: {result.optimal_values}")
```

## Project Structure

```
sim/
├── config/          # Configuration dataclasses
├── core/            # Simulation engine, events, state
├── symbolic/        # SymVal and symbolic combinators
├── workers/         # Prefill and decode workers
├── schedulers/      # Continuous batching scheduler
├── cache/           # Radix cache implementation
├── routing/         # Cache-aware router
├── parallel/        # TP/DP/EP group abstractions
├── network/         # Interconnect and KV transfer
├── kernels/         # Attention, MLP, MoE kernels
├── models/          # Operator graph templates
├── metrics/         # Metric collection and symbolic throughput
├── optimizer/       # Scipy/grid search interface
└── workload/        # Synthetic and trace-based workloads
```

## Examples

See the `examples/` directory for:
- `basic_simulation.py` - Full simulation workflow
- `optimize_pd_ratio.py` - P/D ratio optimization with symbolic expressions
- `sweep_chunk_size.py` - Chunk size analysis and cache effects

## Running Tests

```bash
python -m pytest tests/ -v
```

```

---

### AI

Let me verify the complete directory structure:

---

### AI

The implementation is complete. Here's a summary of all the files created:

**Core Implementation (49 Python files):**

| Module | Files | Description |
|--------|-------|-------------|
| `sim/symbolic/` | `expr.py`, `symbols.py` | SymVal class with dual numeric+symbolic evaluation, config symbols |
| `sim/config/` | `model.py`, `cluster.py`, `scheduler.py`, `cache.py` | Configuration dataclasses for models, clusters, scheduling, caching |
| `sim/core/` | `engine.py`, `events.py`, `request.py`, `state.py` | Discrete-event simulation engine with priority queue, event types, request lifecycle |
| `sim/workers/` | `prefill_worker.py`, `decode_worker.py` | Prefill and decode worker implementations with chunked prefill |
| `sim/schedulers/` | `batch_scheduler.py` | Continuous batching scheduler |
| `sim/cache/` | `radix_cache.py` | LRU-based radix cache for prefix reuse |
| `sim/routing/` | `router.py` | Cache-aware and load-balanced routing policies |
| `sim/parallel/` | `groups.py`, `collectives.py` | TP/DP/EP group abstractions with collective costs |
| `sim/network/` | `interconnect.py`, `kv_transfer.py` | NVLink/IB/Ethernet models, KV transfer manager |
| `sim/kernels/` | `attention.py`, `mlp.py`, `moe.py` | Flash attention, paged attention, MLP, MoE kernel models |
| `sim/models/` | `operator_graph.py` | Full transformer layer latency modeling |
| `sim/metrics/` | `definitions.py`, `symbolic.py` | Metric collection, symbolic throughput builder |
| `sim/optimizer/` | `interface.py` | Grid search and scipy optimization interface |
| `sim/workload/` | `generators.py`, `prefix_groups.py`, `trace_loader.py` | Synthetic and trace-based workload generation |

**Examples:**
- `basic_simulation.py` - Full simulation workflow
- `optimize_pd_ratio.py` - P/D ratio optimization with symbolic expressions  
- `sweep_chunk_size.py` - Chunk size analysis with cache effects

**Tests:**
- `test_symbolic.py` - 13 tests for symbolic expression system

All tests pass and examples run successfully. The simulator produces symbolic throughput expressions like:

```
TP = Min(B_mem*B_tok*N_d/(kv_b*(n_in_avg + n_out_avg/2)), F_gpu*N_p*n_out_avg/(4*H**2*L*c*n_in_avg*ceiling(n_in_avg/c)))
```

```
TP = Min(B_mem*B_tok*N_d/(kv_b*(n_in_avg + n_out_avg/2)), F_gpu*N_p*n_out_avg/(4*H**2*L*c*n_in_avg*ceiling(n_in_avg/c)))
```

---

### User

since i want to approach a problem of finding optimal hyperparameters as a constrained optimization problem in both continuous and discrete variables i also need sympy expressions to be computed for other sensible metrics and for peak memory usage at the accelerators. maybe you even can suggest more features. also add the ability to display compute graph for those metrics as a visual diagram, for example using pyflowchart or whatever you think is appropriate

---

### AI

I'll extend the simulator with additional symbolic metrics, memory modeling, and compute graph visualization. Let me add these features:

---

### AI



```
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

```

---

### AI



```
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

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import sympy
from sympy import Symbol

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder

if TYPE_CHECKING:
    pass


class ConstraintType(Enum):
    INEQUALITY = "ineq"
    EQUALITY = "eq"


@dataclass
class Constraint:
    name: str
    expr: sympy.Expr
    type: ConstraintType
    description: str = ""
    
    def is_satisfied(self, values: dict[Symbol, float]) -> bool:
        evaluated = float(self.expr.subs(list(values.items())))
        if self.type == ConstraintType.EQUALITY:
            return abs(evaluated) < 1e-6
        else:
            return evaluated >= 0
    
    def violation(self, values: dict[Symbol, float]) -> float:
        evaluated = float(self.expr.subs(list(values.items())))
        if self.type == ConstraintType.EQUALITY:
            return abs(evaluated)
        else:
            return max(0, -evaluated)


class ConstraintBuilder:
    def __init__(self, symbols: ConfigSymbols | None = None):
        self.symbols = symbols or ConfigSymbols()
        self.metric_builder = SymbolicMetricBuilder(self.symbols)
        self.constraints: list[Constraint] = []

    def add_memory_constraint(
        self,
        gpu_memory_bytes: int,
        safety_margin: float = 0.9,
    ) -> Constraint:
        peak_memory = self.metric_builder.peak_memory_expression()
        available = gpu_memory_bytes * safety_margin
        
        constraint = Constraint(
            name="memory_limit",
            expr=available - peak_memory,
            type=ConstraintType.INEQUALITY,
            description=f"Peak memory <= {available / 1e9:.1f} GB",
        )
        self.constraints.append(constraint)
        return constraint

    def add_ttft_slo(self, max_ttft_seconds: float) -> Constraint:
        ttft = self.metric_builder.ttft_expression()
        
        constraint = Constraint(
            name="ttft_slo",
            expr=max_ttft_seconds - ttft,
            type=ConstraintType.INEQUALITY,
            description=f"TTFT <= {max_ttft_seconds * 1000:.0f}ms",
        )
        self.constraints.append(constraint)
        return constraint

    def add_tpot_slo(self, max_tpot_seconds: float) -> Constraint:
        tpot = self.metric_builder.tpot_expression()
        
        constraint = Constraint(
            name="tpot_slo",
            expr=max_tpot_seconds - tpot,
            type=ConstraintType.INEQUALITY,
            description=f"TPOT <= {max_tpot_seconds * 1000:.0f}ms",
        )
        self.constraints.append(constraint)
        return constraint

    def add_e2e_latency_slo(self, max_latency_seconds: float) -> Constraint:
        e2e = self.metric_builder.e2e_latency_expression()
        
        constraint = Constraint(
            name="e2e_latency_slo",
            expr=max_latency_seconds - e2e,
            type=ConstraintType.INEQUALITY,
            description=f"E2E Latency <= {max_latency_seconds:.1f}s",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_throughput(self, min_tokens_per_sec: float) -> Constraint:
        throughput = self.metric_builder.throughput_expression()
        
        constraint = Constraint(
            name="min_throughput",
            expr=throughput - min_tokens_per_sec,
            type=ConstraintType.INEQUALITY,
            description=f"Throughput >= {min_tokens_per_sec:.0f} tok/s",
        )
        self.constraints.append(constraint)
        return constraint

    def add_cost_budget(
        self,
        max_cost_per_1k_tokens: float,
        gpu_cost_per_hour: float = 2.0,
    ) -> Constraint:
        cost_per_token = self.metric_builder.cost_per_token_expression(gpu_cost_per_hour)
        max_cost_per_token = max_cost_per_1k_tokens / 1000
        
        constraint = Constraint(
            name="cost_budget",
            expr=max_cost_per_token - cost_per_token,
            type=ConstraintType.INEQUALITY,
            description=f"Cost <= ${max_cost_per_1k_tokens:.4f}/1K tokens",
        )
        self.constraints.append(constraint)
        return constraint

    def add_power_budget(self, max_power_watts: float) -> Constraint:
        power = self.metric_builder.power_consumption_expression()
        
        constraint = Constraint(
            name="power_budget",
            expr=max_power_watts - power,
            type=ConstraintType.INEQUALITY,
            description=f"Power <= {max_power_watts:.0f}W",
        )
        self.constraints.append(constraint)
        return constraint

    def add_total_gpu_constraint(self, max_gpus: int) -> Constraint:
        s = self.symbols
        total_gpus = (s.N_p + s.N_d) * s.tp_size
        
        constraint = Constraint(
            name="max_gpus",
            expr=max_gpus - total_gpus,
            type=ConstraintType.INEQUALITY,
            description=f"Total GPUs <= {max_gpus}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_prefill_workers(self, min_workers: int) -> Constraint:
        s = self.symbols
        
        constraint = Constraint(
            name="min_prefill_workers",
            expr=s.N_p - min_workers,
            type=ConstraintType.INEQUALITY,
            description=f"N_p >= {min_workers}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_decode_workers(self, min_workers: int) -> Constraint:
        s = self.symbols
        
        constraint = Constraint(
            name="min_decode_workers",
            expr=s.N_d - min_workers,
            type=ConstraintType.INEQUALITY,
            description=f"N_d >= {min_workers}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_utilization_constraint(
        self,
        min_compute_util: float = 0.0,
        min_memory_util: float = 0.0,
    ) -> list[Constraint]:
        added = []
        
        if min_compute_util > 0:
            compute_util = self.metric_builder.compute_utilization_expression()
            constraint = Constraint(
                name="min_compute_util",
                expr=compute_util - min_compute_util,
                type=ConstraintType.INEQUALITY,
                description=f"Compute utilization >= {min_compute_util:.0%}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        if min_memory_util > 0:
            mem_util = self.metric_builder.memory_bandwidth_utilization_expression()
            constraint = Constraint(
                name="min_memory_util",
                expr=mem_util - min_memory_util,
                type=ConstraintType.INEQUALITY,
                description=f"Memory BW utilization >= {min_memory_util:.0%}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        return added

    def add_pd_ratio_constraint(
        self,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> list[Constraint]:
        s = self.symbols
        added = []
        
        if min_ratio is not None:
            constraint = Constraint(
                name="min_pd_ratio",
                expr=s.N_p / s.N_d - min_ratio,
                type=ConstraintType.INEQUALITY,
                description=f"N_p/N_d >= {min_ratio:.2f}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        if max_ratio is not None:
            constraint = Constraint(
                name="max_pd_ratio",
                expr=max_ratio - s.N_p / s.N_d,
                type=ConstraintType.INEQUALITY,
                description=f"N_p/N_d <= {max_ratio:.2f}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        return added

    def add_chunk_size_bounds(
        self,
        min_chunk: int,
        max_chunk: int,
    ) -> list[Constraint]:
        s = self.symbols
        
        constraints = [
            Constraint(
                name="min_chunk_size",
                expr=s.chunk_size - min_chunk,
                type=ConstraintType.INEQUALITY,
                description=f"chunk_size >= {min_chunk}",
            ),
            Constraint(
                name="max_chunk_size",
                expr=max_chunk - s.chunk_size,
                type=ConstraintType.INEQUALITY,
                description=f"chunk_size <= {max_chunk}",
            ),
        ]
        self.constraints.extend(constraints)
        return constraints

    def add_batch_size_bounds(
        self,
        min_batch: int,
        max_batch: int,
    ) -> list[Constraint]:
        s = self.symbols
        
        constraints = [
            Constraint(
                name="min_batch_size",
                expr=s.batch_cap_requests - min_batch,
                type=ConstraintType.INEQUALITY,
                description=f"batch_size >= {min_batch}",
            ),
            Constraint(
                name="max_batch_size",
                expr=max_batch - s.batch_cap_requests,
                type=ConstraintType.INEQUALITY,
                description=f"batch_size <= {max_batch}",
            ),
        ]
        self.constraints.extend(constraints)
        return constraints

    def get_all_constraints(self) -> list[Constraint]:
        return self.constraints.copy()

    def get_scipy_constraints(self) -> list[dict]:
        return [
            {
                "type": c.type.value,
                "fun": lambda x, expr=c.expr, syms=self.symbols.decision_vars(): float(
                    expr.subs(list(zip(syms, x)))
                ),
            }
            for c in self.constraints
        ]

    def check_all_constraints(self, values: dict[Symbol, float]) -> dict[str, tuple[bool, float]]:
        results = {}
        for c in self.constraints:
            satisfied = c.is_satisfied(values)
            violation = c.violation(values)
            results[c.name] = (satisfied, violation)
        return results

    def summarize(self) -> str:
        lines = ["Constraints:"]
        for c in self.constraints:
            lines.append(f"  - {c.name}: {c.description}")
        return "\n".join(lines)

```

---

### AI



```
from sim.visualization.graph import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)

__all__ = [
    "ExpressionGraphVisualizer",
    "MetricDependencyGraph",
    "render_expression_tree",
    "render_metric_dependencies",
]

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from io import StringIO
import html

import sympy
from sympy import Symbol, Function, Add, Mul, Pow, Max, Min, ceiling, Piecewise
from sympy.printing.dot import dotprint

if TYPE_CHECKING:
    from sim.metrics.expressions import MetricExpressions
    from sim.metrics.constraints import Constraint


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str
    shape: str = "box"
    color: str = "#ffffff"
    children: list[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str = ""
    style: str = "solid"


class ExpressionGraphVisualizer:
    NODE_COLORS = {
        "symbol": "#e3f2fd",
        "constant": "#fff3e0",
        "operator": "#f3e5f5",
        "function": "#e8f5e9",
        "metric": "#ffebee",
    }
    
    NODE_SHAPES = {
        "symbol": "ellipse",
        "constant": "box",
        "operator": "diamond",
        "function": "hexagon",
        "metric": "doubleoctagon",
    }

    def __init__(self):
        self._node_counter = 0
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    def reset(self) -> None:
        self._node_counter = 0
        self._nodes.clear()
        self._edges.clear()

    def _new_node_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"

    def build_graph(self, expr: sympy.Expr, root_name: str = "result") -> str:
        self.reset()
        root_id = self._visit_expr(expr)
        
        root_node = GraphNode(
            id="root",
            label=root_name,
            node_type="metric",
            shape=self.NODE_SHAPES["metric"],
            color=self.NODE_COLORS["metric"],
        )
        self._nodes["root"] = root_node
        self._edges.append(GraphEdge(source="root", target=root_id))
        
        return root_id

    def _visit_expr(self, expr: sympy.Expr) -> str:
        if isinstance(expr, Symbol):
            return self._add_symbol_node(expr)
        elif isinstance(expr, (int, float, sympy.Integer, sympy.Float, sympy.Rational)):
            return self._add_constant_node(expr)
        elif isinstance(expr, Add):
            return self._add_operator_node("+", list(expr.args))
        elif isinstance(expr, Mul):
            return self._add_operator_node("×", list(expr.args))
        elif isinstance(expr, Pow):
            base, exp = expr.args
            return self._add_function_node(f"^{exp}", [base])
        elif isinstance(expr, Max):
            return self._add_function_node("Max", list(expr.args))
        elif isinstance(expr, Min):
            return self._add_function_node("Min", list(expr.args))
        elif isinstance(expr, ceiling):
            return self._add_function_node("⌈⌉", list(expr.args))
        elif isinstance(expr, Piecewise):
            return self._add_piecewise_node(expr)
        elif hasattr(expr, 'func') and hasattr(expr, 'args'):
            func_name = expr.func.__name__
            return self._add_function_node(func_name, list(expr.args))
        else:
            return self._add_constant_node(expr)

    def _add_symbol_node(self, sym: Symbol) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=str(sym),
            node_type="symbol",
            shape=self.NODE_SHAPES["symbol"],
            color=self.NODE_COLORS["symbol"],
        )
        return node_id

    def _add_constant_node(self, value: Any) -> str:
        node_id = self._new_node_id()
        if isinstance(value, float):
            label = f"{value:.4g}"
        else:
            label = str(value)
        
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=label,
            node_type="constant",
            shape=self.NODE_SHAPES["constant"],
            color=self.NODE_COLORS["constant"],
        )
        return node_id

    def _add_operator_node(self, op: str, args: list) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=op,
            node_type="operator",
            shape=self.NODE_SHAPES["operator"],
            color=self.NODE_COLORS["operator"],
        )
        
        for arg in args:
            child_id = self._visit_expr(arg)
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id))
        
        return node_id

    def _add_function_node(self, func_name: str, args: list) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=func_name,
            node_type="function",
            shape=self.NODE_SHAPES["function"],
            color=self.NODE_COLORS["function"],
        )
        
        for arg in args:
            child_id = self._visit_expr(arg)
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id))
        
        return node_id

    def _add_piecewise_node(self, pw: Piecewise) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label="Piecewise",
            node_type="function",
            shape="house",
            color=self.NODE_COLORS["function"],
        )
        
        for expr, cond in pw.args:
            child_id = self._visit_expr(expr)
            cond_label = str(cond)[:20]
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id, label=cond_label))
        
        return node_id

    def to_dot(self, title: str = "Expression Graph") -> str:
        lines = [
            "digraph G {",
            f'    label="{title}";',
            "    labelloc=t;",
            "    rankdir=TB;",
            '    node [fontname="Helvetica", fontsize=10];',
            '    edge [fontname="Helvetica", fontsize=8];',
            "",
        ]
        
        for node in self._nodes.values():
            escaped_label = html.escape(node.label)
            lines.append(
                f'    {node.id} [label="{escaped_label}", '
                f'shape={node.shape}, style=filled, fillcolor="{node.color}"];'
            )
        
        lines.append("")
        
        for edge in self._edges:
            if edge.label:
                escaped_label = html.escape(edge.label)
                lines.append(
                    f'    {edge.source} -> {edge.target} [label="{escaped_label}"];'
                )
            else:
                lines.append(f"    {edge.source} -> {edge.target};")
        
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, title: str = "Expression Graph") -> str:
        lines = [
            "```mermaid",
            "flowchart TB",
            f"    subgraph title [{title}]",
        ]
        
        shape_map = {
            "ellipse": ("((", "))"),
            "box": ("[", "]"),
            "diamond": ("{", "}"),
            "hexagon": ("{{", "}}"),
            "doubleoctagon": ("[[", "]]"),
            "house": ("[/", "/]"),
        }
        
        for node in self._nodes.values():
            left, right = shape_map.get(node.shape, ("[", "]"))
            label = node.label.replace('"', "'")
            lines.append(f"        {node.id}{left}{label}{right}")
        
        for edge in self._edges:
            if edge.label:
                label = edge.label.replace('"', "'")
                lines.append(f'        {edge.source} -->|"{label}"| {edge.target}')
            else:
                lines.append(f"        {edge.source} --> {edge.target}")
        
        lines.append("    end")
        lines.append("```")
        return "\n".join(lines)

    def render_to_file(
        self,
        expr: sympy.Expr,
        filename: str,
        title: str = "Expression Graph",
        format: str = "png",
    ) -> str:
        self.build_graph(expr, title)
        dot_content = self.to_dot(title)
        
        dot_filename = filename.rsplit(".", 1)[0] + ".dot"
        with open(dot_filename, "w") as f:
            f.write(dot_content)
        
        return dot_filename


class MetricDependencyGraph:
    def __init__(self):
        self.nodes: dict[str, set[str]] = {}
        self.edges: list[tuple[str, str]] = []

    def add_metric(self, name: str, expr: sympy.Expr) -> None:
        symbols = {str(s) for s in expr.free_symbols}
        self.nodes[name] = symbols
        
        for sym in symbols:
            self.edges.append((sym, name))

    def add_metrics(self, metrics: dict[str, sympy.Expr]) -> None:
        for name, expr in metrics.items():
            self.add_metric(name, expr)

    def find_shared_dependencies(self) -> dict[str, list[str]]:
        symbol_to_metrics: dict[str, list[str]] = {}
        
        for metric, symbols in self.nodes.items():
            for sym in symbols:
                if sym not in symbol_to_metrics:
                    symbol_to_metrics[sym] = []
                symbol_to_metrics[sym].append(metric)
        
        return {sym: metrics for sym, metrics in symbol_to_metrics.items() if len(metrics) > 1}

    def to_dot(self, title: str = "Metric Dependencies") -> str:
        lines = [
            "digraph MetricDependencies {",
            f'    label="{title}";',
            "    labelloc=t;",
            "    rankdir=LR;",
            '    node [fontname="Helvetica"];',
            "",
            "    // Symbol nodes",
            '    subgraph cluster_symbols {',
            '        label="Configuration Variables";',
            '        style=dashed;',
            '        node [shape=ellipse, style=filled, fillcolor="#e3f2fd"];',
        ]
        
        all_symbols = set()
        for symbols in self.nodes.values():
            all_symbols.update(symbols)
        
        for sym in sorted(all_symbols):
            lines.append(f'        "{sym}";')
        
        lines.extend([
            "    }",
            "",
            "    // Metric nodes",
            '    subgraph cluster_metrics {',
            '        label="Metrics";',
            '        style=dashed;',
            '        node [shape=box, style=filled, fillcolor="#ffebee"];',
        ])
        
        for metric in sorted(self.nodes.keys()):
            lines.append(f'        "{metric}";')
        
        lines.extend([
            "    }",
            "",
            "    // Edges",
        ])
        
        for sym, metric in self.edges:
            lines.append(f'    "{sym}" -> "{metric}";')
        
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, title: str = "Metric Dependencies") -> str:
        lines = [
            "```mermaid",
            "flowchart LR",
            f"    subgraph title [{title}]",
            "    subgraph symbols [Configuration Variables]",
        ]
        
        all_symbols = set()
        for symbols in self.nodes.values():
            all_symbols.update(symbols)
        
        for sym in sorted(all_symbols):
            safe_id = sym.replace("_", "")
            lines.append(f"        {safe_id}(({sym}))")
        
        lines.append("    end")
        lines.append("    subgraph metrics [Metrics]")
        
        for metric in sorted(self.nodes.keys()):
            safe_id = metric.replace("_", "")
            lines.append(f"        {safe_id}[{metric}]")
        
        lines.append("    end")
        
        for sym, metric in self.edges:
            safe_sym = sym.replace("_", "")
            safe_metric = metric.replace("_", "")
            lines.append(f"    {safe_sym} --> {safe_metric}")
        
        lines.append("    end")
        lines.append("```")
        return "\n".join(lines)


def render_expression_tree(
    expr: sympy.Expr,
    title: str = "Expression",
    output_format: str = "dot",
) -> str:
    viz = ExpressionGraphVisualizer()
    viz.build_graph(expr, title)
    
    if output_format == "mermaid":
        return viz.to_mermaid(title)
    else:
        return viz.to_dot(title)


def render_metric_dependencies(
    metrics: dict[str, sympy.Expr],
    title: str = "Metric Dependencies",
    output_format: str = "dot",
) -> str:
    graph = MetricDependencyGraph()
    graph.add_metrics(metrics)
    
    if output_format == "mermaid":
        return graph.to_mermaid(title)
    else:
        return graph.to_dot(title)


def sympy_dotprint(expr: sympy.Expr) -> str:
    try:
        return dotprint(expr)
    except Exception:
        viz = ExpressionGraphVisualizer()
        viz.build_graph(expr)
        return viz.to_dot()

```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from itertools import product
from enum import Enum

import sympy
from sympy import Symbol
import numpy as np

from sim.symbolic.symbols import ConfigSymbols


class ObjectiveDirection(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class OptimizationResult:
    optimal_values: dict[str, float]
    optimal_objective: float
    success: bool
    message: str
    all_evaluations: list[tuple[dict, float]] = field(default_factory=list)
    constraint_violations: dict[str, float] = field(default_factory=dict)
    pareto_front: list[dict] = field(default_factory=list)


@dataclass
class MultiObjectiveResult:
    pareto_front: list[dict[str, float]]
    pareto_objectives: list[dict[str, float]]
    all_evaluations: list[tuple[dict, dict[str, float]]] = field(default_factory=list)


@dataclass
class Constraint:
    expr: sympy.Expr
    type: str  # "eq" for equality, "ineq" for inequality (>= 0)
    name: str = ""


@dataclass
class ObjectiveSpec:
    name: str
    expr: sympy.Expr
    direction: ObjectiveDirection
    weight: float = 1.0


class OptimizerInterface:
    def __init__(
        self,
        objective: sympy.Expr,
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
        maximize: bool = True,
    ):
        self.objective = objective
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self.maximize = maximize

        self._decision_vars: list[Symbol] = []
        self._frozen_params: dict[Symbol, float] = {}

    def set_decision_vars(self, vars: list[Symbol]) -> None:
        self._decision_vars = vars

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def get_decision_vars(self) -> list[Symbol]:
        if self._decision_vars:
            return self._decision_vars
        return self.symbols.decision_vars()

    def get_frozen_params(self) -> list[Symbol]:
        return self.symbols.frozen_params()

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def to_scipy_minimize(
        self,
        bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        decision_vars = self.get_decision_vars()

        substituted_obj = self.substitute_frozen(self.objective)

        if self.maximize:
            obj_to_minimize = -substituted_obj
        else:
            obj_to_minimize = substituted_obj

        obj_func = sympy.lambdify(decision_vars, obj_to_minimize, modules=["numpy"])

        def objective_wrapper(x: np.ndarray) -> float:
            return float(obj_func(*x))

        scipy_constraints = []
        for constraint in self.constraints:
            substituted = self.substitute_frozen(constraint.expr)
            constraint_func = sympy.lambdify(decision_vars, substituted, modules=["numpy"])

            scipy_constraints.append({
                "type": constraint.type,
                "fun": lambda x, f=constraint_func: float(f(*x)),
            })

        scipy_bounds = None
        if bounds:
            scipy_bounds = [
                bounds.get(var, (None, None)) for var in decision_vars
            ]

        return {
            "fun": objective_wrapper,
            "x0": np.ones(len(decision_vars)),
            "bounds": scipy_bounds,
            "constraints": scipy_constraints,
            "method": "SLSQP",
        }

    def enumerate_discrete_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> list[dict[Symbol, int]]:
        regimes = []
        var_names = list(discrete_vars.keys())
        var_values = list(discrete_vars.values())

        for combo in product(*var_values):
            regime = dict(zip(var_names, combo))
            regimes.append(regime)

        return regimes

    def optimize_over_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> OptimizationResult:
        from scipy.optimize import minimize

        regimes = self.enumerate_discrete_regimes(discrete_vars, continuous_bounds)
        best_result: OptimizationResult | None = None
        all_evaluations: list[tuple[dict, float]] = []

        decision_vars = self.get_decision_vars()
        continuous_vars = [v for v in decision_vars if v not in discrete_vars]

        for regime in regimes:
            regime_objective = self.objective.subs(list(regime.items()))
            regime_objective = self.substitute_frozen(regime_objective)

            if not continuous_vars:
                if self.maximize:
                    obj_val = -float(regime_objective)
                else:
                    obj_val = float(regime_objective)

                result_dict = {str(k): v for k, v in regime.items()}
                all_evaluations.append((result_dict, -obj_val if self.maximize else obj_val))

                if best_result is None or (self.maximize and -obj_val > best_result.optimal_objective) or \
                   (not self.maximize and obj_val < best_result.optimal_objective):
                    best_result = OptimizationResult(
                        optimal_values=result_dict,
                        optimal_objective=-obj_val if self.maximize else obj_val,
                        success=True,
                        message=f"Regime: {regime}",
                    )
            else:
                obj_func = sympy.lambdify(continuous_vars, regime_objective, modules=["numpy"])

                if self.maximize:
                    def objective_wrapper(x):
                        return -float(obj_func(*x))
                else:
                    def objective_wrapper(x):
                        return float(obj_func(*x))

                bounds_list = [
                    continuous_bounds.get(v, (1, 100)) if continuous_bounds else (1, 100)
                    for v in continuous_vars
                ]

                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_list])

                try:
                    result = minimize(
                        objective_wrapper,
                        x0,
                        bounds=bounds_list,
                        method="L-BFGS-B",
                    )

                    result_dict = {str(k): v for k, v in regime.items()}
                    for var, val in zip(continuous_vars, result.x):
                        result_dict[str(var)] = float(val)

                    obj_val = -result.fun if self.maximize else result.fun
                    all_evaluations.append((result_dict, obj_val))

                    if best_result is None or \
                       (self.maximize and obj_val > best_result.optimal_objective) or \
                       (not self.maximize and obj_val < best_result.optimal_objective):
                        best_result = OptimizationResult(
                            optimal_values=result_dict,
                            optimal_objective=obj_val,
                            success=result.success,
                            message=result.message,
                        )
                except Exception as e:
                    continue

        if best_result is None:
            return OptimizationResult(
                optimal_values={},
                optimal_objective=float("inf") if not self.maximize else float("-inf"),
                success=False,
                message="No valid regime found",
                all_evaluations=all_evaluations,
            )

        best_result.all_evaluations = all_evaluations
        return best_result

    def grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())

        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []

        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])

        for combo in product(*var_values):
            try:
                obj_val = float(obj_func(*combo))
                if self.maximize:
                    obj_val = obj_val
                else:
                    obj_val = obj_val

                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))

                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue

        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Grid search complete",
            all_evaluations=all_evaluations,
        )


```

---

### AI



```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from itertools import product
from enum import Enum

import sympy
from sympy import Symbol
import numpy as np

from sim.symbolic.symbols import ConfigSymbols


class ObjectiveDirection(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class OptimizationResult:
    optimal_values: dict[str, float]
    optimal_objective: float
    success: bool
    message: str
    all_evaluations: list[tuple[dict, float]] = field(default_factory=list)
    constraint_violations: dict[str, float] = field(default_factory=dict)
    pareto_front: list[dict] = field(default_factory=list)


@dataclass
class MultiObjectiveResult:
    pareto_front: list[dict[str, float]]
    pareto_objectives: list[dict[str, float]]
    all_evaluations: list[tuple[dict, dict[str, float]]] = field(default_factory=list)


@dataclass
class Constraint:
    expr: sympy.Expr
    type: str  # "eq" for equality, "ineq" for inequality (>= 0)
    name: str = ""


@dataclass
class ObjectiveSpec:
    name: str
    expr: sympy.Expr
    direction: ObjectiveDirection
    weight: float = 1.0


class OptimizerInterface:
    def __init__(
        self,
        objective: sympy.Expr,
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
        maximize: bool = True,
    ):
        self.objective = objective
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self.maximize = maximize

        self._decision_vars: list[Symbol] = []
        self._frozen_params: dict[Symbol, float] = {}

    def set_decision_vars(self, vars: list[Symbol]) -> None:
        self._decision_vars = vars

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def get_decision_vars(self) -> list[Symbol]:
        if self._decision_vars:
            return self._decision_vars
        return self.symbols.decision_vars()

    def get_frozen_params(self) -> list[Symbol]:
        return self.symbols.frozen_params()

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def to_scipy_minimize(
        self,
        bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        decision_vars = self.get_decision_vars()

        substituted_obj = self.substitute_frozen(self.objective)

        if self.maximize:
            obj_to_minimize = -substituted_obj
        else:
            obj_to_minimize = substituted_obj

        obj_func = sympy.lambdify(decision_vars, obj_to_minimize, modules=["numpy"])

        def objective_wrapper(x: np.ndarray) -> float:
            return float(obj_func(*x))

        scipy_constraints = []
        for constraint in self.constraints:
            substituted = self.substitute_frozen(constraint.expr)
            constraint_func = sympy.lambdify(decision_vars, substituted, modules=["numpy"])

            scipy_constraints.append({
                "type": constraint.type,
                "fun": lambda x, f=constraint_func: float(f(*x)),
            })

        scipy_bounds = None
        if bounds:
            scipy_bounds = [
                bounds.get(var, (None, None)) for var in decision_vars
            ]

        return {
            "fun": objective_wrapper,
            "x0": np.ones(len(decision_vars)),
            "bounds": scipy_bounds,
            "constraints": scipy_constraints,
            "method": "SLSQP",
        }

    def enumerate_discrete_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> list[dict[Symbol, int]]:
        regimes = []
        var_names = list(discrete_vars.keys())
        var_values = list(discrete_vars.values())

        for combo in product(*var_values):
            regime = dict(zip(var_names, combo))
            regimes.append(regime)

        return regimes

    def optimize_over_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> OptimizationResult:
        from scipy.optimize import minimize

        regimes = self.enumerate_discrete_regimes(discrete_vars, continuous_bounds)
        best_result: OptimizationResult | None = None
        all_evaluations: list[tuple[dict, float]] = []

        decision_vars = self.get_decision_vars()
        continuous_vars = [v for v in decision_vars if v not in discrete_vars]

        for regime in regimes:
            regime_objective = self.objective.subs(list(regime.items()))
            regime_objective = self.substitute_frozen(regime_objective)

            if not continuous_vars:
                if self.maximize:
                    obj_val = -float(regime_objective)
                else:
                    obj_val = float(regime_objective)

                result_dict = {str(k): v for k, v in regime.items()}
                all_evaluations.append((result_dict, -obj_val if self.maximize else obj_val))

                if best_result is None or (self.maximize and -obj_val > best_result.optimal_objective) or \
                   (not self.maximize and obj_val < best_result.optimal_objective):
                    best_result = OptimizationResult(
                        optimal_values=result_dict,
                        optimal_objective=-obj_val if self.maximize else obj_val,
                        success=True,
                        message=f"Regime: {regime}",
                    )
            else:
                obj_func = sympy.lambdify(continuous_vars, regime_objective, modules=["numpy"])

                if self.maximize:
                    def objective_wrapper(x):
                        return -float(obj_func(*x))
                else:
                    def objective_wrapper(x):
                        return float(obj_func(*x))

                bounds_list = [
                    continuous_bounds.get(v, (1, 100)) if continuous_bounds else (1, 100)
                    for v in continuous_vars
                ]

                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_list])

                try:
                    result = minimize(
                        objective_wrapper,
                        x0,
                        bounds=bounds_list,
                        method="L-BFGS-B",
                    )

                    result_dict = {str(k): v for k, v in regime.items()}
                    for var, val in zip(continuous_vars, result.x):
                        result_dict[str(var)] = float(val)

                    obj_val = -result.fun if self.maximize else result.fun
                    all_evaluations.append((result_dict, obj_val))

                    if best_result is None or \
                       (self.maximize and obj_val > best_result.optimal_objective) or \
                       (not self.maximize and obj_val < best_result.optimal_objective):
                        best_result = OptimizationResult(
                            optimal_values=result_dict,
                            optimal_objective=obj_val,
                            success=result.success,
                            message=result.message,
                        )
                except Exception as e:
                    continue

        if best_result is None:
            return OptimizationResult(
                optimal_values={},
                optimal_objective=float("inf") if not self.maximize else float("-inf"),
                success=False,
                message="No valid regime found",
                all_evaluations=all_evaluations,
            )

        best_result.all_evaluations = all_evaluations
        return best_result

    def grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())

        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []

        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])

        for combo in product(*var_values):
            try:
                obj_val = float(obj_func(*combo))
                if self.maximize:
                    obj_val = obj_val
                else:
                    obj_val = obj_val

                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))

                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue

        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Grid search complete",
            all_evaluations=all_evaluations,
        )

    def constrained_grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
        constraints: list[Constraint],
    ) -> OptimizationResult:
        from sim.metrics.constraints import Constraint as MetricConstraint
        
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())
        
        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []
        
        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])
        
        constraint_funcs = []
        for c in constraints:
            substituted = self.substitute_frozen(c.expr)
            constraint_funcs.append(
                (c.name, c.type, sympy.lambdify(var_names, substituted, modules=["numpy"]))
            )
        
        for combo in product(*var_values):
            try:
                feasible = True
                violations = {}
                
                for name, ctype, cfunc in constraint_funcs:
                    c_val = float(cfunc(*combo))
                    if ctype == "ineq" and c_val < 0:
                        feasible = False
                        violations[name] = -c_val
                    elif ctype == "eq" and abs(c_val) > 1e-6:
                        feasible = False
                        violations[name] = abs(c_val)
                
                if not feasible:
                    continue
                
                obj_val = float(obj_func(*combo))
                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))
                
                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue
        
        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Constrained grid search complete",
            all_evaluations=all_evaluations,
        )


class MultiObjectiveOptimizer:
    def __init__(
        self,
        objectives: list[ObjectiveSpec],
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
    ):
        self.objectives = objectives
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self._frozen_params: dict[Symbol, float] = {}

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def weighted_sum_optimize(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        combined_obj = sympy.Integer(0)
        
        for obj in self.objectives:
            weight = obj.weight
            expr = obj.expr
            if obj.direction == ObjectiveDirection.MINIMIZE:
                combined_obj = combined_obj - weight * expr
            else:
                combined_obj = combined_obj + weight * expr
        
        optimizer = OptimizerInterface(
            objective=combined_obj,
            constraints=self.constraints,
            symbols=self.symbols,
            maximize=True,
        )
        optimizer.set_frozen_params(self._frozen_params)
        
        return optimizer.grid_search(param_grid)

    def pareto_grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> MultiObjectiveResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())
        
        obj_funcs = []
        for obj in self.objectives:
            substituted = self.substitute_frozen(obj.expr)
            obj_funcs.append((
                obj.name,
                obj.direction,
                sympy.lambdify(var_names, substituted, modules=["numpy"])
            ))
        
        constraint_funcs = []
        for c in self.constraints:
            substituted = self.substitute_frozen(c.expr)
            constraint_funcs.append(
                (c.name, c.type, sympy.lambdify(var_names, substituted, modules=["numpy"]))
            )
        
        all_evaluations: list[tuple[dict, dict[str, float]]] = []
        feasible_solutions: list[tuple[dict[str, float], dict[str, float]]] = []
        
        for combo in product(*var_values):
            try:
                feasible = True
                for name, ctype, cfunc in constraint_funcs:
                    c_val = float(cfunc(*combo))
                    if ctype == "ineq" and c_val < 0:
                        feasible = False
                        break
                    elif ctype == "eq" and abs(c_val) > 1e-6:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                obj_vals = {}
                for name, direction, ofunc in obj_funcs:
                    obj_vals[name] = float(ofunc(*combo))
                
                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_vals))
                feasible_solutions.append((result_dict, obj_vals))
            except Exception:
                continue
        
        pareto_front = self._compute_pareto_front(feasible_solutions)
        
        return MultiObjectiveResult(
            pareto_front=[p[0] for p in pareto_front],
            pareto_objectives=[p[1] for p in pareto_front],
            all_evaluations=all_evaluations,
        )

    def _compute_pareto_front(
        self,
        solutions: list[tuple[dict[str, float], dict[str, float]]],
    ) -> list[tuple[dict[str, float], dict[str, float]]]:
        if not solutions:
            return []
        
        pareto = []
        
        for i, (params_i, objs_i) in enumerate(solutions):
            is_dominated = False
            
            for j, (params_j, objs_j) in enumerate(solutions):
                if i == j:
                    continue
                
                if self._dominates(objs_j, objs_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append((params_i, objs_i))
        
        return pareto

    def _dominates(
        self,
        objs_a: dict[str, float],
        objs_b: dict[str, float],
    ) -> bool:
        dominated = True
        strictly_better = False
        
        for obj in self.objectives:
            name = obj.name
            val_a = objs_a[name]
            val_b = objs_b[name]
            
            if obj.direction == ObjectiveDirection.MAXIMIZE:
                if val_a < val_b:
                    dominated = False
                    break
                if val_a > val_b:
                    strictly_better = True
            else:
                if val_a > val_b:
                    dominated = False
                    break
                if val_a < val_b:
                    strictly_better = True
        
        return dominated and strictly_better

    def epsilon_constraint(
        self,
        primary_objective: str,
        epsilon_bounds: dict[str, tuple[float, float]],
        param_grid: dict[Symbol, list[float]],
    ) -> list[OptimizationResult]:
        primary = next(o for o in self.objectives if o.name == primary_objective)
        
        results = []
        
        for other_obj in self.objectives:
            if other_obj.name == primary_objective:
                continue
            
            if other_obj.name in epsilon_bounds:
                lb, ub = epsilon_bounds[other_obj.name]
                
                if other_obj.direction == ObjectiveDirection.MAXIMIZE:
                    constraint = Constraint(
                        expr=other_obj.expr - lb,
                        type="ineq",
                        name=f"{other_obj.name}_lb",
                    )
                else:
                    constraint = Constraint(
                        expr=ub - other_obj.expr,
                        type="ineq",
                        name=f"{other_obj.name}_ub",
                    )
                
                optimizer = OptimizerInterface(
                    objective=primary.expr,
                    constraints=self.constraints + [constraint],
                    symbols=self.symbols,
                    maximize=primary.direction == ObjectiveDirection.MAXIMIZE,
                )
                optimizer.set_frozen_params(self._frozen_params)
                
                result = optimizer.constrained_grid_search(param_grid, [constraint])
                results.append(result)
        
        return results


```

---

### AI



```
from sim.optimizer.interface import (
    OptimizerInterface,
    OptimizationResult,
    MultiObjectiveOptimizer,
    MultiObjectiveResult,
    ObjectiveSpec,
    ObjectiveDirection,
    Constraint,
)

__all__ = [
    "OptimizerInterface",
    "OptimizationResult",
    "MultiObjectiveOptimizer",
    "MultiObjectiveResult",
    "ObjectiveSpec",
    "ObjectiveDirection",
    "Constraint",
]


```

---

### AI



```
from sim.metrics.definitions import MetricCollector, RequestMetrics
from sim.metrics.symbolic import SymbolicThroughputBuilder
from sim.metrics.expressions import SymbolicMetricBuilder, MetricExpressions
from sim.metrics.memory import SymbolicMemoryModel, MemoryBreakdown
from sim.metrics.constraints import ConstraintBuilder, Constraint, ConstraintType

__all__ = [
    "MetricCollector",
    "RequestMetrics",
    "SymbolicThroughputBuilder",
    "SymbolicMetricBuilder",
    "MetricExpressions",
    "SymbolicMemoryModel",
    "MemoryBreakdown",
    "ConstraintBuilder",
    "Constraint",
    "ConstraintType",
]


```

---

### AI



```
#!/usr/bin/env python3
"""Example demonstrating multi-objective optimization with constraints and visualization."""

import sys
sys.path.insert(0, "..")

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder, MetricExpressions
from sim.metrics.memory import SymbolicMemoryModel
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer.interface import (
    MultiObjectiveOptimizer,
    ObjectiveSpec,
    ObjectiveDirection,
)
from sim.visualization.graph import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)


def build_optimization_problem():
    symbols = ConfigSymbols()
    metric_builder = SymbolicMetricBuilder(symbols)
    
    metrics = metric_builder.build_all_expressions()
    
    print("=== Available Metrics ===")
    for name, expr in metrics.as_dict().items():
        print(f"  {name}")
    
    return symbols, metric_builder, metrics


def run_constrained_optimization():
    symbols, metric_builder, metrics = build_optimization_problem()
    
    constraint_builder = ConstraintBuilder(symbols)
    
    constraint_builder.add_memory_constraint(
        gpu_memory_bytes=80 * 1024**3,
        safety_margin=0.9,
    )
    constraint_builder.add_ttft_slo(max_ttft_seconds=0.5)
    constraint_builder.add_tpot_slo(max_tpot_seconds=0.05)
    constraint_builder.add_total_gpu_constraint(max_gpus=16)
    constraint_builder.add_min_throughput(min_tokens_per_sec=100)
    
    print("\n=== Constraints ===")
    print(constraint_builder.summarize())
    
    objectives = [
        ObjectiveSpec(
            name="throughput",
            expr=metrics.throughput,
            direction=ObjectiveDirection.MAXIMIZE,
            weight=1.0,
        ),
        ObjectiveSpec(
            name="cost_per_token",
            expr=metrics.cost_per_token,
            direction=ObjectiveDirection.MINIMIZE,
            weight=0.5,
        ),
        ObjectiveSpec(
            name="e2e_latency",
            expr=metrics.e2e_latency,
            direction=ObjectiveDirection.MINIMIZE,
            weight=0.3,
        ),
    ]
    
    optimizer = MultiObjectiveOptimizer(
        objectives=objectives,
        constraints=constraint_builder.get_all_constraints(),
        symbols=symbols,
    )
    
    frozen_params = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.network_bw: 100e9,
        symbols.network_latency: 10e-6,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.head_dim: 128,
        symbols.num_heads: 32,
        symbols.num_experts: 1,
        symbols.kv_bytes_per_token: 256,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.cache_hit_rate: 0.3,
        symbols.avg_prefix_match: 0.5,
    }
    optimizer.set_frozen_params(frozen_params)
    
    param_grid = {
        symbols.N_p: [1, 2, 4, 8],
        symbols.N_d: [1, 2, 4, 8],
        symbols.chunk_size: [4096, 8192, 16384],
        symbols.batch_cap_requests: [64, 128, 256],
        symbols.tp_size: [1, 2, 4],
    }
    
    print("\n=== Running Pareto Optimization ===")
    result = optimizer.pareto_grid_search(param_grid)
    
    print(f"\nFound {len(result.pareto_front)} Pareto-optimal solutions")
    print(f"Total feasible solutions evaluated: {len(result.all_evaluations)}")
    
    print("\n=== Pareto Front (Top 5) ===")
    sorted_pareto = sorted(
        zip(result.pareto_front, result.pareto_objectives),
        key=lambda x: x[1].get("throughput", 0),
        reverse=True,
    )[:5]
    
    for i, (params, objs) in enumerate(sorted_pareto, 1):
        print(f"\n{i}. Configuration:")
        print(f"   N_p={params.get('N_p')}, N_d={params.get('N_d')}, "
              f"TP={params.get('TP')}, chunk={params.get('c')}")
        print(f"   Objectives:")
        for name, val in objs.items():
            if "latency" in name or "ttft" in name or "tpot" in name:
                print(f"     {name}: {val*1000:.2f}ms")
            elif "cost" in name:
                print(f"     {name}: ${val*1000:.4f}/1K tokens")
            else:
                print(f"     {name}: {val:.2f}")
    
    return result


def visualize_metrics():
    symbols = ConfigSymbols()
    metric_builder = SymbolicMetricBuilder(symbols)
    metrics = metric_builder.build_all_expressions()
    
    print("\n=== Generating Metric Dependency Graph ===")
    dep_graph = MetricDependencyGraph()
    dep_graph.add_metrics({
        "throughput": metrics.throughput,
        "ttft": metrics.ttft,
        "tpot": metrics.tpot,
        "peak_memory": metrics.peak_memory,
        "cost_per_token": metrics.cost_per_token,
    })
    
    shared = dep_graph.find_shared_dependencies()
    print(f"\nShared dependencies affecting multiple metrics:")
    for sym, affected_metrics in list(shared.items())[:10]:
        print(f"  {sym}: affects {affected_metrics}")
    
    mermaid = dep_graph.to_mermaid("Metric Dependencies")
    print(f"\n=== Mermaid Diagram ===")
    print(mermaid[:500] + "..." if len(mermaid) > 500 else mermaid)
    
    print("\n=== Throughput Expression Tree ===")
    viz = ExpressionGraphVisualizer()
    viz.build_graph(metrics.throughput, "Throughput")
    dot = viz.to_dot("Throughput Expression")
    print(f"DOT graph generated ({len(dot)} chars)")
    print("First 300 chars:")
    print(dot[:300] + "...")
    
    return dep_graph, viz


def analyze_memory_scaling():
    symbols = ConfigSymbols()
    mem_model = SymbolicMemoryModel(symbols)
    
    print("\n=== Memory Scaling Analysis ===")
    
    peak_mem_expr = mem_model.symbolic_peak_memory()
    
    base_params = {
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.num_heads: 32,
        symbols.head_dim: 128,
        symbols.num_experts: 1,
        symbols.chunk_size: 8192,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.tp_size: 1,
    }
    
    print("\nPeak Memory vs Batch Size:")
    for batch in [32, 64, 128, 256, 512]:
        params = base_params.copy()
        params[symbols.batch_cap_requests] = batch
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  Batch {batch:3d}: {mem_gb:.1f} GB")
    
    print("\nPeak Memory vs TP Size (batch=256):")
    base_params[symbols.batch_cap_requests] = 256
    for tp in [1, 2, 4, 8]:
        params = base_params.copy()
        params[symbols.tp_size] = tp
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  TP={tp}: {mem_gb:.1f} GB per GPU")
    
    print("\nPeak Memory vs Sequence Length (batch=128):")
    base_params[symbols.batch_cap_requests] = 128
    base_params[symbols.tp_size] = 1
    for seq_len in [256, 512, 1024, 2048, 4096]:
        params = base_params.copy()
        params[symbols.avg_prompt_len] = seq_len
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  Seq len {seq_len:4d}: {mem_gb:.1f} GB")


if __name__ == "__main__":
    run_constrained_optimization()
    visualize_metrics()
    analyze_memory_scaling()

```

---

### AI



```
[project]
name = "sglang-sim"
version = "0.1.0"
description = "LLM inference simulator with symbolic tracing for SGLang-style workloads"
requires-python = ">=3.10"
dependencies = [
    "sympy>=1.12",
    "numpy>=1.24",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]
viz = [
    "graphviz>=0.20",
    "matplotlib>=3.7",
]
all = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "graphviz>=0.20",
    "matplotlib>=3.7",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]


```

---

### AI



```
# SGLang LLM Inference Simulator

A discrete-event simulator for LLM inference with symbolic tracing, designed to model SGLang-style workloads with prefill/decode disaggregation, prefix caching, and parallel execution.

## Features

- **Symbolic Metric Tracing**: All latency computations produce both numeric values and SymPy expressions for optimization
- **P/D Disaggregation**: Separate prefill and decode worker pools with explicit KV transfer modeling
- **Prefix Cache (RadixAttention)**: LRU-based radix cache with cache-aware routing
- **Chunked Prefill**: Configurable chunk sizes for TTFT/throughput tradeoffs
- **Parallelism Support**: Tensor Parallelism (TP), Data Parallelism (DP), Expert Parallelism (EP)
- **MoE Models**: Full support for Mixture-of-Experts with all-to-all communication
- **Configurable Interconnects**: NVLink, InfiniBand, Ethernet with proper latency/bandwidth modeling
- **Multi-Objective Optimization**: Pareto-optimal search with weighted-sum and epsilon-constraint methods
- **Constraint Handling**: Memory limits, SLO targets, cost budgets, GPU constraints
- **Compute Graph Visualization**: DOT and Mermaid diagrams for expression trees and metric dependencies

## Symbolic Metrics Available

| Metric | Description |
|--------|-------------|
| `throughput` | Tokens per second |
| `ttft` | Time to First Token |
| `tpot` | Time per Output Token |
| `e2e_latency` | End-to-end request latency |
| `peak_memory` | Peak GPU memory usage |
| `compute_utilization` | GPU compute utilization |
| `memory_bandwidth_utilization` | Memory bandwidth utilization |
| `network_utilization` | Network bandwidth utilization |
| `cost_per_token` | Cost efficiency metric |
| `power_consumption` | Power consumption estimate |

## Installation

```bash
cd sglang_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# For visualization support
pip install -e ".[viz]"
```

## Quick Start

```python
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution

# Configure simulation
model = ModelConfig.llama_7b()
cluster = ClusterConfig.single_node_8gpu()
scheduler = SchedulerConfig.default()
cache = CacheConfig.h100_default()

# Create engine and run
engine = SimulationEngine(model, cluster, scheduler, cache)
# ... initialize workers, router, etc.
result = engine.run_until_idle()

print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
```

## Symbolic Throughput Optimization

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.optimizer.interface import OptimizerInterface

symbols = ConfigSymbols()

# Build symbolic throughput expression
throughput_expr = ...  # Your throughput formula

# Optimize over configuration space
optimizer = OptimizerInterface(objective=throughput_expr, maximize=True)
result = optimizer.grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
})

print(f"Optimal config: {result.optimal_values}")
```

## Project Structure

```
sim/
├── config/          # Configuration dataclasses
├── core/            # Simulation engine, events, state
├── symbolic/        # SymVal and symbolic combinators
├── workers/         # Prefill and decode workers
├── schedulers/      # Continuous batching scheduler
├── cache/           # Radix cache implementation
├── routing/         # Cache-aware router
├── parallel/        # TP/DP/EP group abstractions
├── network/         # Interconnect and KV transfer
├── kernels/         # Attention, MLP, MoE kernels
├── models/          # Operator graph templates
├── metrics/         # Metric collection and symbolic throughput
├── optimizer/       # Scipy/grid search interface
└── workload/        # Synthetic and trace-based workloads
```

## Examples

See the `examples/` directory for:
- `basic_simulation.py` - Full simulation workflow
- `optimize_pd_ratio.py` - P/D ratio optimization with symbolic expressions
- `sweep_chunk_size.py` - Chunk size analysis and cache effects

## Running Tests

```bash
python -m pytest tests/ -v
```


```

---

### AI



```
# SGLang LLM Inference Simulator

A discrete-event simulator for LLM inference with symbolic tracing, designed to model SGLang-style workloads with prefill/decode disaggregation, prefix caching, and parallel execution.

## Features

- **Symbolic Metric Tracing**: All latency computations produce both numeric values and SymPy expressions for optimization
- **P/D Disaggregation**: Separate prefill and decode worker pools with explicit KV transfer modeling
- **Prefix Cache (RadixAttention)**: LRU-based radix cache with cache-aware routing
- **Chunked Prefill**: Configurable chunk sizes for TTFT/throughput tradeoffs
- **Parallelism Support**: Tensor Parallelism (TP), Data Parallelism (DP), Expert Parallelism (EP)
- **MoE Models**: Full support for Mixture-of-Experts with all-to-all communication
- **Configurable Interconnects**: NVLink, InfiniBand, Ethernet with proper latency/bandwidth modeling
- **Multi-Objective Optimization**: Pareto-optimal search with weighted-sum and epsilon-constraint methods
- **Constraint Handling**: Memory limits, SLO targets, cost budgets, GPU constraints
- **Compute Graph Visualization**: DOT and Mermaid diagrams for expression trees and metric dependencies

## Symbolic Metrics Available

| Metric | Description |
|--------|-------------|
| `throughput` | Tokens per second |
| `ttft` | Time to First Token |
| `tpot` | Time per Output Token |
| `e2e_latency` | End-to-end request latency |
| `peak_memory` | Peak GPU memory usage |
| `compute_utilization` | GPU compute utilization |
| `memory_bandwidth_utilization` | Memory bandwidth utilization |
| `network_utilization` | Network bandwidth utilization |
| `cost_per_token` | Cost efficiency metric |
| `power_consumption` | Power consumption estimate |

## Installation

```bash
cd sglang_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# For visualization support
pip install -e ".[viz]"
```

## Quick Start

```python
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution

# Configure simulation
model = ModelConfig.llama_7b()
cluster = ClusterConfig.single_node_8gpu()
scheduler = SchedulerConfig.default()
cache = CacheConfig.h100_default()

# Create engine and run
engine = SimulationEngine(model, cluster, scheduler, cache)
# ... initialize workers, router, etc.
result = engine.run_until_idle()

print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
```

## Symbolic Throughput Optimization

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.optimizer.interface import OptimizerInterface

symbols = ConfigSymbols()

# Build symbolic throughput expression
throughput_expr = ...  # Your throughput formula

# Optimize over configuration space
optimizer = OptimizerInterface(objective=throughput_expr, maximize=True)
result = optimizer.grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
})

print(f"Optimal config: {result.optimal_values}")
```

## Project Structure

```
sim/
├── config/          # Configuration dataclasses
├── core/            # Simulation engine, events, state
├── symbolic/        # SymVal and symbolic combinators
├── workers/         # Prefill and decode workers
├── schedulers/      # Continuous batching scheduler
├── cache/           # Radix cache implementation
├── routing/         # Cache-aware router
├── parallel/        # TP/DP/EP group abstractions
├── network/         # Interconnect and KV transfer
├── kernels/         # Attention, MLP, MoE kernels
├── models/          # Operator graph templates
├── metrics/         # Metric collection and symbolic throughput
├── optimizer/       # Scipy/grid search interface
└── workload/        # Synthetic and trace-based workloads
```

## Multi-Objective Optimization with Constraints

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer.interface import (
    MultiObjectiveOptimizer,
    ObjectiveSpec,
    ObjectiveDirection,
)

symbols = ConfigSymbols()
metric_builder = SymbolicMetricBuilder(symbols)
metrics = metric_builder.build_all_expressions()

# Define constraints
constraints = ConstraintBuilder(symbols)
constraints.add_memory_constraint(80 * 1024**3)  # 80GB H100
constraints.add_ttft_slo(0.5)  # 500ms TTFT
constraints.add_tpot_slo(0.05)  # 50ms TPOT
constraints.add_total_gpu_constraint(16)

# Multi-objective optimization
optimizer = MultiObjectiveOptimizer(
    objectives=[
        ObjectiveSpec("throughput", metrics.throughput, ObjectiveDirection.MAXIMIZE),
        ObjectiveSpec("cost", metrics.cost_per_token, ObjectiveDirection.MINIMIZE),
        ObjectiveSpec("latency", metrics.e2e_latency, ObjectiveDirection.MINIMIZE),
    ],
    constraints=constraints.get_all_constraints(),
)

# Find Pareto-optimal configurations
result = optimizer.pareto_grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
    symbols.batch_cap_requests: [64, 128, 256],
})

print(f"Found {len(result.pareto_front)} Pareto-optimal solutions")
```

## Compute Graph Visualization

```python
from sim.visualization import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)

# Visualize expression tree
viz = ExpressionGraphVisualizer()
viz.build_graph(metrics.throughput, "Throughput")
dot_code = viz.to_dot("Throughput Expression")  # For Graphviz
mermaid = viz.to_mermaid("Throughput Expression")  # For Markdown

# Visualize metric dependencies
dep_graph = MetricDependencyGraph()
dep_graph.add_metrics(metrics.as_dict())
print(dep_graph.find_shared_dependencies())  # Variables affecting multiple metrics
```

## Memory Analysis

```python
from sim.metrics.memory import SymbolicMemoryModel

mem_model = SymbolicMemoryModel(symbols)

# Get symbolic expression for peak memory
peak_mem = mem_model.symbolic_peak_memory()

# Evaluate for specific configuration
config = {symbols.batch_cap_requests: 256, symbols.tp_size: 4, ...}
memory_gb = float(peak_mem.subs(list(config.items()))) / 1e9
```

## Examples

See the `examples/` directory for:
- `basic_simulation.py` - Full simulation workflow
- `optimize_pd_ratio.py` - P/D ratio optimization with symbolic expressions
- `sweep_chunk_size.py` - Chunk size analysis and cache effects
- `multi_objective_optimization.py` - Pareto optimization with constraints and visualization

## Running Tests

```bash
python -m pytest tests/ -v
```


```

---

### AI



```
import pytest
import sympy

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder, MetricExpressions
from sim.metrics.memory import SymbolicMemoryModel
from sim.metrics.constraints import ConstraintBuilder, ConstraintType


class TestSymbolicMetricBuilder:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.builder = SymbolicMetricBuilder(self.symbols)

    def test_ttft_expression_is_sympy_expr(self):
        ttft = self.builder.ttft_expression()
        assert isinstance(ttft, sympy.Expr)

    def test_tpot_expression_is_sympy_expr(self):
        tpot = self.builder.tpot_expression()
        assert isinstance(tpot, sympy.Expr)

    def test_e2e_latency_expression_is_sympy_expr(self):
        e2e = self.builder.e2e_latency_expression()
        assert isinstance(e2e, sympy.Expr)

    def test_throughput_expression_is_sympy_expr(self):
        throughput = self.builder.throughput_expression()
        assert isinstance(throughput, sympy.Expr)

    def test_peak_memory_expression_is_sympy_expr(self):
        peak_mem = self.builder.peak_memory_expression()
        assert isinstance(peak_mem, sympy.Expr)

    def test_build_all_expressions(self):
        metrics = self.builder.build_all_expressions()
        assert isinstance(metrics, MetricExpressions)
        assert metrics.ttft is not None
        assert metrics.tpot is not None
        assert metrics.throughput is not None
        assert metrics.peak_memory is not None

    def test_expression_caching(self):
        ttft1 = self.builder.ttft_expression()
        ttft2 = self.builder.ttft_expression()
        assert ttft1 is ttft2

    def test_cache_clear(self):
        self.builder.ttft_expression()
        assert "ttft" in self.builder._cache
        self.builder.clear_cache()
        assert "ttft" not in self.builder._cache

    def test_get_expression_by_name(self):
        ttft = self.builder.get_expression("ttft")
        assert isinstance(ttft, sympy.Expr)

    def test_get_expression_unknown_raises(self):
        with pytest.raises(ValueError):
            self.builder.get_expression("unknown_metric")


class TestSymbolicMemoryModel:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.model = SymbolicMemoryModel(self.symbols)

    def test_symbolic_peak_memory_is_sympy_expr(self):
        peak_mem = self.model.symbolic_peak_memory()
        assert isinstance(peak_mem, sympy.Expr)

    def test_memory_constraint_is_sympy_expr(self):
        constraint = self.model.memory_constraint(80e9)
        assert isinstance(constraint, sympy.Expr)

    def test_peak_memory_scales_with_batch(self):
        s = self.symbols
        peak_mem = self.model.symbolic_peak_memory()
        
        base_params = {
            s.num_layers: 32,
            s.hidden_dim: 4096,
            s.num_heads: 32,
            s.head_dim: 128,
            s.num_experts: 1,
            s.chunk_size: 8192,
            s.avg_prompt_len: 512,
            s.avg_output_len: 128,
            s.tp_size: 1,
        }
        
        mem_batch_64 = float(peak_mem.subs({**base_params, s.batch_cap_requests: 64}.items()))
        mem_batch_128 = float(peak_mem.subs({**base_params, s.batch_cap_requests: 128}.items()))
        
        assert mem_batch_128 > mem_batch_64


class TestConstraintBuilder:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.builder = ConstraintBuilder(self.symbols)

    def test_add_memory_constraint(self):
        constraint = self.builder.add_memory_constraint(80e9)
        assert constraint.name == "memory_limit"
        assert constraint.type == ConstraintType.INEQUALITY

    def test_add_ttft_slo(self):
        constraint = self.builder.add_ttft_slo(0.5)
        assert constraint.name == "ttft_slo"
        assert "500ms" in constraint.description

    def test_add_tpot_slo(self):
        constraint = self.builder.add_tpot_slo(0.05)
        assert constraint.name == "tpot_slo"
        assert "50ms" in constraint.description

    def test_add_min_throughput(self):
        constraint = self.builder.add_min_throughput(1000)
        assert constraint.name == "min_throughput"

    def test_add_total_gpu_constraint(self):
        constraint = self.builder.add_total_gpu_constraint(16)
        assert constraint.name == "max_gpus"
        assert "16" in constraint.description

    def test_get_all_constraints(self):
        self.builder.add_memory_constraint(80e9)
        self.builder.add_ttft_slo(0.5)
        constraints = self.builder.get_all_constraints()
        assert len(constraints) == 2

    def test_summarize(self):
        self.builder.add_memory_constraint(80e9)
        summary = self.builder.summarize()
        assert "memory_limit" in summary

```

---

### AI



```
import pytest
import sympy
from sympy import Symbol

from sim.visualization.graph import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)


class TestExpressionGraphVisualizer:
    def setup_method(self):
        self.viz = ExpressionGraphVisualizer()

    def test_simple_expression(self):
        x = Symbol("x")
        y = Symbol("y")
        expr = x + y
        
        self.viz.build_graph(expr, "sum")
        assert len(self.viz._nodes) > 0
        assert len(self.viz._edges) > 0

    def test_complex_expression(self):
        x, y, z = sympy.symbols("x y z")
        expr = sympy.Max(x * y, z + 2)
        
        self.viz.build_graph(expr, "complex")
        dot = self.viz.to_dot("Test Expression")
        
        assert "digraph G" in dot
        assert "Max" in dot

    def test_to_dot_format(self):
        x = Symbol("x")
        expr = x * 2 + 1
        
        self.viz.build_graph(expr)
        dot = self.viz.to_dot()
        
        assert dot.startswith("digraph G {")
        assert "}" in dot
        assert "node [" in dot

    def test_to_mermaid_format(self):
        x = Symbol("x")
        expr = x + 1
        
        self.viz.build_graph(expr)
        mermaid = self.viz.to_mermaid()
        
        assert "```mermaid" in mermaid
        assert "flowchart TB" in mermaid
        assert "```" in mermaid

    def test_reset_clears_state(self):
        x = Symbol("x")
        self.viz.build_graph(x + 1)
        assert len(self.viz._nodes) > 0
        
        self.viz.reset()
        assert len(self.viz._nodes) == 0
        assert len(self.viz._edges) == 0


class TestMetricDependencyGraph:
    def setup_method(self):
        self.graph = MetricDependencyGraph()

    def test_add_metric(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("test", x + y)
        
        assert "test" in self.graph.nodes
        assert "x" in self.graph.nodes["test"]
        assert "y" in self.graph.nodes["test"]

    def test_add_multiple_metrics(self):
        x, y, z = sympy.symbols("x y z")
        self.graph.add_metrics({
            "metric1": x + y,
            "metric2": y + z,
        })
        
        assert len(self.graph.nodes) == 2
        assert len(self.graph.edges) == 4

    def test_find_shared_dependencies(self):
        x, y, z = sympy.symbols("x y z")
        self.graph.add_metrics({
            "metric1": x + y,
            "metric2": y + z,
            "metric3": z * 2,
        })
        
        shared = self.graph.find_shared_dependencies()
        assert "y" in shared
        assert "z" in shared
        assert "x" not in shared

    def test_to_dot_format(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("throughput", x * y)
        
        dot = self.graph.to_dot()
        assert "digraph" in dot
        assert "throughput" in dot

    def test_to_mermaid_format(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("latency", x + y)
        
        mermaid = self.graph.to_mermaid()
        assert "mermaid" in mermaid
        assert "flowchart" in mermaid


class TestRenderFunctions:
    def test_render_expression_tree_dot(self):
        x = Symbol("x")
        dot = render_expression_tree(x + 1, "Test", "dot")
        assert "digraph" in dot

    def test_render_expression_tree_mermaid(self):
        x = Symbol("x")
        mermaid = render_expression_tree(x + 1, "Test", "mermaid")
        assert "mermaid" in mermaid

    def test_render_metric_dependencies_dot(self):
        x, y = sympy.symbols("x y")
        metrics = {"m1": x, "m2": y}
        dot = render_metric_dependencies(metrics, output_format="dot")
        assert "digraph" in dot

    def test_render_metric_dependencies_mermaid(self):
        x, y = sympy.symbols("x y")
        metrics = {"m1": x, "m2": y}
        mermaid = render_metric_dependencies(metrics, output_format="mermaid")
        assert "mermaid" in mermaid

```

---

### AI

All 47 tests pass! Let me show you the final summary of the added features:

---

### AI

I've extended the simulator with comprehensive support for constrained optimization. Here's a summary of the new features:

## New Modules Added

### 1. `sim/metrics/expressions.py` - Symbolic Metric Expressions
Full symbolic SymPy expressions for all key metrics:
- **TTFT** (Time to First Token) - considers cache hit rate, chunked prefill
- **TPOT** (Time per Output Token) - memory-bound decode analysis
- **E2E Latency** - full request lifecycle including KV transfer
- **Throughput** - bottleneck analysis of prefill vs decode capacity
- **Peak Memory** - model weights + KV cache + activations
- **Compute Utilization** - achieved vs theoretical FLOPs
- **Memory Bandwidth Utilization** - KV read/write patterns
- **Network Utilization** - P/D transfer bandwidth
- **Cost per Token** - $/token based on GPU hours
- **Power Consumption** - utilization-weighted TDP estimate

### 2. `sim/metrics/memory.py` - Memory Modeling
- `SymbolicMemoryModel` class with detailed breakdown
- Per-component memory: weights, KV cache, activations, workspace
- TP/batch/sequence scaling analysis
- Memory constraint expressions

### 3. `sim/metrics/constraints.py` - Constraint Builder
Pre-built constraints for constrained optimization:
- Memory limits (e.g., 80GB H100 with safety margin)
- SLO targets (TTFT, TPOT, E2E latency)
- Minimum throughput requirements
- Cost budgets ($/1K tokens)
- Power budgets
- GPU count limits
- P/D ratio bounds
- Chunk size and batch size bounds

### 4. `sim/visualization/graph.py` - Compute Graph Visualization
- `ExpressionGraphVisualizer` - builds DAG from SymPy expression tree
- `MetricDependencyGraph` - shows which variables affect which metrics
- Output formats: **DOT** (for Graphviz) and **Mermaid** (for Markdown)
- Finds shared dependencies across metrics

### 5. `sim/optimizer/interface.py` - Multi-Objective Optimization
Extended with:
- `MultiObjectiveOptimizer` class
- Pareto front computation via grid enumeration
- Weighted-sum scalarization
- Epsilon-constraint method
- Constrained grid search

## Example Usage

```python
from sim.metrics.expressions import SymbolicMetricBuilder
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer import MultiObjectiveOptimizer, ObjectiveSpec, ObjectiveDirection
from sim.visualization import render_expression_tree, MetricDependencyGraph

# Build all metrics as SymPy expressions
metrics = SymbolicMetricBuilder(symbols).build_all_expressions()

# Add constraints
constraints = ConstraintBuilder(symbols)
constraints.add_memory_constraint(80e9)
constraints.add_ttft_slo(0.5)
constraints.add_total_gpu_constraint(16)

# Multi-objective Pareto optimization
optimizer = MultiObjectiveOptimizer([
    ObjectiveSpec("throughput", metrics.throughput, ObjectiveDirection.MAXIMIZE),
    ObjectiveSpec("cost", metrics.cost_per_token, ObjectiveDirection.MINIMIZE),
], constraints=constraints.get_all_constraints())

result = optimizer.pareto_grid_search(param_grid)

# Visualize
print(render_expression_tree(metrics.throughput, format="mermaid"))
```

```
from sim.metrics.expressions import SymbolicMetricBuilder
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer import MultiObjectiveOptimizer, ObjectiveSpec, ObjectiveDirection
from sim.visualization import render_expression_tree, MetricDependencyGraph

# Build all metrics as SymPy expressions
metrics = SymbolicMetricBuilder(symbols).build_all_expressions()

# Add constraints
constraints = ConstraintBuilder(symbols)
constraints.add_memory_constraint(80e9)
constraints.add_ttft_slo(0.5)
constraints.add_total_gpu_constraint(16)

# Multi-objective Pareto optimization
optimizer = MultiObjectiveOptimizer([
    ObjectiveSpec("throughput", metrics.throughput, ObjectiveDirection.MAXIMIZE),
    ObjectiveSpec("cost", metrics.cost_per_token, ObjectiveDirection.MINIMIZE),
], constraints=constraints.get_all_constraints())

result = optimizer.pareto_grid_search(param_grid)

# Visualize
print(render_expression_tree(metrics.throughput, format="mermaid"))
```

---

### AI

**Tool Action:**

**Code Changes:**
```
[project]
name = "sglang-sim"
version = "0.1.0"
description = "LLM inference simulator with symbolic tracing for SGLang-style workloads"
requires-python = ">=3.10"
dependencies = [
    "sympy>=1.12",
    "numpy>=1.24",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.symbolic.expr import SymVal
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig, GPUSpec, InterconnectConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig

__all__ = [
    "SymVal",
    "ModelConfig",
    "ClusterConfig",
    "GPUSpec",
    "InterconnectConfig",
    "SchedulerConfig",
    "CacheConfig",
]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.symbolic.expr import (
    SymVal,
    sym_add,
    sym_sub,
    sym_mul,
    sym_div,
    sym_max,
    sym_min,
    sym_piecewise,
    sym_sum,
    sym_const,
)
from sim.symbolic.symbols import ConfigSymbols

__all__ = [
    "SymVal",
    "sym_add",
    "sym_sub",
    "sym_mul",
    "sym_div",
    "sym_max",
    "sym_min",
    "sym_piecewise",
    "sym_sum",
    "sym_const",
    "ConfigSymbols",
]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import sympy
from sympy import Max, Min, Piecewise, Sum, Symbol, ceiling, floor


@dataclass
class SymVal:
    expr: sympy.Expr
    val: float
    meta: dict = field(default_factory=dict)

    def __add__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr + other.expr,
                val=self.val + other.val,
                meta={"op": "add", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr + other,
            val=self.val + float(other),
            meta={"op": "add_scalar", "operands": [self.meta, other]},
        )

    def __radd__(self, other: float | int) -> SymVal:
        return self.__add__(other)

    def __sub__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr - other.expr,
                val=self.val - other.val,
                meta={"op": "sub", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr - other,
            val=self.val - float(other),
            meta={"op": "sub_scalar", "operands": [self.meta, other]},
        )

    def __rsub__(self, other: float | int) -> SymVal:
        return SymVal(
            expr=other - self.expr,
            val=float(other) - self.val,
            meta={"op": "rsub_scalar", "operands": [other, self.meta]},
        )

    def __mul__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr * other.expr,
                val=self.val * other.val,
                meta={"op": "mul", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr * other,
            val=self.val * float(other),
            meta={"op": "mul_scalar", "operands": [self.meta, other]},
        )

    def __rmul__(self, other: float | int) -> SymVal:
        return self.__mul__(other)

    def __truediv__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr / other.expr,
                val=self.val / other.val if other.val != 0 else float("inf"),
                meta={"op": "div", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr / other,
            val=self.val / float(other) if other != 0 else float("inf"),
            meta={"op": "div_scalar", "operands": [self.meta, other]},
        )

    def __rtruediv__(self, other: float | int) -> SymVal:
        return SymVal(
            expr=other / self.expr,
            val=float(other) / self.val if self.val != 0 else float("inf"),
            meta={"op": "rdiv_scalar", "operands": [other, self.meta]},
        )

    def __neg__(self) -> SymVal:
        return SymVal(
            expr=-self.expr,
            val=-self.val,
            meta={"op": "neg", "operands": [self.meta]},
        )

    def __repr__(self) -> str:
        return f"SymVal(expr={self.expr}, val={self.val:.6g})"

    def lambdify(self, symbols: list[Symbol]) -> Callable[..., float]:
        return sympy.lambdify(symbols, self.expr, modules=["numpy"])

    def simplify(self) -> SymVal:
        return SymVal(
            expr=sympy.simplify(self.expr),
            val=self.val,
            meta=self.meta,
        )


def sym_const(value: float, name: str | None = None) -> SymVal:
    if name:
        return SymVal(expr=Symbol(name), val=value, meta={"type": "symbol", "name": name})
    return SymVal(expr=sympy.Float(value), val=value, meta={"type": "const", "value": value})


def sym_add(a: SymVal, b: SymVal) -> SymVal:
    return a + b


def sym_sub(a: SymVal, b: SymVal) -> SymVal:
    return a - b


def sym_mul(a: SymVal, b: SymVal) -> SymVal:
    return a * b


def sym_div(a: SymVal, b: SymVal) -> SymVal:
    return a / b


def sym_max(a: SymVal, b: SymVal) -> SymVal:
    return SymVal(
        expr=Max(a.expr, b.expr),
        val=max(a.val, b.val),
        meta={"op": "max", "operands": [a.meta, b.meta]},
    )


def sym_min(a: SymVal, b: SymVal) -> SymVal:
    return SymVal(
        expr=Min(a.expr, b.expr),
        val=min(a.val, b.val),
        meta={"op": "min", "operands": [a.meta, b.meta]},
    )


def sym_piecewise(
    cases: list[tuple[sympy.Expr | bool, SymVal]],
    numeric_condition_results: list[bool],
) -> SymVal:
    pw_args = [(case[1].expr, case[0]) for case in cases]
    pw_expr = Piecewise(*pw_args)

    result_val = cases[-1][1].val
    for i, cond_result in enumerate(numeric_condition_results):
        if cond_result:
            result_val = cases[i][1].val
            break

    return SymVal(
        expr=pw_expr,
        val=result_val,
        meta={
            "op": "piecewise",
            "cases": [(str(c[0]), c[1].meta) for c in cases],
        },
    )


def sym_sum(
    body_fn: Callable[[Symbol], SymVal],
    var: Symbol,
    start: int,
    end: int,
) -> SymVal:
    symbolic_body = body_fn(var)
    sum_expr = Sum(symbolic_body.expr, (var, start, end))

    numeric_sum = 0.0
    for i in range(start, end + 1):
        step_val = body_fn(sympy.Integer(i))
        numeric_sum += step_val.val

    return SymVal(
        expr=sum_expr,
        val=numeric_sum,
        meta={
            "op": "sum",
            "var": str(var),
            "range": (start, end),
            "body_meta": symbolic_body.meta,
        },
    )


def sym_ceiling(a: SymVal) -> SymVal:
    return SymVal(
        expr=ceiling(a.expr),
        val=float(int(a.val) + (1 if a.val > int(a.val) else 0)),
        meta={"op": "ceiling", "operands": [a.meta]},
    )


def sym_floor(a: SymVal) -> SymVal:
    return SymVal(
        expr=floor(a.expr),
        val=float(int(a.val)),
        meta={"op": "floor", "operands": [a.meta]},
    )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig, GPUSpec, InterconnectConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig

__all__ = [
    "ModelConfig",
    "ClusterConfig",
    "GPUSpec",
    "InterconnectConfig",
    "SchedulerConfig",
    "CacheConfig",
]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    vocab_size: int
    is_moe: bool = False
    num_experts: int = 1
    top_k_experts: int = 1
    intermediate_dim: int | None = None
    dtype_bytes: int = 2  # fp16 by default

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.head_dim * self.num_heads * self.dtype_bytes

    @property
    def actual_intermediate_dim(self) -> int:
        return self.intermediate_dim or 4 * self.hidden_dim

    @classmethod
    def llama_7b(cls) -> ModelConfig:
        return cls(
            name="llama-7b",
            num_layers=32,
            hidden_dim=4096,
            num_heads=32,
            head_dim=128,
            vocab_size=32000,
            intermediate_dim=11008,
        )

    @classmethod
    def llama_70b(cls) -> ModelConfig:
        return cls(
            name="llama-70b",
            num_layers=80,
            hidden_dim=8192,
            num_heads=64,
            head_dim=128,
            vocab_size=32000,
            intermediate_dim=28672,
        )

    @classmethod
    def mixtral_8x7b(cls) -> ModelConfig:
        return cls(
            name="mixtral-8x7b",
            num_layers=32,
            hidden_dim=4096,
            num_heads=32,
            head_dim=128,
            vocab_size=32000,
            is_moe=True,
            num_experts=8,
            top_k_experts=2,
            intermediate_dim=14336,
        )

    @classmethod
    def deepseek_v3(cls) -> ModelConfig:
        return cls(
            name="deepseek-v3",
            num_layers=61,
            hidden_dim=7168,
            num_heads=128,
            head_dim=128,
            vocab_size=129280,
            is_moe=True,
            num_experts=256,
            top_k_experts=8,
            intermediate_dim=18432,
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class InterconnectType(Enum):
    NVLINK = "nvlink"
    ETHERNET = "ethernet"
    INFINIBAND = "infiniband"


@dataclass
class InterconnectConfig:
    type: InterconnectType
    bandwidth_gbps: float
    latency_us: float

    @property
    def bandwidth_bytes_per_sec(self) -> float:
        return self.bandwidth_gbps * 1e9 / 8

    @property
    def latency_seconds(self) -> float:
        return self.latency_us * 1e-6

    def transfer_time_seconds(self, bytes_to_transfer: int) -> float:
        return self.latency_seconds + bytes_to_transfer / self.bandwidth_bytes_per_sec

    @classmethod
    def nvlink_4(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.NVLINK,
            bandwidth_gbps=900,  # NVLink 4.0: ~900 Gbps bidirectional
            latency_us=1.0,
        )

    @classmethod
    def infiniband_hdr(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.INFINIBAND,
            bandwidth_gbps=200,  # HDR InfiniBand
            latency_us=2.0,
        )

    @classmethod
    def ethernet_100g(cls) -> InterconnectConfig:
        return cls(
            type=InterconnectType.ETHERNET,
            bandwidth_gbps=100,
            latency_us=10.0,
        )


@dataclass
class GPUSpec:
    name: str
    flops_fp16: float  # FLOP/s
    memory_bandwidth: float  # bytes/s
    memory_capacity: int  # bytes

    @classmethod
    def h100_sxm(cls) -> GPUSpec:
        return cls(
            name="H100-SXM",
            flops_fp16=1979e12,  # ~2 PFLOP/s FP16
            memory_bandwidth=3.35e12,  # 3.35 TB/s
            memory_capacity=80 * 1024**3,  # 80 GB
        )

    @classmethod
    def a100_80g(cls) -> GPUSpec:
        return cls(
            name="A100-80G",
            flops_fp16=312e12,  # 312 TFLOP/s FP16
            memory_bandwidth=2.0e12,  # 2 TB/s
            memory_capacity=80 * 1024**3,
        )

    @classmethod
    def h200(cls) -> GPUSpec:
        return cls(
            name="H200",
            flops_fp16=1979e12,
            memory_bandwidth=4.8e12,  # 4.8 TB/s with HBM3e
            memory_capacity=141 * 1024**3,  # 141 GB
        )


@dataclass
class ClusterConfig:
    num_prefill_workers: int
    num_decode_workers: int
    gpus_per_worker: int
    gpu_spec: GPUSpec
    intra_node_interconnect: InterconnectConfig
    inter_node_interconnect: InterconnectConfig

    @property
    def total_workers(self) -> int:
        return self.num_prefill_workers + self.num_decode_workers

    @property
    def total_gpus(self) -> int:
        return self.total_workers * self.gpus_per_worker

    @classmethod
    def single_node_8gpu(cls) -> ClusterConfig:
        return cls(
            num_prefill_workers=4,
            num_decode_workers=4,
            gpus_per_worker=1,
            gpu_spec=GPUSpec.h100_sxm(),
            intra_node_interconnect=InterconnectConfig.nvlink_4(),
            inter_node_interconnect=InterconnectConfig.nvlink_4(),
        )

    @classmethod
    def multi_node_16gpu(cls) -> ClusterConfig:
        return cls(
            num_prefill_workers=8,
            num_decode_workers=8,
            gpus_per_worker=1,
            gpu_spec=GPUSpec.h100_sxm(),
            intra_node_interconnect=InterconnectConfig.nvlink_4(),
            inter_node_interconnect=InterconnectConfig.infiniband_hdr(),
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    chunk_size: int
    max_batch_tokens: int
    max_batch_requests: int
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    max_prefill_tokens_per_iter: int | None = None
    enable_chunked_prefill: bool = True

    @property
    def effective_max_prefill_tokens(self) -> int:
        return self.max_prefill_tokens_per_iter or self.chunk_size

    @classmethod
    def default(cls) -> SchedulerConfig:
        return cls(
            chunk_size=8192,
            max_batch_tokens=32768,
            max_batch_requests=256,
            tp_size=1,
            dp_size=1,
            ep_size=1,
        )

    @classmethod
    def high_throughput(cls) -> SchedulerConfig:
        return cls(
            chunk_size=16384,
            max_batch_tokens=65536,
            max_batch_requests=512,
            tp_size=8,
            dp_size=1,
            ep_size=1,
        )

    @classmethod
    def moe_optimized(cls) -> SchedulerConfig:
        return cls(
            chunk_size=8192,
            max_batch_tokens=32768,
            max_batch_requests=256,
            tp_size=1,
            dp_size=8,
            ep_size=8,
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class EvictionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"


@dataclass
class CacheConfig:
    capacity_bytes: int
    block_size_tokens: int
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    max_tree_depth: int = 256

    @property
    def capacity_blocks(self) -> int:
        return self.capacity_bytes // self.block_size_tokens

    @classmethod
    def default_for_gpu(cls, gpu_memory_bytes: int, cache_fraction: float = 0.8) -> CacheConfig:
        return cls(
            capacity_bytes=int(gpu_memory_bytes * cache_fraction),
            block_size_tokens=16,
            eviction_policy=EvictionPolicy.LRU,
        )

    @classmethod
    def h100_default(cls) -> CacheConfig:
        return cls(
            capacity_bytes=64 * 1024**3,  # 64 GB for KV cache
            block_size_tokens=16,
            eviction_policy=EvictionPolicy.LRU,
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.core.events import (
    Event,
    RequestArrival,
    RouterDispatch,
    PrefillIterationStart,
    PrefillChunkComplete,
    KVTransferStart,
    KVTransferComplete,
    DecodeIterationStart,
    TokenEmit,
    RequestComplete,
)
from sim.core.engine import SimulationEngine, SimulationResult
from sim.core.state import ClusterState, WorkerState
from sim.core.request import Request, PrefillTask, DecodeTask, KVHandle, RequestPhase

__all__ = [
    "Event",
    "RequestArrival",
    "RouterDispatch",
    "PrefillIterationStart",
    "PrefillChunkComplete",
    "KVTransferStart",
    "KVTransferComplete",
    "DecodeIterationStart",
    "TokenEmit",
    "RequestComplete",
    "SimulationEngine",
    "SimulationResult",
    "ClusterState",
    "WorkerState",
    "Request",
    "PrefillTask",
    "DecodeTask",
    "KVHandle",
    "RequestPhase",
]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestPhase(Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    TRANSFERRING = "transferring"
    DECODING = "decoding"
    COMPLETE = "complete"


@dataclass
class Request:
    id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_time: float
    stream: bool = False
    prefix_group_id: int | None = None
    sampling_params: dict[str, Any] = field(default_factory=dict)

    phase: RequestPhase = field(default=RequestPhase.QUEUED)
    prefill_worker_id: int | None = None
    decode_worker_id: int | None = None

    first_token_time: float | None = None
    complete_time: float | None = None
    tokens_generated: int = 0
    token_times: list[float] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def e2e_latency(self) -> float | None:
        if self.complete_time is None:
            return None
        return self.complete_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        if len(self.token_times) < 2:
            return None
        inter_token_delays = [
            self.token_times[i] - self.token_times[i - 1]
            for i in range(1, len(self.token_times))
        ]
        return sum(inter_token_delays) / len(inter_token_delays) if inter_token_delays else None


@dataclass
class KVHandle:
    request_id: int
    kv_bytes: int
    source_worker_id: int
    dest_worker_id: int | None = None
    transfer_started: float | None = None
    transfer_complete: float | None = None

    @property
    def is_transferred(self) -> bool:
        return self.transfer_complete is not None


@dataclass
class PrefillTask:
    request: Request
    remaining_prompt_tokens: int
    chunk_size: int
    kv_bytes_produced: int = 0
    prefix_match_len: int = 0
    chunks_completed: int = 0
    current_chunk_start: float | None = None

    @property
    def is_complete(self) -> bool:
        return self.remaining_prompt_tokens <= 0

    @property
    def tokens_to_prefill_this_chunk(self) -> int:
        return min(self.chunk_size, self.remaining_prompt_tokens)

    def complete_chunk(self, tokens: int, kv_bytes: int) -> None:
        self.remaining_prompt_tokens -= tokens
        self.kv_bytes_produced += kv_bytes
        self.chunks_completed += 1


@dataclass
class DecodeTask:
    request: Request
    kv_handle: KVHandle
    remaining_tokens: int
    current_kv_len: int
    tokens_emitted: int = 0
    iteration_start: float | None = None

    @property
    def is_complete(self) -> bool:
        return self.remaining_tokens <= 0

    def emit_token(self, kv_bytes_per_token: int) -> None:
        self.remaining_tokens -= 1
        self.tokens_emitted += 1
        self.current_kv_len += 1
        self.kv_handle.kv_bytes += kv_bytes_per_token

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, PrefillTask, DecodeTask, KVHandle
    from sim.core.engine import SimulationEngine


@dataclass(order=True)
class Event(ABC):
    time: float
    priority: int = field(default=0, compare=True)

    @abstractmethod
    def process(self, engine: SimulationEngine) -> list[Event]:
        pass


@dataclass(order=True)
class RequestArrival(Event):
    request: Request = field(compare=False)
    priority: int = field(default=10, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        engine.state.pending_requests.append(self.request)
        worker_id = engine.router.route_to_prefill(self.request)
        return [RouterDispatch(time=self.time, request=self.request, worker_id=worker_id)]


@dataclass(order=True)
class RouterDispatch(Event):
    request: Request = field(compare=False)
    worker_id: int = field(compare=False)
    priority: int = field(default=20, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.request.phase = RequestPhase.PREFILLING
        self.request.prefill_worker_id = self.worker_id
        worker = engine.prefill_workers[self.worker_id]
        worker.enqueue_request(self.request, engine.scheduler_config)

        if not worker.is_iteration_scheduled:
            worker.is_iteration_scheduled = True
            return [PrefillIterationStart(time=self.time, worker_id=self.worker_id)]
        return []


@dataclass(order=True)
class PrefillIterationStart(Event):
    worker_id: int = field(compare=False)
    priority: int = field(default=30, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        worker = engine.prefill_workers[self.worker_id]
        iteration_result = worker.run_iteration(self.time, engine)

        events: list[Event] = []
        for task in iteration_result.completed_tasks:
            events.append(
                PrefillChunkComplete(
                    time=self.time + iteration_result.iteration_time.val,
                    task=task,
                    worker_id=self.worker_id,
                )
            )

        if worker.has_pending_work():
            events.append(
                PrefillIterationStart(
                    time=self.time + iteration_result.iteration_time.val,
                    worker_id=self.worker_id,
                )
            )
        else:
            worker.is_iteration_scheduled = False

        return events


@dataclass(order=True)
class PrefillChunkComplete(Event):
    task: PrefillTask = field(compare=False)
    worker_id: int = field(compare=False)
    priority: int = field(default=40, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase, KVHandle

        if self.task.is_complete:
            self.task.request.phase = RequestPhase.TRANSFERRING
            kv_handle = KVHandle(
                request_id=self.task.request.id,
                kv_bytes=self.task.kv_bytes_produced,
                source_worker_id=self.worker_id,
            )
            dest_worker = engine.router.route_to_decode(kv_handle)
            kv_handle.dest_worker_id = dest_worker
            return [
                KVTransferStart(
                    time=self.time,
                    kv_handle=kv_handle,
                    request=self.task.request,
                )
            ]
        return []


@dataclass(order=True)
class KVTransferStart(Event):
    kv_handle: KVHandle = field(compare=False)
    request: Request = field(compare=False)
    priority: int = field(default=50, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        self.kv_handle.transfer_started = self.time
        transfer_time = engine.kv_transfer_manager.initiate_transfer(
            self.kv_handle, engine.cluster_config
        )
        return [
            KVTransferComplete(
                time=self.time + transfer_time.val,
                kv_handle=self.kv_handle,
                request=self.request,
            )
        ]


@dataclass(order=True)
class KVTransferComplete(Event):
    kv_handle: KVHandle = field(compare=False)
    request: Request = field(compare=False)
    priority: int = field(default=60, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.kv_handle.transfer_complete = self.time
        self.request.phase = RequestPhase.DECODING
        self.request.decode_worker_id = self.kv_handle.dest_worker_id

        worker = engine.decode_workers[self.kv_handle.dest_worker_id]
        worker.enqueue_request(self.request, self.kv_handle, engine.scheduler_config)

        if not worker.is_iteration_scheduled:
            worker.is_iteration_scheduled = True
            return [
                DecodeIterationStart(
                    time=self.time, worker_id=self.kv_handle.dest_worker_id
                )
            ]
        return []


@dataclass(order=True)
class DecodeIterationStart(Event):
    worker_id: int = field(compare=False)
    priority: int = field(default=70, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        worker = engine.decode_workers[self.worker_id]
        iteration_result = worker.run_iteration(self.time, engine)

        events: list[Event] = []
        for task in iteration_result.token_emitted_tasks:
            if task.request.tokens_generated == 1:
                task.request.first_token_time = self.time + iteration_result.iteration_time.val

            events.append(
                TokenEmit(
                    time=self.time + iteration_result.iteration_time.val,
                    task=task,
                    worker_id=self.worker_id,
                )
            )

        if worker.has_pending_work():
            events.append(
                DecodeIterationStart(
                    time=self.time + iteration_result.iteration_time.val,
                    worker_id=self.worker_id,
                )
            )
        else:
            worker.is_iteration_scheduled = False

        return events


@dataclass(order=True)
class TokenEmit(Event):
    task: DecodeTask = field(compare=False)
    worker_id: int = field(compare=False)
    priority: int = field(default=80, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        self.task.request.token_times.append(self.time)

        if self.task.is_complete:
            return [
                RequestComplete(
                    time=self.time,
                    request=self.task.request,
                )
            ]
        return []


@dataclass(order=True)
class RequestComplete(Event):
    request: Request = field(compare=False)
    priority: int = field(default=90, compare=True)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.request.phase = RequestPhase.COMPLETE
        self.request.complete_time = self.time
        engine.metrics.record_request_complete(self.request)
        return []

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, PrefillTask, DecodeTask


@dataclass
class WorkerState:
    worker_id: int
    is_prefill: bool
    waiting_queue: list[Request] = field(default_factory=list)
    running_tasks: list[PrefillTask | DecodeTask] = field(default_factory=list)
    total_tokens_processed: int = 0
    total_iterations: int = 0

    @property
    def queue_length(self) -> int:
        return len(self.waiting_queue) + len(self.running_tasks)

    @property
    def is_idle(self) -> bool:
        return len(self.waiting_queue) == 0 and len(self.running_tasks) == 0


@dataclass
class ClusterState:
    prefill_worker_states: list[WorkerState] = field(default_factory=list)
    decode_worker_states: list[WorkerState] = field(default_factory=list)
    pending_requests: list[Request] = field(default_factory=list)
    completed_requests: list[Request] = field(default_factory=list)
    total_tokens_generated: int = 0

    @classmethod
    def initialize(cls, num_prefill: int, num_decode: int) -> ClusterState:
        return cls(
            prefill_worker_states=[
                WorkerState(worker_id=i, is_prefill=True) for i in range(num_prefill)
            ],
            decode_worker_states=[
                WorkerState(worker_id=i, is_prefill=False) for i in range(num_decode)
            ],
        )

    def get_prefill_queue_lengths(self) -> list[int]:
        return [w.queue_length for w in self.prefill_worker_states]

    def get_decode_queue_lengths(self) -> list[int]:
        return [w.queue_length for w in self.decode_worker_states]

    def is_balanced(self, threshold: float = 2.0) -> bool:
        prefill_lengths = self.get_prefill_queue_lengths()
        if not prefill_lengths or max(prefill_lengths) == 0:
            return True
        return max(prefill_lengths) / max(1, min(prefill_lengths)) <= threshold

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import TYPE_CHECKING

from sim.core.state import ClusterState
from sim.core.events import Event

if TYPE_CHECKING:
    from sim.config.model import ModelConfig
    from sim.config.cluster import ClusterConfig
    from sim.config.scheduler import SchedulerConfig
    from sim.config.cache import CacheConfig
    from sim.workers.prefill_worker import PrefillWorker
    from sim.workers.decode_worker import DecodeWorker
    from sim.routing.router import Router
    from sim.network.kv_transfer import KVTransferManager
    from sim.metrics.definitions import MetricCollector


@dataclass
class SimulationResult:
    total_requests: int
    completed_requests: int
    total_tokens_generated: int
    simulation_time: float
    throughput_tokens_per_sec: float
    avg_ttft: float
    avg_tpot: float
    avg_e2e_latency: float
    p50_ttft: float
    p99_ttft: float
    p50_tpot: float
    p99_tpot: float


class SimulationEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        cluster_config: ClusterConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ):
        self.model_config = model_config
        self.cluster_config = cluster_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.clock: float = 0.0
        self.event_queue: list[tuple[float, int, Event]] = []
        self.event_counter: int = 0

        self.state = ClusterState.initialize(
            cluster_config.num_prefill_workers,
            cluster_config.num_decode_workers,
        )

        self.prefill_workers: list[PrefillWorker] = []
        self.decode_workers: list[DecodeWorker] = []
        self.router: Router | None = None
        self.kv_transfer_manager: KVTransferManager | None = None
        self.metrics: MetricCollector | None = None

    def initialize_components(
        self,
        prefill_workers: list[PrefillWorker],
        decode_workers: list[DecodeWorker],
        router: Router,
        kv_transfer_manager: KVTransferManager,
        metrics: MetricCollector,
    ) -> None:
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.router = router
        self.kv_transfer_manager = kv_transfer_manager
        self.metrics = metrics

    def schedule_event(self, event: Event) -> None:
        self.event_counter += 1
        heappush(self.event_queue, (event.time, event.priority, self.event_counter, event))

    def schedule_events(self, events: list[Event]) -> None:
        for event in events:
            self.schedule_event(event)

    def run(self, until: float | None = None) -> SimulationResult:
        while self.event_queue:
            time, priority, _, event = heappop(self.event_queue)

            if until is not None and time > until:
                break

            self.clock = time
            new_events = event.process(self)
            self.schedule_events(new_events)

        return self._compute_results()

    def run_until_idle(self) -> SimulationResult:
        while self.event_queue:
            time, priority, _, event = heappop(self.event_queue)
            self.clock = time
            new_events = event.process(self)
            self.schedule_events(new_events)

        return self._compute_results()

    def _compute_results(self) -> SimulationResult:
        if self.metrics is None:
            raise RuntimeError("MetricCollector not initialized")

        return SimulationResult(
            total_requests=self.metrics.total_requests,
            completed_requests=self.metrics.completed_requests,
            total_tokens_generated=self.metrics.total_tokens,
            simulation_time=self.clock,
            throughput_tokens_per_sec=self.metrics.throughput(self.clock),
            avg_ttft=self.metrics.avg_ttft(),
            avg_tpot=self.metrics.avg_tpot(),
            avg_e2e_latency=self.metrics.avg_e2e_latency(),
            p50_ttft=self.metrics.percentile_ttft(0.5),
            p99_ttft=self.metrics.percentile_ttft(0.99),
            p50_tpot=self.metrics.percentile_tpot(0.5),
            p99_tpot=self.metrics.percentile_tpot(0.99),
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.schedulers.batch_scheduler import ContinuousBatchScheduler, BatchSelectionResult

__all__ = ["ContinuousBatchScheduler", "BatchSelectionResult"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_add, sym_const

if TYPE_CHECKING:
    from sim.core.request import PrefillTask, DecodeTask
    from sim.config.scheduler import SchedulerConfig


@dataclass
class BatchSelectionResult:
    selected_tasks: list[PrefillTask | DecodeTask]
    total_tokens: int
    estimated_cost: SymVal


class ContinuousBatchScheduler:
    def select_prefill_batch(
        self,
        waiting_tasks: list[PrefillTask],
        running_tasks: list[PrefillTask],
        config: SchedulerConfig,
    ) -> BatchSelectionResult:
        selected: list[PrefillTask] = []
        total_tokens = 0
        max_tokens = config.max_batch_tokens
        max_requests = config.max_batch_requests

        running_tokens = sum(t.tokens_to_prefill_this_chunk for t in running_tasks)
        current_tokens = running_tokens

        for task in waiting_tasks:
            chunk_tokens = task.tokens_to_prefill_this_chunk

            if len(selected) + len(running_tasks) >= max_requests:
                break
            if current_tokens + chunk_tokens > max_tokens:
                break

            selected.append(task)
            current_tokens += chunk_tokens
            total_tokens += chunk_tokens

        return BatchSelectionResult(
            selected_tasks=selected,
            total_tokens=total_tokens,
            estimated_cost=sym_const(0.0),  # Will be computed by worker
        )

    def select_decode_batch(
        self,
        waiting_tasks: list[DecodeTask],
        running_tasks: list[DecodeTask],
        config: SchedulerConfig,
    ) -> BatchSelectionResult:
        selected: list[DecodeTask] = []
        max_requests = config.max_batch_requests

        running_count = len(running_tasks)

        for task in waiting_tasks:
            if len(selected) + running_count >= max_requests:
                break
            selected.append(task)

        total_tokens = len(selected) + running_count

        return BatchSelectionResult(
            selected_tasks=selected,
            total_tokens=total_tokens,
            estimated_cost=sym_const(0.0),
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.workers.prefill_worker import PrefillWorker, PrefillIterationResult
from sim.workers.decode_worker import DecodeWorker, DecodeIterationResult

__all__ = [
    "PrefillWorker",
    "PrefillIterationResult",
    "DecodeWorker",
    "DecodeIterationResult",
]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.core.request import Request, PrefillTask
from sim.schedulers.batch_scheduler import ContinuousBatchScheduler
from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_max

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.config.scheduler import SchedulerConfig
    from sim.cache.radix_cache import RadixCache
    from sim.parallel.groups import TPGroup, EPGroup


@dataclass
class PrefillIterationResult:
    iteration_time: SymVal
    completed_tasks: list[PrefillTask]
    tokens_processed: int


class PrefillWorker:
    def __init__(
        self,
        worker_id: int,
        cache: RadixCache | None = None,
        tp_group: TPGroup | None = None,
        ep_group: EPGroup | None = None,
    ):
        self.worker_id = worker_id
        self.cache = cache
        self.tp_group = tp_group
        self.ep_group = ep_group

        self.waiting_queue: deque[PrefillTask] = deque()
        self.running_batch: list[PrefillTask] = []
        self.scheduler = ContinuousBatchScheduler()
        self.is_iteration_scheduled: bool = False
        self.total_tokens_processed: int = 0
        self.total_iterations: int = 0

    def enqueue_request(self, request: Request, config: SchedulerConfig) -> PrefillTask:
        prefix_match_len = 0
        if self.cache is not None:
            prefix_match_len, _ = self.cache.match_prefix(request.prompt_tokens)

        task = PrefillTask(
            request=request,
            remaining_prompt_tokens=request.prompt_len - prefix_match_len,
            chunk_size=config.chunk_size,
            prefix_match_len=prefix_match_len,
        )
        self.waiting_queue.append(task)
        return task

    def has_pending_work(self) -> bool:
        return len(self.waiting_queue) > 0 or len(self.running_batch) > 0

    def run_iteration(
        self,
        current_time: float,
        engine: SimulationEngine,
    ) -> PrefillIterationResult:
        config = engine.scheduler_config

        selection = self.scheduler.select_prefill_batch(
            waiting_tasks=list(self.waiting_queue),
            running_tasks=self.running_batch,
            config=config,
        )

        for task in selection.selected_tasks:
            self.waiting_queue.remove(task)
            self.running_batch.append(task)
            task.current_chunk_start = current_time

        all_active = self.running_batch.copy()
        total_tokens = sum(t.tokens_to_prefill_this_chunk for t in all_active)

        iteration_time = self._compute_iteration_time(
            total_tokens, all_active, engine
        )

        completed_tasks: list[PrefillTask] = []
        for task in all_active:
            chunk_tokens = task.tokens_to_prefill_this_chunk
            kv_bytes = chunk_tokens * engine.model_config.kv_bytes_per_token
            task.complete_chunk(chunk_tokens, kv_bytes)

            if task.is_complete:
                completed_tasks.append(task)
                self.running_batch.remove(task)

                if self.cache is not None:
                    self.cache.insert(
                        task.request.prompt_tokens,
                        task.kv_bytes_produced,
                    )

        self.total_tokens_processed += total_tokens
        self.total_iterations += 1

        return PrefillIterationResult(
            iteration_time=iteration_time,
            completed_tasks=completed_tasks,
            tokens_processed=total_tokens,
        )

    def _compute_iteration_time(
        self,
        total_tokens: int,
        tasks: list[PrefillTask],
        engine: SimulationEngine,
    ) -> SymVal:
        if total_tokens == 0:
            return sym_const(0.0)

        model = engine.model_config
        gpu = engine.cluster_config.gpu_spec

        flops_per_token = (
            2 * model.num_layers * (
                4 * model.hidden_dim * model.hidden_dim +
                2 * model.hidden_dim * model.actual_intermediate_dim
            )
        )
        total_flops = flops_per_token * total_tokens

        attention_flops_per_layer = 2 * total_tokens * total_tokens * model.hidden_dim
        total_attention_flops = attention_flops_per_layer * model.num_layers
        total_flops += total_attention_flops

        effective_flops = gpu.flops_fp16
        if self.tp_group is not None:
            effective_flops *= self.tp_group.size

        compute_time = total_flops / effective_flops

        kv_bytes = total_tokens * model.kv_bytes_per_token
        memory_time = kv_bytes / gpu.memory_bandwidth

        compute_sym = sym_const(compute_time, "t_compute_prefill")
        memory_sym = sym_const(memory_time, "t_memory_prefill")

        return sym_max(compute_sym, memory_sym)

    @property
    def queue_length(self) -> int:
        return len(self.waiting_queue) + len(self.running_batch)

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.core.request import Request, DecodeTask, KVHandle
from sim.schedulers.batch_scheduler import ContinuousBatchScheduler
from sim.symbolic.expr import SymVal, sym_const, sym_add, sym_max

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.config.scheduler import SchedulerConfig
    from sim.parallel.groups import TPGroup, EPGroup


@dataclass
class DecodeIterationResult:
    iteration_time: SymVal
    token_emitted_tasks: list[DecodeTask]
    tokens_generated: int


class DecodeWorker:
    def __init__(
        self,
        worker_id: int,
        tp_group: TPGroup | None = None,
        ep_group: EPGroup | None = None,
    ):
        self.worker_id = worker_id
        self.tp_group = tp_group
        self.ep_group = ep_group

        self.waiting_queue: deque[DecodeTask] = deque()
        self.running_batch: list[DecodeTask] = []
        self.scheduler = ContinuousBatchScheduler()
        self.is_iteration_scheduled: bool = False
        self.total_tokens_generated: int = 0
        self.total_iterations: int = 0

    def enqueue_request(
        self,
        request: Request,
        kv_handle: KVHandle,
        config: SchedulerConfig,
    ) -> DecodeTask:
        task = DecodeTask(
            request=request,
            kv_handle=kv_handle,
            remaining_tokens=request.max_new_tokens,
            current_kv_len=request.prompt_len,
        )
        self.waiting_queue.append(task)
        return task

    def has_pending_work(self) -> bool:
        return len(self.waiting_queue) > 0 or len(self.running_batch) > 0

    def run_iteration(
        self,
        current_time: float,
        engine: SimulationEngine,
    ) -> DecodeIterationResult:
        config = engine.scheduler_config

        selection = self.scheduler.select_decode_batch(
            waiting_tasks=list(self.waiting_queue),
            running_tasks=self.running_batch,
            config=config,
        )

        for task in selection.selected_tasks:
            self.waiting_queue.remove(task)
            self.running_batch.append(task)
            task.iteration_start = current_time

        all_active = self.running_batch.copy()
        batch_size = len(all_active)

        iteration_time = self._compute_iteration_time(
            all_active, engine
        )

        completed_tasks: list[DecodeTask] = []
        for task in all_active:
            task.emit_token(engine.model_config.kv_bytes_per_token)
            task.request.tokens_generated += 1

            if task.is_complete:
                completed_tasks.append(task)
                self.running_batch.remove(task)

        self.total_tokens_generated += batch_size
        self.total_iterations += 1

        return DecodeIterationResult(
            iteration_time=iteration_time,
            token_emitted_tasks=all_active,
            tokens_generated=batch_size,
        )

    def _compute_iteration_time(
        self,
        tasks: list[DecodeTask],
        engine: SimulationEngine,
    ) -> SymVal:
        if not tasks:
            return sym_const(0.0)

        model = engine.model_config
        gpu = engine.cluster_config.gpu_spec
        batch_size = len(tasks)

        total_kv_len = sum(t.current_kv_len for t in tasks)
        avg_kv_len = total_kv_len / batch_size if batch_size > 0 else 0

        kv_bytes_to_read = total_kv_len * model.kv_bytes_per_token
        memory_time = kv_bytes_to_read / gpu.memory_bandwidth

        flops_per_token = (
            2 * model.num_layers * (
                4 * model.hidden_dim * model.hidden_dim +
                2 * model.hidden_dim * model.actual_intermediate_dim
            )
        )
        total_flops = flops_per_token * batch_size

        attention_flops = 2 * batch_size * avg_kv_len * model.hidden_dim * model.num_layers
        total_flops += attention_flops

        effective_flops = gpu.flops_fp16
        if self.tp_group is not None:
            effective_flops *= self.tp_group.size

        compute_time = total_flops / effective_flops

        memory_sym = sym_const(memory_time, "t_memory_decode")
        compute_sym = sym_const(compute_time, "t_compute_decode")

        return sym_max(memory_sym, compute_sym)

    @property
    def queue_length(self) -> int:
        return len(self.waiting_queue) + len(self.running_batch)

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.network.interconnect import InterconnectModel
from sim.network.kv_transfer import KVTransferManager

__all__ = ["InterconnectModel", "KVTransferManager"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add
from sim.config.cluster import InterconnectConfig, InterconnectType

if TYPE_CHECKING:
    pass


class InterconnectModel:
    def __init__(self, config: InterconnectConfig):
        self.config = config

    def transfer_time(self, bytes_to_transfer: int) -> SymVal:
        latency = self.config.latency_seconds
        bandwidth = self.config.bandwidth_bytes_per_sec

        transfer_time = latency + bytes_to_transfer / bandwidth

        latency_sym = sym_const(latency, "L_net")
        transfer_sym = sym_const(bytes_to_transfer / bandwidth, "t_transfer")

        return sym_add(latency_sym, transfer_sym)

    def collective_time(
        self,
        message_bytes: int,
        num_participants: int,
        op: str,
    ) -> SymVal:
        bandwidth = self.config.bandwidth_bytes_per_sec
        latency = self.config.latency_seconds

        if op == "all_reduce":
            effective_bytes = 2 * message_bytes * (num_participants - 1) / num_participants
        elif op == "all_gather":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        elif op == "reduce_scatter":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        elif op == "all_to_all":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        else:
            effective_bytes = message_bytes

        alpha_beta_time = latency * num_participants + effective_bytes / bandwidth

        return sym_const(alpha_beta_time, f"t_{op}")

    @classmethod
    def for_intra_node(cls, config: InterconnectConfig) -> InterconnectModel:
        return cls(config)

    @classmethod
    def for_inter_node(cls, config: InterconnectConfig) -> InterconnectModel:
        return cls(config)

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add
from sim.network.interconnect import InterconnectModel

if TYPE_CHECKING:
    from sim.core.request import KVHandle
    from sim.config.cluster import ClusterConfig


@dataclass
class KVTransfer:
    request_id: int
    kv_bytes: int
    source_worker: int
    dest_worker: int
    start_time: float
    estimated_end_time: float


class KVTransferManager:
    def __init__(
        self,
        intra_node_interconnect: InterconnectModel,
        inter_node_interconnect: InterconnectModel,
        gpus_per_node: int = 8,
    ):
        self.intra_node_interconnect = intra_node_interconnect
        self.inter_node_interconnect = inter_node_interconnect
        self.gpus_per_node = gpus_per_node

        self.pending_transfers: dict[int, KVTransfer] = {}
        self.completed_transfers: dict[int, KVTransfer] = {}

    def initiate_transfer(
        self,
        kv_handle: KVHandle,
        cluster_config: ClusterConfig,
    ) -> SymVal:
        source_node = kv_handle.source_worker_id // self.gpus_per_node
        dest_node = kv_handle.dest_worker_id // self.gpus_per_node

        is_intra_node = source_node == dest_node

        serialization_overhead = 0.0001  # 100 microseconds
        deserialization_overhead = 0.0001

        if is_intra_node:
            transfer_time = self.intra_node_interconnect.transfer_time(kv_handle.kv_bytes)
        else:
            transfer_time = self.inter_node_interconnect.transfer_time(kv_handle.kv_bytes)

        overhead_sym = sym_const(serialization_overhead + deserialization_overhead, "t_serde")
        total_time = sym_add(transfer_time, overhead_sym)

        return total_time

    def is_same_node(self, worker_a: int, worker_b: int) -> bool:
        return worker_a // self.gpus_per_node == worker_b // self.gpus_per_node

    @classmethod
    def from_cluster_config(cls, cluster_config: ClusterConfig) -> KVTransferManager:
        intra = InterconnectModel(cluster_config.intra_node_interconnect)
        inter = InterconnectModel(cluster_config.inter_node_interconnect)
        return cls(
            intra_node_interconnect=intra,
            inter_node_interconnect=inter,
            gpus_per_node=cluster_config.gpus_per_worker,
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.cache.radix_cache import RadixCache, RadixNode

__all__ = ["RadixCache", "RadixNode"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from sim.config.cache import CacheConfig


@dataclass
class RadixNode:
    token: int | None = None
    kv_bytes: int = 0
    last_access: float = 0.0
    children: dict[int, RadixNode] = field(default_factory=dict)
    parent: RadixNode | None = None
    depth: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_descendant_bytes(self) -> int:
        total = self.kv_bytes
        for child in self.children.values():
            total += child.total_descendant_bytes()
        return total


class RadixCache:
    def __init__(self, config: CacheConfig, current_time: float = 0.0):
        self.config = config
        self.root = RadixNode(depth=0)
        self.capacity = config.capacity_bytes
        self.used_bytes = 0
        self.current_time = current_time

        self.access_order: OrderedDict[int, RadixNode] = OrderedDict()
        self._node_id_counter = 0

    def match_prefix(self, tokens: list[int]) -> tuple[int, int]:
        matched_tokens = 0
        matched_bytes = 0
        current = self.root

        for token in tokens:
            if token in current.children:
                current = current.children[token]
                matched_tokens += 1
                matched_bytes += current.kv_bytes
                self._update_access(current)
            else:
                break

        return matched_tokens, matched_bytes

    def insert(self, tokens: list[int], total_kv_bytes: int) -> bool:
        if not tokens:
            return True

        bytes_per_token = total_kv_bytes // len(tokens) if tokens else 0
        current = self.root

        for i, token in enumerate(tokens):
            if token in current.children:
                current = current.children[token]
                self._update_access(current)
            else:
                new_node = RadixNode(
                    token=token,
                    kv_bytes=bytes_per_token,
                    last_access=self.current_time,
                    parent=current,
                    depth=current.depth + 1,
                )

                while self.used_bytes + bytes_per_token > self.capacity:
                    if not self._evict_lru():
                        return False

                current.children[token] = new_node
                self.used_bytes += bytes_per_token
                self._register_node(new_node)
                current = new_node

        return True

    def _update_access(self, node: RadixNode) -> None:
        node.last_access = self.current_time
        node_id = id(node)
        if node_id in self.access_order:
            self.access_order.move_to_end(node_id)

    def _register_node(self, node: RadixNode) -> None:
        self.access_order[id(node)] = node

    def _evict_lru(self) -> bool:
        if not self.access_order:
            return False

        candidates = [
            node for node in self.access_order.values()
            if node.is_leaf() and node.parent is not None
        ]

        if not candidates:
            return False

        oldest_node = min(candidates, key=lambda n: n.last_access)
        return self._remove_node(oldest_node)

    def _remove_node(self, node: RadixNode) -> bool:
        if node.parent is None:
            return False

        if not node.is_leaf():
            return False

        parent = node.parent
        if node.token is not None and node.token in parent.children:
            del parent.children[node.token]

        self.used_bytes -= node.kv_bytes
        node_id = id(node)
        if node_id in self.access_order:
            del self.access_order[node_id]

        return True

    def update_time(self, new_time: float) -> None:
        self.current_time = new_time

    def get_hit_rate_estimate(self) -> float:
        if self.capacity == 0:
            return 0.0
        return min(1.0, self.used_bytes / self.capacity)

    @property
    def utilization(self) -> float:
        return self.used_bytes / self.capacity if self.capacity > 0 else 0.0

    def get_all_prefixes(self) -> list[tuple[list[int], int]]:
        prefixes: list[tuple[list[int], int]] = []
        self._collect_prefixes(self.root, [], prefixes)
        return prefixes

    def _collect_prefixes(
        self,
        node: RadixNode,
        current_path: list[int],
        prefixes: list[tuple[list[int], int]],
    ) -> None:
        if node.is_leaf() and current_path:
            total_bytes = sum(
                self._get_path_bytes(current_path[:i+1])
                for i in range(len(current_path))
            )
            prefixes.append((current_path.copy(), node.kv_bytes))

        for token, child in node.children.items():
            current_path.append(token)
            self._collect_prefixes(child, current_path, prefixes)
            current_path.pop()

    def _get_path_bytes(self, path: list[int]) -> int:
        current = self.root
        for token in path:
            if token in current.children:
                current = current.children[token]
            else:
                return 0
        return current.kv_bytes

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.routing.router import Router, RoutingPolicy

__all__ = ["Router", "RoutingPolicy"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, KVHandle
    from sim.workers.prefill_worker import PrefillWorker
    from sim.workers.decode_worker import DecodeWorker
    from sim.cache.radix_cache import RadixCache


class RoutingPolicy(Enum):
    ROUND_ROBIN = "round_robin"
    SHORTEST_QUEUE = "shortest_queue"
    CACHE_AWARE = "cache_aware"


@dataclass
class ApproxRadixTree:
    worker_id: int
    known_prefixes: list[tuple[list[int], int]] = field(default_factory=list)
    last_updated: float = 0.0

    def estimate_prefix_match(self, tokens: list[int]) -> tuple[int, int]:
        best_match_len = 0
        best_match_bytes = 0

        for prefix, kv_bytes in self.known_prefixes:
            match_len = 0
            for i, (a, b) in enumerate(zip(tokens, prefix)):
                if a == b:
                    match_len = i + 1
                else:
                    break

            if match_len > best_match_len:
                best_match_len = match_len
                # Estimate bytes proportionally
                if len(prefix) > 0:
                    best_match_bytes = int(kv_bytes * match_len / len(prefix))

        return best_match_len, best_match_bytes


class Router:
    def __init__(
        self,
        prefill_workers: list[PrefillWorker],
        decode_workers: list[DecodeWorker],
        prefill_policy: RoutingPolicy = RoutingPolicy.CACHE_AWARE,
        decode_policy: RoutingPolicy = RoutingPolicy.ROUND_ROBIN,
        cache_threshold: float = 0.5,
        balance_threshold: float = 2.0,
    ):
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.prefill_policy = prefill_policy
        self.decode_policy = decode_policy
        self.cache_threshold = cache_threshold
        self.balance_threshold = balance_threshold

        self.approx_trees: dict[int, ApproxRadixTree] = {
            w.worker_id: ApproxRadixTree(worker_id=w.worker_id)
            for w in prefill_workers
        }

        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

    def route_to_prefill(self, request: Request) -> int:
        if self.prefill_policy == RoutingPolicy.ROUND_ROBIN:
            return self._round_robin_prefill()
        elif self.prefill_policy == RoutingPolicy.SHORTEST_QUEUE:
            return self._shortest_queue_prefill()
        elif self.prefill_policy == RoutingPolicy.CACHE_AWARE:
            return self._cache_aware_prefill(request)
        return self._round_robin_prefill()

    def route_to_decode(self, kv_handle: KVHandle) -> int:
        if self.decode_policy == RoutingPolicy.ROUND_ROBIN:
            return self._round_robin_decode()
        elif self.decode_policy == RoutingPolicy.SHORTEST_QUEUE:
            return self._shortest_queue_decode()
        return self._round_robin_decode()

    def _round_robin_prefill(self) -> int:
        worker_id = self.prefill_rr_counter % len(self.prefill_workers)
        self.prefill_rr_counter += 1
        return worker_id

    def _round_robin_decode(self) -> int:
        worker_id = self.decode_rr_counter % len(self.decode_workers)
        self.decode_rr_counter += 1
        return worker_id

    def _shortest_queue_prefill(self) -> int:
        return min(
            range(len(self.prefill_workers)),
            key=lambda i: self.prefill_workers[i].queue_length,
        )

    def _shortest_queue_decode(self) -> int:
        return min(
            range(len(self.decode_workers)),
            key=lambda i: self.decode_workers[i].queue_length,
        )

    def _cache_aware_prefill(self, request: Request) -> int:
        queue_lengths = [w.queue_length for w in self.prefill_workers]
        min_queue = min(queue_lengths) if queue_lengths else 0
        max_queue = max(queue_lengths) if queue_lengths else 0

        if max_queue > 0 and max_queue / max(1, min_queue) > self.balance_threshold:
            return self._shortest_queue_prefill()

        best_worker = 0
        best_match_ratio = 0.0

        for worker_id, worker in enumerate(self.prefill_workers):
            if worker.cache is not None:
                matched_tokens, _ = worker.cache.match_prefix(request.prompt_tokens)
                match_ratio = matched_tokens / len(request.prompt_tokens) if request.prompt_tokens else 0
            else:
                approx_tree = self.approx_trees.get(worker_id)
                if approx_tree:
                    matched_tokens, _ = approx_tree.estimate_prefix_match(request.prompt_tokens)
                    match_ratio = matched_tokens / len(request.prompt_tokens) if request.prompt_tokens else 0
                else:
                    match_ratio = 0.0

            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_worker = worker_id

        if best_match_ratio >= self.cache_threshold:
            return best_worker

        return self._worker_with_most_capacity()

    def _worker_with_most_capacity(self) -> int:
        best_worker = 0
        best_capacity = 0

        for worker_id, worker in enumerate(self.prefill_workers):
            if worker.cache is not None:
                remaining = worker.cache.capacity - worker.cache.used_bytes
            else:
                remaining = float('inf')

            if remaining > best_capacity:
                best_capacity = remaining
                best_worker = worker_id

        return best_worker

    def update_approx_tree(self, worker_id: int, prefixes: list[tuple[list[int], int]], time: float) -> None:
        if worker_id in self.approx_trees:
            self.approx_trees[worker_id].known_prefixes = prefixes
            self.approx_trees[worker_id].last_updated = time

    def is_balanced(self) -> bool:
        queue_lengths = [w.queue_length for w in self.prefill_workers]
        if not queue_lengths or max(queue_lengths) == 0:
            return True
        return max(queue_lengths) / max(1, min(queue_lengths)) <= self.balance_threshold

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.parallel.groups import TPGroup, DPGroup, EPGroup
from sim.parallel.collectives import CollectiveOp, collective_cost

__all__ = ["TPGroup", "DPGroup", "EPGroup", "CollectiveOp", "collective_cost"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add

if TYPE_CHECKING:
    from sim.config.cluster import InterconnectConfig


class CollectiveOp(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


def collective_cost(
    op: CollectiveOp,
    message_bytes: int,
    num_participants: int,
    bandwidth_bytes_per_sec: float,
    latency_seconds: float,
) -> SymVal:
    if num_participants <= 1:
        return sym_const(0.0, f"t_{op.value}_trivial")

    if op == CollectiveOp.ALL_REDUCE:
        effective_bytes = 2 * message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.ALL_GATHER:
        effective_bytes = message_bytes * (num_participants - 1)
    elif op == CollectiveOp.REDUCE_SCATTER:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.ALL_TO_ALL:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.BROADCAST:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    else:
        effective_bytes = message_bytes

    alpha = latency_seconds
    beta_inv = bandwidth_bytes_per_sec

    log_p = 1
    p = num_participants
    while p > 1:
        p //= 2
        log_p += 1

    total_time = log_p * alpha + effective_bytes / beta_inv

    return sym_const(total_time, f"t_{op.value}")


def ring_all_reduce_cost(
    message_bytes: int,
    num_participants: int,
    bandwidth_bytes_per_sec: float,
    latency_seconds: float,
) -> SymVal:
    if num_participants <= 1:
        return sym_const(0.0, "t_ring_ar_trivial")

    n = num_participants
    effective_bytes = 2 * message_bytes * (n - 1) / n
    total_time = 2 * (n - 1) * latency_seconds + effective_bytes / bandwidth_bytes_per_sec

    return sym_const(total_time, "t_ring_all_reduce")

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.kernels.attention import AttentionKernel
from sim.kernels.mlp import MLPKernel
from sim.kernels.moe import MoEKernel

__all__ = ["AttentionKernel", "MLPKernel", "MoEKernel"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.models.operator_graph import OperatorGraph

__all__ = ["OperatorGraph"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.metrics.definitions import MetricCollector, RequestMetrics
from sim.metrics.symbolic import SymbolicThroughputBuilder

__all__ = ["MetricCollector", "RequestMetrics", "SymbolicThroughputBuilder"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import statistics

if TYPE_CHECKING:
    from sim.core.request import Request


@dataclass
class RequestMetrics:
    request_id: int
    ttft: float
    tpot: float
    e2e_latency: float
    queue_time: float
    prefill_time: float
    transfer_time: float
    decode_time: float
    prompt_len: int
    output_len: int


class MetricCollector:
    def __init__(self):
        self.request_metrics: list[RequestMetrics] = []
        self.total_requests: int = 0
        self.completed_requests: int = 0
        self.total_tokens: int = 0

        self._ttfts: list[float] = []
        self._tpots: list[float] = []
        self._e2e_latencies: list[float] = []
        self._queue_times: list[float] = []

    def record_request_complete(self, request: Request) -> None:
        self.completed_requests += 1
        self.total_tokens += request.tokens_generated

        if request.ttft is not None:
            self._ttfts.append(request.ttft)

        if request.tpot is not None:
            self._tpots.append(request.tpot)

        if request.e2e_latency is not None:
            self._e2e_latencies.append(request.e2e_latency)

        metrics = RequestMetrics(
            request_id=request.id,
            ttft=request.ttft or 0.0,
            tpot=request.tpot or 0.0,
            e2e_latency=request.e2e_latency or 0.0,
            queue_time=0.0,
            prefill_time=0.0,
            transfer_time=0.0,
            decode_time=0.0,
            prompt_len=request.prompt_len,
            output_len=request.tokens_generated,
        )
        self.request_metrics.append(metrics)

    def record_request_arrival(self, request: Request) -> None:
        self.total_requests += 1

    def throughput(self, elapsed_time: float) -> float:
        if elapsed_time <= 0:
            return 0.0
        return self.total_tokens / elapsed_time

    def avg_ttft(self) -> float:
        if not self._ttfts:
            return 0.0
        return statistics.mean(self._ttfts)

    def avg_tpot(self) -> float:
        if not self._tpots:
            return 0.0
        return statistics.mean(self._tpots)

    def avg_e2e_latency(self) -> float:
        if not self._e2e_latencies:
            return 0.0
        return statistics.mean(self._e2e_latencies)

    def percentile_ttft(self, p: float) -> float:
        if not self._ttfts:
            return 0.0
        return self._percentile(self._ttfts, p)

    def percentile_tpot(self, p: float) -> float:
        if not self._tpots:
            return 0.0
        return self._percentile(self._tpots, p)

    def percentile_e2e(self, p: float) -> float:
        if not self._e2e_latencies:
            return 0.0
        return self._percentile(self._e2e_latencies, p)

    def _percentile(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        if f == c:
            return sorted_data[f]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "total_tokens": self.total_tokens,
            "avg_ttft": self.avg_ttft(),
            "avg_tpot": self.avg_tpot(),
            "avg_e2e_latency": self.avg_e2e_latency(),
            "p50_ttft": self.percentile_ttft(0.5),
            "p99_ttft": self.percentile_ttft(0.99),
            "p50_tpot": self.percentile_tpot(0.5),
            "p99_tpot": self.percentile_tpot(0.99),
        }

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.optimizer.interface import OptimizerInterface, OptimizationResult

__all__ = ["OptimizerInterface", "OptimizationResult"]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from itertools import product

import sympy
from sympy import Symbol
import numpy as np

from sim.symbolic.symbols import ConfigSymbols


@dataclass
class OptimizationResult:
    optimal_values: dict[str, float]
    optimal_objective: float
    success: bool
    message: str
    all_evaluations: list[tuple[dict, float]] = field(default_factory=list)


@dataclass
class Constraint:
    expr: sympy.Expr
    type: str  # "eq" for equality, "ineq" for inequality (>= 0)
    name: str = ""


class OptimizerInterface:
    def __init__(
        self,
        objective: sympy.Expr,
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
        maximize: bool = True,
    ):
        self.objective = objective
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self.maximize = maximize

        self._decision_vars: list[Symbol] = []
        self._frozen_params: dict[Symbol, float] = {}

    def set_decision_vars(self, vars: list[Symbol]) -> None:
        self._decision_vars = vars

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def get_decision_vars(self) -> list[Symbol]:
        if self._decision_vars:
            return self._decision_vars
        return self.symbols.decision_vars()

    def get_frozen_params(self) -> list[Symbol]:
        return self.symbols.frozen_params()

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def to_scipy_minimize(
        self,
        bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        decision_vars = self.get_decision_vars()

        substituted_obj = self.substitute_frozen(self.objective)

        if self.maximize:
            obj_to_minimize = -substituted_obj
        else:
            obj_to_minimize = substituted_obj

        obj_func = sympy.lambdify(decision_vars, obj_to_minimize, modules=["numpy"])

        def objective_wrapper(x: np.ndarray) -> float:
            return float(obj_func(*x))

        scipy_constraints = []
        for constraint in self.constraints:
            substituted = self.substitute_frozen(constraint.expr)
            constraint_func = sympy.lambdify(decision_vars, substituted, modules=["numpy"])

            scipy_constraints.append({
                "type": constraint.type,
                "fun": lambda x, f=constraint_func: float(f(*x)),
            })

        scipy_bounds = None
        if bounds:
            scipy_bounds = [
                bounds.get(var, (None, None)) for var in decision_vars
            ]

        return {
            "fun": objective_wrapper,
            "x0": np.ones(len(decision_vars)),
            "bounds": scipy_bounds,
            "constraints": scipy_constraints,
            "method": "SLSQP",
        }

    def enumerate_discrete_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> list[dict[Symbol, int]]:
        regimes = []
        var_names = list(discrete_vars.keys())
        var_values = list(discrete_vars.values())

        for combo in product(*var_values):
            regime = dict(zip(var_names, combo))
            regimes.append(regime)

        return regimes

    def optimize_over_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> OptimizationResult:
        from scipy.optimize import minimize

        regimes = self.enumerate_discrete_regimes(discrete_vars, continuous_bounds)
        best_result: OptimizationResult | None = None
        all_evaluations: list[tuple[dict, float]] = []

        decision_vars = self.get_decision_vars()
        continuous_vars = [v for v in decision_vars if v not in discrete_vars]

        for regime in regimes:
            regime_objective = self.objective.subs(list(regime.items()))
            regime_objective = self.substitute_frozen(regime_objective)

            if not continuous_vars:
                if self.maximize:
                    obj_val = -float(regime_objective)
                else:
                    obj_val = float(regime_objective)

                result_dict = {str(k): v for k, v in regime.items()}
                all_evaluations.append((result_dict, -obj_val if self.maximize else obj_val))

                if best_result is None or (self.maximize and -obj_val > best_result.optimal_objective) or \
                   (not self.maximize and obj_val < best_result.optimal_objective):
                    best_result = OptimizationResult(
                        optimal_values=result_dict,
                        optimal_objective=-obj_val if self.maximize else obj_val,
                        success=True,
                        message=f"Regime: {regime}",
                    )
            else:
                obj_func = sympy.lambdify(continuous_vars, regime_objective, modules=["numpy"])

                if self.maximize:
                    def objective_wrapper(x):
                        return -float(obj_func(*x))
                else:
                    def objective_wrapper(x):
                        return float(obj_func(*x))

                bounds_list = [
                    continuous_bounds.get(v, (1, 100)) if continuous_bounds else (1, 100)
                    for v in continuous_vars
                ]

                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_list])

                try:
                    result = minimize(
                        objective_wrapper,
                        x0,
                        bounds=bounds_list,
                        method="L-BFGS-B",
                    )

                    result_dict = {str(k): v for k, v in regime.items()}
                    for var, val in zip(continuous_vars, result.x):
                        result_dict[str(var)] = float(val)

                    obj_val = -result.fun if self.maximize else result.fun
                    all_evaluations.append((result_dict, obj_val))

                    if best_result is None or \
                       (self.maximize and obj_val > best_result.optimal_objective) or \
                       (not self.maximize and obj_val < best_result.optimal_objective):
                        best_result = OptimizationResult(
                            optimal_values=result_dict,
                            optimal_objective=obj_val,
                            success=result.success,
                            message=result.message,
                        )
                except Exception as e:
                    continue

        if best_result is None:
            return OptimizationResult(
                optimal_values={},
                optimal_objective=float("inf") if not self.maximize else float("-inf"),
                success=False,
                message="No valid regime found",
                all_evaluations=all_evaluations,
            )

        best_result.all_evaluations = all_evaluations
        return best_result

    def grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())

        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []

        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])

        for combo in product(*var_values):
            try:
                obj_val = float(obj_func(*combo))
                if self.maximize:
                    obj_val = obj_val
                else:
                    obj_val = obj_val

                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))

                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue

        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Grid search complete",
            all_evaluations=all_evaluations,
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution
from sim.workload.trace_loader import TraceWorkloadLoader
from sim.workload.prefix_groups import PrefixGroup

__all__ = [
    "SyntheticWorkloadGenerator",
    "Distribution",
    "TraceWorkloadLoader",
    "PrefixGroup",
]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
import random


@dataclass
class PrefixGroup:
    prefix_tokens: list[int]
    share_ratio: float
    group_id: int = 0

    @property
    def prefix_length(self) -> int:
        return len(self.prefix_tokens)

    def sample_tokens(self, additional_length: int, vocab_size: int = 32000) -> list[int]:
        suffix = [random.randint(0, vocab_size - 1) for _ in range(additional_length)]
        return self.prefix_tokens + suffix


@dataclass
class PrefixGroupManager:
    groups: list[PrefixGroup] = field(default_factory=list)
    vocab_size: int = 32000

    def add_group(
        self,
        prefix_length: int,
        share_ratio: float,
        group_id: int | None = None,
    ) -> PrefixGroup:
        if group_id is None:
            group_id = len(self.groups)

        prefix_tokens = [random.randint(0, self.vocab_size - 1) for _ in range(prefix_length)]
        group = PrefixGroup(
            prefix_tokens=prefix_tokens,
            share_ratio=share_ratio,
            group_id=group_id,
        )
        self.groups.append(group)
        return group

    def sample_group(self) -> PrefixGroup | None:
        if not self.groups:
            return None

        total_ratio = sum(g.share_ratio for g in self.groups)
        if total_ratio == 0:
            return None

        r = random.random() * total_ratio
        cumulative = 0.0
        for group in self.groups:
            cumulative += group.share_ratio
            if r <= cumulative:
                return group

        return self.groups[-1] if self.groups else None

    def generate_prompt(self, total_length: int) -> tuple[list[int], int | None]:
        group = self.sample_group()
        if group is None:
            tokens = [random.randint(0, self.vocab_size - 1) for _ in range(total_length)]
            return tokens, None

        if group.prefix_length >= total_length:
            return group.prefix_tokens[:total_length], group.group_id

        additional = total_length - group.prefix_length
        tokens = group.sample_tokens(additional, self.vocab_size)
        return tokens, group.group_id

    @classmethod
    def create_uniform_groups(
        cls,
        num_groups: int,
        prefix_length: int,
        vocab_size: int = 32000,
    ) -> PrefixGroupManager:
        manager = cls(vocab_size=vocab_size)
        share_per_group = 1.0 / num_groups
        for i in range(num_groups):
            manager.add_group(prefix_length, share_per_group, group_id=i)
        return manager

    @classmethod
    def create_zipf_groups(
        cls,
        num_groups: int,
        prefix_length: int,
        zipf_param: float = 1.0,
        vocab_size: int = 32000,
    ) -> PrefixGroupManager:
        manager = cls(vocab_size=vocab_size)

        weights = [1.0 / (i + 1) ** zipf_param for i in range(num_groups)]
        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]

        for i, ratio in enumerate(normalized):
            manager.add_group(prefix_length, ratio, group_id=i)

        return manager

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from typing import TYPE_CHECKING

from sim.core.request import Request
from sim.workload.prefix_groups import PrefixGroupManager

if TYPE_CHECKING:
    pass


class DistributionType(Enum):
    CONSTANT = "constant"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    ZIPF = "zipf"


@dataclass
class Distribution:
    type: DistributionType
    params: dict = field(default_factory=dict)

    def sample(self) -> float:
        if self.type == DistributionType.CONSTANT:
            return self.params.get("value", 0.0)

        elif self.type == DistributionType.UNIFORM:
            low = self.params.get("low", 0.0)
            high = self.params.get("high", 1.0)
            return random.uniform(low, high)

        elif self.type == DistributionType.EXPONENTIAL:
            rate = self.params.get("rate", 1.0)
            return random.expovariate(rate)

        elif self.type == DistributionType.NORMAL:
            mean = self.params.get("mean", 0.0)
            std = self.params.get("std", 1.0)
            return random.gauss(mean, std)

        elif self.type == DistributionType.LOGNORMAL:
            mean = self.params.get("mean", 0.0)
            sigma = self.params.get("sigma", 1.0)
            return random.lognormvariate(mean, sigma)

        elif self.type == DistributionType.ZIPF:
            alpha = self.params.get("alpha", 1.5)
            n = self.params.get("n", 1000)
            return self._zipf_sample(alpha, n)

        return 0.0

    def sample_int(self, min_val: int = 1, max_val: int | None = None) -> int:
        value = int(self.sample())
        value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
        return value

    def _zipf_sample(self, alpha: float, n: int) -> float:
        harmonic = sum(1.0 / (i ** alpha) for i in range(1, n + 1))
        u = random.random()
        cumulative = 0.0
        for i in range(1, n + 1):
            cumulative += 1.0 / (i ** alpha) / harmonic
            if u <= cumulative:
                return float(i)
        return float(n)

    @classmethod
    def constant(cls, value: float) -> Distribution:
        return cls(DistributionType.CONSTANT, {"value": value})

    @classmethod
    def uniform(cls, low: float, high: float) -> Distribution:
        return cls(DistributionType.UNIFORM, {"low": low, "high": high})

    @classmethod
    def exponential(cls, rate: float) -> Distribution:
        return cls(DistributionType.EXPONENTIAL, {"rate": rate})

    @classmethod
    def normal(cls, mean: float, std: float) -> Distribution:
        return cls(DistributionType.NORMAL, {"mean": mean, "std": std})

    @classmethod
    def lognormal(cls, mean: float, sigma: float) -> Distribution:
        return cls(DistributionType.LOGNORMAL, {"mean": mean, "sigma": sigma})


class SyntheticWorkloadGenerator:
    def __init__(
        self,
        arrival_rate: float,
        prompt_len_dist: Distribution,
        output_len_dist: Distribution,
        prefix_groups: PrefixGroupManager | None = None,
        vocab_size: int = 32000,
        seed: int | None = None,
    ):
        self.arrival_rate = arrival_rate
        self.prompt_len_dist = prompt_len_dist
        self.output_len_dist = output_len_dist
        self.prefix_groups = prefix_groups
        self.vocab_size = vocab_size

        if seed is not None:
            random.seed(seed)

        self._request_counter = 0

    def generate(self, duration: float) -> list[Request]:
        requests: list[Request] = []
        current_time = 0.0

        while current_time < duration:
            inter_arrival = random.expovariate(self.arrival_rate)
            current_time += inter_arrival

            if current_time >= duration:
                break

            prompt_len = self.prompt_len_dist.sample_int(min_val=1, max_val=32768)
            output_len = self.output_len_dist.sample_int(min_val=1, max_val=4096)

            if self.prefix_groups is not None:
                prompt_tokens, prefix_group_id = self.prefix_groups.generate_prompt(prompt_len)
            else:
                prompt_tokens = [
                    random.randint(0, self.vocab_size - 1) for _ in range(prompt_len)
                ]
                prefix_group_id = None

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
                prefix_group_id=prefix_group_id,
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    def generate_batch(self, num_requests: int, start_time: float = 0.0) -> list[Request]:
        requests: list[Request] = []
        current_time = start_time

        for _ in range(num_requests):
            inter_arrival = random.expovariate(self.arrival_rate)
            current_time += inter_arrival

            prompt_len = self.prompt_len_dist.sample_int(min_val=1, max_val=32768)
            output_len = self.output_len_dist.sample_int(min_val=1, max_val=4096)

            if self.prefix_groups is not None:
                prompt_tokens, prefix_group_id = self.prefix_groups.generate_prompt(prompt_len)
            else:
                prompt_tokens = [
                    random.randint(0, self.vocab_size - 1) for _ in range(prompt_len)
                ]
                prefix_group_id = None

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
                prefix_group_id=prefix_group_id,
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    @classmethod
    def default(cls, arrival_rate: float = 10.0) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=arrival_rate,
            prompt_len_dist=Distribution.uniform(256, 2048),
            output_len_dist=Distribution.uniform(64, 512),
        )

    @classmethod
    def chat_workload(cls, arrival_rate: float = 5.0) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=arrival_rate,
            prompt_len_dist=Distribution.lognormal(mean=6.0, sigma=1.0),  # ~400 avg
            output_len_dist=Distribution.lognormal(mean=5.0, sigma=1.0),  # ~150 avg
        )

    @classmethod
    def batch_inference(cls, num_requests: int = 1000) -> SyntheticWorkloadGenerator:
        return cls(
            arrival_rate=float("inf"),
            prompt_len_dist=Distribution.uniform(512, 4096),
            output_len_dist=Distribution.uniform(128, 1024),
        )

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass
import json
import csv
from pathlib import Path
from typing import TYPE_CHECKING

from sim.core.request import Request

if TYPE_CHECKING:
    pass


@dataclass
class TraceFormat:
    VIDUR = "vidur"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class TraceWorkloadLoader:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._request_counter = 0

    def load(self, path: str, format: str = "auto") -> list[Request]:
        file_path = Path(path)

        if format == "auto":
            format = self._detect_format(file_path)

        if format == TraceFormat.VIDUR:
            return self._load_vidur(file_path)
        elif format == TraceFormat.SHAREGPT:
            return self._load_sharegpt(file_path)
        elif format == TraceFormat.CUSTOM:
            return self._load_custom(file_path)
        else:
            raise ValueError(f"Unknown trace format: {format}")

    def _detect_format(self, path: Path) -> str:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return TraceFormat.VIDUR
        elif suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if "conversations" in data[0]:
                        return TraceFormat.SHAREGPT
            return TraceFormat.CUSTOM
        return TraceFormat.CUSTOM

    def _load_vidur(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt_len = int(row.get("prompt_len", row.get("input_tokens", 512)))
                output_len = int(row.get("output_len", row.get("output_tokens", 128)))
                arrival_time = float(row.get("arrival_time", 0.0))

                prompt_tokens = self._generate_dummy_tokens(prompt_len)

                request = Request(
                    id=self._request_counter,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=output_len,
                    arrival_time=arrival_time,
                )
                requests.append(request)
                self._request_counter += 1

        return requests

    def _load_sharegpt(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            data = json.load(f)

        current_time = 0.0
        inter_arrival = 0.1

        for item in data:
            conversations = item.get("conversations", [])

            prompt_len = 0
            output_len = 0

            for conv in conversations:
                if conv.get("from") == "human":
                    prompt_len += len(conv.get("value", "").split())
                elif conv.get("from") == "gpt":
                    output_len += len(conv.get("value", "").split())

            prompt_len = max(1, int(prompt_len * 1.3))
            output_len = max(1, int(output_len * 1.3))

            prompt_tokens = self._generate_dummy_tokens(prompt_len)

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=output_len,
                arrival_time=current_time,
            )
            requests.append(request)
            self._request_counter += 1
            current_time += inter_arrival

        return requests

    def _load_custom(self, path: Path) -> list[Request]:
        requests = []

        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("requests", [])

        for item in data:
            prompt_tokens = item.get("prompt_tokens")
            if prompt_tokens is None:
                prompt_len = item.get("prompt_len", 512)
                prompt_tokens = self._generate_dummy_tokens(prompt_len)

            request = Request(
                id=self._request_counter,
                prompt_tokens=prompt_tokens,
                max_new_tokens=item.get("max_new_tokens", item.get("output_len", 128)),
                arrival_time=item.get("arrival_time", 0.0),
                stream=item.get("stream", False),
                prefix_group_id=item.get("prefix_group_id"),
            )
            requests.append(request)
            self._request_counter += 1

        return requests

    def _generate_dummy_tokens(self, length: int) -> list[int]:
        import random
        return [random.randint(0, self.vocab_size - 1) for _ in range(length)]

    def save_requests(self, requests: list[Request], path: str) -> None:
        data = {
            "requests": [
                {
                    "id": r.id,
                    "prompt_len": r.prompt_len,
                    "max_new_tokens": r.max_new_tokens,
                    "arrival_time": r.arrival_time,
                    "stream": r.stream,
                    "prefix_group_id": r.prefix_group_id,
                }
                for r in requests
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

```

---

### AI

**Tool Action:**

**Code Changes:**
```
#!/usr/bin/env python3
"""Basic simulation example showing the core workflow of the LLM inference simulator."""

import sys
sys.path.insert(0, "..")

from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig, GPUSpec, InterconnectConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.core.events import RequestArrival
from sim.workers.prefill_worker import PrefillWorker
from sim.workers.decode_worker import DecodeWorker
from sim.routing.router import Router, RoutingPolicy
from sim.network.kv_transfer import KVTransferManager
from sim.cache.radix_cache import RadixCache
from sim.metrics.definitions import MetricCollector
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution


def run_basic_simulation():
    model_config = ModelConfig.llama_7b()
    cluster_config = ClusterConfig.single_node_8gpu()
    scheduler_config = SchedulerConfig.default()
    cache_config = CacheConfig.h100_default()

    engine = SimulationEngine(
        model_config=model_config,
        cluster_config=cluster_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )

    prefill_workers = []
    for i in range(cluster_config.num_prefill_workers):
        cache = RadixCache(cache_config)
        worker = PrefillWorker(worker_id=i, cache=cache)
        prefill_workers.append(worker)

    decode_workers = []
    for i in range(cluster_config.num_decode_workers):
        worker = DecodeWorker(worker_id=i)
        decode_workers.append(worker)

    router = Router(
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        prefill_policy=RoutingPolicy.CACHE_AWARE,
        decode_policy=RoutingPolicy.ROUND_ROBIN,
    )

    kv_transfer = KVTransferManager.from_cluster_config(cluster_config)
    metrics = MetricCollector()

    engine.initialize_components(
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        router=router,
        kv_transfer_manager=kv_transfer,
        metrics=metrics,
    )

    workload_gen = SyntheticWorkloadGenerator(
        arrival_rate=5.0,
        prompt_len_dist=Distribution.uniform(256, 1024),
        output_len_dist=Distribution.uniform(64, 256),
        seed=42,
    )

    requests = workload_gen.generate(duration=10.0)
    print(f"Generated {len(requests)} requests")

    for request in requests:
        engine.schedule_event(RequestArrival(time=request.arrival_time, request=request))
        metrics.record_request_arrival(request)

    result = engine.run_until_idle()

    print("\n=== Simulation Results ===")
    print(f"Total requests: {result.total_requests}")
    print(f"Completed requests: {result.completed_requests}")
    print(f"Total tokens generated: {result.total_tokens_generated}")
    print(f"Simulation time: {result.simulation_time:.4f}s")
    print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"Avg TTFT: {result.avg_ttft * 1000:.2f}ms")
    print(f"Avg TPOT: {result.avg_tpot * 1000:.4f}ms")
    print(f"Avg E2E Latency: {result.avg_e2e_latency * 1000:.2f}ms")
    print(f"P50 TTFT: {result.p50_ttft * 1000:.2f}ms")
    print(f"P99 TTFT: {result.p99_ttft * 1000:.2f}ms")

    return result


if __name__ == "__main__":
    run_basic_simulation()

```

---

### AI

**Tool Action:**

**Code Changes:**
```
#!/usr/bin/env python3
"""Example demonstrating P/D ratio optimization using symbolic throughput expressions."""

import sys
sys.path.insert(0, "..")

import sympy
from sympy import Symbol, ceiling

from sim.symbolic.symbols import ConfigSymbols
from sim.symbolic.expr import SymVal, sym_const
from sim.metrics.symbolic import SymbolicThroughputBuilder
from sim.optimizer.interface import OptimizerInterface, Constraint
from sim.config.model import ModelConfig
from sim.config.cluster import GPUSpec


def build_throughput_model():
    symbols = ConfigSymbols()

    N_p = symbols.N_p
    N_d = symbols.N_d
    chunk_size = symbols.chunk_size
    batch_cap = symbols.batch_cap_tokens

    gpu_flops = symbols.gpu_flops
    gpu_mem_bw = symbols.gpu_mem_bw

    avg_prompt_len = symbols.avg_prompt_len
    avg_output_len = symbols.avg_output_len

    num_layers = symbols.num_layers
    hidden_dim = symbols.hidden_dim
    kv_bytes = symbols.kv_bytes_per_token

    prefill_flops_per_chunk = 4 * num_layers * hidden_dim * hidden_dim * chunk_size
    prefill_time_per_chunk = prefill_flops_per_chunk / gpu_flops

    num_prefill_chunks = ceiling(avg_prompt_len / chunk_size)
    total_prefill_time = num_prefill_chunks * prefill_time_per_chunk

    prefill_capacity = N_p / total_prefill_time

    avg_kv_len = avg_prompt_len + avg_output_len / 2
    kv_read_bytes = avg_kv_len * kv_bytes
    decode_time_per_token = kv_read_bytes / gpu_mem_bw

    decode_capacity = N_d * batch_cap / decode_time_per_token

    arrival_rate = Symbol("lambda", positive=True)

    input_rate = arrival_rate * avg_prompt_len
    output_rate = arrival_rate * avg_output_len

    prefill_util = input_rate / prefill_capacity
    decode_util = output_rate / decode_capacity

    max_throughput = sympy.Min(
        prefill_capacity * avg_output_len / avg_prompt_len,
        decode_capacity,
    )

    return max_throughput, symbols


def optimize_pd_ratio():
    throughput_expr, symbols = build_throughput_model()

    frozen_params = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.kv_bytes_per_token: 256,
        symbols.chunk_size: 8192,
        symbols.batch_cap_tokens: 256,
    }

    optimizer = OptimizerInterface(
        objective=throughput_expr,
        symbols=symbols,
        maximize=True,
    )
    optimizer.set_frozen_params(frozen_params)

    result = optimizer.grid_search(
        param_grid={
            symbols.N_p: [1, 2, 3, 4, 5, 6, 7, 8],
            symbols.N_d: [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    print("\n=== P/D Ratio Optimization Results ===")
    print(f"Optimal N_p: {result.optimal_values.get('N_p', 'N/A')}")
    print(f"Optimal N_d: {result.optimal_values.get('N_d', 'N/A')}")
    print(f"Optimal throughput: {result.optimal_objective:.2f} tokens/sec")

    print("\n=== Top 5 Configurations ===")
    sorted_evals = sorted(result.all_evaluations, key=lambda x: x[1], reverse=True)[:5]
    for i, (config, throughput) in enumerate(sorted_evals, 1):
        print(f"{i}. N_p={config['N_p']}, N_d={config['N_d']}: {throughput:.2f} tok/s")

    print("\n=== Symbolic Throughput Expression ===")
    print(f"TP = {throughput_expr}")

    return result


def analyze_scaling():
    throughput_expr, symbols = build_throughput_model()

    frozen_base = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.kv_bytes_per_token: 256,
        symbols.chunk_size: 8192,
        symbols.batch_cap_tokens: 256,
        symbols.N_p: 4,
        symbols.N_d: 4,
    }

    print("\n=== Throughput vs Prompt Length ===")
    for prompt_len in [128, 256, 512, 1024, 2048, 4096]:
        params = frozen_base.copy()
        params[symbols.avg_prompt_len] = prompt_len
        params[symbols.avg_output_len] = 128

        tp = float(throughput_expr.subs(list(params.items())))
        print(f"Prompt len {prompt_len:4d}: {tp:10.2f} tok/s")

    print("\n=== Throughput vs Output Length ===")
    for output_len in [32, 64, 128, 256, 512, 1024]:
        params = frozen_base.copy()
        params[symbols.avg_prompt_len] = 512
        params[symbols.avg_output_len] = output_len

        tp = float(throughput_expr.subs(list(params.items())))
        print(f"Output len {output_len:4d}: {tp:10.2f} tok/s")


if __name__ == "__main__":
    optimize_pd_ratio()
    analyze_scaling()

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
# Tests for SGLang LLM Inference Simulator

```

---

### AI

**Tool Action:**

**Code Changes:**
```
import pytest
import sympy
from sympy import Symbol

from sim.symbolic.expr import (
    SymVal,
    sym_add,
    sym_sub,
    sym_mul,
    sym_div,
    sym_max,
    sym_min,
    sym_const,
    sym_piecewise,
    sym_sum,
)
from sim.symbolic.symbols import ConfigSymbols


class TestSymVal:
    def test_creation(self):
        sv = SymVal(expr=sympy.Float(1.5), val=1.5)
        assert sv.val == 1.5
        assert sv.expr == sympy.Float(1.5)

    def test_add(self):
        a = sym_const(2.0)
        b = sym_const(3.0)
        result = sym_add(a, b)
        assert result.val == 5.0

    def test_sub(self):
        a = sym_const(5.0)
        b = sym_const(3.0)
        result = sym_sub(a, b)
        assert result.val == 2.0

    def test_mul(self):
        a = sym_const(2.0)
        b = sym_const(3.0)
        result = sym_mul(a, b)
        assert result.val == 6.0

    def test_div(self):
        a = sym_const(6.0)
        b = sym_const(2.0)
        result = sym_div(a, b)
        assert result.val == 3.0

    def test_max(self):
        a = sym_const(2.0)
        b = sym_const(5.0)
        result = sym_max(a, b)
        assert result.val == 5.0

    def test_min(self):
        a = sym_const(2.0)
        b = sym_const(5.0)
        result = sym_min(a, b)
        assert result.val == 2.0

    def test_symbolic_expression(self):
        x = Symbol("x")
        sv = SymVal(expr=x * 2, val=4.0, meta={"name": "test"})
        assert sv.expr == x * 2
        assert sv.val == 4.0

    def test_piecewise(self):
        x = Symbol("x")
        condition = x > 5
        sv_true = sym_const(10.0)
        sv_false = sym_const(1.0)

        result = sym_piecewise(
            [(condition, sv_true), (True, sv_false)],
            numeric_condition_results=[True],
        )
        assert result.val == 10.0

    def test_lambdify(self):
        x = Symbol("x")
        sv = SymVal(expr=x**2, val=4.0)
        func = sv.lambdify([x])
        assert func(3) == 9


class TestConfigSymbols:
    def test_decision_vars(self):
        symbols = ConfigSymbols()
        dvars = symbols.decision_vars()
        assert len(dvars) > 0
        assert symbols.N_p in dvars
        assert symbols.N_d in dvars

    def test_frozen_params(self):
        symbols = ConfigSymbols()
        fparams = symbols.frozen_params()
        assert symbols.cache_hit_rate in fparams

    def test_all_symbols(self):
        symbols = ConfigSymbols()
        all_syms = symbols.all_symbols()
        assert len(all_syms) > 10

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, PrefillTask, DecodeTask, KVHandle
    from sim.core.engine import SimulationEngine


@dataclass(order=True)
class Event(ABC):
    time: float
    priority: int = field(default=0, compare=True)

    @abstractmethod
    def process(self, engine: SimulationEngine) -> list[Event]:
        pass


@dataclass(order=True)
class RequestArrival(Event):
    time: float = field(compare=True)
    priority: int = field(default=10, compare=True)
    request: Request = field(default=None, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        engine.state.pending_requests.append(self.request)
        worker_id = engine.router.route_to_prefill(self.request)
        return [RouterDispatch(time=self.time, request=self.request, worker_id=worker_id)]


@dataclass(order=True)
class RouterDispatch(Event):
    time: float = field(compare=True)
    priority: int = field(default=20, compare=True)
    request: Request = field(default=None, compare=False)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.request.phase = RequestPhase.PREFILLING
        self.request.prefill_worker_id = self.worker_id
        worker = engine.prefill_workers[self.worker_id]
        worker.enqueue_request(self.request, engine.scheduler_config)

        if not worker.is_iteration_scheduled:
            worker.is_iteration_scheduled = True
            return [PrefillIterationStart(time=self.time, worker_id=self.worker_id)]
        return []


@dataclass(order=True)
class PrefillIterationStart(Event):
    time: float = field(compare=True)
    priority: int = field(default=30, compare=True)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        worker = engine.prefill_workers[self.worker_id]
        iteration_result = worker.run_iteration(self.time, engine)

        events: list[Event] = []
        for task in iteration_result.completed_tasks:
            events.append(
                PrefillChunkComplete(
                    time=self.time + iteration_result.iteration_time.val,
                    task=task,
                    worker_id=self.worker_id,
                )
            )

        if worker.has_pending_work():
            events.append(
                PrefillIterationStart(
                    time=self.time + iteration_result.iteration_time.val,
                    worker_id=self.worker_id,
                )
            )
        else:
            worker.is_iteration_scheduled = False

        return events


@dataclass(order=True)
class PrefillChunkComplete(Event):
    time: float = field(compare=True)
    priority: int = field(default=40, compare=True)
    task: PrefillTask = field(default=None, compare=False)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase, KVHandle

        if self.task.is_complete:
            self.task.request.phase = RequestPhase.TRANSFERRING
            kv_handle = KVHandle(
                request_id=self.task.request.id,
                kv_bytes=self.task.kv_bytes_produced,
                source_worker_id=self.worker_id,
            )
            dest_worker = engine.router.route_to_decode(kv_handle)
            kv_handle.dest_worker_id = dest_worker
            return [
                KVTransferStart(
                    time=self.time,
                    kv_handle=kv_handle,
                    request=self.task.request,
                )
            ]
        return []


@dataclass(order=True)
class KVTransferStart(Event):
    time: float = field(compare=True)
    priority: int = field(default=50, compare=True)
    kv_handle: KVHandle = field(default=None, compare=False)
    request: Request = field(default=None, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        self.kv_handle.transfer_started = self.time
        transfer_time = engine.kv_transfer_manager.initiate_transfer(
            self.kv_handle, engine.cluster_config
        )
        return [
            KVTransferComplete(
                time=self.time + transfer_time.val,
                kv_handle=self.kv_handle,
                request=self.request,
            )
        ]


@dataclass(order=True)
class KVTransferComplete(Event):
    time: float = field(compare=True)
    priority: int = field(default=60, compare=True)
    kv_handle: KVHandle = field(default=None, compare=False)
    request: Request = field(default=None, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.kv_handle.transfer_complete = self.time
        self.request.phase = RequestPhase.DECODING
        self.request.decode_worker_id = self.kv_handle.dest_worker_id

        worker = engine.decode_workers[self.kv_handle.dest_worker_id]
        worker.enqueue_request(self.request, self.kv_handle, engine.scheduler_config)

        if not worker.is_iteration_scheduled:
            worker.is_iteration_scheduled = True
            return [
                DecodeIterationStart(
                    time=self.time, worker_id=self.kv_handle.dest_worker_id
                )
            ]
        return []


@dataclass(order=True)
class DecodeIterationStart(Event):
    time: float = field(compare=True)
    priority: int = field(default=70, compare=True)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        worker = engine.decode_workers[self.worker_id]
        iteration_result = worker.run_iteration(self.time, engine)

        events: list[Event] = []
        for task in iteration_result.token_emitted_tasks:
            if task.request.tokens_generated == 1:
                task.request.first_token_time = self.time + iteration_result.iteration_time.val

            events.append(
                TokenEmit(
                    time=self.time + iteration_result.iteration_time.val,
                    task=task,
                    worker_id=self.worker_id,
                )
            )

        if worker.has_pending_work():
            events.append(
                DecodeIterationStart(
                    time=self.time + iteration_result.iteration_time.val,
                    worker_id=self.worker_id,
                )
            )
        else:
            worker.is_iteration_scheduled = False

        return events


@dataclass(order=True)
class TokenEmit(Event):
    time: float = field(compare=True)
    priority: int = field(default=80, compare=True)
    task: DecodeTask = field(default=None, compare=False)
    worker_id: int = field(default=0, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        self.task.request.token_times.append(self.time)

        if self.task.is_complete:
            return [
                RequestComplete(
                    time=self.time,
                    request=self.task.request,
                )
            ]
        return []


@dataclass(order=True)
class RequestComplete(Event):
    time: float = field(compare=True)
    priority: int = field(default=90, compare=True)
    request: Request = field(default=None, compare=False)

    def process(self, engine: SimulationEngine) -> list[Event]:
        from sim.core.request import RequestPhase

        self.request.phase = RequestPhase.COMPLETE
        self.request.complete_time = self.time
        engine.metrics.record_request_complete(self.request)
        return []

```

---

### AI

**Tool Action:**

**Code Changes:**
```
# SGLang LLM Inference Simulator

A discrete-event simulator for LLM inference with symbolic tracing, designed to model SGLang-style workloads with prefill/decode disaggregation, prefix caching, and parallel execution.

## Features

- **Symbolic Metric Tracing**: All latency computations produce both numeric values and SymPy expressions for optimization
- **P/D Disaggregation**: Separate prefill and decode worker pools with explicit KV transfer modeling
- **Prefix Cache (RadixAttention)**: LRU-based radix cache with cache-aware routing
- **Chunked Prefill**: Configurable chunk sizes for TTFT/throughput tradeoffs
- **Parallelism Support**: Tensor Parallelism (TP), Data Parallelism (DP), Expert Parallelism (EP)
- **MoE Models**: Full support for Mixture-of-Experts with all-to-all communication
- **Configurable Interconnects**: NVLink, InfiniBand, Ethernet with proper latency/bandwidth modeling

## Installation

```bash
cd sglang_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```python
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution

# Configure simulation
model = ModelConfig.llama_7b()
cluster = ClusterConfig.single_node_8gpu()
scheduler = SchedulerConfig.default()
cache = CacheConfig.h100_default()

# Create engine and run
engine = SimulationEngine(model, cluster, scheduler, cache)
# ... initialize workers, router, etc.
result = engine.run_until_idle()

print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
```

## Symbolic Throughput Optimization

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.optimizer.interface import OptimizerInterface

symbols = ConfigSymbols()

# Build symbolic throughput expression
throughput_expr = ...  # Your throughput formula

# Optimize over configuration space
optimizer = OptimizerInterface(objective=throughput_expr, maximize=True)
result = optimizer.grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
})

print(f"Optimal config: {result.optimal_values}")
```

## Project Structure

```
sim/
├── config/          # Configuration dataclasses
├── core/            # Simulation engine, events, state
├── symbolic/        # SymVal and symbolic combinators
├── workers/         # Prefill and decode workers
├── schedulers/      # Continuous batching scheduler
├── cache/           # Radix cache implementation
├── routing/         # Cache-aware router
├── parallel/        # TP/DP/EP group abstractions
├── network/         # Interconnect and KV transfer
├── kernels/         # Attention, MLP, MoE kernels
├── models/          # Operator graph templates
├── metrics/         # Metric collection and symbolic throughput
├── optimizer/       # Scipy/grid search interface
└── workload/        # Synthetic and trace-based workloads
```

## Examples

See the `examples/` directory for:
- `basic_simulation.py` - Full simulation workflow
- `optimize_pd_ratio.py` - P/D ratio optimization with symbolic expressions
- `sweep_chunk_size.py` - Chunk size analysis and cache effects

## Running Tests

```bash
python -m pytest tests/ -v
```

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
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

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import sympy
from sympy import Symbol

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder

if TYPE_CHECKING:
    pass


class ConstraintType(Enum):
    INEQUALITY = "ineq"
    EQUALITY = "eq"


@dataclass
class Constraint:
    name: str
    expr: sympy.Expr
    type: ConstraintType
    description: str = ""
    
    def is_satisfied(self, values: dict[Symbol, float]) -> bool:
        evaluated = float(self.expr.subs(list(values.items())))
        if self.type == ConstraintType.EQUALITY:
            return abs(evaluated) < 1e-6
        else:
            return evaluated >= 0
    
    def violation(self, values: dict[Symbol, float]) -> float:
        evaluated = float(self.expr.subs(list(values.items())))
        if self.type == ConstraintType.EQUALITY:
            return abs(evaluated)
        else:
            return max(0, -evaluated)


class ConstraintBuilder:
    def __init__(self, symbols: ConfigSymbols | None = None):
        self.symbols = symbols or ConfigSymbols()
        self.metric_builder = SymbolicMetricBuilder(self.symbols)
        self.constraints: list[Constraint] = []

    def add_memory_constraint(
        self,
        gpu_memory_bytes: int,
        safety_margin: float = 0.9,
    ) -> Constraint:
        peak_memory = self.metric_builder.peak_memory_expression()
        available = gpu_memory_bytes * safety_margin
        
        constraint = Constraint(
            name="memory_limit",
            expr=available - peak_memory,
            type=ConstraintType.INEQUALITY,
            description=f"Peak memory <= {available / 1e9:.1f} GB",
        )
        self.constraints.append(constraint)
        return constraint

    def add_ttft_slo(self, max_ttft_seconds: float) -> Constraint:
        ttft = self.metric_builder.ttft_expression()
        
        constraint = Constraint(
            name="ttft_slo",
            expr=max_ttft_seconds - ttft,
            type=ConstraintType.INEQUALITY,
            description=f"TTFT <= {max_ttft_seconds * 1000:.0f}ms",
        )
        self.constraints.append(constraint)
        return constraint

    def add_tpot_slo(self, max_tpot_seconds: float) -> Constraint:
        tpot = self.metric_builder.tpot_expression()
        
        constraint = Constraint(
            name="tpot_slo",
            expr=max_tpot_seconds - tpot,
            type=ConstraintType.INEQUALITY,
            description=f"TPOT <= {max_tpot_seconds * 1000:.0f}ms",
        )
        self.constraints.append(constraint)
        return constraint

    def add_e2e_latency_slo(self, max_latency_seconds: float) -> Constraint:
        e2e = self.metric_builder.e2e_latency_expression()
        
        constraint = Constraint(
            name="e2e_latency_slo",
            expr=max_latency_seconds - e2e,
            type=ConstraintType.INEQUALITY,
            description=f"E2E Latency <= {max_latency_seconds:.1f}s",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_throughput(self, min_tokens_per_sec: float) -> Constraint:
        throughput = self.metric_builder.throughput_expression()
        
        constraint = Constraint(
            name="min_throughput",
            expr=throughput - min_tokens_per_sec,
            type=ConstraintType.INEQUALITY,
            description=f"Throughput >= {min_tokens_per_sec:.0f} tok/s",
        )
        self.constraints.append(constraint)
        return constraint

    def add_cost_budget(
        self,
        max_cost_per_1k_tokens: float,
        gpu_cost_per_hour: float = 2.0,
    ) -> Constraint:
        cost_per_token = self.metric_builder.cost_per_token_expression(gpu_cost_per_hour)
        max_cost_per_token = max_cost_per_1k_tokens / 1000
        
        constraint = Constraint(
            name="cost_budget",
            expr=max_cost_per_token - cost_per_token,
            type=ConstraintType.INEQUALITY,
            description=f"Cost <= ${max_cost_per_1k_tokens:.4f}/1K tokens",
        )
        self.constraints.append(constraint)
        return constraint

    def add_power_budget(self, max_power_watts: float) -> Constraint:
        power = self.metric_builder.power_consumption_expression()
        
        constraint = Constraint(
            name="power_budget",
            expr=max_power_watts - power,
            type=ConstraintType.INEQUALITY,
            description=f"Power <= {max_power_watts:.0f}W",
        )
        self.constraints.append(constraint)
        return constraint

    def add_total_gpu_constraint(self, max_gpus: int) -> Constraint:
        s = self.symbols
        total_gpus = (s.N_p + s.N_d) * s.tp_size
        
        constraint = Constraint(
            name="max_gpus",
            expr=max_gpus - total_gpus,
            type=ConstraintType.INEQUALITY,
            description=f"Total GPUs <= {max_gpus}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_prefill_workers(self, min_workers: int) -> Constraint:
        s = self.symbols
        
        constraint = Constraint(
            name="min_prefill_workers",
            expr=s.N_p - min_workers,
            type=ConstraintType.INEQUALITY,
            description=f"N_p >= {min_workers}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_decode_workers(self, min_workers: int) -> Constraint:
        s = self.symbols
        
        constraint = Constraint(
            name="min_decode_workers",
            expr=s.N_d - min_workers,
            type=ConstraintType.INEQUALITY,
            description=f"N_d >= {min_workers}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_utilization_constraint(
        self,
        min_compute_util: float = 0.0,
        min_memory_util: float = 0.0,
    ) -> list[Constraint]:
        added = []
        
        if min_compute_util > 0:
            compute_util = self.metric_builder.compute_utilization_expression()
            constraint = Constraint(
                name="min_compute_util",
                expr=compute_util - min_compute_util,
                type=ConstraintType.INEQUALITY,
                description=f"Compute utilization >= {min_compute_util:.0%}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        if min_memory_util > 0:
            mem_util = self.metric_builder.memory_bandwidth_utilization_expression()
            constraint = Constraint(
                name="min_memory_util",
                expr=mem_util - min_memory_util,
                type=ConstraintType.INEQUALITY,
                description=f"Memory BW utilization >= {min_memory_util:.0%}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        return added

    def add_pd_ratio_constraint(
        self,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> list[Constraint]:
        s = self.symbols
        added = []
        
        if min_ratio is not None:
            constraint = Constraint(
                name="min_pd_ratio",
                expr=s.N_p / s.N_d - min_ratio,
                type=ConstraintType.INEQUALITY,
                description=f"N_p/N_d >= {min_ratio:.2f}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        if max_ratio is not None:
            constraint = Constraint(
                name="max_pd_ratio",
                expr=max_ratio - s.N_p / s.N_d,
                type=ConstraintType.INEQUALITY,
                description=f"N_p/N_d <= {max_ratio:.2f}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        return added

    def add_chunk_size_bounds(
        self,
        min_chunk: int,
        max_chunk: int,
    ) -> list[Constraint]:
        s = self.symbols
        
        constraints = [
            Constraint(
                name="min_chunk_size",
                expr=s.chunk_size - min_chunk,
                type=ConstraintType.INEQUALITY,
                description=f"chunk_size >= {min_chunk}",
            ),
            Constraint(
                name="max_chunk_size",
                expr=max_chunk - s.chunk_size,
                type=ConstraintType.INEQUALITY,
                description=f"chunk_size <= {max_chunk}",
            ),
        ]
        self.constraints.extend(constraints)
        return constraints

    def add_batch_size_bounds(
        self,
        min_batch: int,
        max_batch: int,
    ) -> list[Constraint]:
        s = self.symbols
        
        constraints = [
            Constraint(
                name="min_batch_size",
                expr=s.batch_cap_requests - min_batch,
                type=ConstraintType.INEQUALITY,
                description=f"batch_size >= {min_batch}",
            ),
            Constraint(
                name="max_batch_size",
                expr=max_batch - s.batch_cap_requests,
                type=ConstraintType.INEQUALITY,
                description=f"batch_size <= {max_batch}",
            ),
        ]
        self.constraints.extend(constraints)
        return constraints

    def get_all_constraints(self) -> list[Constraint]:
        return self.constraints.copy()

    def get_scipy_constraints(self) -> list[dict]:
        return [
            {
                "type": c.type.value,
                "fun": lambda x, expr=c.expr, syms=self.symbols.decision_vars(): float(
                    expr.subs(list(zip(syms, x)))
                ),
            }
            for c in self.constraints
        ]

    def check_all_constraints(self, values: dict[Symbol, float]) -> dict[str, tuple[bool, float]]:
        results = {}
        for c in self.constraints:
            satisfied = c.is_satisfied(values)
            violation = c.violation(values)
            results[c.name] = (satisfied, violation)
        return results

    def summarize(self) -> str:
        lines = ["Constraints:"]
        for c in self.constraints:
            lines.append(f"  - {c.name}: {c.description}")
        return "\n".join(lines)

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.visualization.graph import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)

__all__ = [
    "ExpressionGraphVisualizer",
    "MetricDependencyGraph",
    "render_expression_tree",
    "render_metric_dependencies",
]

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from io import StringIO
import html

import sympy
from sympy import Symbol, Function, Add, Mul, Pow, Max, Min, ceiling, Piecewise
from sympy.printing.dot import dotprint

if TYPE_CHECKING:
    from sim.metrics.expressions import MetricExpressions
    from sim.metrics.constraints import Constraint


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str
    shape: str = "box"
    color: str = "#ffffff"
    children: list[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str = ""
    style: str = "solid"


class ExpressionGraphVisualizer:
    NODE_COLORS = {
        "symbol": "#e3f2fd",
        "constant": "#fff3e0",
        "operator": "#f3e5f5",
        "function": "#e8f5e9",
        "metric": "#ffebee",
    }
    
    NODE_SHAPES = {
        "symbol": "ellipse",
        "constant": "box",
        "operator": "diamond",
        "function": "hexagon",
        "metric": "doubleoctagon",
    }

    def __init__(self):
        self._node_counter = 0
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    def reset(self) -> None:
        self._node_counter = 0
        self._nodes.clear()
        self._edges.clear()

    def _new_node_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"

    def build_graph(self, expr: sympy.Expr, root_name: str = "result") -> str:
        self.reset()
        root_id = self._visit_expr(expr)
        
        root_node = GraphNode(
            id="root",
            label=root_name,
            node_type="metric",
            shape=self.NODE_SHAPES["metric"],
            color=self.NODE_COLORS["metric"],
        )
        self._nodes["root"] = root_node
        self._edges.append(GraphEdge(source="root", target=root_id))
        
        return root_id

    def _visit_expr(self, expr: sympy.Expr) -> str:
        if isinstance(expr, Symbol):
            return self._add_symbol_node(expr)
        elif isinstance(expr, (int, float, sympy.Integer, sympy.Float, sympy.Rational)):
            return self._add_constant_node(expr)
        elif isinstance(expr, Add):
            return self._add_operator_node("+", list(expr.args))
        elif isinstance(expr, Mul):
            return self._add_operator_node("×", list(expr.args))
        elif isinstance(expr, Pow):
            base, exp = expr.args
            return self._add_function_node(f"^{exp}", [base])
        elif isinstance(expr, Max):
            return self._add_function_node("Max", list(expr.args))
        elif isinstance(expr, Min):
            return self._add_function_node("Min", list(expr.args))
        elif isinstance(expr, ceiling):
            return self._add_function_node("⌈⌉", list(expr.args))
        elif isinstance(expr, Piecewise):
            return self._add_piecewise_node(expr)
        elif hasattr(expr, 'func') and hasattr(expr, 'args'):
            func_name = expr.func.__name__
            return self._add_function_node(func_name, list(expr.args))
        else:
            return self._add_constant_node(expr)

    def _add_symbol_node(self, sym: Symbol) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=str(sym),
            node_type="symbol",
            shape=self.NODE_SHAPES["symbol"],
            color=self.NODE_COLORS["symbol"],
        )
        return node_id

    def _add_constant_node(self, value: Any) -> str:
        node_id = self._new_node_id()
        if isinstance(value, float):
            label = f"{value:.4g}"
        else:
            label = str(value)
        
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=label,
            node_type="constant",
            shape=self.NODE_SHAPES["constant"],
            color=self.NODE_COLORS["constant"],
        )
        return node_id

    def _add_operator_node(self, op: str, args: list) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=op,
            node_type="operator",
            shape=self.NODE_SHAPES["operator"],
            color=self.NODE_COLORS["operator"],
        )
        
        for arg in args:
            child_id = self._visit_expr(arg)
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id))
        
        return node_id

    def _add_function_node(self, func_name: str, args: list) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=func_name,
            node_type="function",
            shape=self.NODE_SHAPES["function"],
            color=self.NODE_COLORS["function"],
        )
        
        for arg in args:
            child_id = self._visit_expr(arg)
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id))
        
        return node_id

    def _add_piecewise_node(self, pw: Piecewise) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label="Piecewise",
            node_type="function",
            shape="house",
            color=self.NODE_COLORS["function"],
        )
        
        for expr, cond in pw.args:
            child_id = self._visit_expr(expr)
            cond_label = str(cond)[:20]
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id, label=cond_label))
        
        return node_id

    def to_dot(self, title: str = "Expression Graph") -> str:
        lines = [
            "digraph G {",
            f'    label="{title}";',
            "    labelloc=t;",
            "    rankdir=TB;",
            '    node [fontname="Helvetica", fontsize=10];',
            '    edge [fontname="Helvetica", fontsize=8];',
            "",
        ]
        
        for node in self._nodes.values():
            escaped_label = html.escape(node.label)
            lines.append(
                f'    {node.id} [label="{escaped_label}", '
                f'shape={node.shape}, style=filled, fillcolor="{node.color}"];'
            )
        
        lines.append("")
        
        for edge in self._edges:
            if edge.label:
                escaped_label = html.escape(edge.label)
                lines.append(
                    f'    {edge.source} -> {edge.target} [label="{escaped_label}"];'
                )
            else:
                lines.append(f"    {edge.source} -> {edge.target};")
        
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, title: str = "Expression Graph") -> str:
        lines = [
            "```mermaid",
            "flowchart TB",
            f"    subgraph title [{title}]",
        ]
        
        shape_map = {
            "ellipse": ("((", "))"),
            "box": ("[", "]"),
            "diamond": ("{", "}"),
            "hexagon": ("{{", "}}"),
            "doubleoctagon": ("[[", "]]"),
            "house": ("[/", "/]"),
        }
        
        for node in self._nodes.values():
            left, right = shape_map.get(node.shape, ("[", "]"))
            label = node.label.replace('"', "'")
            lines.append(f"        {node.id}{left}{label}{right}")
        
        for edge in self._edges:
            if edge.label:
                label = edge.label.replace('"', "'")
                lines.append(f'        {edge.source} -->|"{label}"| {edge.target}')
            else:
                lines.append(f"        {edge.source} --> {edge.target}")
        
        lines.append("    end")
        lines.append("```")
        return "\n".join(lines)

    def render_to_file(
        self,
        expr: sympy.Expr,
        filename: str,
        title: str = "Expression Graph",
        format: str = "png",
    ) -> str:
        self.build_graph(expr, title)
        dot_content = self.to_dot(title)
        
        dot_filename = filename.rsplit(".", 1)[0] + ".dot"
        with open(dot_filename, "w") as f:
            f.write(dot_content)
        
        return dot_filename


class MetricDependencyGraph:
    def __init__(self):
        self.nodes: dict[str, set[str]] = {}
        self.edges: list[tuple[str, str]] = []

    def add_metric(self, name: str, expr: sympy.Expr) -> None:
        symbols = {str(s) for s in expr.free_symbols}
        self.nodes[name] = symbols
        
        for sym in symbols:
            self.edges.append((sym, name))

    def add_metrics(self, metrics: dict[str, sympy.Expr]) -> None:
        for name, expr in metrics.items():
            self.add_metric(name, expr)

    def find_shared_dependencies(self) -> dict[str, list[str]]:
        symbol_to_metrics: dict[str, list[str]] = {}
        
        for metric, symbols in self.nodes.items():
            for sym in symbols:
                if sym not in symbol_to_metrics:
                    symbol_to_metrics[sym] = []
                symbol_to_metrics[sym].append(metric)
        
        return {sym: metrics for sym, metrics in symbol_to_metrics.items() if len(metrics) > 1}

    def to_dot(self, title: str = "Metric Dependencies") -> str:
        lines = [
            "digraph MetricDependencies {",
            f'    label="{title}";',
            "    labelloc=t;",
            "    rankdir=LR;",
            '    node [fontname="Helvetica"];',
            "",
            "    // Symbol nodes",
            '    subgraph cluster_symbols {',
            '        label="Configuration Variables";',
            '        style=dashed;',
            '        node [shape=ellipse, style=filled, fillcolor="#e3f2fd"];',
        ]
        
        all_symbols = set()
        for symbols in self.nodes.values():
            all_symbols.update(symbols)
        
        for sym in sorted(all_symbols):
            lines.append(f'        "{sym}";')
        
        lines.extend([
            "    }",
            "",
            "    // Metric nodes",
            '    subgraph cluster_metrics {',
            '        label="Metrics";',
            '        style=dashed;',
            '        node [shape=box, style=filled, fillcolor="#ffebee"];',
        ])
        
        for metric in sorted(self.nodes.keys()):
            lines.append(f'        "{metric}";')
        
        lines.extend([
            "    }",
            "",
            "    // Edges",
        ])
        
        for sym, metric in self.edges:
            lines.append(f'    "{sym}" -> "{metric}";')
        
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, title: str = "Metric Dependencies") -> str:
        lines = [
            "```mermaid",
            "flowchart LR",
            f"    subgraph title [{title}]",
            "    subgraph symbols [Configuration Variables]",
        ]
        
        all_symbols = set()
        for symbols in self.nodes.values():
            all_symbols.update(symbols)
        
        for sym in sorted(all_symbols):
            safe_id = sym.replace("_", "")
            lines.append(f"        {safe_id}(({sym}))")
        
        lines.append("    end")
        lines.append("    subgraph metrics [Metrics]")
        
        for metric in sorted(self.nodes.keys()):
            safe_id = metric.replace("_", "")
            lines.append(f"        {safe_id}[{metric}]")
        
        lines.append("    end")
        
        for sym, metric in self.edges:
            safe_sym = sym.replace("_", "")
            safe_metric = metric.replace("_", "")
            lines.append(f"    {safe_sym} --> {safe_metric}")
        
        lines.append("    end")
        lines.append("```")
        return "\n".join(lines)


def render_expression_tree(
    expr: sympy.Expr,
    title: str = "Expression",
    output_format: str = "dot",
) -> str:
    viz = ExpressionGraphVisualizer()
    viz.build_graph(expr, title)
    
    if output_format == "mermaid":
        return viz.to_mermaid(title)
    else:
        return viz.to_dot(title)


def render_metric_dependencies(
    metrics: dict[str, sympy.Expr],
    title: str = "Metric Dependencies",
    output_format: str = "dot",
) -> str:
    graph = MetricDependencyGraph()
    graph.add_metrics(metrics)
    
    if output_format == "mermaid":
        return graph.to_mermaid(title)
    else:
        return graph.to_dot(title)


def sympy_dotprint(expr: sympy.Expr) -> str:
    try:
        return dotprint(expr)
    except Exception:
        viz = ExpressionGraphVisualizer()
        viz.build_graph(expr)
        return viz.to_dot()

```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from itertools import product
from enum import Enum

import sympy
from sympy import Symbol
import numpy as np

from sim.symbolic.symbols import ConfigSymbols


class ObjectiveDirection(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class OptimizationResult:
    optimal_values: dict[str, float]
    optimal_objective: float
    success: bool
    message: str
    all_evaluations: list[tuple[dict, float]] = field(default_factory=list)
    constraint_violations: dict[str, float] = field(default_factory=dict)
    pareto_front: list[dict] = field(default_factory=list)


@dataclass
class MultiObjectiveResult:
    pareto_front: list[dict[str, float]]
    pareto_objectives: list[dict[str, float]]
    all_evaluations: list[tuple[dict, dict[str, float]]] = field(default_factory=list)


@dataclass
class Constraint:
    expr: sympy.Expr
    type: str  # "eq" for equality, "ineq" for inequality (>= 0)
    name: str = ""


@dataclass
class ObjectiveSpec:
    name: str
    expr: sympy.Expr
    direction: ObjectiveDirection
    weight: float = 1.0


class OptimizerInterface:
    def __init__(
        self,
        objective: sympy.Expr,
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
        maximize: bool = True,
    ):
        self.objective = objective
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self.maximize = maximize

        self._decision_vars: list[Symbol] = []
        self._frozen_params: dict[Symbol, float] = {}

    def set_decision_vars(self, vars: list[Symbol]) -> None:
        self._decision_vars = vars

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def get_decision_vars(self) -> list[Symbol]:
        if self._decision_vars:
            return self._decision_vars
        return self.symbols.decision_vars()

    def get_frozen_params(self) -> list[Symbol]:
        return self.symbols.frozen_params()

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def to_scipy_minimize(
        self,
        bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        decision_vars = self.get_decision_vars()

        substituted_obj = self.substitute_frozen(self.objective)

        if self.maximize:
            obj_to_minimize = -substituted_obj
        else:
            obj_to_minimize = substituted_obj

        obj_func = sympy.lambdify(decision_vars, obj_to_minimize, modules=["numpy"])

        def objective_wrapper(x: np.ndarray) -> float:
            return float(obj_func(*x))

        scipy_constraints = []
        for constraint in self.constraints:
            substituted = self.substitute_frozen(constraint.expr)
            constraint_func = sympy.lambdify(decision_vars, substituted, modules=["numpy"])

            scipy_constraints.append({
                "type": constraint.type,
                "fun": lambda x, f=constraint_func: float(f(*x)),
            })

        scipy_bounds = None
        if bounds:
            scipy_bounds = [
                bounds.get(var, (None, None)) for var in decision_vars
            ]

        return {
            "fun": objective_wrapper,
            "x0": np.ones(len(decision_vars)),
            "bounds": scipy_bounds,
            "constraints": scipy_constraints,
            "method": "SLSQP",
        }

    def enumerate_discrete_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> list[dict[Symbol, int]]:
        regimes = []
        var_names = list(discrete_vars.keys())
        var_values = list(discrete_vars.values())

        for combo in product(*var_values):
            regime = dict(zip(var_names, combo))
            regimes.append(regime)

        return regimes

    def optimize_over_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> OptimizationResult:
        from scipy.optimize import minimize

        regimes = self.enumerate_discrete_regimes(discrete_vars, continuous_bounds)
        best_result: OptimizationResult | None = None
        all_evaluations: list[tuple[dict, float]] = []

        decision_vars = self.get_decision_vars()
        continuous_vars = [v for v in decision_vars if v not in discrete_vars]

        for regime in regimes:
            regime_objective = self.objective.subs(list(regime.items()))
            regime_objective = self.substitute_frozen(regime_objective)

            if not continuous_vars:
                if self.maximize:
                    obj_val = -float(regime_objective)
                else:
                    obj_val = float(regime_objective)

                result_dict = {str(k): v for k, v in regime.items()}
                all_evaluations.append((result_dict, -obj_val if self.maximize else obj_val))

                if best_result is None or (self.maximize and -obj_val > best_result.optimal_objective) or \
                   (not self.maximize and obj_val < best_result.optimal_objective):
                    best_result = OptimizationResult(
                        optimal_values=result_dict,
                        optimal_objective=-obj_val if self.maximize else obj_val,
                        success=True,
                        message=f"Regime: {regime}",
                    )
            else:
                obj_func = sympy.lambdify(continuous_vars, regime_objective, modules=["numpy"])

                if self.maximize:
                    def objective_wrapper(x):
                        return -float(obj_func(*x))
                else:
                    def objective_wrapper(x):
                        return float(obj_func(*x))

                bounds_list = [
                    continuous_bounds.get(v, (1, 100)) if continuous_bounds else (1, 100)
                    for v in continuous_vars
                ]

                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_list])

                try:
                    result = minimize(
                        objective_wrapper,
                        x0,
                        bounds=bounds_list,
                        method="L-BFGS-B",
                    )

                    result_dict = {str(k): v for k, v in regime.items()}
                    for var, val in zip(continuous_vars, result.x):
                        result_dict[str(var)] = float(val)

                    obj_val = -result.fun if self.maximize else result.fun
                    all_evaluations.append((result_dict, obj_val))

                    if best_result is None or \
                       (self.maximize and obj_val > best_result.optimal_objective) or \
                       (not self.maximize and obj_val < best_result.optimal_objective):
                        best_result = OptimizationResult(
                            optimal_values=result_dict,
                            optimal_objective=obj_val,
                            success=result.success,
                            message=result.message,
                        )
                except Exception as e:
                    continue

        if best_result is None:
            return OptimizationResult(
                optimal_values={},
                optimal_objective=float("inf") if not self.maximize else float("-inf"),
                success=False,
                message="No valid regime found",
                all_evaluations=all_evaluations,
            )

        best_result.all_evaluations = all_evaluations
        return best_result

    def grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())

        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []

        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])

        for combo in product(*var_values):
            try:
                obj_val = float(obj_func(*combo))
                if self.maximize:
                    obj_val = obj_val
                else:
                    obj_val = obj_val

                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))

                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue

        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Grid search complete",
            all_evaluations=all_evaluations,
        )


```

---

### AI

**Tool Action:**

**Code Changes:**
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from itertools import product
from enum import Enum

import sympy
from sympy import Symbol
import numpy as np

from sim.symbolic.symbols import ConfigSymbols


class ObjectiveDirection(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class OptimizationResult:
    optimal_values: dict[str, float]
    optimal_objective: float
    success: bool
    message: str
    all_evaluations: list[tuple[dict, float]] = field(default_factory=list)
    constraint_violations: dict[str, float] = field(default_factory=dict)
    pareto_front: list[dict] = field(default_factory=list)


@dataclass
class MultiObjectiveResult:
    pareto_front: list[dict[str, float]]
    pareto_objectives: list[dict[str, float]]
    all_evaluations: list[tuple[dict, dict[str, float]]] = field(default_factory=list)


@dataclass
class Constraint:
    expr: sympy.Expr
    type: str  # "eq" for equality, "ineq" for inequality (>= 0)
    name: str = ""


@dataclass
class ObjectiveSpec:
    name: str
    expr: sympy.Expr
    direction: ObjectiveDirection
    weight: float = 1.0


class OptimizerInterface:
    def __init__(
        self,
        objective: sympy.Expr,
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
        maximize: bool = True,
    ):
        self.objective = objective
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self.maximize = maximize

        self._decision_vars: list[Symbol] = []
        self._frozen_params: dict[Symbol, float] = {}

    def set_decision_vars(self, vars: list[Symbol]) -> None:
        self._decision_vars = vars

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def get_decision_vars(self) -> list[Symbol]:
        if self._decision_vars:
            return self._decision_vars
        return self.symbols.decision_vars()

    def get_frozen_params(self) -> list[Symbol]:
        return self.symbols.frozen_params()

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def to_scipy_minimize(
        self,
        bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        decision_vars = self.get_decision_vars()

        substituted_obj = self.substitute_frozen(self.objective)

        if self.maximize:
            obj_to_minimize = -substituted_obj
        else:
            obj_to_minimize = substituted_obj

        obj_func = sympy.lambdify(decision_vars, obj_to_minimize, modules=["numpy"])

        def objective_wrapper(x: np.ndarray) -> float:
            return float(obj_func(*x))

        scipy_constraints = []
        for constraint in self.constraints:
            substituted = self.substitute_frozen(constraint.expr)
            constraint_func = sympy.lambdify(decision_vars, substituted, modules=["numpy"])

            scipy_constraints.append({
                "type": constraint.type,
                "fun": lambda x, f=constraint_func: float(f(*x)),
            })

        scipy_bounds = None
        if bounds:
            scipy_bounds = [
                bounds.get(var, (None, None)) for var in decision_vars
            ]

        return {
            "fun": objective_wrapper,
            "x0": np.ones(len(decision_vars)),
            "bounds": scipy_bounds,
            "constraints": scipy_constraints,
            "method": "SLSQP",
        }

    def enumerate_discrete_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> list[dict[Symbol, int]]:
        regimes = []
        var_names = list(discrete_vars.keys())
        var_values = list(discrete_vars.values())

        for combo in product(*var_values):
            regime = dict(zip(var_names, combo))
            regimes.append(regime)

        return regimes

    def optimize_over_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> OptimizationResult:
        from scipy.optimize import minimize

        regimes = self.enumerate_discrete_regimes(discrete_vars, continuous_bounds)
        best_result: OptimizationResult | None = None
        all_evaluations: list[tuple[dict, float]] = []

        decision_vars = self.get_decision_vars()
        continuous_vars = [v for v in decision_vars if v not in discrete_vars]

        for regime in regimes:
            regime_objective = self.objective.subs(list(regime.items()))
            regime_objective = self.substitute_frozen(regime_objective)

            if not continuous_vars:
                if self.maximize:
                    obj_val = -float(regime_objective)
                else:
                    obj_val = float(regime_objective)

                result_dict = {str(k): v for k, v in regime.items()}
                all_evaluations.append((result_dict, -obj_val if self.maximize else obj_val))

                if best_result is None or (self.maximize and -obj_val > best_result.optimal_objective) or \
                   (not self.maximize and obj_val < best_result.optimal_objective):
                    best_result = OptimizationResult(
                        optimal_values=result_dict,
                        optimal_objective=-obj_val if self.maximize else obj_val,
                        success=True,
                        message=f"Regime: {regime}",
                    )
            else:
                obj_func = sympy.lambdify(continuous_vars, regime_objective, modules=["numpy"])

                if self.maximize:
                    def objective_wrapper(x):
                        return -float(obj_func(*x))
                else:
                    def objective_wrapper(x):
                        return float(obj_func(*x))

                bounds_list = [
                    continuous_bounds.get(v, (1, 100)) if continuous_bounds else (1, 100)
                    for v in continuous_vars
                ]

                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_list])

                try:
                    result = minimize(
                        objective_wrapper,
                        x0,
                        bounds=bounds_list,
                        method="L-BFGS-B",
                    )

                    result_dict = {str(k): v for k, v in regime.items()}
                    for var, val in zip(continuous_vars, result.x):
                        result_dict[str(var)] = float(val)

                    obj_val = -result.fun if self.maximize else result.fun
                    all_evaluations.append((result_dict, obj_val))

                    if best_result is None or \
                       (self.maximize and obj_val > best_result.optimal_objective) or \
                       (not self.maximize and obj_val < best_result.optimal_objective):
                        best_result = OptimizationResult(
                            optimal_values=result_dict,
                            optimal_objective=obj_val,
                            success=result.success,
                            message=result.message,
                        )
                except Exception as e:
                    continue

        if best_result is None:
            return OptimizationResult(
                optimal_values={},
                optimal_objective=float("inf") if not self.maximize else float("-inf"),
                success=False,
                message="No valid regime found",
                all_evaluations=all_evaluations,
            )

        best_result.all_evaluations = all_evaluations
        return best_result

    def grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())

        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []

        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])

        for combo in product(*var_values):
            try:
                obj_val = float(obj_func(*combo))
                if self.maximize:
                    obj_val = obj_val
                else:
                    obj_val = obj_val

                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))

                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue

        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Grid search complete",
            all_evaluations=all_evaluations,
        )

    def constrained_grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
        constraints: list[Constraint],
    ) -> OptimizationResult:
        from sim.metrics.constraints import Constraint as MetricConstraint
        
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())
        
        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []
        
        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])
        
        constraint_funcs = []
        for c in constraints:
            substituted = self.substitute_frozen(c.expr)
            constraint_funcs.append(
                (c.name, c.type, sympy.lambdify(var_names, substituted, modules=["numpy"]))
            )
        
        for combo in product(*var_values):
            try:
                feasible = True
                violations = {}
                
                for name, ctype, cfunc in constraint_funcs:
                    c_val = float(cfunc(*combo))
                    if ctype == "ineq" and c_val < 0:
                        feasible = False
                        violations[name] = -c_val
                    elif ctype == "eq" and abs(c_val) > 1e-6:
                        feasible = False
                        violations[name] = abs(c_val)
                
                if not feasible:
                    continue
                
                obj_val = float(obj_func(*combo))
                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))
                
                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue
        
        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Constrained grid search complete",
            all_evaluations=all_evaluations,
        )


class MultiObjectiveOptimizer:
    def __init__(
        self,
        objectives: list[ObjectiveSpec],
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
    ):
        self.objectives = objectives
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self._frozen_params: dict[Symbol, float] = {}

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def weighted_sum_optimize(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        combined_obj = sympy.Integer(0)
        
        for obj in self.objectives:
            weight = obj.weight
            expr = obj.expr
            if obj.direction == ObjectiveDirection.MINIMIZE:
                combined_obj = combined_obj - weight * expr
            else:
                combined_obj = combined_obj + weight * expr
        
        optimizer = OptimizerInterface(
            objective=combined_obj,
            constraints=self.constraints,
            symbols=self.symbols,
            maximize=True,
        )
        optimizer.set_frozen_params(self._frozen_params)
        
        return optimizer.grid_search(param_grid)

    def pareto_grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> MultiObjectiveResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())
        
        obj_funcs = []
        for obj in self.objectives:
            substituted = self.substitute_frozen(obj.expr)
            obj_funcs.append((
                obj.name,
                obj.direction,
                sympy.lambdify(var_names, substituted, modules=["numpy"])
            ))
        
        constraint_funcs = []
        for c in self.constraints:
            substituted = self.substitute_frozen(c.expr)
            constraint_funcs.append(
                (c.name, c.type, sympy.lambdify(var_names, substituted, modules=["numpy"]))
            )
        
        all_evaluations: list[tuple[dict, dict[str, float]]] = []
        feasible_solutions: list[tuple[dict[str, float], dict[str, float]]] = []
        
        for combo in product(*var_values):
            try:
                feasible = True
                for name, ctype, cfunc in constraint_funcs:
                    c_val = float(cfunc(*combo))
                    if ctype == "ineq" and c_val < 0:
                        feasible = False
                        break
                    elif ctype == "eq" and abs(c_val) > 1e-6:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                obj_vals = {}
                for name, direction, ofunc in obj_funcs:
                    obj_vals[name] = float(ofunc(*combo))
                
                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_vals))
                feasible_solutions.append((result_dict, obj_vals))
            except Exception:
                continue
        
        pareto_front = self._compute_pareto_front(feasible_solutions)
        
        return MultiObjectiveResult(
            pareto_front=[p[0] for p in pareto_front],
            pareto_objectives=[p[1] for p in pareto_front],
            all_evaluations=all_evaluations,
        )

    def _compute_pareto_front(
        self,
        solutions: list[tuple[dict[str, float], dict[str, float]]],
    ) -> list[tuple[dict[str, float], dict[str, float]]]:
        if not solutions:
            return []
        
        pareto = []
        
        for i, (params_i, objs_i) in enumerate(solutions):
            is_dominated = False
            
            for j, (params_j, objs_j) in enumerate(solutions):
                if i == j:
                    continue
                
                if self._dominates(objs_j, objs_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append((params_i, objs_i))
        
        return pareto

    def _dominates(
        self,
        objs_a: dict[str, float],
        objs_b: dict[str, float],
    ) -> bool:
        dominated = True
        strictly_better = False
        
        for obj in self.objectives:
            name = obj.name
            val_a = objs_a[name]
            val_b = objs_b[name]
            
            if obj.direction == ObjectiveDirection.MAXIMIZE:
                if val_a < val_b:
                    dominated = False
                    break
                if val_a > val_b:
                    strictly_better = True
            else:
                if val_a > val_b:
                    dominated = False
                    break
                if val_a < val_b:
                    strictly_better = True
        
        return dominated and strictly_better

    def epsilon_constraint(
        self,
        primary_objective: str,
        epsilon_bounds: dict[str, tuple[float, float]],
        param_grid: dict[Symbol, list[float]],
    ) -> list[OptimizationResult]:
        primary = next(o for o in self.objectives if o.name == primary_objective)
        
        results = []
        
        for other_obj in self.objectives:
            if other_obj.name == primary_objective:
                continue
            
            if other_obj.name in epsilon_bounds:
                lb, ub = epsilon_bounds[other_obj.name]
                
                if other_obj.direction == ObjectiveDirection.MAXIMIZE:
                    constraint = Constraint(
                        expr=other_obj.expr - lb,
                        type="ineq",
                        name=f"{other_obj.name}_lb",
                    )
                else:
                    constraint = Constraint(
                        expr=ub - other_obj.expr,
                        type="ineq",
                        name=f"{other_obj.name}_ub",
                    )
                
                optimizer = OptimizerInterface(
                    objective=primary.expr,
                    constraints=self.constraints + [constraint],
                    symbols=self.symbols,
                    maximize=primary.direction == ObjectiveDirection.MAXIMIZE,
                )
                optimizer.set_frozen_params(self._frozen_params)
                
                result = optimizer.constrained_grid_search(param_grid, [constraint])
                results.append(result)
        
        return results


```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.optimizer.interface import (
    OptimizerInterface,
    OptimizationResult,
    MultiObjectiveOptimizer,
    MultiObjectiveResult,
    ObjectiveSpec,
    ObjectiveDirection,
    Constraint,
)

__all__ = [
    "OptimizerInterface",
    "OptimizationResult",
    "MultiObjectiveOptimizer",
    "MultiObjectiveResult",
    "ObjectiveSpec",
    "ObjectiveDirection",
    "Constraint",
]


```

---

### AI

**Tool Action:**

**Code Changes:**
```
from sim.metrics.definitions import MetricCollector, RequestMetrics
from sim.metrics.symbolic import SymbolicThroughputBuilder
from sim.metrics.expressions import SymbolicMetricBuilder, MetricExpressions
from sim.metrics.memory import SymbolicMemoryModel, MemoryBreakdown
from sim.metrics.constraints import ConstraintBuilder, Constraint, ConstraintType

__all__ = [
    "MetricCollector",
    "RequestMetrics",
    "SymbolicThroughputBuilder",
    "SymbolicMetricBuilder",
    "MetricExpressions",
    "SymbolicMemoryModel",
    "MemoryBreakdown",
    "ConstraintBuilder",
    "Constraint",
    "ConstraintType",
]


```

---

### AI

**Tool Action:**

**Code Changes:**
```
#!/usr/bin/env python3
"""Example demonstrating multi-objective optimization with constraints and visualization."""

import sys
sys.path.insert(0, "..")

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder, MetricExpressions
from sim.metrics.memory import SymbolicMemoryModel
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer.interface import (
    MultiObjectiveOptimizer,
    ObjectiveSpec,
    ObjectiveDirection,
)
from sim.visualization.graph import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)


def build_optimization_problem():
    symbols = ConfigSymbols()
    metric_builder = SymbolicMetricBuilder(symbols)
    
    metrics = metric_builder.build_all_expressions()
    
    print("=== Available Metrics ===")
    for name, expr in metrics.as_dict().items():
        print(f"  {name}")
    
    return symbols, metric_builder, metrics


def run_constrained_optimization():
    symbols, metric_builder, metrics = build_optimization_problem()
    
    constraint_builder = ConstraintBuilder(symbols)
    
    constraint_builder.add_memory_constraint(
        gpu_memory_bytes=80 * 1024**3,
        safety_margin=0.9,
    )
    constraint_builder.add_ttft_slo(max_ttft_seconds=0.5)
    constraint_builder.add_tpot_slo(max_tpot_seconds=0.05)
    constraint_builder.add_total_gpu_constraint(max_gpus=16)
    constraint_builder.add_min_throughput(min_tokens_per_sec=100)
    
    print("\n=== Constraints ===")
    print(constraint_builder.summarize())
    
    objectives = [
        ObjectiveSpec(
            name="throughput",
            expr=metrics.throughput,
            direction=ObjectiveDirection.MAXIMIZE,
            weight=1.0,
        ),
        ObjectiveSpec(
            name="cost_per_token",
            expr=metrics.cost_per_token,
            direction=ObjectiveDirection.MINIMIZE,
            weight=0.5,
        ),
        ObjectiveSpec(
            name="e2e_latency",
            expr=metrics.e2e_latency,
            direction=ObjectiveDirection.MINIMIZE,
            weight=0.3,
        ),
    ]
    
    optimizer = MultiObjectiveOptimizer(
        objectives=objectives,
        constraints=constraint_builder.get_all_constraints(),
        symbols=symbols,
    )
    
    frozen_params = {
        symbols.gpu_flops: 2e15,
        symbols.gpu_mem_bw: 3.35e12,
        symbols.network_bw: 100e9,
        symbols.network_latency: 10e-6,
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.head_dim: 128,
        symbols.num_heads: 32,
        symbols.num_experts: 1,
        symbols.kv_bytes_per_token: 256,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.cache_hit_rate: 0.3,
        symbols.avg_prefix_match: 0.5,
    }
    optimizer.set_frozen_params(frozen_params)
    
    param_grid = {
        symbols.N_p: [1, 2, 4, 8],
        symbols.N_d: [1, 2, 4, 8],
        symbols.chunk_size: [4096, 8192, 16384],
        symbols.batch_cap_requests: [64, 128, 256],
        symbols.tp_size: [1, 2, 4],
    }
    
    print("\n=== Running Pareto Optimization ===")
    result = optimizer.pareto_grid_search(param_grid)
    
    print(f"\nFound {len(result.pareto_front)} Pareto-optimal solutions")
    print(f"Total feasible solutions evaluated: {len(result.all_evaluations)}")
    
    print("\n=== Pareto Front (Top 5) ===")
    sorted_pareto = sorted(
        zip(result.pareto_front, result.pareto_objectives),
        key=lambda x: x[1].get("throughput", 0),
        reverse=True,
    )[:5]
    
    for i, (params, objs) in enumerate(sorted_pareto, 1):
        print(f"\n{i}. Configuration:")
        print(f"   N_p={params.get('N_p')}, N_d={params.get('N_d')}, "
              f"TP={params.get('TP')}, chunk={params.get('c')}")
        print(f"   Objectives:")
        for name, val in objs.items():
            if "latency" in name or "ttft" in name or "tpot" in name:
                print(f"     {name}: {val*1000:.2f}ms")
            elif "cost" in name:
                print(f"     {name}: ${val*1000:.4f}/1K tokens")
            else:
                print(f"     {name}: {val:.2f}")
    
    return result


def visualize_metrics():
    symbols = ConfigSymbols()
    metric_builder = SymbolicMetricBuilder(symbols)
    metrics = metric_builder.build_all_expressions()
    
    print("\n=== Generating Metric Dependency Graph ===")
    dep_graph = MetricDependencyGraph()
    dep_graph.add_metrics({
        "throughput": metrics.throughput,
        "ttft": metrics.ttft,
        "tpot": metrics.tpot,
        "peak_memory": metrics.peak_memory,
        "cost_per_token": metrics.cost_per_token,
    })
    
    shared = dep_graph.find_shared_dependencies()
    print(f"\nShared dependencies affecting multiple metrics:")
    for sym, affected_metrics in list(shared.items())[:10]:
        print(f"  {sym}: affects {affected_metrics}")
    
    mermaid = dep_graph.to_mermaid("Metric Dependencies")
    print(f"\n=== Mermaid Diagram ===")
    print(mermaid[:500] + "..." if len(mermaid) > 500 else mermaid)
    
    print("\n=== Throughput Expression Tree ===")
    viz = ExpressionGraphVisualizer()
    viz.build_graph(metrics.throughput, "Throughput")
    dot = viz.to_dot("Throughput Expression")
    print(f"DOT graph generated ({len(dot)} chars)")
    print("First 300 chars:")
    print(dot[:300] + "...")
    
    return dep_graph, viz


def analyze_memory_scaling():
    symbols = ConfigSymbols()
    mem_model = SymbolicMemoryModel(symbols)
    
    print("\n=== Memory Scaling Analysis ===")
    
    peak_mem_expr = mem_model.symbolic_peak_memory()
    
    base_params = {
        symbols.num_layers: 32,
        symbols.hidden_dim: 4096,
        symbols.num_heads: 32,
        symbols.head_dim: 128,
        symbols.num_experts: 1,
        symbols.chunk_size: 8192,
        symbols.avg_prompt_len: 512,
        symbols.avg_output_len: 128,
        symbols.tp_size: 1,
    }
    
    print("\nPeak Memory vs Batch Size:")
    for batch in [32, 64, 128, 256, 512]:
        params = base_params.copy()
        params[symbols.batch_cap_requests] = batch
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  Batch {batch:3d}: {mem_gb:.1f} GB")
    
    print("\nPeak Memory vs TP Size (batch=256):")
    base_params[symbols.batch_cap_requests] = 256
    for tp in [1, 2, 4, 8]:
        params = base_params.copy()
        params[symbols.tp_size] = tp
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  TP={tp}: {mem_gb:.1f} GB per GPU")
    
    print("\nPeak Memory vs Sequence Length (batch=128):")
    base_params[symbols.batch_cap_requests] = 128
    base_params[symbols.tp_size] = 1
    for seq_len in [256, 512, 1024, 2048, 4096]:
        params = base_params.copy()
        params[symbols.avg_prompt_len] = seq_len
        mem_gb = float(peak_mem_expr.subs(list(params.items()))) / 1e9
        print(f"  Seq len {seq_len:4d}: {mem_gb:.1f} GB")


if __name__ == "__main__":
    run_constrained_optimization()
    visualize_metrics()
    analyze_memory_scaling()

```

---

### AI

**Tool Action:**

**Code Changes:**
```
[project]
name = "sglang-sim"
version = "0.1.0"
description = "LLM inference simulator with symbolic tracing for SGLang-style workloads"
requires-python = ">=3.10"
dependencies = [
    "sympy>=1.12",
    "numpy>=1.24",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]
viz = [
    "graphviz>=0.20",
    "matplotlib>=3.7",
]
all = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "graphviz>=0.20",
    "matplotlib>=3.7",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]


```

---

### AI

**Tool Action:**

**Code Changes:**
```
# SGLang LLM Inference Simulator

A discrete-event simulator for LLM inference with symbolic tracing, designed to model SGLang-style workloads with prefill/decode disaggregation, prefix caching, and parallel execution.

## Features

- **Symbolic Metric Tracing**: All latency computations produce both numeric values and SymPy expressions for optimization
- **P/D Disaggregation**: Separate prefill and decode worker pools with explicit KV transfer modeling
- **Prefix Cache (RadixAttention)**: LRU-based radix cache with cache-aware routing
- **Chunked Prefill**: Configurable chunk sizes for TTFT/throughput tradeoffs
- **Parallelism Support**: Tensor Parallelism (TP), Data Parallelism (DP), Expert Parallelism (EP)
- **MoE Models**: Full support for Mixture-of-Experts with all-to-all communication
- **Configurable Interconnects**: NVLink, InfiniBand, Ethernet with proper latency/bandwidth modeling
- **Multi-Objective Optimization**: Pareto-optimal search with weighted-sum and epsilon-constraint methods
- **Constraint Handling**: Memory limits, SLO targets, cost budgets, GPU constraints
- **Compute Graph Visualization**: DOT and Mermaid diagrams for expression trees and metric dependencies

## Symbolic Metrics Available

| Metric | Description |
|--------|-------------|
| `throughput` | Tokens per second |
| `ttft` | Time to First Token |
| `tpot` | Time per Output Token |
| `e2e_latency` | End-to-end request latency |
| `peak_memory` | Peak GPU memory usage |
| `compute_utilization` | GPU compute utilization |
| `memory_bandwidth_utilization` | Memory bandwidth utilization |
| `network_utilization` | Network bandwidth utilization |
| `cost_per_token` | Cost efficiency metric |
| `power_consumption` | Power consumption estimate |

## Installation

```bash
cd sglang_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# For visualization support
pip install -e ".[viz]"
```

## Quick Start

```python
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution

# Configure simulation
model = ModelConfig.llama_7b()
cluster = ClusterConfig.single_node_8gpu()
scheduler = SchedulerConfig.default()
cache = CacheConfig.h100_default()

# Create engine and run
engine = SimulationEngine(model, cluster, scheduler, cache)
# ... initialize workers, router, etc.
result = engine.run_until_idle()

print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
```

## Symbolic Throughput Optimization

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.optimizer.interface import OptimizerInterface

symbols = ConfigSymbols()

# Build symbolic throughput expression
throughput_expr = ...  # Your throughput formula

# Optimize over configuration space
optimizer = OptimizerInterface(objective=throughput_expr, maximize=True)
result = optimizer.grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
})

print(f"Optimal config: {result.optimal_values}")
```

## Project Structure

```
sim/
├── config/          # Configuration dataclasses
├── core/            # Simulation engine, events, state
├── symbolic/        # SymVal and symbolic combinators
├── workers/         # Prefill and decode workers
├── schedulers/      # Continuous batching scheduler
├── cache/           # Radix cache implementation
├── routing/         # Cache-aware router
├── parallel/        # TP/DP/EP group abstractions
├── network/         # Interconnect and KV transfer
├── kernels/         # Attention, MLP, MoE kernels
├── models/          # Operator graph templates
├── metrics/         # Metric collection and symbolic throughput
├── optimizer/       # Scipy/grid search interface
└── workload/        # Synthetic and trace-based workloads
```

## Examples

See the `examples/` directory for:
- `basic_simulation.py` - Full simulation workflow
- `optimize_pd_ratio.py` - P/D ratio optimization with symbolic expressions
- `sweep_chunk_size.py` - Chunk size analysis and cache effects

## Running Tests

```bash
python -m pytest tests/ -v
```


```

---

### AI

**Tool Action:**

**Code Changes:**
```
# SGLang LLM Inference Simulator

A discrete-event simulator for LLM inference with symbolic tracing, designed to model SGLang-style workloads with prefill/decode disaggregation, prefix caching, and parallel execution.

## Features

- **Symbolic Metric Tracing**: All latency computations produce both numeric values and SymPy expressions for optimization
- **P/D Disaggregation**: Separate prefill and decode worker pools with explicit KV transfer modeling
- **Prefix Cache (RadixAttention)**: LRU-based radix cache with cache-aware routing
- **Chunked Prefill**: Configurable chunk sizes for TTFT/throughput tradeoffs
- **Parallelism Support**: Tensor Parallelism (TP), Data Parallelism (DP), Expert Parallelism (EP)
- **MoE Models**: Full support for Mixture-of-Experts with all-to-all communication
- **Configurable Interconnects**: NVLink, InfiniBand, Ethernet with proper latency/bandwidth modeling
- **Multi-Objective Optimization**: Pareto-optimal search with weighted-sum and epsilon-constraint methods
- **Constraint Handling**: Memory limits, SLO targets, cost budgets, GPU constraints
- **Compute Graph Visualization**: DOT and Mermaid diagrams for expression trees and metric dependencies

## Symbolic Metrics Available

| Metric | Description |
|--------|-------------|
| `throughput` | Tokens per second |
| `ttft` | Time to First Token |
| `tpot` | Time per Output Token |
| `e2e_latency` | End-to-end request latency |
| `peak_memory` | Peak GPU memory usage |
| `compute_utilization` | GPU compute utilization |
| `memory_bandwidth_utilization` | Memory bandwidth utilization |
| `network_utilization` | Network bandwidth utilization |
| `cost_per_token` | Cost efficiency metric |
| `power_consumption` | Power consumption estimate |

## Installation

```bash
cd sglang_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# For visualization support
pip install -e ".[viz]"
```

## Quick Start

```python
from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution

# Configure simulation
model = ModelConfig.llama_7b()
cluster = ClusterConfig.single_node_8gpu()
scheduler = SchedulerConfig.default()
cache = CacheConfig.h100_default()

# Create engine and run
engine = SimulationEngine(model, cluster, scheduler, cache)
# ... initialize workers, router, etc.
result = engine.run_until_idle()

print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
```

## Symbolic Throughput Optimization

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.optimizer.interface import OptimizerInterface

symbols = ConfigSymbols()

# Build symbolic throughput expression
throughput_expr = ...  # Your throughput formula

# Optimize over configuration space
optimizer = OptimizerInterface(objective=throughput_expr, maximize=True)
result = optimizer.grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
})

print(f"Optimal config: {result.optimal_values}")
```

## Project Structure

```
sim/
├── config/          # Configuration dataclasses
├── core/            # Simulation engine, events, state
├── symbolic/        # SymVal and symbolic combinators
├── workers/         # Prefill and decode workers
├── schedulers/      # Continuous batching scheduler
├── cache/           # Radix cache implementation
├── routing/         # Cache-aware router
├── parallel/        # TP/DP/EP group abstractions
├── network/         # Interconnect and KV transfer
├── kernels/         # Attention, MLP, MoE kernels
├── models/          # Operator graph templates
├── metrics/         # Metric collection and symbolic throughput
├── optimizer/       # Scipy/grid search interface
└── workload/        # Synthetic and trace-based workloads
```

## Multi-Objective Optimization with Constraints

```python
from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer.interface import (
    MultiObjectiveOptimizer,
    ObjectiveSpec,
    ObjectiveDirection,
)

symbols = ConfigSymbols()
metric_builder = SymbolicMetricBuilder(symbols)
metrics = metric_builder.build_all_expressions()

# Define constraints
constraints = ConstraintBuilder(symbols)
constraints.add_memory_constraint(80 * 1024**3)  # 80GB H100
constraints.add_ttft_slo(0.5)  # 500ms TTFT
constraints.add_tpot_slo(0.05)  # 50ms TPOT
constraints.add_total_gpu_constraint(16)

# Multi-objective optimization
optimizer = MultiObjectiveOptimizer(
    objectives=[
        ObjectiveSpec("throughput", metrics.throughput, ObjectiveDirection.MAXIMIZE),
        ObjectiveSpec("cost", metrics.cost_per_token, ObjectiveDirection.MINIMIZE),
        ObjectiveSpec("latency", metrics.e2e_latency, ObjectiveDirection.MINIMIZE),
    ],
    constraints=constraints.get_all_constraints(),
)

# Find Pareto-optimal configurations
result = optimizer.pareto_grid_search({
    symbols.N_p: [1, 2, 4, 8],
    symbols.N_d: [1, 2, 4, 8],
    symbols.chunk_size: [4096, 8192, 16384],
    symbols.batch_cap_requests: [64, 128, 256],
})

print(f"Found {len(result.pareto_front)} Pareto-optimal solutions")
```

## Compute Graph Visualization

```python
from sim.visualization import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)

# Visualize expression tree
viz = ExpressionGraphVisualizer()
viz.build_graph(metrics.throughput, "Throughput")
dot_code = viz.to_dot("Throughput Expression")  # For Graphviz
mermaid = viz.to_mermaid("Throughput Expression")  # For Markdown

# Visualize metric dependencies
dep_graph = MetricDependencyGraph()
dep_graph.add_metrics(metrics.as_dict())
print(dep_graph.find_shared_dependencies())  # Variables affecting multiple metrics
```

## Memory Analysis

```python
from sim.metrics.memory import SymbolicMemoryModel

mem_model = SymbolicMemoryModel(symbols)

# Get symbolic expression for peak memory
peak_mem = mem_model.symbolic_peak_memory()

# Evaluate for specific configuration
config = {symbols.batch_cap_requests: 256, symbols.tp_size: 4, ...}
memory_gb = float(peak_mem.subs(list(config.items()))) / 1e9
```

## Examples

See the `examples/` directory for:
- `basic_simulation.py` - Full simulation workflow
- `optimize_pd_ratio.py` - P/D ratio optimization with symbolic expressions
- `sweep_chunk_size.py` - Chunk size analysis and cache effects
- `multi_objective_optimization.py` - Pareto optimization with constraints and visualization

## Running Tests

```bash
python -m pytest tests/ -v
```


```

---

### AI

**Tool Action:**

**Code Changes:**
```
import pytest
import sympy

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder, MetricExpressions
from sim.metrics.memory import SymbolicMemoryModel
from sim.metrics.constraints import ConstraintBuilder, ConstraintType


class TestSymbolicMetricBuilder:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.builder = SymbolicMetricBuilder(self.symbols)

    def test_ttft_expression_is_sympy_expr(self):
        ttft = self.builder.ttft_expression()
        assert isinstance(ttft, sympy.Expr)

    def test_tpot_expression_is_sympy_expr(self):
        tpot = self.builder.tpot_expression()
        assert isinstance(tpot, sympy.Expr)

    def test_e2e_latency_expression_is_sympy_expr(self):
        e2e = self.builder.e2e_latency_expression()
        assert isinstance(e2e, sympy.Expr)

    def test_throughput_expression_is_sympy_expr(self):
        throughput = self.builder.throughput_expression()
        assert isinstance(throughput, sympy.Expr)

    def test_peak_memory_expression_is_sympy_expr(self):
        peak_mem = self.builder.peak_memory_expression()
        assert isinstance(peak_mem, sympy.Expr)

    def test_build_all_expressions(self):
        metrics = self.builder.build_all_expressions()
        assert isinstance(metrics, MetricExpressions)
        assert metrics.ttft is not None
        assert metrics.tpot is not None
        assert metrics.throughput is not None
        assert metrics.peak_memory is not None

    def test_expression_caching(self):
        ttft1 = self.builder.ttft_expression()
        ttft2 = self.builder.ttft_expression()
        assert ttft1 is ttft2

    def test_cache_clear(self):
        self.builder.ttft_expression()
        assert "ttft" in self.builder._cache
        self.builder.clear_cache()
        assert "ttft" not in self.builder._cache

    def test_get_expression_by_name(self):
        ttft = self.builder.get_expression("ttft")
        assert isinstance(ttft, sympy.Expr)

    def test_get_expression_unknown_raises(self):
        with pytest.raises(ValueError):
            self.builder.get_expression("unknown_metric")


class TestSymbolicMemoryModel:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.model = SymbolicMemoryModel(self.symbols)

    def test_symbolic_peak_memory_is_sympy_expr(self):
        peak_mem = self.model.symbolic_peak_memory()
        assert isinstance(peak_mem, sympy.Expr)

    def test_memory_constraint_is_sympy_expr(self):
        constraint = self.model.memory_constraint(80e9)
        assert isinstance(constraint, sympy.Expr)

    def test_peak_memory_scales_with_batch(self):
        s = self.symbols
        peak_mem = self.model.symbolic_peak_memory()
        
        base_params = {
            s.num_layers: 32,
            s.hidden_dim: 4096,
            s.num_heads: 32,
            s.head_dim: 128,
            s.num_experts: 1,
            s.chunk_size: 8192,
            s.avg_prompt_len: 512,
            s.avg_output_len: 128,
            s.tp_size: 1,
        }
        
        mem_batch_64 = float(peak_mem.subs({**base_params, s.batch_cap_requests: 64}.items()))
        mem_batch_128 = float(peak_mem.subs({**base_params, s.batch_cap_requests: 128}.items()))
        
        assert mem_batch_128 > mem_batch_64


class TestConstraintBuilder:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.builder = ConstraintBuilder(self.symbols)

    def test_add_memory_constraint(self):
        constraint = self.builder.add_memory_constraint(80e9)
        assert constraint.name == "memory_limit"
        assert constraint.type == ConstraintType.INEQUALITY

    def test_add_ttft_slo(self):
        constraint = self.builder.add_ttft_slo(0.5)
        assert constraint.name == "ttft_slo"
        assert "500ms" in constraint.description

    def test_add_tpot_slo(self):
        constraint = self.builder.add_tpot_slo(0.05)
        assert constraint.name == "tpot_slo"
        assert "50ms" in constraint.description

    def test_add_min_throughput(self):
        constraint = self.builder.add_min_throughput(1000)
        assert constraint.name == "min_throughput"

    def test_add_total_gpu_constraint(self):
        constraint = self.builder.add_total_gpu_constraint(16)
        assert constraint.name == "max_gpus"
        assert "16" in constraint.description

    def test_get_all_constraints(self):
        self.builder.add_memory_constraint(80e9)
        self.builder.add_ttft_slo(0.5)
        constraints = self.builder.get_all_constraints()
        assert len(constraints) == 2

    def test_summarize(self):
        self.builder.add_memory_constraint(80e9)
        summary = self.builder.summarize()
        assert "memory_limit" in summary

```

---

### AI

**Tool Action:**

**Code Changes:**
```
import pytest
import sympy
from sympy import Symbol

from sim.visualization.graph import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)


class TestExpressionGraphVisualizer:
    def setup_method(self):
        self.viz = ExpressionGraphVisualizer()

    def test_simple_expression(self):
        x = Symbol("x")
        y = Symbol("y")
        expr = x + y
        
        self.viz.build_graph(expr, "sum")
        assert len(self.viz._nodes) > 0
        assert len(self.viz._edges) > 0

    def test_complex_expression(self):
        x, y, z = sympy.symbols("x y z")
        expr = sympy.Max(x * y, z + 2)
        
        self.viz.build_graph(expr, "complex")
        dot = self.viz.to_dot("Test Expression")
        
        assert "digraph G" in dot
        assert "Max" in dot

    def test_to_dot_format(self):
        x = Symbol("x")
        expr = x * 2 + 1
        
        self.viz.build_graph(expr)
        dot = self.viz.to_dot()
        
        assert dot.startswith("digraph G {")
        assert "}" in dot
        assert "node [" in dot

    def test_to_mermaid_format(self):
        x = Symbol("x")
        expr = x + 1
        
        self.viz.build_graph(expr)
        mermaid = self.viz.to_mermaid()
        
        assert "```mermaid" in mermaid
        assert "flowchart TB" in mermaid
        assert "```" in mermaid

    def test_reset_clears_state(self):
        x = Symbol("x")
        self.viz.build_graph(x + 1)
        assert len(self.viz._nodes) > 0
        
        self.viz.reset()
        assert len(self.viz._nodes) == 0
        assert len(self.viz._edges) == 0


class TestMetricDependencyGraph:
    def setup_method(self):
        self.graph = MetricDependencyGraph()

    def test_add_metric(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("test", x + y)
        
        assert "test" in self.graph.nodes
        assert "x" in self.graph.nodes["test"]
        assert "y" in self.graph.nodes["test"]

    def test_add_multiple_metrics(self):
        x, y, z = sympy.symbols("x y z")
        self.graph.add_metrics({
            "metric1": x + y,
            "metric2": y + z,
        })
        
        assert len(self.graph.nodes) == 2
        assert len(self.graph.edges) == 4

    def test_find_shared_dependencies(self):
        x, y, z = sympy.symbols("x y z")
        self.graph.add_metrics({
            "metric1": x + y,
            "metric2": y + z,
            "metric3": z * 2,
        })
        
        shared = self.graph.find_shared_dependencies()
        assert "y" in shared
        assert "z" in shared
        assert "x" not in shared

    def test_to_dot_format(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("throughput", x * y)
        
        dot = self.graph.to_dot()
        assert "digraph" in dot
        assert "throughput" in dot

    def test_to_mermaid_format(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("latency", x + y)
        
        mermaid = self.graph.to_mermaid()
        assert "mermaid" in mermaid
        assert "flowchart" in mermaid


class TestRenderFunctions:
    def test_render_expression_tree_dot(self):
        x = Symbol("x")
        dot = render_expression_tree(x + 1, "Test", "dot")
        assert "digraph" in dot

    def test_render_expression_tree_mermaid(self):
        x = Symbol("x")
        mermaid = render_expression_tree(x + 1, "Test", "mermaid")
        assert "mermaid" in mermaid

    def test_render_metric_dependencies_dot(self):
        x, y = sympy.symbols("x y")
        metrics = {"m1": x, "m2": y}
        dot = render_metric_dependencies(metrics, output_format="dot")
        assert "digraph" in dot

    def test_render_metric_dependencies_mermaid(self):
        x, y = sympy.symbols("x y")
        metrics = {"m1": x, "m2": y}
        mermaid = render_metric_dependencies(metrics, output_format="mermaid")
        assert "mermaid" in mermaid

```

---

