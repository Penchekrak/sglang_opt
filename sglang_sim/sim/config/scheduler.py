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

