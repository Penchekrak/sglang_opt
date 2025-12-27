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

