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

