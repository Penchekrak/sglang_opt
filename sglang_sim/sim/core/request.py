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

