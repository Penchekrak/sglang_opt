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

