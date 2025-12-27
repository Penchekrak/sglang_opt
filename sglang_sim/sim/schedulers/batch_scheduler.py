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

