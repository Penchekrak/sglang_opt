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

