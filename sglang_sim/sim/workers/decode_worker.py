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

