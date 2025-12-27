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
