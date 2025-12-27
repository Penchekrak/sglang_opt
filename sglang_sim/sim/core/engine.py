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

