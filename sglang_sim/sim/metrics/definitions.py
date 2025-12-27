from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import statistics

if TYPE_CHECKING:
    from sim.core.request import Request


@dataclass
class RequestMetrics:
    request_id: int
    ttft: float
    tpot: float
    e2e_latency: float
    queue_time: float
    prefill_time: float
    transfer_time: float
    decode_time: float
    prompt_len: int
    output_len: int


class MetricCollector:
    def __init__(self):
        self.request_metrics: list[RequestMetrics] = []
        self.total_requests: int = 0
        self.completed_requests: int = 0
        self.total_tokens: int = 0

        self._ttfts: list[float] = []
        self._tpots: list[float] = []
        self._e2e_latencies: list[float] = []
        self._queue_times: list[float] = []

    def record_request_complete(self, request: Request) -> None:
        self.completed_requests += 1
        self.total_tokens += request.tokens_generated

        if request.ttft is not None:
            self._ttfts.append(request.ttft)

        if request.tpot is not None:
            self._tpots.append(request.tpot)

        if request.e2e_latency is not None:
            self._e2e_latencies.append(request.e2e_latency)

        metrics = RequestMetrics(
            request_id=request.id,
            ttft=request.ttft or 0.0,
            tpot=request.tpot or 0.0,
            e2e_latency=request.e2e_latency or 0.0,
            queue_time=0.0,
            prefill_time=0.0,
            transfer_time=0.0,
            decode_time=0.0,
            prompt_len=request.prompt_len,
            output_len=request.tokens_generated,
        )
        self.request_metrics.append(metrics)

    def record_request_arrival(self, request: Request) -> None:
        self.total_requests += 1

    def throughput(self, elapsed_time: float) -> float:
        if elapsed_time <= 0:
            return 0.0
        return self.total_tokens / elapsed_time

    def avg_ttft(self) -> float:
        if not self._ttfts:
            return 0.0
        return statistics.mean(self._ttfts)

    def avg_tpot(self) -> float:
        if not self._tpots:
            return 0.0
        return statistics.mean(self._tpots)

    def avg_e2e_latency(self) -> float:
        if not self._e2e_latencies:
            return 0.0
        return statistics.mean(self._e2e_latencies)

    def percentile_ttft(self, p: float) -> float:
        if not self._ttfts:
            return 0.0
        return self._percentile(self._ttfts, p)

    def percentile_tpot(self, p: float) -> float:
        if not self._tpots:
            return 0.0
        return self._percentile(self._tpots, p)

    def percentile_e2e(self, p: float) -> float:
        if not self._e2e_latencies:
            return 0.0
        return self._percentile(self._e2e_latencies, p)

    def _percentile(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        if f == c:
            return sorted_data[f]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "total_tokens": self.total_tokens,
            "avg_ttft": self.avg_ttft(),
            "avg_tpot": self.avg_tpot(),
            "avg_e2e_latency": self.avg_e2e_latency(),
            "p50_ttft": self.percentile_ttft(0.5),
            "p99_ttft": self.percentile_ttft(0.99),
            "p50_tpot": self.percentile_tpot(0.5),
            "p99_tpot": self.percentile_tpot(0.99),
        }

