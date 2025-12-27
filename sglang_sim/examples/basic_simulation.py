#!/usr/bin/env python3
"""Basic simulation example showing the core workflow of the LLM inference simulator."""

import sys
sys.path.insert(0, "..")

from sim.config.model import ModelConfig
from sim.config.cluster import ClusterConfig, GPUSpec, InterconnectConfig
from sim.config.scheduler import SchedulerConfig
from sim.config.cache import CacheConfig
from sim.core.engine import SimulationEngine
from sim.core.events import RequestArrival
from sim.workers.prefill_worker import PrefillWorker
from sim.workers.decode_worker import DecodeWorker
from sim.routing.router import Router, RoutingPolicy
from sim.network.kv_transfer import KVTransferManager
from sim.cache.radix_cache import RadixCache
from sim.metrics.definitions import MetricCollector
from sim.workload.generators import SyntheticWorkloadGenerator, Distribution


def run_basic_simulation():
    model_config = ModelConfig.llama_7b()
    cluster_config = ClusterConfig.single_node_8gpu()
    scheduler_config = SchedulerConfig.default()
    cache_config = CacheConfig.h100_default()

    engine = SimulationEngine(
        model_config=model_config,
        cluster_config=cluster_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )

    prefill_workers = []
    for i in range(cluster_config.num_prefill_workers):
        cache = RadixCache(cache_config)
        worker = PrefillWorker(worker_id=i, cache=cache)
        prefill_workers.append(worker)

    decode_workers = []
    for i in range(cluster_config.num_decode_workers):
        worker = DecodeWorker(worker_id=i)
        decode_workers.append(worker)

    router = Router(
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        prefill_policy=RoutingPolicy.CACHE_AWARE,
        decode_policy=RoutingPolicy.ROUND_ROBIN,
    )

    kv_transfer = KVTransferManager.from_cluster_config(cluster_config)
    metrics = MetricCollector()

    engine.initialize_components(
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        router=router,
        kv_transfer_manager=kv_transfer,
        metrics=metrics,
    )

    workload_gen = SyntheticWorkloadGenerator(
        arrival_rate=5.0,
        prompt_len_dist=Distribution.uniform(256, 1024),
        output_len_dist=Distribution.uniform(64, 256),
        seed=42,
    )

    requests = workload_gen.generate(duration=10.0)
    print(f"Generated {len(requests)} requests")

    for request in requests:
        engine.schedule_event(RequestArrival(time=request.arrival_time, request=request))
        metrics.record_request_arrival(request)

    result = engine.run_until_idle()

    print("\n=== Simulation Results ===")
    print(f"Total requests: {result.total_requests}")
    print(f"Completed requests: {result.completed_requests}")
    print(f"Total tokens generated: {result.total_tokens_generated}")
    print(f"Simulation time: {result.simulation_time:.4f}s")
    print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"Avg TTFT: {result.avg_ttft * 1000:.2f}ms")
    print(f"Avg TPOT: {result.avg_tpot * 1000:.4f}ms")
    print(f"Avg E2E Latency: {result.avg_e2e_latency * 1000:.2f}ms")
    print(f"P50 TTFT: {result.p50_ttft * 1000:.2f}ms")
    print(f"P99 TTFT: {result.p99_ttft * 1000:.2f}ms")

    return result


if __name__ == "__main__":
    run_basic_simulation()

