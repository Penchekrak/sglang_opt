from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.request import Request, KVHandle
    from sim.workers.prefill_worker import PrefillWorker
    from sim.workers.decode_worker import DecodeWorker
    from sim.cache.radix_cache import RadixCache


class RoutingPolicy(Enum):
    ROUND_ROBIN = "round_robin"
    SHORTEST_QUEUE = "shortest_queue"
    CACHE_AWARE = "cache_aware"


@dataclass
class ApproxRadixTree:
    worker_id: int
    known_prefixes: list[tuple[list[int], int]] = field(default_factory=list)
    last_updated: float = 0.0

    def estimate_prefix_match(self, tokens: list[int]) -> tuple[int, int]:
        best_match_len = 0
        best_match_bytes = 0

        for prefix, kv_bytes in self.known_prefixes:
            match_len = 0
            for i, (a, b) in enumerate(zip(tokens, prefix)):
                if a == b:
                    match_len = i + 1
                else:
                    break

            if match_len > best_match_len:
                best_match_len = match_len
                # Estimate bytes proportionally
                if len(prefix) > 0:
                    best_match_bytes = int(kv_bytes * match_len / len(prefix))

        return best_match_len, best_match_bytes


class Router:
    def __init__(
        self,
        prefill_workers: list[PrefillWorker],
        decode_workers: list[DecodeWorker],
        prefill_policy: RoutingPolicy = RoutingPolicy.CACHE_AWARE,
        decode_policy: RoutingPolicy = RoutingPolicy.ROUND_ROBIN,
        cache_threshold: float = 0.5,
        balance_threshold: float = 2.0,
    ):
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.prefill_policy = prefill_policy
        self.decode_policy = decode_policy
        self.cache_threshold = cache_threshold
        self.balance_threshold = balance_threshold

        self.approx_trees: dict[int, ApproxRadixTree] = {
            w.worker_id: ApproxRadixTree(worker_id=w.worker_id)
            for w in prefill_workers
        }

        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

    def route_to_prefill(self, request: Request) -> int:
        if self.prefill_policy == RoutingPolicy.ROUND_ROBIN:
            return self._round_robin_prefill()
        elif self.prefill_policy == RoutingPolicy.SHORTEST_QUEUE:
            return self._shortest_queue_prefill()
        elif self.prefill_policy == RoutingPolicy.CACHE_AWARE:
            return self._cache_aware_prefill(request)
        return self._round_robin_prefill()

    def route_to_decode(self, kv_handle: KVHandle) -> int:
        if self.decode_policy == RoutingPolicy.ROUND_ROBIN:
            return self._round_robin_decode()
        elif self.decode_policy == RoutingPolicy.SHORTEST_QUEUE:
            return self._shortest_queue_decode()
        return self._round_robin_decode()

    def _round_robin_prefill(self) -> int:
        worker_id = self.prefill_rr_counter % len(self.prefill_workers)
        self.prefill_rr_counter += 1
        return worker_id

    def _round_robin_decode(self) -> int:
        worker_id = self.decode_rr_counter % len(self.decode_workers)
        self.decode_rr_counter += 1
        return worker_id

    def _shortest_queue_prefill(self) -> int:
        return min(
            range(len(self.prefill_workers)),
            key=lambda i: self.prefill_workers[i].queue_length,
        )

    def _shortest_queue_decode(self) -> int:
        return min(
            range(len(self.decode_workers)),
            key=lambda i: self.decode_workers[i].queue_length,
        )

    def _cache_aware_prefill(self, request: Request) -> int:
        queue_lengths = [w.queue_length for w in self.prefill_workers]
        min_queue = min(queue_lengths) if queue_lengths else 0
        max_queue = max(queue_lengths) if queue_lengths else 0

        if max_queue > 0 and max_queue / max(1, min_queue) > self.balance_threshold:
            return self._shortest_queue_prefill()

        best_worker = 0
        best_match_ratio = 0.0

        for worker_id, worker in enumerate(self.prefill_workers):
            if worker.cache is not None:
                matched_tokens, _ = worker.cache.match_prefix(request.prompt_tokens)
                match_ratio = matched_tokens / len(request.prompt_tokens) if request.prompt_tokens else 0
            else:
                approx_tree = self.approx_trees.get(worker_id)
                if approx_tree:
                    matched_tokens, _ = approx_tree.estimate_prefix_match(request.prompt_tokens)
                    match_ratio = matched_tokens / len(request.prompt_tokens) if request.prompt_tokens else 0
                else:
                    match_ratio = 0.0

            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_worker = worker_id

        if best_match_ratio >= self.cache_threshold:
            return best_worker

        return self._worker_with_most_capacity()

    def _worker_with_most_capacity(self) -> int:
        best_worker = 0
        best_capacity = 0

        for worker_id, worker in enumerate(self.prefill_workers):
            if worker.cache is not None:
                remaining = worker.cache.capacity - worker.cache.used_bytes
            else:
                remaining = float('inf')

            if remaining > best_capacity:
                best_capacity = remaining
                best_worker = worker_id

        return best_worker

    def update_approx_tree(self, worker_id: int, prefixes: list[tuple[list[int], int]], time: float) -> None:
        if worker_id in self.approx_trees:
            self.approx_trees[worker_id].known_prefixes = prefixes
            self.approx_trees[worker_id].last_updated = time

    def is_balanced(self) -> bool:
        queue_lengths = [w.queue_length for w in self.prefill_workers]
        if not queue_lengths or max(queue_lengths) == 0:
            return True
        return max(queue_lengths) / max(1, min(queue_lengths)) <= self.balance_threshold

