from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from sim.config.cache import CacheConfig


@dataclass
class RadixNode:
    token: int | None = None
    kv_bytes: int = 0
    last_access: float = 0.0
    children: dict[int, RadixNode] = field(default_factory=dict)
    parent: RadixNode | None = None
    depth: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_descendant_bytes(self) -> int:
        total = self.kv_bytes
        for child in self.children.values():
            total += child.total_descendant_bytes()
        return total


class RadixCache:
    def __init__(self, config: CacheConfig, current_time: float = 0.0):
        self.config = config
        self.root = RadixNode(depth=0)
        self.capacity = config.capacity_bytes
        self.used_bytes = 0
        self.current_time = current_time

        self.access_order: OrderedDict[int, RadixNode] = OrderedDict()
        self._node_id_counter = 0

    def match_prefix(self, tokens: list[int]) -> tuple[int, int]:
        matched_tokens = 0
        matched_bytes = 0
        current = self.root

        for token in tokens:
            if token in current.children:
                current = current.children[token]
                matched_tokens += 1
                matched_bytes += current.kv_bytes
                self._update_access(current)
            else:
                break

        return matched_tokens, matched_bytes

    def insert(self, tokens: list[int], total_kv_bytes: int) -> bool:
        if not tokens:
            return True

        bytes_per_token = total_kv_bytes // len(tokens) if tokens else 0
        current = self.root

        for i, token in enumerate(tokens):
            if token in current.children:
                current = current.children[token]
                self._update_access(current)
            else:
                new_node = RadixNode(
                    token=token,
                    kv_bytes=bytes_per_token,
                    last_access=self.current_time,
                    parent=current,
                    depth=current.depth + 1,
                )

                while self.used_bytes + bytes_per_token > self.capacity:
                    if not self._evict_lru():
                        return False

                current.children[token] = new_node
                self.used_bytes += bytes_per_token
                self._register_node(new_node)
                current = new_node

        return True

    def _update_access(self, node: RadixNode) -> None:
        node.last_access = self.current_time
        node_id = id(node)
        if node_id in self.access_order:
            self.access_order.move_to_end(node_id)

    def _register_node(self, node: RadixNode) -> None:
        self.access_order[id(node)] = node

    def _evict_lru(self) -> bool:
        if not self.access_order:
            return False

        candidates = [
            node for node in self.access_order.values()
            if node.is_leaf() and node.parent is not None
        ]

        if not candidates:
            return False

        oldest_node = min(candidates, key=lambda n: n.last_access)
        return self._remove_node(oldest_node)

    def _remove_node(self, node: RadixNode) -> bool:
        if node.parent is None:
            return False

        if not node.is_leaf():
            return False

        parent = node.parent
        if node.token is not None and node.token in parent.children:
            del parent.children[node.token]

        self.used_bytes -= node.kv_bytes
        node_id = id(node)
        if node_id in self.access_order:
            del self.access_order[node_id]

        return True

    def update_time(self, new_time: float) -> None:
        self.current_time = new_time

    def get_hit_rate_estimate(self) -> float:
        if self.capacity == 0:
            return 0.0
        return min(1.0, self.used_bytes / self.capacity)

    @property
    def utilization(self) -> float:
        return self.used_bytes / self.capacity if self.capacity > 0 else 0.0

    def get_all_prefixes(self) -> list[tuple[list[int], int]]:
        prefixes: list[tuple[list[int], int]] = []
        self._collect_prefixes(self.root, [], prefixes)
        return prefixes

    def _collect_prefixes(
        self,
        node: RadixNode,
        current_path: list[int],
        prefixes: list[tuple[list[int], int]],
    ) -> None:
        if node.is_leaf() and current_path:
            total_bytes = sum(
                self._get_path_bytes(current_path[:i+1])
                for i in range(len(current_path))
            )
            prefixes.append((current_path.copy(), node.kv_bytes))

        for token, child in node.children.items():
            current_path.append(token)
            self._collect_prefixes(child, current_path, prefixes)
            current_path.pop()

    def _get_path_bytes(self, path: list[int]) -> int:
        current = self.root
        for token in path:
            if token in current.children:
                current = current.children[token]
            else:
                return 0
        return current.kv_bytes

