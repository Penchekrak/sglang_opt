from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add
from sim.network.interconnect import InterconnectModel

if TYPE_CHECKING:
    from sim.core.request import KVHandle
    from sim.config.cluster import ClusterConfig


@dataclass
class KVTransfer:
    request_id: int
    kv_bytes: int
    source_worker: int
    dest_worker: int
    start_time: float
    estimated_end_time: float


class KVTransferManager:
    def __init__(
        self,
        intra_node_interconnect: InterconnectModel,
        inter_node_interconnect: InterconnectModel,
        gpus_per_node: int = 8,
    ):
        self.intra_node_interconnect = intra_node_interconnect
        self.inter_node_interconnect = inter_node_interconnect
        self.gpus_per_node = gpus_per_node

        self.pending_transfers: dict[int, KVTransfer] = {}
        self.completed_transfers: dict[int, KVTransfer] = {}

    def initiate_transfer(
        self,
        kv_handle: KVHandle,
        cluster_config: ClusterConfig,
    ) -> SymVal:
        source_node = kv_handle.source_worker_id // self.gpus_per_node
        dest_node = kv_handle.dest_worker_id // self.gpus_per_node

        is_intra_node = source_node == dest_node

        serialization_overhead = 0.0001  # 100 microseconds
        deserialization_overhead = 0.0001

        if is_intra_node:
            transfer_time = self.intra_node_interconnect.transfer_time(kv_handle.kv_bytes)
        else:
            transfer_time = self.inter_node_interconnect.transfer_time(kv_handle.kv_bytes)

        overhead_sym = sym_const(serialization_overhead + deserialization_overhead, "t_serde")
        total_time = sym_add(transfer_time, overhead_sym)

        return total_time

    def is_same_node(self, worker_a: int, worker_b: int) -> bool:
        return worker_a // self.gpus_per_node == worker_b // self.gpus_per_node

    @classmethod
    def from_cluster_config(cls, cluster_config: ClusterConfig) -> KVTransferManager:
        intra = InterconnectModel(cluster_config.intra_node_interconnect)
        inter = InterconnectModel(cluster_config.inter_node_interconnect)
        return cls(
            intra_node_interconnect=intra,
            inter_node_interconnect=inter,
            gpus_per_node=cluster_config.gpus_per_worker,
        )

