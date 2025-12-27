from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add
from sim.config.cluster import InterconnectConfig, InterconnectType

if TYPE_CHECKING:
    pass


class InterconnectModel:
    def __init__(self, config: InterconnectConfig):
        self.config = config

    def transfer_time(self, bytes_to_transfer: int) -> SymVal:
        latency = self.config.latency_seconds
        bandwidth = self.config.bandwidth_bytes_per_sec

        transfer_time = latency + bytes_to_transfer / bandwidth

        latency_sym = sym_const(latency, "L_net")
        transfer_sym = sym_const(bytes_to_transfer / bandwidth, "t_transfer")

        return sym_add(latency_sym, transfer_sym)

    def collective_time(
        self,
        message_bytes: int,
        num_participants: int,
        op: str,
    ) -> SymVal:
        bandwidth = self.config.bandwidth_bytes_per_sec
        latency = self.config.latency_seconds

        if op == "all_reduce":
            effective_bytes = 2 * message_bytes * (num_participants - 1) / num_participants
        elif op == "all_gather":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        elif op == "reduce_scatter":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        elif op == "all_to_all":
            effective_bytes = message_bytes * (num_participants - 1) / num_participants
        else:
            effective_bytes = message_bytes

        alpha_beta_time = latency * num_participants + effective_bytes / bandwidth

        return sym_const(alpha_beta_time, f"t_{op}")

    @classmethod
    def for_intra_node(cls, config: InterconnectConfig) -> InterconnectModel:
        return cls(config)

    @classmethod
    def for_inter_node(cls, config: InterconnectConfig) -> InterconnectModel:
        return cls(config)

