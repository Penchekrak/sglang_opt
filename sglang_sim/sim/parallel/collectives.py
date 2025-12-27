from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING

from sim.symbolic.expr import SymVal, sym_const, sym_add

if TYPE_CHECKING:
    from sim.config.cluster import InterconnectConfig


class CollectiveOp(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


def collective_cost(
    op: CollectiveOp,
    message_bytes: int,
    num_participants: int,
    bandwidth_bytes_per_sec: float,
    latency_seconds: float,
) -> SymVal:
    if num_participants <= 1:
        return sym_const(0.0, f"t_{op.value}_trivial")

    if op == CollectiveOp.ALL_REDUCE:
        effective_bytes = 2 * message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.ALL_GATHER:
        effective_bytes = message_bytes * (num_participants - 1)
    elif op == CollectiveOp.REDUCE_SCATTER:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.ALL_TO_ALL:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    elif op == CollectiveOp.BROADCAST:
        effective_bytes = message_bytes * (num_participants - 1) / num_participants
    else:
        effective_bytes = message_bytes

    alpha = latency_seconds
    beta_inv = bandwidth_bytes_per_sec

    log_p = 1
    p = num_participants
    while p > 1:
        p //= 2
        log_p += 1

    total_time = log_p * alpha + effective_bytes / beta_inv

    return sym_const(total_time, f"t_{op.value}")


def ring_all_reduce_cost(
    message_bytes: int,
    num_participants: int,
    bandwidth_bytes_per_sec: float,
    latency_seconds: float,
) -> SymVal:
    if num_participants <= 1:
        return sym_const(0.0, "t_ring_ar_trivial")

    n = num_participants
    effective_bytes = 2 * message_bytes * (n - 1) / n
    total_time = 2 * (n - 1) * latency_seconds + effective_bytes / bandwidth_bytes_per_sec

    return sym_const(total_time, "t_ring_all_reduce")

