from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import sympy
from sympy import Symbol

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder

if TYPE_CHECKING:
    pass


class ConstraintType(Enum):
    INEQUALITY = "ineq"
    EQUALITY = "eq"


@dataclass
class Constraint:
    name: str
    expr: sympy.Expr
    type: ConstraintType
    description: str = ""
    
    def is_satisfied(self, values: dict[Symbol, float]) -> bool:
        evaluated = float(self.expr.subs(list(values.items())))
        if self.type == ConstraintType.EQUALITY:
            return abs(evaluated) < 1e-6
        else:
            return evaluated >= 0
    
    def violation(self, values: dict[Symbol, float]) -> float:
        evaluated = float(self.expr.subs(list(values.items())))
        if self.type == ConstraintType.EQUALITY:
            return abs(evaluated)
        else:
            return max(0, -evaluated)


class ConstraintBuilder:
    def __init__(self, symbols: ConfigSymbols | None = None):
        self.symbols = symbols or ConfigSymbols()
        self.metric_builder = SymbolicMetricBuilder(self.symbols)
        self.constraints: list[Constraint] = []

    def add_memory_constraint(
        self,
        gpu_memory_bytes: int,
        safety_margin: float = 0.9,
    ) -> Constraint:
        peak_memory = self.metric_builder.peak_memory_expression()
        available = gpu_memory_bytes * safety_margin
        
        constraint = Constraint(
            name="memory_limit",
            expr=available - peak_memory,
            type=ConstraintType.INEQUALITY,
            description=f"Peak memory <= {available / 1e9:.1f} GB",
        )
        self.constraints.append(constraint)
        return constraint

    def add_ttft_slo(self, max_ttft_seconds: float) -> Constraint:
        ttft = self.metric_builder.ttft_expression()
        
        constraint = Constraint(
            name="ttft_slo",
            expr=max_ttft_seconds - ttft,
            type=ConstraintType.INEQUALITY,
            description=f"TTFT <= {max_ttft_seconds * 1000:.0f}ms",
        )
        self.constraints.append(constraint)
        return constraint

    def add_tpot_slo(self, max_tpot_seconds: float) -> Constraint:
        tpot = self.metric_builder.tpot_expression()
        
        constraint = Constraint(
            name="tpot_slo",
            expr=max_tpot_seconds - tpot,
            type=ConstraintType.INEQUALITY,
            description=f"TPOT <= {max_tpot_seconds * 1000:.0f}ms",
        )
        self.constraints.append(constraint)
        return constraint

    def add_e2e_latency_slo(self, max_latency_seconds: float) -> Constraint:
        e2e = self.metric_builder.e2e_latency_expression()
        
        constraint = Constraint(
            name="e2e_latency_slo",
            expr=max_latency_seconds - e2e,
            type=ConstraintType.INEQUALITY,
            description=f"E2E Latency <= {max_latency_seconds:.1f}s",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_throughput(self, min_tokens_per_sec: float) -> Constraint:
        throughput = self.metric_builder.throughput_expression()
        
        constraint = Constraint(
            name="min_throughput",
            expr=throughput - min_tokens_per_sec,
            type=ConstraintType.INEQUALITY,
            description=f"Throughput >= {min_tokens_per_sec:.0f} tok/s",
        )
        self.constraints.append(constraint)
        return constraint

    def add_cost_budget(
        self,
        max_cost_per_1k_tokens: float,
        gpu_cost_per_hour: float = 2.0,
    ) -> Constraint:
        cost_per_token = self.metric_builder.cost_per_token_expression(gpu_cost_per_hour)
        max_cost_per_token = max_cost_per_1k_tokens / 1000
        
        constraint = Constraint(
            name="cost_budget",
            expr=max_cost_per_token - cost_per_token,
            type=ConstraintType.INEQUALITY,
            description=f"Cost <= ${max_cost_per_1k_tokens:.4f}/1K tokens",
        )
        self.constraints.append(constraint)
        return constraint

    def add_power_budget(self, max_power_watts: float) -> Constraint:
        power = self.metric_builder.power_consumption_expression()
        
        constraint = Constraint(
            name="power_budget",
            expr=max_power_watts - power,
            type=ConstraintType.INEQUALITY,
            description=f"Power <= {max_power_watts:.0f}W",
        )
        self.constraints.append(constraint)
        return constraint

    def add_total_gpu_constraint(self, max_gpus: int) -> Constraint:
        s = self.symbols
        total_gpus = (s.N_p + s.N_d) * s.tp_size
        
        constraint = Constraint(
            name="max_gpus",
            expr=max_gpus - total_gpus,
            type=ConstraintType.INEQUALITY,
            description=f"Total GPUs <= {max_gpus}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_prefill_workers(self, min_workers: int) -> Constraint:
        s = self.symbols
        
        constraint = Constraint(
            name="min_prefill_workers",
            expr=s.N_p - min_workers,
            type=ConstraintType.INEQUALITY,
            description=f"N_p >= {min_workers}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_min_decode_workers(self, min_workers: int) -> Constraint:
        s = self.symbols
        
        constraint = Constraint(
            name="min_decode_workers",
            expr=s.N_d - min_workers,
            type=ConstraintType.INEQUALITY,
            description=f"N_d >= {min_workers}",
        )
        self.constraints.append(constraint)
        return constraint

    def add_utilization_constraint(
        self,
        min_compute_util: float = 0.0,
        min_memory_util: float = 0.0,
    ) -> list[Constraint]:
        added = []
        
        if min_compute_util > 0:
            compute_util = self.metric_builder.compute_utilization_expression()
            constraint = Constraint(
                name="min_compute_util",
                expr=compute_util - min_compute_util,
                type=ConstraintType.INEQUALITY,
                description=f"Compute utilization >= {min_compute_util:.0%}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        if min_memory_util > 0:
            mem_util = self.metric_builder.memory_bandwidth_utilization_expression()
            constraint = Constraint(
                name="min_memory_util",
                expr=mem_util - min_memory_util,
                type=ConstraintType.INEQUALITY,
                description=f"Memory BW utilization >= {min_memory_util:.0%}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        return added

    def add_pd_ratio_constraint(
        self,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> list[Constraint]:
        s = self.symbols
        added = []
        
        if min_ratio is not None:
            constraint = Constraint(
                name="min_pd_ratio",
                expr=s.N_p / s.N_d - min_ratio,
                type=ConstraintType.INEQUALITY,
                description=f"N_p/N_d >= {min_ratio:.2f}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        if max_ratio is not None:
            constraint = Constraint(
                name="max_pd_ratio",
                expr=max_ratio - s.N_p / s.N_d,
                type=ConstraintType.INEQUALITY,
                description=f"N_p/N_d <= {max_ratio:.2f}",
            )
            self.constraints.append(constraint)
            added.append(constraint)
        
        return added

    def add_chunk_size_bounds(
        self,
        min_chunk: int,
        max_chunk: int,
    ) -> list[Constraint]:
        s = self.symbols
        
        constraints = [
            Constraint(
                name="min_chunk_size",
                expr=s.chunk_size - min_chunk,
                type=ConstraintType.INEQUALITY,
                description=f"chunk_size >= {min_chunk}",
            ),
            Constraint(
                name="max_chunk_size",
                expr=max_chunk - s.chunk_size,
                type=ConstraintType.INEQUALITY,
                description=f"chunk_size <= {max_chunk}",
            ),
        ]
        self.constraints.extend(constraints)
        return constraints

    def add_batch_size_bounds(
        self,
        min_batch: int,
        max_batch: int,
    ) -> list[Constraint]:
        s = self.symbols
        
        constraints = [
            Constraint(
                name="min_batch_size",
                expr=s.batch_cap_requests - min_batch,
                type=ConstraintType.INEQUALITY,
                description=f"batch_size >= {min_batch}",
            ),
            Constraint(
                name="max_batch_size",
                expr=max_batch - s.batch_cap_requests,
                type=ConstraintType.INEQUALITY,
                description=f"batch_size <= {max_batch}",
            ),
        ]
        self.constraints.extend(constraints)
        return constraints

    def get_all_constraints(self) -> list[Constraint]:
        return self.constraints.copy()

    def get_scipy_constraints(self) -> list[dict]:
        return [
            {
                "type": c.type.value,
                "fun": lambda x, expr=c.expr, syms=self.symbols.decision_vars(): float(
                    expr.subs(list(zip(syms, x)))
                ),
            }
            for c in self.constraints
        ]

    def check_all_constraints(self, values: dict[Symbol, float]) -> dict[str, tuple[bool, float]]:
        results = {}
        for c in self.constraints:
            satisfied = c.is_satisfied(values)
            violation = c.violation(values)
            results[c.name] = (satisfied, violation)
        return results

    def summarize(self) -> str:
        lines = ["Constraints:"]
        for c in self.constraints:
            lines.append(f"  - {c.name}: {c.description}")
        return "\n".join(lines)

