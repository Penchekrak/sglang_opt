from sim.metrics.expressions import SymbolicMetricBuilder
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer import MultiObjectiveOptimizer, ObjectiveSpec, ObjectiveDirection
from sim.visualization import render_expression_tree, MetricDependencyGraph


from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder
from sim.metrics.constraints import ConstraintBuilder
from sim.optimizer.interface import (
    MultiObjectiveOptimizer,
    ObjectiveSpec,
    ObjectiveDirection,
)

from mermaid import Mermaid

symbols = ConfigSymbols()
metric_builder = SymbolicMetricBuilder(symbols)
metrics = metric_builder.build_all_expressions()

# Define constraints
constraints = ConstraintBuilder(symbols)
constraints.add_memory_constraint(80 * 1024**3)  # 80GB H100
constraints.add_ttft_slo(0.5)  # 500ms TTFT
constraints.add_tpot_slo(0.05)  # 50ms TPOT
constraints.add_total_gpu_constraint(16)

# Multi-objective Pareto optimization
optimizer = MultiObjectiveOptimizer([
    ObjectiveSpec("throughput", metrics.throughput, ObjectiveDirection.MAXIMIZE),
    ObjectiveSpec("cost", metrics.cost_per_token, ObjectiveDirection.MINIMIZE),
], constraints=constraints.get_all_constraints())

print('metrics.throughput:\n', metrics.throughput)
for constraint in constraints.get_all_constraints():
    print(f"{constraint.name}:\n", constraint.expr > 0)

# Visualize
# mermaid_code = render_expression_tree(metrics.throughput, output_format="mermaid")
# Mermaid(mermaid_code.lstrip('```mermaid').rstrip('```')).to_svg("throughput.svg")