import pytest
import sympy

from sim.symbolic.symbols import ConfigSymbols
from sim.metrics.expressions import SymbolicMetricBuilder, MetricExpressions
from sim.metrics.memory import SymbolicMemoryModel
from sim.metrics.constraints import ConstraintBuilder, ConstraintType


class TestSymbolicMetricBuilder:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.builder = SymbolicMetricBuilder(self.symbols)

    def test_ttft_expression_is_sympy_expr(self):
        ttft = self.builder.ttft_expression()
        assert isinstance(ttft, sympy.Expr)

    def test_tpot_expression_is_sympy_expr(self):
        tpot = self.builder.tpot_expression()
        assert isinstance(tpot, sympy.Expr)

    def test_e2e_latency_expression_is_sympy_expr(self):
        e2e = self.builder.e2e_latency_expression()
        assert isinstance(e2e, sympy.Expr)

    def test_throughput_expression_is_sympy_expr(self):
        throughput = self.builder.throughput_expression()
        assert isinstance(throughput, sympy.Expr)

    def test_peak_memory_expression_is_sympy_expr(self):
        peak_mem = self.builder.peak_memory_expression()
        assert isinstance(peak_mem, sympy.Expr)

    def test_build_all_expressions(self):
        metrics = self.builder.build_all_expressions()
        assert isinstance(metrics, MetricExpressions)
        assert metrics.ttft is not None
        assert metrics.tpot is not None
        assert metrics.throughput is not None
        assert metrics.peak_memory is not None

    def test_expression_caching(self):
        ttft1 = self.builder.ttft_expression()
        ttft2 = self.builder.ttft_expression()
        assert ttft1 is ttft2

    def test_cache_clear(self):
        self.builder.ttft_expression()
        assert "ttft" in self.builder._cache
        self.builder.clear_cache()
        assert "ttft" not in self.builder._cache

    def test_get_expression_by_name(self):
        ttft = self.builder.get_expression("ttft")
        assert isinstance(ttft, sympy.Expr)

    def test_get_expression_unknown_raises(self):
        with pytest.raises(ValueError):
            self.builder.get_expression("unknown_metric")


class TestSymbolicMemoryModel:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.model = SymbolicMemoryModel(self.symbols)

    def test_symbolic_peak_memory_is_sympy_expr(self):
        peak_mem = self.model.symbolic_peak_memory()
        assert isinstance(peak_mem, sympy.Expr)

    def test_memory_constraint_is_sympy_expr(self):
        constraint = self.model.memory_constraint(80e9)
        assert isinstance(constraint, sympy.Expr)

    def test_peak_memory_scales_with_batch(self):
        s = self.symbols
        peak_mem = self.model.symbolic_peak_memory()
        
        base_params = {
            s.num_layers: 32,
            s.hidden_dim: 4096,
            s.num_heads: 32,
            s.head_dim: 128,
            s.num_experts: 1,
            s.chunk_size: 8192,
            s.avg_prompt_len: 512,
            s.avg_output_len: 128,
            s.tp_size: 1,
        }
        
        mem_batch_64 = float(peak_mem.subs({**base_params, s.batch_cap_requests: 64}.items()))
        mem_batch_128 = float(peak_mem.subs({**base_params, s.batch_cap_requests: 128}.items()))
        
        assert mem_batch_128 > mem_batch_64


class TestConstraintBuilder:
    def setup_method(self):
        self.symbols = ConfigSymbols()
        self.builder = ConstraintBuilder(self.symbols)

    def test_add_memory_constraint(self):
        constraint = self.builder.add_memory_constraint(80e9)
        assert constraint.name == "memory_limit"
        assert constraint.type == ConstraintType.INEQUALITY

    def test_add_ttft_slo(self):
        constraint = self.builder.add_ttft_slo(0.5)
        assert constraint.name == "ttft_slo"
        assert "500ms" in constraint.description

    def test_add_tpot_slo(self):
        constraint = self.builder.add_tpot_slo(0.05)
        assert constraint.name == "tpot_slo"
        assert "50ms" in constraint.description

    def test_add_min_throughput(self):
        constraint = self.builder.add_min_throughput(1000)
        assert constraint.name == "min_throughput"

    def test_add_total_gpu_constraint(self):
        constraint = self.builder.add_total_gpu_constraint(16)
        assert constraint.name == "max_gpus"
        assert "16" in constraint.description

    def test_get_all_constraints(self):
        self.builder.add_memory_constraint(80e9)
        self.builder.add_ttft_slo(0.5)
        constraints = self.builder.get_all_constraints()
        assert len(constraints) == 2

    def test_summarize(self):
        self.builder.add_memory_constraint(80e9)
        summary = self.builder.summarize()
        assert "memory_limit" in summary

