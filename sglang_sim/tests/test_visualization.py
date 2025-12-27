import pytest
import sympy
from sympy import Symbol

from sim.visualization.graph import (
    ExpressionGraphVisualizer,
    MetricDependencyGraph,
    render_expression_tree,
    render_metric_dependencies,
)


class TestExpressionGraphVisualizer:
    def setup_method(self):
        self.viz = ExpressionGraphVisualizer()

    def test_simple_expression(self):
        x = Symbol("x")
        y = Symbol("y")
        expr = x + y
        
        self.viz.build_graph(expr, "sum")
        assert len(self.viz._nodes) > 0
        assert len(self.viz._edges) > 0

    def test_complex_expression(self):
        x, y, z = sympy.symbols("x y z")
        expr = sympy.Max(x * y, z + 2)
        
        self.viz.build_graph(expr, "complex")
        dot = self.viz.to_dot("Test Expression")
        
        assert "digraph G" in dot
        assert "Max" in dot

    def test_to_dot_format(self):
        x = Symbol("x")
        expr = x * 2 + 1
        
        self.viz.build_graph(expr)
        dot = self.viz.to_dot()
        
        assert dot.startswith("digraph G {")
        assert "}" in dot
        assert "node [" in dot

    def test_to_mermaid_format(self):
        x = Symbol("x")
        expr = x + 1
        
        self.viz.build_graph(expr)
        mermaid = self.viz.to_mermaid()
        
        assert "```mermaid" in mermaid
        assert "flowchart TB" in mermaid
        assert "```" in mermaid

    def test_reset_clears_state(self):
        x = Symbol("x")
        self.viz.build_graph(x + 1)
        assert len(self.viz._nodes) > 0
        
        self.viz.reset()
        assert len(self.viz._nodes) == 0
        assert len(self.viz._edges) == 0


class TestMetricDependencyGraph:
    def setup_method(self):
        self.graph = MetricDependencyGraph()

    def test_add_metric(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("test", x + y)
        
        assert "test" in self.graph.nodes
        assert "x" in self.graph.nodes["test"]
        assert "y" in self.graph.nodes["test"]

    def test_add_multiple_metrics(self):
        x, y, z = sympy.symbols("x y z")
        self.graph.add_metrics({
            "metric1": x + y,
            "metric2": y + z,
        })
        
        assert len(self.graph.nodes) == 2
        assert len(self.graph.edges) == 4

    def test_find_shared_dependencies(self):
        x, y, z = sympy.symbols("x y z")
        self.graph.add_metrics({
            "metric1": x + y,
            "metric2": y + z,
            "metric3": z * 2,
        })
        
        shared = self.graph.find_shared_dependencies()
        assert "y" in shared
        assert "z" in shared
        assert "x" not in shared

    def test_to_dot_format(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("throughput", x * y)
        
        dot = self.graph.to_dot()
        assert "digraph" in dot
        assert "throughput" in dot

    def test_to_mermaid_format(self):
        x, y = sympy.symbols("x y")
        self.graph.add_metric("latency", x + y)
        
        mermaid = self.graph.to_mermaid()
        assert "mermaid" in mermaid
        assert "flowchart" in mermaid


class TestRenderFunctions:
    def test_render_expression_tree_dot(self):
        x = Symbol("x")
        dot = render_expression_tree(x + 1, "Test", "dot")
        assert "digraph" in dot

    def test_render_expression_tree_mermaid(self):
        x = Symbol("x")
        mermaid = render_expression_tree(x + 1, "Test", "mermaid")
        assert "mermaid" in mermaid

    def test_render_metric_dependencies_dot(self):
        x, y = sympy.symbols("x y")
        metrics = {"m1": x, "m2": y}
        dot = render_metric_dependencies(metrics, output_format="dot")
        assert "digraph" in dot

    def test_render_metric_dependencies_mermaid(self):
        x, y = sympy.symbols("x y")
        metrics = {"m1": x, "m2": y}
        mermaid = render_metric_dependencies(metrics, output_format="mermaid")
        assert "mermaid" in mermaid

