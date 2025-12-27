from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from io import StringIO
import html

import sympy
from sympy import Symbol, Function, Add, Mul, Pow, Max, Min, ceiling, Piecewise
from sympy.printing.dot import dotprint

if TYPE_CHECKING:
    from sim.metrics.expressions import MetricExpressions
    from sim.metrics.constraints import Constraint


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str
    shape: str = "box"
    color: str = "#ffffff"
    children: list[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str = ""
    style: str = "solid"


class ExpressionGraphVisualizer:
    NODE_COLORS = {
        "symbol": "#e3f2fd",
        "constant": "#fff3e0",
        "operator": "#f3e5f5",
        "function": "#e8f5e9",
        "metric": "#ffebee",
    }
    
    NODE_SHAPES = {
        "symbol": "ellipse",
        "constant": "box",
        "operator": "diamond",
        "function": "hexagon",
        "metric": "doubleoctagon",
    }

    def __init__(self):
        self._node_counter = 0
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    def reset(self) -> None:
        self._node_counter = 0
        self._nodes.clear()
        self._edges.clear()

    def _new_node_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"

    def build_graph(self, expr: sympy.Expr, root_name: str = "result") -> str:
        self.reset()
        root_id = self._visit_expr(expr)
        
        root_node = GraphNode(
            id="root",
            label=root_name,
            node_type="metric",
            shape=self.NODE_SHAPES["metric"],
            color=self.NODE_COLORS["metric"],
        )
        self._nodes["root"] = root_node
        self._edges.append(GraphEdge(source="root", target=root_id))
        
        return root_id

    def _visit_expr(self, expr: sympy.Expr) -> str:
        if isinstance(expr, Symbol):
            return self._add_symbol_node(expr)
        elif isinstance(expr, (int, float, sympy.Integer, sympy.Float, sympy.Rational)):
            return self._add_constant_node(expr)
        elif isinstance(expr, Add):
            return self._add_operator_node("+", list(expr.args))
        elif isinstance(expr, Mul):
            return self._add_operator_node("×", list(expr.args))
        elif isinstance(expr, Pow):
            base, exp = expr.args
            return self._add_function_node(f"^{exp}", [base])
        elif isinstance(expr, Max):
            return self._add_function_node("Max", list(expr.args))
        elif isinstance(expr, Min):
            return self._add_function_node("Min", list(expr.args))
        elif isinstance(expr, ceiling):
            return self._add_function_node("⌈⌉", list(expr.args))
        elif isinstance(expr, Piecewise):
            return self._add_piecewise_node(expr)
        elif hasattr(expr, 'func') and hasattr(expr, 'args'):
            func_name = expr.func.__name__
            return self._add_function_node(func_name, list(expr.args))
        else:
            return self._add_constant_node(expr)

    def _add_symbol_node(self, sym: Symbol) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=str(sym),
            node_type="symbol",
            shape=self.NODE_SHAPES["symbol"],
            color=self.NODE_COLORS["symbol"],
        )
        return node_id

    def _add_constant_node(self, value: Any) -> str:
        node_id = self._new_node_id()
        if isinstance(value, float):
            label = f"{value:.4g}"
        else:
            label = str(value)
        
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=label,
            node_type="constant",
            shape=self.NODE_SHAPES["constant"],
            color=self.NODE_COLORS["constant"],
        )
        return node_id

    def _add_operator_node(self, op: str, args: list) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=op,
            node_type="operator",
            shape=self.NODE_SHAPES["operator"],
            color=self.NODE_COLORS["operator"],
        )
        
        for arg in args:
            child_id = self._visit_expr(arg)
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id))
        
        return node_id

    def _add_function_node(self, func_name: str, args: list) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label=func_name,
            node_type="function",
            shape=self.NODE_SHAPES["function"],
            color=self.NODE_COLORS["function"],
        )
        
        for arg in args:
            child_id = self._visit_expr(arg)
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id))
        
        return node_id

    def _add_piecewise_node(self, pw: Piecewise) -> str:
        node_id = self._new_node_id()
        self._nodes[node_id] = GraphNode(
            id=node_id,
            label="Piecewise",
            node_type="function",
            shape="house",
            color=self.NODE_COLORS["function"],
        )
        
        for expr, cond in pw.args:
            child_id = self._visit_expr(expr)
            cond_label = str(cond)[:20]
            self._nodes[node_id].children.append(child_id)
            self._edges.append(GraphEdge(source=node_id, target=child_id, label=cond_label))
        
        return node_id

    def to_dot(self, title: str = "Expression Graph") -> str:
        lines = [
            "digraph G {",
            f'    label="{title}";',
            "    labelloc=t;",
            "    rankdir=TB;",
            '    node [fontname="Helvetica", fontsize=10];',
            '    edge [fontname="Helvetica", fontsize=8];',
            "",
        ]
        
        for node in self._nodes.values():
            escaped_label = html.escape(node.label)
            lines.append(
                f'    {node.id} [label="{escaped_label}", '
                f'shape={node.shape}, style=filled, fillcolor="{node.color}"];'
            )
        
        lines.append("")
        
        for edge in self._edges:
            if edge.label:
                escaped_label = html.escape(edge.label)
                lines.append(
                    f'    {edge.source} -> {edge.target} [label="{escaped_label}"];'
                )
            else:
                lines.append(f"    {edge.source} -> {edge.target};")
        
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, title: str = "Expression Graph") -> str:
        lines = [
            "```mermaid",
            "flowchart TB",
            f"    subgraph title [{title}]",
        ]
        
        shape_map = {
            "ellipse": ("((", "))"),
            "box": ("[", "]"),
            "diamond": ("{", "}"),
            "hexagon": ("{{", "}}"),
            "doubleoctagon": ("[[", "]]"),
            "house": ("[/", "/]"),
        }
        
        for node in self._nodes.values():
            left, right = shape_map.get(node.shape, ("[", "]"))
            label = node.label.replace('"', "'")
            lines.append(f"        {node.id}{left}{label}{right}")
        
        for edge in self._edges:
            if edge.label:
                label = edge.label.replace('"', "'")
                lines.append(f'        {edge.source} -->|"{label}"| {edge.target}')
            else:
                lines.append(f"        {edge.source} --> {edge.target}")
        
        lines.append("    end")
        lines.append("```")
        return "\n".join(lines)

    def render_to_file(
        self,
        expr: sympy.Expr,
        filename: str,
        title: str = "Expression Graph",
        format: str = "png",
    ) -> str:
        self.build_graph(expr, title)
        dot_content = self.to_dot(title)
        
        dot_filename = filename.rsplit(".", 1)[0] + ".dot"
        with open(dot_filename, "w") as f:
            f.write(dot_content)
        
        return dot_filename


class MetricDependencyGraph:
    def __init__(self):
        self.nodes: dict[str, set[str]] = {}
        self.edges: list[tuple[str, str]] = []

    def add_metric(self, name: str, expr: sympy.Expr) -> None:
        symbols = {str(s) for s in expr.free_symbols}
        self.nodes[name] = symbols
        
        for sym in symbols:
            self.edges.append((sym, name))

    def add_metrics(self, metrics: dict[str, sympy.Expr]) -> None:
        for name, expr in metrics.items():
            self.add_metric(name, expr)

    def find_shared_dependencies(self) -> dict[str, list[str]]:
        symbol_to_metrics: dict[str, list[str]] = {}
        
        for metric, symbols in self.nodes.items():
            for sym in symbols:
                if sym not in symbol_to_metrics:
                    symbol_to_metrics[sym] = []
                symbol_to_metrics[sym].append(metric)
        
        return {sym: metrics for sym, metrics in symbol_to_metrics.items() if len(metrics) > 1}

    def to_dot(self, title: str = "Metric Dependencies") -> str:
        lines = [
            "digraph MetricDependencies {",
            f'    label="{title}";',
            "    labelloc=t;",
            "    rankdir=LR;",
            '    node [fontname="Helvetica"];',
            "",
            "    // Symbol nodes",
            '    subgraph cluster_symbols {',
            '        label="Configuration Variables";',
            '        style=dashed;',
            '        node [shape=ellipse, style=filled, fillcolor="#e3f2fd"];',
        ]
        
        all_symbols = set()
        for symbols in self.nodes.values():
            all_symbols.update(symbols)
        
        for sym in sorted(all_symbols):
            lines.append(f'        "{sym}";')
        
        lines.extend([
            "    }",
            "",
            "    // Metric nodes",
            '    subgraph cluster_metrics {',
            '        label="Metrics";',
            '        style=dashed;',
            '        node [shape=box, style=filled, fillcolor="#ffebee"];',
        ])
        
        for metric in sorted(self.nodes.keys()):
            lines.append(f'        "{metric}";')
        
        lines.extend([
            "    }",
            "",
            "    // Edges",
        ])
        
        for sym, metric in self.edges:
            lines.append(f'    "{sym}" -> "{metric}";')
        
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, title: str = "Metric Dependencies") -> str:
        lines = [
            "```mermaid",
            "flowchart LR",
            f"    subgraph title [{title}]",
            "    subgraph symbols [Configuration Variables]",
        ]
        
        all_symbols = set()
        for symbols in self.nodes.values():
            all_symbols.update(symbols)
        
        for sym in sorted(all_symbols):
            safe_id = sym.replace("_", "")
            lines.append(f"        {safe_id}(({sym}))")
        
        lines.append("    end")
        lines.append("    subgraph metrics [Metrics]")
        
        for metric in sorted(self.nodes.keys()):
            safe_id = metric.replace("_", "")
            lines.append(f"        {safe_id}[{metric}]")
        
        lines.append("    end")
        
        for sym, metric in self.edges:
            safe_sym = sym.replace("_", "")
            safe_metric = metric.replace("_", "")
            lines.append(f"    {safe_sym} --> {safe_metric}")
        
        lines.append("    end")
        lines.append("```")
        return "\n".join(lines)


def render_expression_tree(
    expr: sympy.Expr,
    title: str = "Expression",
    output_format: str = "dot",
) -> str:
    viz = ExpressionGraphVisualizer()
    viz.build_graph(expr, title)
    
    if output_format == "mermaid":
        return viz.to_mermaid(title)
    else:
        return viz.to_dot(title)


def render_metric_dependencies(
    metrics: dict[str, sympy.Expr],
    title: str = "Metric Dependencies",
    output_format: str = "dot",
) -> str:
    graph = MetricDependencyGraph()
    graph.add_metrics(metrics)
    
    if output_format == "mermaid":
        return graph.to_mermaid(title)
    else:
        return graph.to_dot(title)


def sympy_dotprint(expr: sympy.Expr) -> str:
    try:
        return dotprint(expr)
    except Exception:
        viz = ExpressionGraphVisualizer()
        viz.build_graph(expr)
        return viz.to_dot()

