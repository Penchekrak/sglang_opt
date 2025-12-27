from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import sympy
from sympy import Max, Min, Piecewise, Sum, Symbol, ceiling, floor


@dataclass
class SymVal:
    expr: sympy.Expr
    val: float
    meta: dict = field(default_factory=dict)

    def __add__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr + other.expr,
                val=self.val + other.val,
                meta={"op": "add", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr + other,
            val=self.val + float(other),
            meta={"op": "add_scalar", "operands": [self.meta, other]},
        )

    def __radd__(self, other: float | int) -> SymVal:
        return self.__add__(other)

    def __sub__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr - other.expr,
                val=self.val - other.val,
                meta={"op": "sub", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr - other,
            val=self.val - float(other),
            meta={"op": "sub_scalar", "operands": [self.meta, other]},
        )

    def __rsub__(self, other: float | int) -> SymVal:
        return SymVal(
            expr=other - self.expr,
            val=float(other) - self.val,
            meta={"op": "rsub_scalar", "operands": [other, self.meta]},
        )

    def __mul__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr * other.expr,
                val=self.val * other.val,
                meta={"op": "mul", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr * other,
            val=self.val * float(other),
            meta={"op": "mul_scalar", "operands": [self.meta, other]},
        )

    def __rmul__(self, other: float | int) -> SymVal:
        return self.__mul__(other)

    def __truediv__(self, other: SymVal | float | int) -> SymVal:
        if isinstance(other, SymVal):
            return SymVal(
                expr=self.expr / other.expr,
                val=self.val / other.val if other.val != 0 else float("inf"),
                meta={"op": "div", "operands": [self.meta, other.meta]},
            )
        return SymVal(
            expr=self.expr / other,
            val=self.val / float(other) if other != 0 else float("inf"),
            meta={"op": "div_scalar", "operands": [self.meta, other]},
        )

    def __rtruediv__(self, other: float | int) -> SymVal:
        return SymVal(
            expr=other / self.expr,
            val=float(other) / self.val if self.val != 0 else float("inf"),
            meta={"op": "rdiv_scalar", "operands": [other, self.meta]},
        )

    def __neg__(self) -> SymVal:
        return SymVal(
            expr=-self.expr,
            val=-self.val,
            meta={"op": "neg", "operands": [self.meta]},
        )

    def __repr__(self) -> str:
        return f"SymVal(expr={self.expr}, val={self.val:.6g})"

    def lambdify(self, symbols: list[Symbol]) -> Callable[..., float]:
        return sympy.lambdify(symbols, self.expr, modules=["numpy"])

    def simplify(self) -> SymVal:
        return SymVal(
            expr=sympy.simplify(self.expr),
            val=self.val,
            meta=self.meta,
        )


def sym_const(value: float, name: str | None = None) -> SymVal:
    if name:
        return SymVal(expr=Symbol(name), val=value, meta={"type": "symbol", "name": name})
    return SymVal(expr=sympy.Float(value), val=value, meta={"type": "const", "value": value})


def sym_add(a: SymVal, b: SymVal) -> SymVal:
    return a + b


def sym_sub(a: SymVal, b: SymVal) -> SymVal:
    return a - b


def sym_mul(a: SymVal, b: SymVal) -> SymVal:
    return a * b


def sym_div(a: SymVal, b: SymVal) -> SymVal:
    return a / b


def sym_max(a: SymVal, b: SymVal) -> SymVal:
    return SymVal(
        expr=Max(a.expr, b.expr),
        val=max(a.val, b.val),
        meta={"op": "max", "operands": [a.meta, b.meta]},
    )


def sym_min(a: SymVal, b: SymVal) -> SymVal:
    return SymVal(
        expr=Min(a.expr, b.expr),
        val=min(a.val, b.val),
        meta={"op": "min", "operands": [a.meta, b.meta]},
    )


def sym_piecewise(
    cases: list[tuple[sympy.Expr | bool, SymVal]],
    numeric_condition_results: list[bool],
) -> SymVal:
    pw_args = [(case[1].expr, case[0]) for case in cases]
    pw_expr = Piecewise(*pw_args)

    result_val = cases[-1][1].val
    for i, cond_result in enumerate(numeric_condition_results):
        if cond_result:
            result_val = cases[i][1].val
            break

    return SymVal(
        expr=pw_expr,
        val=result_val,
        meta={
            "op": "piecewise",
            "cases": [(str(c[0]), c[1].meta) for c in cases],
        },
    )


def sym_sum(
    body_fn: Callable[[Symbol], SymVal],
    var: Symbol,
    start: int,
    end: int,
) -> SymVal:
    symbolic_body = body_fn(var)
    sum_expr = Sum(symbolic_body.expr, (var, start, end))

    numeric_sum = 0.0
    for i in range(start, end + 1):
        step_val = body_fn(sympy.Integer(i))
        numeric_sum += step_val.val

    return SymVal(
        expr=sum_expr,
        val=numeric_sum,
        meta={
            "op": "sum",
            "var": str(var),
            "range": (start, end),
            "body_meta": symbolic_body.meta,
        },
    )


def sym_ceiling(a: SymVal) -> SymVal:
    return SymVal(
        expr=ceiling(a.expr),
        val=float(int(a.val) + (1 if a.val > int(a.val) else 0)),
        meta={"op": "ceiling", "operands": [a.meta]},
    )


def sym_floor(a: SymVal) -> SymVal:
    return SymVal(
        expr=floor(a.expr),
        val=float(int(a.val)),
        meta={"op": "floor", "operands": [a.meta]},
    )

