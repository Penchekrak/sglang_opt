import pytest
import sympy
from sympy import Symbol

from sim.symbolic.expr import (
    SymVal,
    sym_add,
    sym_sub,
    sym_mul,
    sym_div,
    sym_max,
    sym_min,
    sym_const,
    sym_piecewise,
    sym_sum,
)
from sim.symbolic.symbols import ConfigSymbols


class TestSymVal:
    def test_creation(self):
        sv = SymVal(expr=sympy.Float(1.5), val=1.5)
        assert sv.val == 1.5
        assert sv.expr == sympy.Float(1.5)

    def test_add(self):
        a = sym_const(2.0)
        b = sym_const(3.0)
        result = sym_add(a, b)
        assert result.val == 5.0

    def test_sub(self):
        a = sym_const(5.0)
        b = sym_const(3.0)
        result = sym_sub(a, b)
        assert result.val == 2.0

    def test_mul(self):
        a = sym_const(2.0)
        b = sym_const(3.0)
        result = sym_mul(a, b)
        assert result.val == 6.0

    def test_div(self):
        a = sym_const(6.0)
        b = sym_const(2.0)
        result = sym_div(a, b)
        assert result.val == 3.0

    def test_max(self):
        a = sym_const(2.0)
        b = sym_const(5.0)
        result = sym_max(a, b)
        assert result.val == 5.0

    def test_min(self):
        a = sym_const(2.0)
        b = sym_const(5.0)
        result = sym_min(a, b)
        assert result.val == 2.0

    def test_symbolic_expression(self):
        x = Symbol("x")
        sv = SymVal(expr=x * 2, val=4.0, meta={"name": "test"})
        assert sv.expr == x * 2
        assert sv.val == 4.0

    def test_piecewise(self):
        x = Symbol("x")
        condition = x > 5
        sv_true = sym_const(10.0)
        sv_false = sym_const(1.0)

        result = sym_piecewise(
            [(condition, sv_true), (True, sv_false)],
            numeric_condition_results=[True],
        )
        assert result.val == 10.0

    def test_lambdify(self):
        x = Symbol("x")
        sv = SymVal(expr=x**2, val=4.0)
        func = sv.lambdify([x])
        assert func(3) == 9


class TestConfigSymbols:
    def test_decision_vars(self):
        symbols = ConfigSymbols()
        dvars = symbols.decision_vars()
        assert len(dvars) > 0
        assert symbols.N_p in dvars
        assert symbols.N_d in dvars

    def test_frozen_params(self):
        symbols = ConfigSymbols()
        fparams = symbols.frozen_params()
        assert symbols.cache_hit_rate in fparams

    def test_all_symbols(self):
        symbols = ConfigSymbols()
        all_syms = symbols.all_symbols()
        assert len(all_syms) > 10

