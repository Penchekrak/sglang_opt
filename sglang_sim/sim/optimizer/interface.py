from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from itertools import product
from enum import Enum

import sympy
from sympy import Symbol
import numpy as np

from sim.symbolic.symbols import ConfigSymbols


class ObjectiveDirection(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class OptimizationResult:
    optimal_values: dict[str, float]
    optimal_objective: float
    success: bool
    message: str
    all_evaluations: list[tuple[dict, float]] = field(default_factory=list)
    constraint_violations: dict[str, float] = field(default_factory=dict)
    pareto_front: list[dict] = field(default_factory=list)


@dataclass
class MultiObjectiveResult:
    pareto_front: list[dict[str, float]]
    pareto_objectives: list[dict[str, float]]
    all_evaluations: list[tuple[dict, dict[str, float]]] = field(default_factory=list)


@dataclass
class Constraint:
    expr: sympy.Expr
    type: str  # "eq" for equality, "ineq" for inequality (>= 0)
    name: str = ""


@dataclass
class ObjectiveSpec:
    name: str
    expr: sympy.Expr
    direction: ObjectiveDirection
    weight: float = 1.0


class OptimizerInterface:
    def __init__(
        self,
        objective: sympy.Expr,
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
        maximize: bool = True,
    ):
        self.objective = objective
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self.maximize = maximize

        self._decision_vars: list[Symbol] = []
        self._frozen_params: dict[Symbol, float] = {}

    def set_decision_vars(self, vars: list[Symbol]) -> None:
        self._decision_vars = vars

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def get_decision_vars(self) -> list[Symbol]:
        if self._decision_vars:
            return self._decision_vars
        return self.symbols.decision_vars()

    def get_frozen_params(self) -> list[Symbol]:
        return self.symbols.frozen_params()

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def to_scipy_minimize(
        self,
        bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        decision_vars = self.get_decision_vars()

        substituted_obj = self.substitute_frozen(self.objective)

        if self.maximize:
            obj_to_minimize = -substituted_obj
        else:
            obj_to_minimize = substituted_obj

        obj_func = sympy.lambdify(decision_vars, obj_to_minimize, modules=["numpy"])

        def objective_wrapper(x: np.ndarray) -> float:
            return float(obj_func(*x))

        scipy_constraints = []
        for constraint in self.constraints:
            substituted = self.substitute_frozen(constraint.expr)
            constraint_func = sympy.lambdify(decision_vars, substituted, modules=["numpy"])

            scipy_constraints.append({
                "type": constraint.type,
                "fun": lambda x, f=constraint_func: float(f(*x)),
            })

        scipy_bounds = None
        if bounds:
            scipy_bounds = [
                bounds.get(var, (None, None)) for var in decision_vars
            ]

        return {
            "fun": objective_wrapper,
            "x0": np.ones(len(decision_vars)),
            "bounds": scipy_bounds,
            "constraints": scipy_constraints,
            "method": "SLSQP",
        }

    def enumerate_discrete_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> list[dict[Symbol, int]]:
        regimes = []
        var_names = list(discrete_vars.keys())
        var_values = list(discrete_vars.values())

        for combo in product(*var_values):
            regime = dict(zip(var_names, combo))
            regimes.append(regime)

        return regimes

    def optimize_over_regimes(
        self,
        discrete_vars: dict[Symbol, list[int]],
        continuous_bounds: dict[Symbol, tuple[float, float]] | None = None,
    ) -> OptimizationResult:
        from scipy.optimize import minimize

        regimes = self.enumerate_discrete_regimes(discrete_vars, continuous_bounds)
        best_result: OptimizationResult | None = None
        all_evaluations: list[tuple[dict, float]] = []

        decision_vars = self.get_decision_vars()
        continuous_vars = [v for v in decision_vars if v not in discrete_vars]

        for regime in regimes:
            regime_objective = self.objective.subs(list(regime.items()))
            regime_objective = self.substitute_frozen(regime_objective)

            if not continuous_vars:
                if self.maximize:
                    obj_val = -float(regime_objective)
                else:
                    obj_val = float(regime_objective)

                result_dict = {str(k): v for k, v in regime.items()}
                all_evaluations.append((result_dict, -obj_val if self.maximize else obj_val))

                if best_result is None or (self.maximize and -obj_val > best_result.optimal_objective) or \
                   (not self.maximize and obj_val < best_result.optimal_objective):
                    best_result = OptimizationResult(
                        optimal_values=result_dict,
                        optimal_objective=-obj_val if self.maximize else obj_val,
                        success=True,
                        message=f"Regime: {regime}",
                    )
            else:
                obj_func = sympy.lambdify(continuous_vars, regime_objective, modules=["numpy"])

                if self.maximize:
                    def objective_wrapper(x):
                        return -float(obj_func(*x))
                else:
                    def objective_wrapper(x):
                        return float(obj_func(*x))

                bounds_list = [
                    continuous_bounds.get(v, (1, 100)) if continuous_bounds else (1, 100)
                    for v in continuous_vars
                ]

                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds_list])

                try:
                    result = minimize(
                        objective_wrapper,
                        x0,
                        bounds=bounds_list,
                        method="L-BFGS-B",
                    )

                    result_dict = {str(k): v for k, v in regime.items()}
                    for var, val in zip(continuous_vars, result.x):
                        result_dict[str(var)] = float(val)

                    obj_val = -result.fun if self.maximize else result.fun
                    all_evaluations.append((result_dict, obj_val))

                    if best_result is None or \
                       (self.maximize and obj_val > best_result.optimal_objective) or \
                       (not self.maximize and obj_val < best_result.optimal_objective):
                        best_result = OptimizationResult(
                            optimal_values=result_dict,
                            optimal_objective=obj_val,
                            success=result.success,
                            message=result.message,
                        )
                except Exception as e:
                    continue

        if best_result is None:
            return OptimizationResult(
                optimal_values={},
                optimal_objective=float("inf") if not self.maximize else float("-inf"),
                success=False,
                message="No valid regime found",
                all_evaluations=all_evaluations,
            )

        best_result.all_evaluations = all_evaluations
        return best_result

    def grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())

        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []

        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])

        for combo in product(*var_values):
            try:
                obj_val = float(obj_func(*combo))
                if self.maximize:
                    obj_val = obj_val
                else:
                    obj_val = obj_val

                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))

                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue

        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Grid search complete",
            all_evaluations=all_evaluations,
        )

    def constrained_grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
        constraints: list[Constraint],
    ) -> OptimizationResult:
        from sim.metrics.constraints import Constraint as MetricConstraint
        
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())
        
        best_values: dict[str, float] = {}
        best_objective = float("-inf") if self.maximize else float("inf")
        all_evaluations: list[tuple[dict, float]] = []
        
        substituted_obj = self.substitute_frozen(self.objective)
        obj_func = sympy.lambdify(var_names, substituted_obj, modules=["numpy"])
        
        constraint_funcs = []
        for c in constraints:
            substituted = self.substitute_frozen(c.expr)
            constraint_funcs.append(
                (c.name, c.type, sympy.lambdify(var_names, substituted, modules=["numpy"]))
            )
        
        for combo in product(*var_values):
            try:
                feasible = True
                violations = {}
                
                for name, ctype, cfunc in constraint_funcs:
                    c_val = float(cfunc(*combo))
                    if ctype == "ineq" and c_val < 0:
                        feasible = False
                        violations[name] = -c_val
                    elif ctype == "eq" and abs(c_val) > 1e-6:
                        feasible = False
                        violations[name] = abs(c_val)
                
                if not feasible:
                    continue
                
                obj_val = float(obj_func(*combo))
                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_val))
                
                if (self.maximize and obj_val > best_objective) or \
                   (not self.maximize and obj_val < best_objective):
                    best_objective = obj_val
                    best_values = result_dict
            except Exception:
                continue
        
        return OptimizationResult(
            optimal_values=best_values,
            optimal_objective=best_objective,
            success=len(best_values) > 0,
            message="Constrained grid search complete",
            all_evaluations=all_evaluations,
        )


class MultiObjectiveOptimizer:
    def __init__(
        self,
        objectives: list[ObjectiveSpec],
        constraints: list[Constraint] | None = None,
        symbols: ConfigSymbols | None = None,
    ):
        self.objectives = objectives
        self.constraints = constraints or []
        self.symbols = symbols or ConfigSymbols()
        self._frozen_params: dict[Symbol, float] = {}

    def set_frozen_params(self, params: dict[Symbol, float]) -> None:
        self._frozen_params = params

    def substitute_frozen(self, expr: sympy.Expr) -> sympy.Expr:
        for sym, val in self._frozen_params.items():
            expr = expr.subs(sym, val)
        return expr

    def weighted_sum_optimize(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> OptimizationResult:
        combined_obj = sympy.Integer(0)
        
        for obj in self.objectives:
            weight = obj.weight
            expr = obj.expr
            if obj.direction == ObjectiveDirection.MINIMIZE:
                combined_obj = combined_obj - weight * expr
            else:
                combined_obj = combined_obj + weight * expr
        
        optimizer = OptimizerInterface(
            objective=combined_obj,
            constraints=self.constraints,
            symbols=self.symbols,
            maximize=True,
        )
        optimizer.set_frozen_params(self._frozen_params)
        
        return optimizer.grid_search(param_grid)

    def pareto_grid_search(
        self,
        param_grid: dict[Symbol, list[float]],
    ) -> MultiObjectiveResult:
        var_names = list(param_grid.keys())
        var_values = list(param_grid.values())
        
        obj_funcs = []
        for obj in self.objectives:
            substituted = self.substitute_frozen(obj.expr)
            obj_funcs.append((
                obj.name,
                obj.direction,
                sympy.lambdify(var_names, substituted, modules=["numpy"])
            ))
        
        constraint_funcs = []
        for c in self.constraints:
            substituted = self.substitute_frozen(c.expr)
            constraint_funcs.append(
                (c.name, c.type, sympy.lambdify(var_names, substituted, modules=["numpy"]))
            )
        
        all_evaluations: list[tuple[dict, dict[str, float]]] = []
        feasible_solutions: list[tuple[dict[str, float], dict[str, float]]] = []
        
        for combo in product(*var_values):
            try:
                feasible = True
                for name, ctype, cfunc in constraint_funcs:
                    c_val = float(cfunc(*combo))
                    if ctype == "ineq" and c_val < 0:
                        feasible = False
                        break
                    elif ctype == "eq" and abs(c_val) > 1e-6:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                obj_vals = {}
                for name, direction, ofunc in obj_funcs:
                    obj_vals[name] = float(ofunc(*combo))
                
                result_dict = {str(k): v for k, v in zip(var_names, combo)}
                all_evaluations.append((result_dict, obj_vals))
                feasible_solutions.append((result_dict, obj_vals))
            except Exception:
                continue
        
        pareto_front = self._compute_pareto_front(feasible_solutions)
        
        return MultiObjectiveResult(
            pareto_front=[p[0] for p in pareto_front],
            pareto_objectives=[p[1] for p in pareto_front],
            all_evaluations=all_evaluations,
        )

    def _compute_pareto_front(
        self,
        solutions: list[tuple[dict[str, float], dict[str, float]]],
    ) -> list[tuple[dict[str, float], dict[str, float]]]:
        if not solutions:
            return []
        
        pareto = []
        
        for i, (params_i, objs_i) in enumerate(solutions):
            is_dominated = False
            
            for j, (params_j, objs_j) in enumerate(solutions):
                if i == j:
                    continue
                
                if self._dominates(objs_j, objs_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append((params_i, objs_i))
        
        return pareto

    def _dominates(
        self,
        objs_a: dict[str, float],
        objs_b: dict[str, float],
    ) -> bool:
        dominated = True
        strictly_better = False
        
        for obj in self.objectives:
            name = obj.name
            val_a = objs_a[name]
            val_b = objs_b[name]
            
            if obj.direction == ObjectiveDirection.MAXIMIZE:
                if val_a < val_b:
                    dominated = False
                    break
                if val_a > val_b:
                    strictly_better = True
            else:
                if val_a > val_b:
                    dominated = False
                    break
                if val_a < val_b:
                    strictly_better = True
        
        return dominated and strictly_better

    def epsilon_constraint(
        self,
        primary_objective: str,
        epsilon_bounds: dict[str, tuple[float, float]],
        param_grid: dict[Symbol, list[float]],
    ) -> list[OptimizationResult]:
        primary = next(o for o in self.objectives if o.name == primary_objective)
        
        results = []
        
        for other_obj in self.objectives:
            if other_obj.name == primary_objective:
                continue
            
            if other_obj.name in epsilon_bounds:
                lb, ub = epsilon_bounds[other_obj.name]
                
                if other_obj.direction == ObjectiveDirection.MAXIMIZE:
                    constraint = Constraint(
                        expr=other_obj.expr - lb,
                        type="ineq",
                        name=f"{other_obj.name}_lb",
                    )
                else:
                    constraint = Constraint(
                        expr=ub - other_obj.expr,
                        type="ineq",
                        name=f"{other_obj.name}_ub",
                    )
                
                optimizer = OptimizerInterface(
                    objective=primary.expr,
                    constraints=self.constraints + [constraint],
                    symbols=self.symbols,
                    maximize=primary.direction == ObjectiveDirection.MAXIMIZE,
                )
                optimizer.set_frozen_params(self._frozen_params)
                
                result = optimizer.constrained_grid_search(param_grid, [constraint])
                results.append(result)
        
        return results

