"""Safe arithmetic expression handling for YAML project values."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from typing import Any


class ExpressionError(ValueError):
    """Raised when a project expression is invalid."""


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _linear_map(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        raise ExpressionError("linear_map requires x0 and x1 to be different")
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _transition_erf(
    x: float,
    start: float,
    end: float,
    center: float,
    width: float,
) -> float:
    if width == 0.0:
        raise ExpressionError("transition_erf requires nonzero width")
    fraction = 0.5 * (1.0 + math.erf((x - center) / (math.sqrt(2.0) * width)))
    return (1.0 - fraction) * start + fraction * end


_SAFE_FUNCTIONS = {
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "erf": math.erf,
    "linear_map": _linear_map,
    "transition_erf": _transition_erf,
}


def names_in_expression(value: Any) -> set[str]:
    """Return variable names referenced by a scalar expression."""

    if isinstance(value, (int, float)):
        return set()
    if not isinstance(value, str):
        raise ExpressionError(
            f"expression values must be numbers or strings, got {type(value).__name__}"
        )
    text = _normalize_reference(value)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"invalid expression {value!r}") from error
    for node in ast.walk(tree):
        _validate_ast_node(node, label=value)
    call_function_node_ids = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and id(node) not in call_function_node_ids
    }


def evaluate_number(
    value: Any,
    variables: Mapping[str, float],
    *,
    label: str,
) -> float:
    """Evaluate a scalar number, parameter reference, or safe arithmetic expression."""

    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ExpressionError(f"{label} must be a number or expression string")
    text = _normalize_reference(value)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"invalid expression for {label}: {value!r}") from error
    for node in ast.walk(tree):
        _validate_ast_node(node, label=label)
    return float(_eval_node(tree.body, variables, label=label))


def _normalize_reference(value: str) -> str:
    text = value.strip()
    if text.startswith("$"):
        name = text[1:]
        if not name.isidentifier():
            raise ExpressionError(f"invalid parameter reference {value!r}")
        return name
    return text


def _validate_ast_node(node: ast.AST, *, label: str) -> None:
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.UAdd,
        ast.USub,
    )
    if not isinstance(node, allowed):
        _raise_allowed_expression_error(label)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
            raise ExpressionError(f"unknown function in {label}")
        if node.keywords:
            raise ExpressionError(f"{label} function calls do not accept keyword arguments")


def _eval_node(node: ast.AST, variables: Mapping[str, float], *, label: str) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ExpressionError(f"{label} contains a non-numeric literal")
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ExpressionError(f"unknown parameter or variable {node.id!r} in {label}")
        return float(variables[node.id])
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_node(node.left, variables, label=label)
        right = _eval_node(node.right, variables, label=label)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0.0:
            raise ExpressionError(f"division by zero in {label}")
        return left / right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        value = _eval_node(node.operand, variables, label=label)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _SAFE_FUNCTIONS[node.func.id]
        arguments = [_eval_node(argument, variables, label=label) for argument in node.args]
        try:
            return float(function(*arguments))
        except TypeError as error:
            raise ExpressionError(f"wrong argument count for {node.func.id!r} in {label}") from error
        except ValueError as error:
            raise ExpressionError(f"invalid argument for {node.func.id!r} in {label}") from error
    _raise_allowed_expression_error(label)


def _raise_allowed_expression_error(label: str) -> None:
    functions = ", ".join(sorted(_SAFE_FUNCTIONS))
    raise ExpressionError(
        f"{label} may contain only numbers, parameter names, repeat_index, "
        "repeat_index0, layer_index, +, -, *, /, parentheses, and safe "
        f"functions: {functions}"
    )
