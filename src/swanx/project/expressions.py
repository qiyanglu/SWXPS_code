"""Safe arithmetic expression handling for YAML project values."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any


class ExpressionError(ValueError):
    """Raised when a project expression is invalid."""


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


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
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


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
        raise ExpressionError(
            f"{label} may contain only numbers, parameter names, repeat_index, "
            "layer_index, +, -, *, /, and parentheses"
        )


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
    raise ExpressionError(
        f"{label} may contain only numbers, parameter names, repeat_index, "
        "layer_index, +, -, *, /, and parentheses"
    )
