"""Tools available to the ReAct agent.

Deliberately small and dependency-free — the point of this example is the agent pattern and
its surrounding infrastructure (checkpointing, tracing, evals), not the tools themselves.
"""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from collections.abc import Callable

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
}

_GLOSSARY: dict[str, str] = {
    "42": "The Answer to the Ultimate Question of Life, the Universe, and Everything (Hitchhiker's Guide to the Galaxy).",
    "404": "Not found — often used to mean someone or something is missing or unavailable.",
    "1337": "Leet — internet slang derived from 'elite'.",
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a restricted arithmetic AST (no names, no calls, no attributes)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        binary_fn = _BINARY_OPERATORS[type(node.op)]
        return binary_fn(left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        unary_fn = _UNARY_OPERATORS[type(node.op)]
        return unary_fn(_eval_node(node.operand))
    msg = f"Unsupported expression: {ast.dump(node)}"
    raise ValueError(msg)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: An arithmetic expression using +, -, *, /, ** and parentheses,
            e.g. "47 * 12" or "(3 + 4) / 2".

    Returns:
        The numeric result, as a string.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"Could not evaluate '{expression}': {exc}"
    return str(int(result) if result.is_integer() else result)


@tool
def lookup_glossary_term(term: str) -> str:
    """Look up a term in the internal glossary.

    Args:
        term: The term to look up, e.g. "42" or "404".

    Returns:
        The glossary definition, or a not-found message.
    """
    return _GLOSSARY.get(term.strip(), f"No glossary entry found for '{term}'.")


TOOLS = [calculator, lookup_glossary_term]
