"""Tools available to supervisor-agent's two sub-agents.

Deliberately small and dependency-free, same philosophy as `react_agent.tools` — the point of
this example is the supervisor/delegation pattern and its surrounding infrastructure, not the
tools themselves. Kept local to this package (not imported from `react_agent.tools`) so each
pattern in this repo stays self-contained, depending only on `agents-common`, never on a sibling
pattern package.
"""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING

from langchain_core.tools import tool
import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = structlog.get_logger(__name__)

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
        _logger.warning("calculator_failed", expression=expression, error=str(exc))
        return f"Could not evaluate '{expression}': {exc}"
    result_str = str(int(result) if result.is_integer() else result)
    _logger.info("calculator_evaluated", expression=expression, result=result_str)
    return result_str


@tool
def count_words(text: str) -> str:
    """Count the number of whitespace-separated words in the given text.

    Args:
        text: The text to count words in.

    Returns:
        The word count, as a string.
    """
    count = len(text.split())
    _logger.info("words_counted", count=count)
    return str(count)


@tool
def reverse_text(text: str) -> str:
    """Reverse the characters in the given text.

    Args:
        text: The text to reverse.

    Returns:
        The reversed text.
    """
    _logger.info("text_reversed", length=len(text))
    return text[::-1]


MATH_TOOLS = [calculator]
TEXT_TOOLS = [count_words, reverse_text]
