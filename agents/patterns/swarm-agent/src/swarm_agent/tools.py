"""Tools available to swarm-agent's billing specialist.

Deliberately small, dependency-free, and stateless (no simulated database, no mutation) — same
philosophy as `react_agent.tools`, and kept local to this package rather than imported from a
sibling pattern, same convention as `supervisor_agent.tools`.
"""

from __future__ import annotations

from langchain_core.tools import tool
import structlog

_logger = structlog.get_logger(__name__)

_INVOICES: dict[str, float] = {
    "INV-1001": 49.99,
    "INV-1002": 19.99,
    "INV-1003": 129.00,
}


@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice's amount.

    Args:
        invoice_id: The invoice identifier, e.g. "INV-1001".

    Returns:
        The invoice amount, or a not-found message.
    """
    amount = _INVOICES.get(invoice_id)
    if amount is None:
        _logger.info("invoice_lookup_failed", invoice_id=invoice_id)
        return f"No invoice found for '{invoice_id}'."
    _logger.info("invoice_looked_up", invoice_id=invoice_id, amount=amount)
    return f"{invoice_id}: ${amount:.2f}"


@tool
def issue_refund(invoice_id: str) -> str:
    """Issue a refund for the given invoice's full amount.

    Args:
        invoice_id: The invoice identifier, e.g. "INV-1001".

    Returns:
        A confirmation of the refunded amount, or a not-found message.
    """
    amount = _INVOICES.get(invoice_id)
    if amount is None:
        _logger.info("refund_failed", invoice_id=invoice_id, reason="invoice_not_found")
        return f"No invoice found for '{invoice_id}'."
    _logger.info("refund_issued", invoice_id=invoice_id, amount=amount)
    return f"Refunded ${amount:.2f} for {invoice_id}."


BILLING_TOOLS = [lookup_invoice, issue_refund]
