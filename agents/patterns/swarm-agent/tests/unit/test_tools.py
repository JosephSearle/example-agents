"""Unit tests for swarm_agent.tools — pure functions, no network, no LLM."""

from __future__ import annotations

import pytest
from swarm_agent.tools import issue_refund, lookup_invoice

pytestmark = pytest.mark.unit


def test_lookup_invoice_known_invoice() -> None:
    assert lookup_invoice.invoke({"invoice_id": "INV-1001"}) == "INV-1001: $49.99"


def test_lookup_invoice_unknown_invoice() -> None:
    result = lookup_invoice.invoke({"invoice_id": "does-not-exist"})
    assert "No invoice found" in result


def test_issue_refund_known_invoice() -> None:
    result = issue_refund.invoke({"invoice_id": "INV-1002"})
    assert result == "Refunded $19.99 for INV-1002."


def test_issue_refund_unknown_invoice() -> None:
    result = issue_refund.invoke({"invoice_id": "does-not-exist"})
    assert "No invoice found" in result
