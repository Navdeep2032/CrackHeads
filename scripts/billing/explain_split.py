#!/usr/bin/env python3
"""Generate plain-language explanations for SplitSnap bill splits."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Mapping


def _rupees(value: object) -> str:
    return f"₹{Decimal(str(value or 0)):.2f}"


def explain_split(
    split_result: Mapping[str, Any],
    receipt: Mapping[str, Any],
    repair_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return readable per-person and audit explanations."""
    item_lookup = {
        index: item
        for index, item in enumerate(receipt.get("items", []))  # type: ignore[union-attr]
    }
    people: List[Dict[str, Any]] = []
    for person in split_result.get("people", []):  # type: ignore[union-attr]
        name = str(person["name"])
        amount = person["amount"]
        lines = [f"{name} owes {_rupees(amount)}."]
        for item in split_result.get("item_breakdown", []):  # type: ignore[union-attr]
            if name not in item.get("shares", {}):
                continue
            source = item_lookup.get(item["item_index"], {})
            label = str(source.get("name", f"item {item['item_index']}"))
            lines.append(f"- {label}: {_rupees(item['shares'][name])}")
        for charge in split_result.get("charge_breakdown", []):  # type: ignore[union-attr]
            charge_share = charge.get("shares", {}).get(name, 0)
            if Decimal(str(charge_share)) != 0:
                lines.append(f"- {charge.get('type', 'Charge')}: {_rupees(charge_share)}")
        people.append({"name": name, "amount": amount, "explanation": "\n".join(lines)})

    warnings = list(split_result.get("warnings", []))
    if repair_result:
        warnings.extend(repair_result.get("warnings", []))
        for change in repair_result.get("changes", []):
            warnings.append(
                f"Adjusted {change['field']} from {_rupees(change['old'])} "
                f"to {_rupees(change['new'])} ({change['reason']})."
            )
    return {
        "summary": (
            f"{len(people)} people split {_rupees(split_result.get('receipt_total', 0))}. "
            f"Balanced: {'yes' if split_result.get('balanced') else 'no'}."
        ),
        "people": people,
        "warnings": warnings,
    }
