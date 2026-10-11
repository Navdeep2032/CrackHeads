#!/usr/bin/env python3
"""Deterministic, rounding-safe bill splitting for SplitSnap."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Dict, Iterable, List, Mapping, Sequence


CENT = Decimal("0.01")


def _money(value: object) -> Decimal:
    return Decimal(str(value or 0)).quantize(CENT, rounding=ROUND_HALF_UP)


def _allocate(total: Decimal, weights: Mapping[str, Decimal]) -> Dict[str, Decimal]:
    """Allocate cents by largest remainder so the result sums exactly to total."""
    if not weights:
        return {}
    sign = Decimal("-1") if total < 0 else Decimal("1")
    cents = int((abs(total) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    positive_weights = {name: max(weight, Decimal("0")) for name, weight in weights.items()}
    weight_sum = sum(positive_weights.values(), Decimal("0"))
    if weight_sum <= 0:
        positive_weights = {name: Decimal("1") for name in weights}
        weight_sum = Decimal(len(weights))

    exact = {
        name: Decimal(cents) * weight / weight_sum
        for name, weight in positive_weights.items()
    }
    base = {name: int(value.to_integral_value(rounding=ROUND_DOWN)) for name, value in exact.items()}
    remainder = cents - sum(base.values())
    ranked = sorted(
        exact,
        key=lambda name: (exact[name] - base[name], name),
        reverse=True,
    )
    for name in ranked[:remainder]:
        base[name] += 1
    return {name: sign * Decimal(value) / 100 for name, value in base.items()}


def split_bill(
    receipt: Mapping[str, object],
    people: Sequence[str],
    assignments: Mapping[int, Iterable[str]],
) -> Dict[str, object]:
    """Split a receipt by item index and return exact per-person totals.

    ``assignments`` maps each item index to one or more people. Multiple people
    share that item's amount equally. Unassigned items and invalid people are
    reported as warnings rather than silently dropped.
    """
    names = list(dict.fromkeys(str(person).strip() for person in people if str(person).strip()))
    if not names:
        raise ValueError("At least one person is required.")
    allowed = set(names)
    items = list(receipt.get("items", []))  # type: ignore[arg-type]
    shares = {name: Decimal("0.00") for name in names}
    warnings: List[str] = []
    item_breakdown: List[Dict[str, object]] = []

    for index, item in enumerate(items):
        assigned = list(dict.fromkeys(str(name) for name in assignments.get(index, [])))
        invalid = [name for name in assigned if name not in allowed]
        if invalid:
            warnings.append(f"Item {index} has unknown people: {', '.join(invalid)}.")
            assigned = [name for name in assigned if name in allowed]
        if not assigned:
            warnings.append(f"Item {index} is unassigned.")
            continue
        total = _money(item.get("total", 0))  # type: ignore[union-attr]
        allocation = _allocate(total, {name: Decimal("1") for name in assigned})
        for name, amount in allocation.items():
            shares[name] += amount
        item_breakdown.append(
            {"item_index": index, "total": total, "shares": dict(allocation)}
        )

    assigned_base = sum(shares.values(), Decimal("0"))
    charge_breakdown: List[Dict[str, object]] = []
    for charge in receipt.get("charges", []):  # type: ignore[union-attr]
        amount = _money(charge.get("amount", 0))  # type: ignore[union-attr]
        allocation = _allocate(amount, shares)
        for name, value in allocation.items():
            shares[name] += value
        charge_breakdown.append(
            {
                "type": str(charge.get("type", "Unknown")),  # type: ignore[union-attr]
                "amount": amount,
                "shares": dict(allocation),
            }
        )

    expected_total = _money(receipt.get("grand_total", 0))
    computed_total = sum(shares.values(), Decimal("0")).quantize(CENT)
    if computed_total != expected_total:
        warnings.append(
            f"Assigned total {computed_total} does not equal receipt total {expected_total}."
        )

    return {
        "people": [
            {"name": name, "amount": shares[name].quantize(CENT)}
            for name in names
        ],
        "item_breakdown": item_breakdown,
        "charge_breakdown": charge_breakdown,
        "assigned_base": assigned_base.quantize(CENT),
        "computed_total": computed_total,
        "receipt_total": expected_total,
        "balanced": computed_total == expected_total and not any(
            warning.startswith("Item ") and warning.endswith("unassigned.") for warning in warnings
        ),
        "warnings": warnings,
    }
