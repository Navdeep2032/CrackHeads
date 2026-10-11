from decimal import Decimal

from explain_split import explain_split


def test_explanation_includes_items_charges_and_warnings() -> None:
    receipt = {
        "items": [{"name": "Dosa", "qty": 1, "unit_price": 100, "total": 100}],
        "charges": [{"type": "CGST", "amount": 9}],
        "subtotal": 100,
        "grand_total": 109,
    }
    split = {
        "balanced": True,
        "receipt_total": Decimal("109.00"),
        "people": [{"name": "Asha", "amount": Decimal("109.00")}],
        "item_breakdown": [
            {"item_index": 0, "shares": {"Asha": Decimal("100.00")}}
        ],
        "charge_breakdown": [
            {"type": "CGST", "shares": {"Asha": Decimal("9.00")}}
        ],
        "warnings": [],
    }
    result = explain_split(split, receipt)
    assert "Asha owes ₹109.00." in result["people"][0]["explanation"]
    assert "Dosa: ₹100.00" in result["people"][0]["explanation"]
    assert "CGST: ₹9.00" in result["people"][0]["explanation"]
