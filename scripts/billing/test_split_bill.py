from decimal import Decimal

try:
    from split_bill import split_bill
except ModuleNotFoundError:
    from scripts.billing.split_bill import split_bill


def test_split_bill_balances_shared_item_and_charge() -> None:
    receipt = {
        "items": [
            {"name": "Dosa", "qty": 1, "unit_price": 100, "total": 100},
            {"name": "Biryani", "qty": 1, "unit_price": 201, "total": 201},
        ],
        "charges": [{"type": "CGST", "amount": 30.10}],
        "subtotal": 301,
        "grand_total": 331.10,
    }
    result = split_bill(
        receipt,
        ["Asha", "Ravi"],
        {0: ["Asha", "Ravi"], 1: ["Ravi"]},
    )
    assert result["balanced"] is True
    assert result["computed_total"] == Decimal("331.10")
    assert result["people"] == [
        {"name": "Asha", "amount": Decimal("55.00")},
        {"name": "Ravi", "amount": Decimal("276.10")},
    ]


def test_split_bill_reports_unassigned_items() -> None:
    receipt = {
        "items": [{"name": "Tea", "qty": 1, "unit_price": 50, "total": 50}],
        "charges": [],
        "subtotal": 50,
        "grand_total": 50,
    }
    result = split_bill(receipt, ["Asha"], {})
    assert result["balanced"] is False
    assert "Item 0 is unassigned." in result["warnings"]


def test_split_bill_with_custom_ratios() -> None:
    receipt = {
        "items": [
            {"name": "Large Pizza", "qty": 1, "unit_price": 300, "total": 300},
        ],
        "charges": [{"type": "Service Charge", "amount": 30}],
        "subtotal": 300,
        "grand_total": 330,
    }
    # Asha ate 2 slices (ratio 2), Ravi ate 1 slice (ratio 1) -> 2:1 split
    result = split_bill(
        receipt,
        ["Asha", "Ravi"],
        {0: {"Asha": 2, "Ravi": 1}},
    )
    assert result["balanced"] is True
    assert result["computed_total"] == Decimal("330.00")
    # Asha: 200 item + 20 charge = 220; Ravi: 100 item + 10 charge = 110
    assert result["people"] == [
        {"name": "Asha", "amount": Decimal("220.00")},
        {"name": "Ravi", "amount": Decimal("110.00")},
    ]

