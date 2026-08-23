import pytest
import pandas as pd
from src.data_loader import load_data
from src.validation import validate_and_clean_data, ValidationReport

def test_data_loader_valid():
    cases_df, payments_df = load_data("data")
    assert not cases_df.empty
    assert not payments_df.empty
    assert "case_id" in cases_df.columns
    assert "payment_id" in payments_df.columns

def test_validation_missing_and_malformed():
    raw_cases = pd.DataFrame([{
        "case_id": "C-TEST-1",
        "district": "Calder Central",
        "household_size": "INVALID",  # Should coerce to 1
        "monthly_award": "-500.0",     # Should clip to 0
        "status": "UnknownStatus",    # Should default to Active
        "opened_date": "2024-99-99",  # Malformed date
        "closure_month": "",
        "payment_adjustments": "3",
        "contact_attempts": "5",
        "months_since_review": "10"
    }])
    
    raw_payments = pd.DataFrame([
        {
            "payment_id": "P-100",
            "case_id": "C-TEST-1",
            "pay_month": "2025-08",
            "amount": "800.0",
            "method": "Transfer",
            "adjustment": "N"
        },
        {
            "payment_id": "P-101",
            "case_id": "C-UNKNOWN-CASE",  # Orphaned payment
            "pay_month": "2025-09",
            "amount": "999.0",
            "method": "Card",
            "adjustment": "N"
        },
        {
            "payment_id": "P-102",
            "case_id": "C-TEST-1",
            "pay_month": "2025-10",
            "amount": "INVALID_AMT",      # Invalid numeric
            "method": "Transfer",
            "adjustment": "N"
        }
    ])

    cases_clean, payments_clean, report = validate_and_clean_data(raw_cases, raw_payments)

    assert len(cases_clean) == 1
    assert cases_clean.iloc[0]["household_size"] == 1
    assert cases_clean.iloc[0]["monthly_award"] == 0.0
    assert cases_clean.iloc[0]["status"] == "Active"

    # Invalid payment dropped, orphan payment dropped
    assert len(payments_clean) == 1
    assert payments_clean.iloc[0]["payment_id"] == "P-100"
    assert report.dropped_payments_count == 1
    assert report.orphaned_payments_count == 1

def test_empty_payments_history():
    raw_cases = pd.DataFrame([{
        "case_id": "C-NO-PMT",
        "district": "Northgate",
        "household_size": "2",
        "monthly_award": "1000.0",
        "status": "Active",
        "opened_date": "2024-01-01",
        "closure_month": "",
        "payment_adjustments": "0",
        "contact_attempts": "0",
        "months_since_review": "0"
    }])
    raw_payments = pd.DataFrame(columns=["payment_id", "case_id", "pay_month", "amount", "method", "adjustment"])

    cases_clean, payments_clean, report = validate_and_clean_data(raw_cases, raw_payments)
    assert len(cases_clean) == 1
    assert len(payments_clean) == 0
    assert "C-NO-PMT" in report.cases_without_payments
