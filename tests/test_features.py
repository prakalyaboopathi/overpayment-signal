import pytest
import pandas as pd
from src.features import engineer_features, get_needs_figure, NEEDS_FIGURES

def test_needs_figures_lookup():
    assert get_needs_figure(1) == 1240.0
    assert get_needs_figure(2) == 1670.0
    assert get_needs_figure(3) == 2000.0
    assert get_needs_figure(4) == 2330.0
    assert get_needs_figure(5) == 2660.0
    assert get_needs_figure(6) == 2990.0
    assert get_needs_figure(10) == 2990.0  # Cap at 6+

def test_feature_engineering_calculations():
    cases_df = pd.DataFrame([{
        "case_id": "C-FEAT-1",
        "district": "Calder Central",
        "household_size": 2,
        "monthly_award": 1000.0,
        "status": "Active",
        "opened_date": "2024-01-01",
        "closure_month": "",
        "payment_adjustments": 2,
        "contact_attempts": 4,
        "months_since_review": 6
    }])

    payments_df = pd.DataFrame([
        {"payment_id": "P-1", "case_id": "C-FEAT-1", "pay_month": "2025-07", "amount": 1200.0, "method": "Transfer", "adjustment": "N"},
        {"payment_id": "P-2", "case_id": "C-FEAT-1", "pay_month": "2025-08", "amount": 1400.0, "method": "Card", "adjustment": "Y"},
        {"payment_id": "P-3", "case_id": "C-FEAT-1", "pay_month": "2025-09", "amount": 1200.0, "method": "Card", "adjustment": "N"}
    ])

    feat_df = engineer_features(cases_df, payments_df)

    assert feat_df.iloc[0]["needs_figure"] == 1670.0
    assert feat_df.iloc[0]["award_to_needs_ratio"] == pytest.approx(1000.0 / 1670.0)
    assert feat_df.iloc[0]["mean_payment_amount"] == pytest.approx(1266.66666, rel=1e-3)
    assert feat_df.iloc[0]["max_payment_amount"] == 1400.0
    assert feat_df.iloc[0]["mean_payment_vs_award_diff"] == pytest.approx(266.66666, rel=1e-3)
    assert feat_df.iloc[0]["payment_method_switches"] == 1
    assert feat_df.iloc[0]["dept_adjustments_count"] == 2
    assert feat_df.iloc[0]["dept_contact_attempts"] == 4
