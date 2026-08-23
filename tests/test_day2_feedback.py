import pytest
import pandas as pd
from src.data_loader import load_data
from src.validation import validate_and_clean_data
from src.features import engineer_features
from src.scoring import ScoringEngine
from src.explanations import generate_worklist_explanations

def test_day2_investigator_feedback_c33248_not_in_top_20():
    """
    Day 2 Feedback Regression Test:
    Ensures case C-33248 (busy file with 5 adjustments and 7 contact attempts)
    is NOT incorrectly elevated into the Top 20 due to departmental processing activity.
    """
    cases_df, payments_df = load_data("data")
    cases_clean, payments_clean, _ = validate_and_clean_data(cases_df, payments_df)
    features_df = engineer_features(cases_clean, payments_clean)

    engine = ScoringEngine()
    scored_full_df, _ = engine.compute_scores(features_df)
    top20_df, top20_subscores = engine.get_ranked_worklist(scored_full_df, top_n=20)

    # 1. C-33248 must NOT be in top 20
    top20_case_ids = top20_df['case_id'].tolist()
    assert "C-33248" not in top20_case_ids

    # 2. Verify C-33248 rank is far below top 20 (e.g. rank > 100)
    sorted_df = scored_full_df.sort_values(by=['risk_score', 'case_id'], ascending=[False, True]).reset_index(drop=True)
    sorted_df['rank'] = range(1, len(sorted_df) + 1)
    c33248_row = sorted_df[sorted_df['case_id'] == 'C-33248']
    
    assert not c33248_row.empty
    c33248_rank = c33248_row.iloc[0]['rank']
    assert c33248_rank > 100

def test_departmental_activity_zero_weight_in_scoring():
    """
    Verifies that increasing contact_attempts or payment_adjustments on a case
    does NOT increase its resident risk score.
    """
    base_case = pd.DataFrame([{
        "case_id": "C-TEST-DEPT",
        "district": "Calder Central",
        "household_size": 2,
        "monthly_award": 1000.0,
        "status": "Active",
        "opened_date": "2024-01-01",
        "closure_month": "",
        "payment_adjustments": 0,
        "contact_attempts": 0,
        "months_since_review": 0,
        "needs_figure": 1670.0,
        "award_to_needs_ratio": 0.598,
        "payment_count": 6,
        "mean_payment_amount": 1000.0,
        "max_payment_amount": 1000.0,
        "min_payment_amount": 1000.0,
        "std_payment_amount": 0.0,
        "total_paid_amount": 6000.0,
        "payment_method_switches": 0,
        "mean_payment_vs_award_diff": 0.0,
        "max_payment_vs_award_diff": 0.0,
        "payment_to_award_ratio": 1.0,
        "max_payment_to_needs_ratio": 0.598,
        "payment_exceeds_needs": 0,
        "payment_spread": 0.0,
        "payment_volatility_ratio": 0.0,
        "status_payment_anomaly": 0,
        "dept_adjustments_count": 0,
        "dept_contact_attempts": 0
    }])

    busy_case = base_case.copy()
    busy_case['payment_adjustments'] = 10
    busy_case['contact_attempts'] = 20
    busy_case['dept_adjustments_count'] = 10
    busy_case['dept_contact_attempts'] = 20

    engine = ScoringEngine()
    score_base, _ = engine.compute_scores(base_case)
    score_busy, _ = engine.compute_scores(busy_case)

    # Resident risk score must be identical (departmental activity has 0 weight in risk)
    assert score_base.iloc[0]['risk_score'] == score_busy.iloc[0]['risk_score']
