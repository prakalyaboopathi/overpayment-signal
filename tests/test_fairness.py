import pytest
import pandas as pd
from src.data_loader import load_data
from src.validation import validate_and_clean_data
from src.features import engineer_features
from src.scoring import ScoringEngine
from src.explanations import generate_worklist_explanations
from src.fairness import analyze_fairness, DEMOGRAPHIC_FIELDS

def test_fairness_report_covers_all_demographic_fields():
    cases_df, payments_df = load_data("data")
    cases_clean, payments_clean, _ = validate_and_clean_data(cases_df, payments_df)
    features_df = engineer_features(cases_clean, payments_clean)

    engine = ScoringEngine()
    scored_full_df, _ = engine.compute_scores(features_df)
    top20_df, top20_subscores = engine.get_ranked_worklist(scored_full_df, top_n=20)
    top20_explained = generate_worklist_explanations(top20_df, top20_subscores)

    fairness_df = analyze_fairness(scored_full_df, top20_explained)

    # 1. Required columns present
    required_cols = [
        'demographic_category', 'group_value', 'population_count', 
        'population_share_pct', 'top20_selected_count', 'top20_share_pct', 
        'group_selection_rate_pct', 'relative_selection_ratio', 'group_mean_risk_score'
    ]
    for col in required_cols:
        assert col in fairness_df.columns

    # 2. Every demographic category present
    categories_present = set(fairness_df['demographic_category'])
    for field in DEMOGRAPHIC_FIELDS:
        assert field in categories_present

    # 3. Sum of top20_selected_count per category equals 20
    for field in DEMOGRAPHIC_FIELDS:
        cat_df = fairness_df[fairness_df['demographic_category'] == field]
        assert cat_df['top20_selected_count'].sum() == 20
