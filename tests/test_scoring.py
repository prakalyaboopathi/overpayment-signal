import pytest
import pandas as pd
from src.data_loader import load_data
from src.validation import validate_and_clean_data
from src.features import engineer_features
from src.scoring import ScoringEngine

def test_scoring_returns_exact_top_20():
    cases_df, payments_df = load_data("data")
    cases_clean, payments_clean, _ = validate_and_clean_data(cases_df, payments_df)
    features_df = engineer_features(cases_clean, payments_clean)

    engine = ScoringEngine()
    top20_df, top20_subscores = engine.get_ranked_worklist(features_df, top_n=20)

    # 1. Exactly 20 cases returned
    assert len(top20_df) == 20

    # 2. No duplicate case_ids in top 20
    assert top20_df['case_id'].nunique() == 20

    # 3. Sorted by risk_score descending
    scores = top20_df['risk_score'].tolist()
    assert scores == sorted(scores, reverse=True)

    # 4. Ranks are 1 to 20
    assert top20_df['rank'].tolist() == list(range(1, 21))

def test_scoring_extensibility_custom_rules():
    cases_df, payments_df = load_data("data")
    cases_clean, payments_clean, _ = validate_and_clean_data(cases_df, payments_df)
    features_df = engineer_features(cases_clean, payments_clean)

    # Modular engine supports custom rule injection
    engine = ScoringEngine(rules=[])
    scored_df, subscores = engine.compute_scores(features_df)
    assert (scored_df['risk_score'] == 0).all()
