import pytest
import pandas as pd
from src.data_loader import load_data
from src.validation import validate_and_clean_data
from src.features import engineer_features
from src.scoring import ScoringEngine
from src.explanations import generate_worklist_explanations

def test_explanations_exist_for_all_top_20():
    cases_df, payments_df = load_data("data")
    cases_clean, payments_clean, _ = validate_and_clean_data(cases_df, payments_df)
    features_df = engineer_features(cases_clean, payments_clean)

    engine = ScoringEngine()
    top20_df, top20_subscores = engine.get_ranked_worklist(features_df, top_n=20)
    top20_explained = generate_worklist_explanations(top20_df, top20_subscores)

    assert "explanation" in top20_explained.columns
    assert "evidence" in top20_explained.columns

    for idx, row in top20_explained.iterrows():
        exp = row["explanation"]
        ev = row["evidence"]
        
        # Must be non-empty strings
        assert isinstance(exp, str) and len(exp.strip()) > 10
        assert isinstance(ev, str) and len(ev.strip()) > 5

        # Must not contain ML jargon
        assert "SHAP" not in exp
        assert "feature_importance" not in exp
        assert "classifier" not in exp

        # Must not accuse of fraud
        assert "committed fraud" not in exp.lower()
        assert "guilty of fraud" not in exp.lower()
