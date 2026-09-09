from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

from src.data_loader import load_data
from src.validation import validate_and_clean_data
from src.features import engineer_features
from src.scoring import ScoringEngine
from src.explanations import generate_worklist_explanations
from src.fairness import analyze_fairness
from src.report import save_outputs

app = FastAPI(
    title="The Overpayment Signal API",
    description="Explainable Case Prioritisation & Governance Web API",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount public directory if exists for single-server serving
public_path = Path(__file__).parent.parent / "public"
if public_path.exists():
    app.mount("/app", StaticFiles(directory=str(public_path), html=True), name="static")

# Shared cache for pipeline results to make API endpoints instant
_cached_state: Dict[str, Any] = {}

def run_pipeline_internal(data_dir: str = "data", output_dir: str = "outputs") -> Dict[str, Any]:
    """Runs the existing Python pipeline without duplicating any scoring or validation logic."""
    cases_raw, payments_raw = load_data(data_dir=data_dir)
    cases_clean, payments_clean, validation_report = validate_and_clean_data(cases_raw, payments_raw)
    features_df = engineer_features(cases_clean, payments_clean)

    engine = ScoringEngine()
    scored_full_df, subscores_df = engine.compute_scores(features_df)
    
    top20_raw, top20_subscores = engine.get_ranked_worklist(scored_full_df, top_n=20)
    top20_explained = generate_worklist_explanations(top20_raw, top20_subscores)
    fairness_df = analyze_fairness(scored_full_df, top20_explained)

    top20_file, fairness_file = save_outputs(top20_explained, fairness_df, output_dir=output_dir)

    state = {
        "cases_count": len(cases_clean),
        "payments_count": len(payments_clean),
        "validation": validation_report.summary(),
        "top20_explained": top20_explained,
        "top20_subscores": top20_subscores,
        "fairness_df": fairness_df,
        "scored_full_df": scored_full_df,
        "payments_clean": payments_clean
    }
    _cached_state.update(state)
    return state

def get_or_run_state() -> Dict[str, Any]:
    if "top20_explained" not in _cached_state:
        return run_pipeline_internal()
    return _cached_state

@app.get("/")
def read_root():
    index_file = Path(__file__).parent.parent / "public" / "index.html"
    if index_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(index_file))
    return {"message": "Overpayment Signal API active. Visit /api/health"}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "The Overpayment Signal API",
        "version": "1.0.0"
    }

@app.post("/api/analyze")
def run_analysis():
    state = run_pipeline_internal()
    return {
        "status": "success",
        "message": "Pipeline analysis executed successfully",
        "cases_analyzed": state["cases_count"],
        "payments_analyzed": state["payments_count"],
        "priority_cases_count": len(state["top20_explained"]),
        "warnings_count": state["validation"]["total_warnings"]
    }

@app.get("/api/results")
def get_results():
    state = get_or_run_state()
    top20_df = state["top20_explained"]
    
    # Export clean list matching required schema
    records = []
    for _, row in top20_df.iterrows():
        records.append({
            "rank": int(row["rank"]),
            "case_id": str(row["case_id"]),
            "score": float(row["risk_score"]),
            "confidence": str(row["signal_confidence"]),
            "explanation": str(row["explanation"]),
            "evidence": str(row["evidence"]),
            "district": str(row["district"]),
            "household_size": int(row["household_size"]),
            "age_band": str(row["age_band"]),
            "language_preference": str(row["language_preference"]),
            "tenure": str(row["tenure"]),
            "status": str(row["status"]),
            "monthly_award": float(row["monthly_award"]),
            "mean_payment_amount": float(row["mean_payment_amount"]),
            "max_payment_amount": float(row["max_payment_amount"]),
            "std_payment_amount": float(row["std_payment_amount"]),
            "months_since_review": int(row["months_since_review"]),
            "dept_adjustments_count": int(row["dept_adjustments_count"]),
            "dept_contact_attempts": int(row["dept_contact_attempts"])
        })
    return records

@app.get("/api/fairness")
def get_fairness():
    state = get_or_run_state()
    fairness_df = state["fairness_df"]
    return fairness_df.to_dict(orient="records")

@app.get("/api/cases/{case_id}")
def get_case_detail(case_id: str):
    state = get_or_run_state()
    scored_full_df = state["scored_full_df"]
    payments_clean = state["payments_clean"]

    case_rows = scored_full_df[scored_full_df["case_id"] == case_id]
    if case_rows.empty:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    row = case_rows.iloc[0].to_dict()
    case_pmts = payments_clean[payments_clean["case_id"] == case_id].sort_values("pay_month").to_dict(orient="records")

    return {
        "case_id": str(row["case_id"]),
        "score": float(row["risk_score"]),
        "confidence": str(row["signal_confidence"]),
        "district": str(row["district"]),
        "household_size": int(row["household_size"]),
        "needs_figure": float(row["needs_figure"]),
        "age_band": str(row["age_band"]),
        "language_preference": str(row["language_preference"]),
        "tenure": str(row["tenure"]),
        "status": str(row["status"]),
        "monthly_award": float(row["monthly_award"]),
        "mean_payment_amount": float(row["mean_payment_amount"]),
        "max_payment_amount": float(row["max_payment_amount"]),
        "std_payment_amount": float(row["std_payment_amount"]),
        "months_since_review": int(row["months_since_review"]),
        "dept_adjustments_count": int(row["dept_adjustments_count"]),
        "dept_contact_attempts": int(row["dept_contact_attempts"]),
        "payments_history": case_pmts
    }
