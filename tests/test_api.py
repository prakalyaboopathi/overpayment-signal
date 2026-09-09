import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_api_analyze():
    response = client.post("/api/analyze")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["cases_analyzed"] == 4200
    assert data["priority_cases_count"] == 20

def test_api_results():
    response = client.get("/api/results")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 20
    assert "rank" in data[0]
    assert "case_id" in data[0]
    assert "score" in data[0]
    assert "explanation" in data[0]

def test_api_fairness():
    response = client.get("/api/fairness")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    categories = set(item["demographic_category"] for item in data)
    assert "age_band" in categories
    assert "language_preference" in categories
    assert "district" in categories
    assert "tenure" in categories

def test_api_case_detail():
    response = client.get("/api/cases/C-33201")
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == "C-33201"
    assert "payments_history" in data
