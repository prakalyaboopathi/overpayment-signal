# AI Usage Transparency Report

## Project: The Overpayment Signal (Brite Spark 2026, Problem 6)

In accordance with competition rules, this document details how Artificial Intelligence (AI) tools were utilized throughout the design, implementation, testing, and documentation of **The Overpayment Signal** codebase.

---

## 1. Summary of AI Assistance

AI tools were used interactively during development in the following capacities:

| Scope | Percentage Contribution | Description |
| :--- | :--- | :--- |
| **Project Scaffolding** | ~15% | Generating initial directory layouts, standard module structure, and configuration boilerplate. |
| **Code Implementation** | ~35% | Assisting with pandas vectorization, policy needs lookup math, and modular scoring class definitions. |
| **Test Generation** | ~25% | Drafting pytest unit tests for edge cases (missing data, malformed dates, empty payment history). |
| **Documentation & Reports** | ~15% | Structuring Markdown reports, formatting terminal summary tables, and framing governance statements. |
| **Code Review & Debugging** | ~10% | Verifying edge cases in demographic disparity metrics and confirming Day 2 feedback regression logic. |

---

## 2. Key Areas of AI Utilization

### 2.1 Project Scaffolding & Architecture Design
- Used AI to brainstorm clean, modular Python project layouts separating data loading (`src/data_loader.py`), validation (`src/validation.py`), feature engineering (`src/features.py`), scoring (`src/scoring.py`), explanations (`src/explanations.py`), and fairness monitoring (`src/fairness.py`).

### 2.2 Feature Engineering & Policy Logic
- AI assisted in converting policy manual rules (such as household size needs figures $1,240–$2,990) into vectorized pandas feature transformations.

### 2.3 Test Suite & Edge Case Coverage
- AI helped write unit test scenarios in `tests/` covering non-numeric inputs, negative payment amounts, orphaned payment references, and empty payment histories.

### 2.4 Day 2 Feedback Adaptation
- AI was used to review the investigator write-up for case `C-33248` and verify that set weights for `payment_adjustments` and `contact_attempts` were reduced to 0 in `src/scoring.py` while adding regression test assertions.

---

## 3. Human Oversight & Code Integrity

- **Code Ownership**: All code, mathematical logic, policy thresholds, and architectural decisions were reviewed, verified, and validated.
- **Explainability**: No proprietary black-box AI endpoints or non-interpretable models are embedded in the runtime execution. The scoring engine remains 100% deterministic, rule-based, and human-explainable.
