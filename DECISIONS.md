# Architectural & Strategic Decisions Log

## Project: The Overpayment Signal (Brite Spark 2026, Problem 6)

This document records the design decisions, trade-offs, governance considerations, and response to Day 2 investigator feedback for **The Overpayment Signal** case prioritization system.

---

## 1. Core Modeling Decisions

### 1.1 Why a Transparent Ranking System Instead of Deep Learning / Black-Box Models
* **Decision**: We built a deterministic, rule-based composite risk scoring engine with modular, weighted sub-scores rather than training a deep neural network, random forest, or XGBoost model.
* **Rationale**:
  1. **No Ground-Truth Labels**: The dataset contains 4,200 cases and 24,756 payment records without improper-payment labels. Training a supervised classification model would require inventing synthetic labels, which violates competition floor rules and creates fake precision.
  2. **Investigator Explainability**: Human investigators require clear, actionable, evidence-backed reasons to justify spending review hours on a file. Black-box SHAP/LIME outputs or opaque neural network probabilities are unexplainable to non-technical operational staff.
  3. **Auditability & Regulatory Governance**: Public benefits administration requires complete auditability. Every point in a case's risk score can be directly traced back to specific policy rules (e.g. household needs figure limits, payment-to-award discrepancies).

### 1.2 Why Supervised Classification Was Rejected
* **Decision**: We explicitly rejected fitting a supervised classifier to predict "fraud".
* **Rationale**: Claiming a supervised accuracy score (e.g., "95% accuracy in detecting fraud") on unlabeled synthetic data is deceptive. The system is designed strictly as a **risk prioritisation signal** for allocating human review capacity, not a predictive decision engine.

---

## 2. Feature Engineering & Selection Strategy

### 2.1 Selected Features
* **Financial & Award Discrepancies**:
  - `mean_payment_vs_award_diff`: Difference between observed mean payment received and official recorded monthly award on case file.
  - `max_payment_to_needs_ratio`: Maximum payment received relative to policy household needs figure ($1,240 for 1-person up to $2,990 for 6+ person households).
* **Payment Volatility & Anomalies**:
  - `std_payment_amount`: Standard deviation of monthly payments across 6 months.
  - `payment_method_switches`: Number of times payment delivery flipped between Transfer and Card.
  - `status_payment_anomaly`: Flag for payments issued to Suspended or Closed cases.
* **Administrative Context**:
  - `months_since_review`: Months since last completed review.

### 2.2 Rejected & Excluded Features
* **Demographic Variables (`age_band`, `language_preference`, `district`, `tenure`)**: Excluded from risk scoring to prevent direct demographic discrimination. Kept exclusively in `src/fairness.py` for demographic disparity monitoring.

---

## 3. Day 2 Investigator Feedback Integration

### 3.1 What Changed After Case C-33248 Review
* **Investigator Findings**: Senior Investigator M. Rasmussen reviewed top-ranked case `C-33248` and reported it was a false referral. The case's 5 adjustments were internal department corrections following timely resident income reports, and the 7 contact attempts resulted from administrative language barriers (sending English correspondence to a Spanish-preferring household).
* **Architecture Modification**:
  1. Departmental processing activity (`payment_adjustments` and `contact_attempts`) was **removed** as a positive risk indicator in `src/scoring.py` (given 0 weight).
  2. Departmental activity was re-classified strictly as administrative contextual metadata.
  3. Regression Test (`tests/test_day2_feedback.py`) was introduced to ensure `C-33248` drops out of the Top 20 (it now ranks #501).

### 3.2 What Was Deliberately NOT Changed
* **Base Risk Scoring Architecture**: We did not rewrite the pipeline or introduce a case-specific hardcoded hack (e.g. `if case_id == 'C-33248': score = 0`). Instead, we updated the underlying scoring rule definitions so all similar "busy files" are evaluated correctly.

---

## 4. Governance & "What System Must NEVER Decide"

### 4.1 Automated Decision Boundaries
The system is explicitly prohibited from taking automated administrative actions. It must **NEVER** automatically:
1. Stop, suspend, reduce, or deny a resident's benefit payments.
2. Accuse a resident of fraud or create formal fraud findings.
3. Prioritise cases for punitive action based on demographic traits.
4. Replace human investigator review.

---

## 5. What Would Need to Be True Before Using on Real Residents?

Before deploying this system on real resident data, the following criteria must be satisfied:

1. **Empirical Validation on Historical Audits**: The scoring rules must be validated against a sample of past audited cases to confirm that high-scoring patterns correspond to true overpayments.
2. **Comprehensive Disparity Auditing**: Continuous fairness testing across all protected demographic groups to ensure proxy variables (e.g. zip code or housing type) do not produce systemic bias.
3. **Appeals & Recourse Mechanism**: A transparent, fast-track process for residents to clear administrative misunderstandings (such as language preferences or variable working hours).
4. **Investigator Feedback Loop**: A built-in protocol allowing investigators to flag false positive patterns (like the Day 2 feedback loop) to update scoring rules periodically.

---

## 6. Known Limitations & Future Work

* **System Limitations**: The system evaluates static 6-month historical windows and does not currently connect directly to real-time payroll databases or tax reporting APIs.
* **What We Would Fix First**: Implement automated anomaly detection for multi-member household income pooling and integrate automated translation/correspondence tracking to prevent language-based contact attempt spikes.
