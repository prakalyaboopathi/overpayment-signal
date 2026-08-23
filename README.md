# The Overpayment Signal — Case Prioritisation & Governance System

**Brite Spark 2026 — Problem 6**

An explainable, audit-ready case prioritisation system designed to produce a ranked worklist of the **20 cases most worth reviewing** by benefits investigators, incorporating Day 2 investigator feedback and comprehensive demographic fairness monitoring.

---

## Executive Summary & Approach

Public benefits programs require efficient allocation of limited investigative resources without relying on unexplainable black-box AI models or unvalidated fraud labels.

**Key Design Principles**:
* **No Synthetic Fraud Labels**: Avoids treating unlabeled data as supervised classification.
* **Explainable Policy Scoring**: Transparent, rule-based composite scoring built on policy rules (household needs figures, payment-to-award discrepancies, payment volatility).
* **Day 2 Feedback Architecture**: Distinguishes resident payment risk from departmental processing activity (`payment_adjustments` and `contact_attempts`), eliminating false-positive referrals for administrative "busy files" like case `C-33248`.
* **Demographic Fairness Analysis**: Comprehensive monitoring across `age_band`, `language_preference`, `district`, and `tenure`.
* **Explicit Governance Boundaries**: Clear written rules defining what the system must **NEVER** decide automatically.

---

## Quick Start (Clean Clone Setup)

### 1. Prerequisites
* Python 3.10+ (tested on Python 3.14)

### 2. Create and Activate Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline Demo
```bash
python -m src.cli
```

### 5. Run the Test Suite
```bash
pytest
```

---

## Architecture & Data Flow

```
overpayment-signal/
├── README.md                          # Project documentation & run instructions
├── DECISIONS.md                       # Architectural design decisions & trade-offs
├── AI-USAGE.md                        # Transparency log of AI tools used
├── requirements.txt                   # Dependency manifest (pandas, pytest, numpy)
├── .gitignore
├── data/
│   ├── cases.csv                      # 4,200 cases
│   └── payments.csv                   # 24,756 payment records (July–Dec 2025)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # CSV data ingestion
│   ├── validation.py                  # Input validation & data quality checks
│   ├── features.py                    # Case-level feature engineering & policy math
│   ├── scoring.py                     # Configurable, transparent risk scoring engine
│   ├── explanations.py                # Plain-language reason generator
│   ├── fairness.py                    # Demographic disparity monitoring
│   ├── report.py                      # CSV & human-readable output formatting
│   └── cli.py                         # Command-line entry point
├── tests/
│   ├── test_data_validation.py        # Data quality unit tests
│   ├── test_features.py               # Policy math & feature tests
│   ├── test_scoring.py                # Scoring rules & extensibility tests
│   ├── test_explanations.py           # Explanation text quality tests
│   ├── test_fairness.py               # Demographic monitoring tests
│   └── test_day2_feedback.py          # Day 2 feedback & C-33248 regression tests
└── outputs/
    ├── top20.csv                      # Top 20 ranked worklist with evidence
    └── fairness_report.csv            # Demographic selection ratios & breakdown
```

---

## Policy Needs Figures & How Scoring Works

### 1. Household Policy Needs Table
Monthly needs figure by household size (from official policy manual):

| Household Size | 1 | 2 | 3 | 4 | 5 | 6+ |
|:---|---:|---:|---:|---:|---:|---:|
| **Needs Figure** | $1,240 | $1,670 | $2,000 | $2,330 | $2,660 | $2,990 |

*Policy Rule*: An award should not normally approach the full needs figure, because award = needs figure less household countable income.

### 2. Transparent Scoring Engine (`src/scoring.py`)
The risk score is a composite weighted index (0–100):
1. **Payment vs Award Discrepancy (35% Weight)**: Average monthly payment received exceeds recorded award on case file.
2. **Payment Exceeds Needs Limit (25% Weight)**: Payment amount approaches or exceeds policy household needs limit.
3. **Payment Volatility (15% Weight)**: High standard deviation in monthly payment amounts across the 6-month period.
4. **Overdue Review Anomaly (15% Weight)**: Months since last review combined with payment variance.
5. **Status Anomaly (5% Weight)**: Payments issued to `Suspended` or `Closed` cases.
6. **Payment Method Instability (5% Weight)**: Frequent switches between `Transfer` and `Card`.
7. **Departmental Activity (0% Weight)**: `payment_adjustments` and `contact_attempts` are categorized as administrative processing activity and assigned 0 weight in resident risk ranking.

---

## Day 2 Investigator Feedback Integration

Following Senior Investigator M. Rasmussen's review of case `C-33248`:
* **Issue**: `C-33248` was previously flagged as a "busy file" due to 5 adjustments (department processing corrections) and 7 contact attempts (Spanish language communication barrier).
* **Fix**: Departmental activity variables were removed from positive risk scoring.
* **Verification**: `C-33248` drops out of the Top 20 to rank #501. Regression test `tests/test_day2_feedback.py` ensures this policy is maintained.

---

## Plain-Language Explanations Strategy

Explanations are generated for investigators without ML jargon or accusations of fraud.

*Example Output*:
> "Observed monthly payments consistently exceeded the official recorded monthly award on the case file. Payment amounts approached or exceeded the policy monthly needs limit for a household size of 2. Average monthly payment received ($1,420.00) exceeded recorded award ($950.00) by $470.00 per month."

---

## Mandatory Governance Boundary

This system is strictly an investigative prioritisation tool and **MUST NEVER** be allowed to automatically:
1. Stop, suspend, reduce, or deny a resident's benefit payment.
2. Accuse a resident of fraud or generate legal findings.
3. Prioritise cases for punitive action based on demographic traits.

---

## Outputs

After running `python -m src.cli`, the following files are produced in `outputs/`:

1. `outputs/top20.csv`: Exactly 20 cases with columns `rank`, `case_id`, `score`, `confidence`, `explanation`, `evidence`.
2. `outputs/fairness_report.csv`: Complete disparity metrics across `age_band`, `language_preference`, `district`, and `tenure`.

---

## Repository

* GitHub Repository: `https://github.com/user/overpayment-signal` *(Remote repository ready for submission)*
