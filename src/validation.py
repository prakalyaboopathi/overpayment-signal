import logging
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ValidationReport:
    def __init__(self):
        self.warnings: List[str] = []
        self.dropped_payments_count: int = 0
        self.dropped_cases_count: int = 0
        self.orphaned_payments_count: int = 0
        self.cases_without_payments: List[str] = []

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_warnings": len(self.warnings),
            "dropped_payments": self.dropped_payments_count,
            "dropped_cases": self.dropped_cases_count,
            "orphaned_payments": self.orphaned_payments_count,
            "cases_without_payments_count": len(self.cases_without_payments),
            "warnings_list": self.warnings
        }

def validate_and_clean_data(cases_raw: pd.DataFrame, payments_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """
    Validates cases and payments dataframes, checks quality rules, cleans/parses types,
    and records warnings/exclusions in a ValidationReport.
    """
    report = ValidationReport()
    cases = cases_raw.copy()
    payments = payments_raw.copy()

    # 1. Duplicate Check
    if cases['case_id'].duplicated().any():
        dups = cases[cases['case_id'].duplicated()]['case_id'].tolist()
        report.add_warning(f"Duplicate case_ids found in cases.csv: {dups}")
        cases = cases.drop_duplicates(subset=['case_id'], keep='first')

    if payments['payment_id'].duplicated().any():
        dup_p = payments[payments['payment_id'].duplicated()]['payment_id'].tolist()
        report.add_warning(f"Duplicate payment_ids found in payments.csv: {dup_p}. Deduplicating...")
        payments = payments.drop_duplicates(subset=['payment_id'], keep='first')

    # 2. Check numeric coercions for numeric fields in cases
    for num_col in ['household_size', 'monthly_award', 'payment_adjustments', 'contact_attempts', 'months_since_review']:
        if num_col in cases.columns:
            invalid_mask = pd.to_numeric(cases[num_col], errors='coerce').isna() & cases[num_col].notna() & (cases[num_col] != '')
            if invalid_mask.any():
                invalid_ids = cases[invalid_mask]['case_id'].tolist()
                report.add_warning(f"Non-numeric values in cases.{num_col} for case_ids: {invalid_ids}")

    cases['household_size'] = pd.to_numeric(cases['household_size'], errors='coerce').fillna(1).astype(int)
    cases['monthly_award'] = pd.to_numeric(cases['monthly_award'], errors='coerce').fillna(0.0).astype(float)
    cases['payment_adjustments'] = pd.to_numeric(cases['payment_adjustments'], errors='coerce').fillna(0).astype(int)
    cases['contact_attempts'] = pd.to_numeric(cases['contact_attempts'], errors='coerce').fillna(0).astype(int)
    cases['months_since_review'] = pd.to_numeric(cases['months_since_review'], errors='coerce').fillna(0).astype(int)

    # Validate household size bounds
    invalid_hh = cases[(cases['household_size'] < 1) | (cases['household_size'] > 10)]
    if not invalid_hh.empty:
        report.add_warning(f"Cases with out-of-range household_size (<1 or >10): {invalid_hh['case_id'].tolist()}")
        cases['household_size'] = cases['household_size'].clip(lower=1, upper=10)

    # Validate monthly_award non-negative
    neg_award = cases[cases['monthly_award'] < 0]
    if not neg_award.empty:
        report.add_warning(f"Cases with negative monthly_award: {neg_award['case_id'].tolist()}")
        cases.loc[cases['monthly_award'] < 0, 'monthly_award'] = 0.0

    # Validate Status category
    valid_statuses = {'Active', 'Suspended', 'Closed'}
    invalid_status = cases[~cases['status'].isin(valid_statuses)]
    if not invalid_status.empty:
        report.add_warning(f"Cases with unexpected status: {invalid_status[['case_id', 'status']].to_dict('records')}")
        cases.loc[~cases['status'].isin(valid_statuses), 'status'] = 'Active'

    # Validate Date formats
    for date_col in ['opened_date', 'closure_month']:
        if date_col in cases.columns:
            # check malformed
            non_empty = cases[date_col].notna() & (cases[date_col] != '')
            parsed = pd.to_datetime(cases.loc[non_empty, date_col], errors='coerce')
            malformed = cases.loc[non_empty & parsed.isna(), 'case_id'].tolist()
            if malformed:
                report.add_warning(f"Malformed date in cases.{date_col} for case_ids: {malformed}")

    # 3. Clean and validate Payments
    invalid_amt = pd.to_numeric(payments['amount'], errors='coerce').isna() & payments['amount'].notna()
    if invalid_amt.any():
        bad_p_ids = payments[invalid_amt]['payment_id'].tolist()
        report.add_warning(f"Non-numeric payment amount for payment_ids: {bad_p_ids}. Dropping invalid payments.")
        payments = payments[~invalid_amt]
        report.dropped_payments_count += invalid_amt.sum()

    payments['amount'] = pd.to_numeric(payments['amount'], errors='coerce').fillna(0.0).astype(float)
    
    neg_pmt = payments[payments['amount'] < 0]
    if not neg_pmt.empty:
        report.add_warning(f"Negative payment amounts detected for payment_ids: {neg_pmt['payment_id'].tolist()}")
        payments.loc[payments['amount'] < 0, 'amount'] = 0.0

    # Validate pay_month format
    valid_pay_month = pd.to_datetime(payments['pay_month'], format='%Y-%m', errors='coerce').notna()
    if (~valid_pay_month).any():
        bad_months = payments[~valid_pay_month]['payment_id'].tolist()
        report.add_warning(f"Malformed pay_month for payment_ids: {bad_months}")

    # Unknown case_ids in payments
    known_cases = set(cases['case_id'])
    unknown_pmt_cases = set(payments['case_id']) - known_cases
    if unknown_pmt_cases:
        report.orphaned_payments_count = len(payments[payments['case_id'].isin(unknown_pmt_cases)])
        report.add_warning(f"Payments referencing unknown case_ids ({len(unknown_pmt_cases)} unique case_ids). Dropping orphan payments.")
        payments = payments[payments['case_id'].isin(known_cases)]

    # Cases without payments
    cases_with_pmts = set(payments['case_id'])
    cases_no_pmts = list(known_cases - cases_with_pmts)
    if cases_no_pmts:
        report.cases_without_payments = cases_no_pmts
        report.add_warning(f"Cases without any payment records: {len(cases_no_pmts)}")

    return cases, payments, report
