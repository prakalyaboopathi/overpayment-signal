import pandas as pd
from typing import List, Dict, Any

def generate_case_explanation(row: pd.Series, subscores_row: pd.Series) -> Tuple[str, str]:
    """
    Generates a plain-language explanation and supporting evidence for a case record.
    
    Returns:
        tuple[str, str]: (plain_language_explanation, supporting_evidence)
    """
    reasons = []
    evidence = []

    case_id = row['case_id']
    monthly_award = row['monthly_award']
    mean_pmt = row['mean_payment_amount']
    max_pmt = row['max_payment_amount']
    needs_fig = row['needs_figure']
    hh_size = int(row['household_size'])
    months_rev = int(row['months_since_review'])
    status = row['status']
    pmt_diff = row['mean_payment_vs_award_diff']
    std_pmt = row['std_payment_amount']
    switches = int(row['payment_method_switches'])

    # 1. Payment vs Award Discrepancy
    if subscores_row.get('payment_vs_award_discrepancy', 0) > 25.0 or pmt_diff > 50:
        reasons.append("Observed monthly payments consistently exceeded the official recorded monthly award on the case file.")
        evidence.append(f"Average monthly payment received (${mean_pmt:,.2f}) exceeded recorded award (${monthly_award:,.2f}) by ${pmt_diff:,.2f} per month.")

    # 2. Payment Exceeds Household Needs Figure
    if subscores_row.get('payment_exceeds_needs', 0) > 25.0 or max_pmt > (needs_fig * 0.85):
        pct_needs = (max_pmt / needs_fig) * 100
        reasons.append(f"Payment amounts approached or exceeded the policy monthly needs limit for a household size of {hh_size}.")
        evidence.append(f"Maximum payment (${max_pmt:,.2f}) reached {pct_needs:.1f}% of the household needs figure (${needs_fig:,.2f}).")

    # 3. Payment Volatility / Swings
    if subscores_row.get('payment_volatility', 0) > 25.0 or std_pmt > 75:
        reasons.append("Payment amounts fluctuated significantly across the 6-month period while the recorded award remained constant.")
        evidence.append(f"Payment volatility standard deviation was ${std_pmt:,.2f} (min payment: ${row['min_payment_amount']:,.2f}, max payment: ${max_pmt:,.2f}).")

    # 4. Overdue Review Anomaly
    if subscores_row.get('overdue_review', 0) > 25.0 or months_rev >= 12:
        reasons.append(f"The case has gone {months_rev} months without a completed review while exhibiting active payment pattern changes.")
        evidence.append(f"Last formal review was completed {months_rev} months ago (policy review interval is typically 12 months).")

    # 5. Status Anomaly
    if subscores_row.get('status_anomaly', 0) > 0:
        reasons.append(f"Payments were issued to a case currently flagged as {status} in the system.")
        evidence.append(f"Case status is registered as '{status}' but received {int(row['payment_count'])} payments totaling ${row['total_paid_amount']:,.2f}.")

    # 6. Payment Method Switches
    if subscores_row.get('method_instability', 0) > 25.0:
        reasons.append("Payment distribution method changed multiple times across recent months.")
        evidence.append(f"Payment delivery method switched {switches} times between Transfer and Card.")

    # Primary plain-language explanation synthesis
    if reasons:
        explanation = " ".join(reasons) + " This case has a payment pattern worth reviewing against official case documentation."
    else:
        explanation = "Minor payment variance observed. This case is flagged for routine quality assurance review."

    if evidence:
        evidence_str = "; ".join(evidence)
    else:
        evidence_str = f"Recorded award: ${monthly_award:,.2f}, Mean payment: ${mean_pmt:,.2f}, Household size: {hh_size}."

    return explanation, evidence_str


def generate_worklist_explanations(top_df: pd.DataFrame, top_subscores: pd.DataFrame) -> pd.DataFrame:
    """
    Appends plain-language explanation and evidence columns to the top N worklist.
    """
    explanations = []
    evidences = []

    for idx, row in top_df.iterrows():
        sub_row = top_subscores.loc[idx] if idx in top_subscores.index else pd.Series()
        exp, ev = generate_case_explanation(row, sub_row)
        explanations.append(exp)
        evidences.append(ev)

    result_df = top_df.copy()
    result_df['explanation'] = explanations
    result_df['evidence'] = evidences

    return result_df
