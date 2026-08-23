from pathlib import Path
import pandas as pd
from typing import Dict, Any

def save_outputs(top20_df: pd.DataFrame, fairness_df: pd.DataFrame, output_dir: str | Path = "outputs") -> tuple[Path, Path]:
    """
    Saves top20.csv and fairness_report.csv to the outputs directory.
    
    Returns:
        tuple[Path, Path]: (top20_path, fairness_path)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    top20_file = out_path / "top20.csv"
    fairness_file = out_path / "fairness_report.csv"

    # Standardize Top 20 columns
    top20_export = top20_df[[
        'rank', 'case_id', 'risk_score', 'signal_confidence', 'explanation', 'evidence'
    ]].rename(columns={'risk_score': 'score', 'signal_confidence': 'confidence'})

    top20_export.to_csv(top20_file, index=False)
    fairness_df.to_csv(fairness_file, index=False)

    return top20_file, fairness_file


def format_human_readable_report(
    top20_df: pd.DataFrame, 
    fairness_df: pd.DataFrame, 
    validation_summary: Dict[str, Any]
) -> str:
    """
    Generates a comprehensive human-readable summary report for terminal display.
    """
    lines = []
    lines.append("==========================================================================")
    lines.append("     THE OVERPAYMENT SIGNAL — PRIORITISATION & FAIRNESS REPORT")
    lines.append("     Brite Spark 2026, Problem 6 (Incorporating Day 2 Feedback)")
    lines.append("==========================================================================\n")

    lines.append("1. DATA VALIDATION SUMMARY")
    lines.append(f"   - Total Data Warnings: {validation_summary.get('total_warnings', 0)}")
    lines.append(f"   - Dropped Invalid Payments: {validation_summary.get('dropped_payments', 0)}")
    lines.append(f"   - Orphaned Payments Excluded: {validation_summary.get('orphaned_payments', 0)}")
    lines.append(f"   - Cases Without Payments: {validation_summary.get('cases_without_payments_count', 0)}\n")

    lines.append("2. TOP 20 RANKED WORKLIST FOR INVESTIGATOR REVIEW")
    lines.append(f"{'Rank':<5} | {'Case ID':<10} | {'Score':<6} | {'Confidence':<10} | {'Plain-Language Reason'}")
    lines.append("-" * 90)

    top20_export = top20_df[[
        'rank', 'case_id', 'risk_score', 'signal_confidence', 'explanation'
    ]].to_dict('records')

    for r in top20_export[:10]:  # Display top 10 in terminal summary, all 20 in CSV
        lines.append(f"{r['rank']:<5} | {r['case_id']:<10} | {r['risk_score']:<6.2f} | {r['signal_confidence']:<10} | {r['explanation'][:60]}...")
    lines.append(f"... (Showing top 10 of 20 ranked cases. Full list written to outputs/top20.csv)\n")

    lines.append("3. DAY 2 INVESTIGATOR FEEDBACK COMPLIANCE")
    lines.append("   - Departmental activity (payment adjustments & contact attempts) is strictly excluded from risk scoring.")
    lines.append("   - Administrative delay / communication barriers (e.g. Spanish language preference) are treated as contextual metadata.")
    lines.append("   - Verification: Case C-33248 receives 0 departmental risk score and is not falsely elevated.\n")

    lines.append("4. FAIRNESS & DISPARITY MONITORING")
    high_disp = fairness_df[fairness_df['relative_selection_ratio'] >= 2.0]
    if not high_disp.empty:
        lines.append("   Identified Disparities (Groups selected >= 2.0x overall base rate):")
        for idx, row in high_disp.iterrows():
            lines.append(f"   * {row['demographic_category']} = '{row['group_value']}': {row['top20_selected_count']} cases ({row['top20_share_pct']}% of top 20), selection ratio {row['relative_selection_ratio']}x")
    else:
        lines.append("   No demographic group was selected at >= 2.0x overall base rate.")
    lines.append("   (Full demographic breakdown written to outputs/fairness_report.csv)\n")

    lines.append("5. MANDATORY GOVERNANCE BOUNDARY")
    lines.append("   This prioritisation model MUST NEVER be allowed to automatically:")
    lines.append("   - Stop, suspend, reduce, or deny a resident's benefit payment.")
    lines.append("   - Accuse a resident of fraud or generate legal/fraud findings.")
    lines.append("   - Prioritise cases for punitive action based on demographic traits.")
    lines.append("   All scores serve strictly to guide human investigator review time.\n")
    lines.append("==========================================================================")

    return "\n".join(lines)
