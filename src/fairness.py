import pandas as pd
import numpy as np
from typing import Dict, List, Any

DEMOGRAPHIC_FIELDS = ['age_band', 'language_preference', 'district', 'tenure']

def analyze_fairness(full_df: pd.DataFrame, top_20_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes fairness and disparity metrics across all 4 demographic fields:
    age_band, language_preference, district, tenure.
    
    Returns a unified summary DataFrame.
    """
    total_population = len(full_df)
    total_selected = len(top_20_df)
    overall_selection_rate = total_selected / total_population if total_population > 0 else 0.0

    reports = []

    for field in DEMOGRAPHIC_FIELDS:
        if field not in full_df.columns:
            continue

        pop_counts = full_df[field].value_counts(dropna=False)
        top_counts = top_20_df[field].value_counts(dropna=False)
        group_means = full_df.groupby(field, dropna=False)['risk_score'].mean()

        for group, pop_cnt in pop_counts.items():
            sel_cnt = top_counts.get(group, 0)
            pop_share = pop_cnt / total_population
            top20_share = sel_cnt / total_selected if total_selected > 0 else 0.0
            group_sel_rate = sel_cnt / pop_cnt if pop_cnt > 0 else 0.0
            
            # Selection ratio relative to overall population selection rate
            relative_ratio = group_sel_rate / overall_selection_rate if overall_selection_rate > 0 else 0.0
            
            mean_score = group_means.get(group, 0.0)

            reports.append({
                'demographic_category': field,
                'group_value': str(group),
                'population_count': pop_cnt,
                'population_share_pct': round(pop_share * 100, 2),
                'top20_selected_count': sel_cnt,
                'top20_share_pct': round(top20_share * 100, 2),
                'group_selection_rate_pct': round(group_sel_rate * 100, 4),
                'relative_selection_ratio': round(relative_ratio, 2),
                'group_mean_risk_score': round(mean_score, 2),
                'disparity_flag': 'High Disparity (>2.0x)' if relative_ratio >= 2.0 else ('Underrepresented (<0.5x)' if relative_ratio <= 0.5 and sel_cnt == 0 else 'Balanced')
            })

    return pd.DataFrame(reports)


def generate_fairness_narrative(fairness_df: pd.DataFrame) -> str:
    """
    Generates a clear human-readable narrative summarizing fairness findings and potential proxy effects.
    """
    high_disparities = fairness_df[fairness_df['relative_selection_ratio'] >= 2.0]
    
    lines = []
    lines.append("=== FAIRNESS & GOVERNANCE DISPARITY REPORT ===")
    lines.append("Demographic monitoring evaluated across: age_band, language_preference, district, tenure.\n")

    if not high_disparities.empty:
        lines.append("DISPARITIES DETECTED:")
        for idx, row in high_disparities.iterrows():
            lines.append(
                f"- Category: {row['demographic_category']} | Group: {row['group_value']} "
                f"| Pop Share: {row['population_share_pct']}% | Top 20 Share: {row['top20_share_pct']}% "
                f"| Relative Selection Ratio: {row['relative_selection_ratio']}x overall base rate."
            )
    else:
        lines.append("No demographic group was selected at >= 2.0x overall base rate.")

    lines.append("\nPROXY EFFECT ANALYSIS & GOVERNANCE STATEMENT:")
    lines.append(
        "Demographic variables are strictly excluded from the risk scoring model. However, proxy correlation "
        "may exist between financial risk signals (e.g. award-to-needs ratio, review delays) and specific housing tenures or districts. "
        "Per governance policy, this model provides prioritisation signals for human investigation ONLY and must never be used "
        "to trigger automated benefit adjustments or punitive decisions."
    )

    return "\n".join(lines)
