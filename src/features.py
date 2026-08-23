import pandas as pd
import numpy as np
from typing import Dict

# Policy Needs Figures lookup by household size
NEEDS_FIGURES: Dict[int, float] = {
    1: 1240.0,
    2: 1670.0,
    3: 2000.0,
    4: 2330.0,
    5: 2660.0,
    6: 2990.0
}

def get_needs_figure(household_size: int) -> float:
    """Returns the policy monthly needs figure for a given household size."""
    size = int(household_size)
    if size <= 1:
        return NEEDS_FIGURES[1]
    elif size >= 6:
        return NEEDS_FIGURES[6]
    return NEEDS_FIGURES.get(size, NEEDS_FIGURES[6])

def engineer_features(cases_df: pd.DataFrame, payments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers case-level risk features from cases and payment records.
    
    Returns a unified DataFrame indexed by case_id containing:
    - Base demographic & case attributes
    - Derived behavioral & policy risk features
    - Segregated departmental activity features
    """
    df = cases_df.copy()

    # 1. Policy Needs Figure & Award Ratio
    df['needs_figure'] = df['household_size'].apply(get_needs_figure)
    df['award_to_needs_ratio'] = df['monthly_award'] / df['needs_figure']

    # 2. Payment Summaries per Case
    if not payments_df.empty:
        # Sort payments chronologically by pay_month
        pmts_sorted = payments_df.sort_values(['case_id', 'pay_month'])
        
        # Aggregate stats per case
        pmt_aggs = pmts_sorted.groupby('case_id').agg(
            payment_count=('amount', 'count'),
            mean_payment_amount=('amount', 'mean'),
            max_payment_amount=('amount', 'max'),
            min_payment_amount=('amount', 'min'),
            std_payment_amount=('amount', lambda x: x.std(ddof=0) if len(x) > 1 else 0.0),
            total_paid_amount=('amount', 'sum')
        ).reset_index()

        # Method switch count
        def calc_method_switches(methods: pd.Series) -> int:
            if len(methods) <= 1:
                return 0
            return int((methods.values[:-1] != methods.values[1:]).sum())

        method_switches = pmts_sorted.groupby('case_id')['method'].apply(calc_method_switches).reset_index()
        method_switches.columns = ['case_id', 'payment_method_switches']

        # Merge payment aggs
        df = df.merge(pmt_aggs, on='case_id', how='left')
        df = df.merge(method_switches, on='case_id', how='left')
    else:
        # Fallback if payments empty
        for col in ['payment_count', 'mean_payment_amount', 'max_payment_amount', 'min_payment_amount', 'std_payment_amount', 'total_paid_amount', 'payment_method_switches']:
            df[col] = 0.0

    # Fill NaNs for cases without payments
    df['payment_count'] = df['payment_count'].fillna(0).astype(int)
    df['mean_payment_amount'] = df['mean_payment_amount'].fillna(0.0)
    df['max_payment_amount'] = df['max_payment_amount'].fillna(0.0)
    df['min_payment_amount'] = df['min_payment_amount'].fillna(0.0)
    df['std_payment_amount'] = df['std_payment_amount'].fillna(0.0)
    df['total_paid_amount'] = df['total_paid_amount'].fillna(0.0)
    df['payment_method_switches'] = df['payment_method_switches'].fillna(0).astype(int)

    # 3. Behavioral Risk Indicators
    # Discrepancy: Observed payment vs recorded award
    df['mean_payment_vs_award_diff'] = df['mean_payment_amount'] - df['monthly_award']
    df['max_payment_vs_award_diff'] = df['max_payment_amount'] - df['monthly_award']
    df['payment_to_award_ratio'] = np.where(df['monthly_award'] > 0, df['mean_payment_amount'] / df['monthly_award'], 0.0)
    
    # Exceeding Needs Figure
    df['max_payment_to_needs_ratio'] = df['max_payment_amount'] / df['needs_figure']
    df['payment_exceeds_needs'] = (df['max_payment_amount'] > df['needs_figure']).astype(int)

    # Payment Jumps / Volatility
    df['payment_spread'] = df['max_payment_amount'] - df['min_payment_amount']
    df['payment_volatility_ratio'] = np.where(df['mean_payment_amount'] > 0, df['std_payment_amount'] / df['mean_payment_amount'], 0.0)

    # Status Anomalies (e.g. Payments made while Suspended or Closed)
    df['status_payment_anomaly'] = np.where(
        (df['status'].isin(['Suspended', 'Closed'])) & (df['payment_count'] > 0), 1, 0
    )

    # 4. Departmental Activity Features (Categorized as Contextual, NOT Resident Overpayment Risk)
    # payment_adjustments and contact_attempts are departmental logs
    df['dept_adjustments_count'] = df['payment_adjustments'].fillna(0).astype(int)
    df['dept_contact_attempts'] = df['contact_attempts'].fillna(0).astype(int)

    return df
