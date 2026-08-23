import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class ScoringRule:
    """Base class for configurable, modular scoring rules."""
    def __init__(self, name: str, weight: float, description: str):
        self.name = name
        self.weight = weight
        self.description = description

    def compute(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement compute()")

class PaymentVsAwardDiscrepancyRule(ScoringRule):
    """Flags cases where observed payments exceed the official recorded award."""
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Score proportional to how much average payment exceeds monthly award
        diff = df['mean_payment_vs_award_diff'].clip(lower=0)
        # Normalize: $500 diff yields max subscore of 100
        score = (diff / 500.0).clip(upper=1.0) * 100.0
        return score

class PaymentExceedsNeedsRule(ScoringRule):
    """Flags cases where payments or award approach or exceed the policy needs figure."""
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Ratio of max payment to household needs figure
        ratio = df['max_payment_to_needs_ratio']
        # If ratio > 0.7, score ramps up. If ratio > 1.0 (exceeds total needs figure), max score
        score = np.where(ratio > 0.7, ((ratio - 0.7) / 0.3).clip(upper=1.0) * 100.0, 0.0)
        return pd.Series(score, index=df.index)

class PaymentVolatilityRule(ScoringRule):
    """Flags cases with unexplained payment instability or large month-over-month swings."""
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Standard deviation of payments normalized by mean payment
        volatility = df['std_payment_amount']
        # std of $200+ yields max subscore of 100
        score = (volatility / 200.0).clip(upper=1.0) * 100.0
        return score

class OverdueReviewAnomalyRule(ScoringRule):
    """Flags cases with long periods since last review combined with payment changes."""
    def compute(self, df: pd.DataFrame) -> pd.Series:
        months = df['months_since_review']
        has_variance = df['std_payment_amount'] > 30.0
        # Ramps up for review > 12 months if payment variance exists
        months_score = (months / 24.0).clip(upper=1.0) * 100.0
        score = np.where(has_variance, months_score, months_score * 0.3)
        return pd.Series(score, index=df.index)

class StatusAnomalyRule(ScoringRule):
    """Flags payments issued to Suspended or Closed cases."""
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return (df['status_payment_anomaly'] * 100.0).astype(float)

class PaymentMethodInstabilityRule(ScoringRule):
    """Flags cases with multiple payment method changes across months."""
    def compute(self, df: pd.DataFrame) -> pd.Series:
        switches = df['payment_method_switches']
        score = (switches / 3.0).clip(upper=1.0) * 100.0
        return score

# Day 2 Departmental Activity Rule (For explicit monitoring, given 0 weight in resident risk score)
class DepartmentalActivityContextRule(ScoringRule):
    """Departmental activity (contact attempts & adjustments) - strictly 0 weight for risk ranking."""
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Computes contextual departmental churn metric
        adj_score = (df['dept_adjustments_count'] / 5.0).clip(upper=1.0) * 50.0
        cnt_score = (df['dept_contact_attempts'] / 10.0).clip(upper=1.0) * 50.0
        return adj_score + cnt_score


class ScoringEngine:
    """Configurable scoring engine that combines multiple scoring rules."""
    def __init__(self, rules: List[ScoringRule] = None):
        if rules is None:
            # Default production scoring rules (Day 2 Feedback compliant)
            self.rules = [
                PaymentVsAwardDiscrepancyRule("payment_vs_award_discrepancy", weight=0.35, description="Payment exceeds recorded monthly award"),
                PaymentExceedsNeedsRule("payment_exceeds_needs", weight=0.25, description="Payment approaches or exceeds household needs figure"),
                PaymentVolatilityRule("payment_volatility", weight=0.15, description="Unexplained payment volatility across months"),
                OverdueReviewAnomalyRule("overdue_review", weight=0.15, description="Overdue case review combined with active payment changes"),
                StatusAnomalyRule("status_anomaly", weight=0.05, description="Payments made on Suspended or Closed case"),
                PaymentMethodInstabilityRule("method_instability", weight=0.05, description="Frequent payment delivery method switches"),
                DepartmentalActivityContextRule("departmental_activity", weight=0.00, description="Departmental administrative activity (0 weight in risk)")
            ]
        else:
            self.rules = rules

    def compute_scores(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes composite risk score and sub-scores for each case.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (scored_df, subscores_df)
        """
        scored_df = df.copy()
        subscores = pd.DataFrame(index=df.index)

        total_weight = sum(rule.weight for rule in self.rules if rule.weight > 0)
        composite_score = pd.Series(0.0, index=df.index)

        for rule in self.rules:
            rule_subscore = rule.compute(df)
            subscores[rule.name] = rule_subscore
            if rule.weight > 0:
                composite_score += (rule_subscore * rule.weight)

        if total_weight > 0:
            composite_score = composite_score / total_weight

        scored_df['risk_score'] = composite_score.round(2)
        
        # Calculate confidence / signal strength indicator
        # High confidence if score is driven by multiple agreeing behavioral signals
        active_signals = (subscores[[r.name for r in self.rules if r.weight > 0]] > 30.0).sum(axis=1)
        scored_df['signal_confidence'] = np.where(active_signals >= 3, 'High', np.where(active_signals >= 2, 'Medium', 'Moderate'))

        return scored_df, subscores

    def get_ranked_worklist(self, df: pd.DataFrame, top_n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates the top N ranked cases sorted by risk_score descending.
        """
        scored_df, subscores = self.compute_scores(df)
        
        # Rank cases (deterministic tie-breaking by case_id)
        ranked_df = scored_df.sort_values(by=['risk_score', 'case_id'], ascending=[False, True]).reset_index(drop=True)
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        
        top_n_df = ranked_df.head(top_n).copy()
        top_n_subscores = subscores.loc[top_n_df.index].copy()
        
        return top_n_df, top_n_subscores
