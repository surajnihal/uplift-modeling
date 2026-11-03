"""
Verification: Thompson vs. Response-Only backtest numbers.

Prints:
1. frequent_payer contact rate for Thompson vs Response-Only
2. Best warm-started arm per segment
3. Full backtest results table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.dgp import DataSimulator
from src.modeling import ResponseModel, UpliftModel
from src.bandit import ThompsonSamplingPolicy, ResponseOnlyPolicy, ARM_IDX
from src.guardrails import Guardrails
from src.backtest import BacktestEngine, compare_policies, mean_ci

def main():
    print("="*70)
    print("Verification: No-Contact Boost Fix")
    print("="*70)

    # Setup
    sim = DataSimulator(seed=42)
    df_historical = sim.simulate_cohort(3000, role='historical', horizon=8)
    df_backtest = sim.simulate_cohort(200, role='backtest', horizon=8)
    df_backtest_features = df_backtest.drop_duplicates(subset=['debtor_id']).reset_index(drop=True)

    response_model = ResponseModel(model_type='gb')
    response_model.fit(df_historical)

    uplift_model = UpliftModel(learner_type='t')
    uplift_model.fit(df_historical)

    # Policy 1: Thompson (same contact budget as Response-Only below, for an
    # equal-contact-volume comparison)
    policy_thompson = ThompsonSamplingPolicy(
        seed=42,
        response_model=response_model,
        uplift_model=uplift_model,
        contact_budget_fraction=0.5,
    )
    df_hist_features = df_historical.drop_duplicates(subset=['debtor_id'])
    policy_thompson.fit(
        df_historical[['balance', 'days_overdue', 'prior_payments', 'contactability']].values,
        df_historical['action_contact'].values,
        df_historical['outcome_paid'].values,
        features_df=df_hist_features
    )

    # Policy 2: Response-Only
    policy_response = ResponseOnlyPolicy(response_model=response_model, contact_fraction=0.5)

    # Print verification metrics
    print("\n[1/3] Baseline P(pay) by segment (from ResponseModel):")
    for segment in ['high_risk', 'medium_risk', 'low_risk', 'frequent_payer']:
        seg_features = df_hist_features[df_hist_features['segment'] == segment]
        if len(seg_features) > 0:
            baseline = response_model.predict_proba(seg_features).mean()
            print(f"  {segment:20s}: {baseline:.3f}")

    print("\n[2/3] Best warm-started arm per segment (Thompson):")
    for segment in ['high_risk', 'medium_risk', 'low_risk', 'frequent_payer']:
        best_arm_idx = None
        best_mean = -1
        for arm_idx in range(len(ARM_IDX)):
            key = (segment, arm_idx)
            if key in policy_thompson.posteriors:
                alpha = policy_thompson.posteriors[key]['alpha']
                beta = policy_thompson.posteriors[key]['beta']
                mean = alpha / (alpha + beta)
                if mean > best_mean:
                    best_mean = mean
                    best_arm_idx = arm_idx

        if best_arm_idx is not None:
            best_arm = ARM_IDX[best_arm_idx]
            print(f"  {segment:20s}: Arm {best_arm_idx:2d} {str(best_arm):50s} μ={best_mean:.3f}")

    print("\n[3/3] Backtest Results Comparison:")

    guardrails = Guardrails(
        max_contacts_per_week=1,
        max_total_contacts=4,
        rest_period_weeks=1,
        hard_stop_on_optout=True,
    )

    n_seeds = 30
    engine_thompson = BacktestEngine(sim, policy_thompson, guardrails, df_backtest_features, horizon=8, verbose=False)
    results_thompson = engine_thompson.run_multiple(n_seeds=n_seeds)

    engine_response = BacktestEngine(sim, policy_response, guardrails, df_backtest_features, horizon=8, verbose=False)
    results_response = engine_response.run_multiple(n_seeds=n_seeds)

    print(f"\nACTUAL RESULTS (n_seeds={n_seeds}, ± is 95% CI on the mean):")
    print("-" * 70)

    thompson_recovery, thompson_ci = mean_ci([r.total_recovery for r in results_thompson])
    thompson_contacts = np.mean([r.total_contacts for r in results_thompson])

    response_recovery, response_ci = mean_ci([r.total_recovery for r in results_response])
    response_contacts = np.mean([r.total_contacts for r in results_response])

    print(f"Thompson:       {thompson_recovery:,.0f} ± {thompson_ci:,.0f} ({thompson_contacts:.1f} contacts)")
    print(f"Response-Only:  {response_recovery:,.0f} ± {response_ci:,.0f} ({response_contacts:.1f} contacts)")
    print(f"Recovery gap:   {thompson_recovery - response_recovery:,.0f} ({100*(thompson_recovery - response_recovery)/response_recovery:.1f}% vs Response-Only)")
    print(f"Contact gap:    {thompson_contacts - response_contacts:+.1f} contacts ({100*(thompson_contacts - response_contacts)/response_contacts:.1f}% vs Response-Only)")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
