"""
Demo: Bandit Policy + Guardrails + Backtest

Shows how to:
1. Fit response and uplift models on historical data
2. Run backtest with multiple policies (random, response-only, Thompson Sampling)
3. Compare results and select best policy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dgp import DataSimulator
from src.modeling import ResponseModel, UpliftModel
from src.bandit import RandomPolicy, ThompsonSamplingPolicy, ResponseOnlyPolicy
from src.guardrails import Guardrails
from src.backtest import BacktestEngine, compare_policies


def main():
    print("="*70)
    print("Bandit Policy + Guardrails + Backtest Demo")
    print("="*70)

    # Generate data
    print("\n[1/5] Generating historical and backtest cohorts...")
    sim = DataSimulator(seed=42)

    df_historical = sim.simulate_cohort(3000, role='historical', horizon=8)
    print(f"  Historical: {len(df_historical)} records from {df_historical['debtor_id'].nunique()} debtors")

    df_backtest = sim.simulate_cohort(200, role='backtest', horizon=8)
    df_backtest_features = df_backtest.drop_duplicates(subset=['debtor_id']).reset_index(drop=True)
    print(f"  Backtest: {len(df_backtest_features)} unique debtors")

    # Fit models
    print("\n[2/5] Fitting response and uplift models...")
    response_model = ResponseModel(model_type='gb')
    response_model.fit(df_historical)
    print(f"  Response model accuracy: {response_model.score(df_historical):.3f}")

    uplift_model = UpliftModel(learner_type='t')
    uplift_model.fit(df_historical)
    print(f"  Uplift model fitted (T-learner)")

    # Define guardrails
    guardrails = Guardrails(
        max_contacts_per_week=1,
        max_total_contacts=4,
        rest_period_weeks=1,
        hard_stop_on_optout=True,
    )
    print(f"\n[3/5] Guardrails configured:")
    for key, val in guardrails.get_summary().items():
        print(f"  - {key}: {val}")

    # Create policies
    print("\n[4/5] Running backtest with 3 policies...")
    policies_dict = {}

    # Policy 1: Random (baseline)
    policy_random = RandomPolicy()
    engine_random = BacktestEngine(
        sim, policy_random, guardrails, df_backtest_features, horizon=8
    )
    print("\n  [Policy 1/3] Running RandomPolicy...")
    results_random = engine_random.run_multiple(n_seeds=30)
    print(results_random[0])
    policies_dict['RandomPolicy'] = results_random

    # Policy 2: Response-Only (current practice)
    policy_response = ResponseOnlyPolicy(response_model=response_model, contact_fraction=0.5)
    engine_response = BacktestEngine(
        sim, policy_response, guardrails, df_backtest_features, horizon=8
    )
    print("\n  [Policy 2/3] Running ResponseOnlyPolicy...")
    results_response = engine_response.run_multiple(n_seeds=30)
    print(results_response[0])
    policies_dict['ResponseOnlyPolicy'] = results_response

    # Policy 3: Thompson Sampling (uplift-bandit)
    # Same contact budget as ResponseOnlyPolicy (contact_fraction=0.5 above),
    # so the comparison is at equal contact volume rather than letting each
    # policy freely decide its own total contact count.
    policy_thompson = ThompsonSamplingPolicy(
        seed=42,
        response_model=response_model,
        uplift_model=uplift_model,
        contact_budget_fraction=0.5,
    )
    # Extract features for warm-start (drop duplicates to get unique debtors)
    df_hist_features = df_historical.drop_duplicates(subset=['debtor_id'])
    policy_thompson.fit(
        df_historical[['balance', 'days_overdue', 'prior_payments', 'contactability']].values,
        df_historical['action_contact'].values,
        df_historical['outcome_paid'].values,
        features_df=df_hist_features
    )
    engine_thompson = BacktestEngine(
        sim, policy_thompson, guardrails, df_backtest_features, horizon=8
    )
    print("\n  [Policy 3/3] Running ThompsonSamplingPolicy...")
    results_thompson = engine_thompson.run_multiple(n_seeds=30)
    print(results_thompson[0])
    policies_dict['ThompsonSamplingPolicy'] = results_thompson

    # Compare policies
    print("\n[5/5] Policy Comparison:")
    print("="*70)
    comparison_df = compare_policies(policies_dict)
    print(comparison_df.to_string(index=False))

    print("\n" + "="*70)
    print("Demo Complete!")
    print("\nKey Insights:")
    print("- Random policy: baseline sanity check")
    print("- Response-only policy: current state-of-practice")
    print("- Thompson Sampling: uplift-bandit with online learning")
    print("- Guardrails enforce business constraints across all policies")
    print("="*70)


if __name__ == '__main__':
    main()
