"""
Diagnose Thompson Sampling: Is it personalizing by channel?

Shows what channel Thompson selects for each segment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.dgp import DataSimulator, DebtorState
from src.modeling import ResponseModel, UpliftModel
from src.bandit import ThompsonSamplingPolicy

def main():
    print("="*70)
    print("Thompson Sampling Personalization Diagnostic")
    print("="*70)

    # Generate data
    sim = DataSimulator(seed=42)
    df_historical = sim.simulate_cohort(3000, role='historical', horizon=8)

    # Fit models
    response_model = ResponseModel(model_type='gb')
    response_model.fit(df_historical)

    uplift_model = UpliftModel(learner_type='t')
    uplift_model.fit(df_historical)

    # Create Thompson policy
    policy_thompson = ThompsonSamplingPolicy(
        seed=42,
        response_model=response_model,
        uplift_model=uplift_model
    )
    df_hist_features = df_historical.drop_duplicates(subset=['debtor_id'])
    policy_thompson.fit(
        df_historical[['balance', 'days_overdue', 'prior_payments', 'contactability']].values,
        df_historical['action_contact'].values,
        df_historical['outcome_paid'].values,
        features_df=df_hist_features
    )

    print("\nThompson Posterior Means (sample of arms, high_risk segment):")
    from src.bandit import ARM_IDX
    segment = 'high_risk'
    for arm_idx in [0, 1, 2, 3, 4, 5]:  # First 6 arms
        arm = ARM_IDX[arm_idx]
        key = (segment, arm_idx)
        if key in policy_thompson.posteriors:
            alpha = policy_thompson.posteriors[key]['alpha']
            beta = policy_thompson.posteriors[key]['beta']
            mean = alpha / (alpha + beta)
            print(f"  Arm {arm_idx}: {str(arm):50s} μ={mean:.3f} (α={alpha:.0f}, β={beta:.0f})")

    # Test on synthetic debtors from each segment
    print("\n\nThompson Channel Selection by Segment:")
    print("(What channel does Thompson pick for each segment?)")
    print("="*70)

    for segment in ['high_risk', 'medium_risk', 'low_risk', 'frequent_payer']:
        channels_selected = {'voice': 0, 'sms': 0, 'whatsapp': 0, 'email': 0}

        for trial in range(50):  # Sample 50 times per segment
            features = pd.Series({
                'balance': 5000 + np.random.randn() * 1000,
                'days_overdue': 60 + np.random.randint(-20, 20),
                'prior_payments': 3 + np.random.randint(0, 10),
                'contactability': 0.7 + np.random.rand() * 0.3,
                'segment': segment,
            })
            state = DebtorState(
                debtor_id=999, balance=5000, days_overdue=60, prior_payments=3,
                contactability=0.8, annoyance=0.2, opted_out=False,
                complained=False, paid=False
            )

            action = policy_thompson.select_arm(features, state, week=0)
            if action.contact:
                channels_selected[action.channel] += 1

        total = sum(channels_selected.values())
        print(f"\n{segment:20s}:")
        for ch, count in sorted(channels_selected.items(), key=lambda x: -x[1]):
            pct = 100 * count / total if total > 0 else 0
            print(f"  {ch:12s}: {pct:5.1f}% ({count}/50)")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
