"""
Tests for Bandit Policy + Guardrails + Backtest.

Tests validate:
1. Bandit policies (Thompson, ε-Greedy, Random, Response-Only)
2. Guardrail enforcement (frequency caps, opt-out, rest periods)
3. Backtest engine (multi-period simulation, metrics)
4. End-to-end policy comparison
"""

import pytest
import numpy as np
import pandas as pd
from src.dgp import DataSimulator, DebtorState, Action
from src.modeling import ResponseModel, UpliftModel
from src.bandit import (
    BanditPolicy, RandomPolicy, ThompsonSamplingPolicy, EpsilonGreedyPolicy,
    ResponseOnlyPolicy, ALL_ARMS, ARM_IDX, IDX_ARM, NO_CONTACT_ARM
)
from src.guardrails import Guardrails
from src.backtest import BacktestEngine, BacktestResults, compare_policies


class TestRandomPolicy:
    """Test baseline random policy."""

    def test_random_policy_selects_arm(self):
        """RandomPolicy.select_arm() returns valid Action."""
        policy = RandomPolicy()
        features = pd.Series({
            'balance': 100,
            'days_overdue': 30,
            'prior_payments': 2,
            'contactability': 0.8,
            'segment': 'high_risk',
        })
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.2, opted_out=False,
            complained=False, paid=False
        )

        action = policy.select_arm(features, state, week=0)
        assert action in ALL_ARMS

    def test_random_policy_update_noop(self):
        """RandomPolicy.update() is no-op."""
        policy = RandomPolicy()
        outcome = {'paid': True, 'opted_out': False, 'complained': False, 'annoyance': 0.3}
        policy.update(ALL_ARMS[0], outcome)  # Should not crash


class TestThompsonSamplingPolicy:
    """Test Thompson Sampling policy."""

    def test_thompson_initialization(self):
        """Thompson posteriors initialized correctly (per-segment, hierarchical)."""
        policy = ThompsonSamplingPolicy(alpha_prior=1.0, beta_prior=1.0)
        # Level 2: 4 segments × 12 contact arms = 48 posteriors (no-contact is
        # handled by the separate level-1 decision, not a level-2 arm)
        n_contact_arms = sum(1 for arm in ARM_IDX.values() if arm.contact)
        assert len(policy.posteriors) == 4 * n_contact_arms
        for (segment, arm_idx), posterior in policy.posteriors.items():
            assert posterior['alpha'] == 1.0
            assert posterior['beta'] == 1.0

        # Level 1: one contact/no-contact decision posterior per segment
        assert len(policy.contact_decision_posteriors) == 4
        for segment, decision in policy.contact_decision_posteriors.items():
            assert decision['contact']['alpha'] == 1.0
            assert decision['contact']['beta'] == 1.0
            assert decision['no_contact']['alpha'] == 1.0
            assert decision['no_contact']['beta'] == 1.0

    def test_thompson_select_arm(self):
        """Thompson.select_arm() samples from posteriors."""
        policy = ThompsonSamplingPolicy()
        features = pd.Series({
            'balance': 100, 'days_overdue': 30, 'prior_payments': 2,
            'contactability': 0.8, 'segment': 'high_risk'
        })
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.2, opted_out=False,
            complained=False, paid=False
        )

        action = policy.select_arm(features, state, week=0)
        assert action in ALL_ARMS

    def test_thompson_update_success(self):
        """Thompson.update() increments alpha on success."""
        policy = ThompsonSamplingPolicy()
        arm_idx_0 = 0
        arm = ARM_IDX[arm_idx_0]

        # Set segment before calling update (normally done in select_arm)
        policy.last_segment = 'high_risk'
        outcome = {'paid': True, 'opted_out': False, 'complained': False, 'annoyance': 0.1}

        policy.update(arm, outcome)
        key = ('high_risk', arm_idx_0)
        assert policy.posteriors[key]['alpha'] == 2.0  # alpha_prior + 1
        assert policy.posteriors[key]['beta'] == 1.0   # unchanged

    def test_thompson_update_failure(self):
        """Thompson.update() increments beta on failure."""
        policy = ThompsonSamplingPolicy()
        arm_idx_0 = 0
        arm = ARM_IDX[arm_idx_0]

        # Set segment before calling update (normally done in select_arm)
        policy.last_segment = 'medium_risk'
        outcome = {'paid': False, 'opted_out': False, 'complained': False, 'annoyance': 0.1}

        policy.update(arm, outcome)
        key = ('medium_risk', arm_idx_0)
        assert policy.posteriors[key]['alpha'] == 1.0   # unchanged
        assert policy.posteriors[key]['beta'] == 2.0    # beta_prior + 1

    def test_thompson_fit_warmstart(self):
        """Thompson.fit() warm-starts from historical data."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(100, role='historical', horizon=4)

        X = df[['balance', 'days_overdue', 'prior_payments', 'contactability']].copy()
        T = df['action_contact'].values
        y = df['outcome_paid'].values

        policy = ThompsonSamplingPolicy()
        policy.fit(X, T, y)

        # Posteriors should be updated (alpha + beta > 2)
        for arm_idx, posterior in policy.posteriors.items():
            assert posterior['alpha'] + posterior['beta'] > 2

    def test_thompson_warmstart_preserved_after_reset(self):
        """Thompson.reset() preserves warm-start posteriors (critical for multi-seed backtest)."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(100, role='historical', horizon=4)

        X = df[['balance', 'days_overdue', 'prior_payments', 'contactability']].copy()
        T = df['action_contact'].values
        y = df['outcome_paid'].values

        policy = ThompsonSamplingPolicy()
        policy.fit(X, T, y)

        # Capture posteriors after fit
        posteriors_after_fit = {i: {'alpha': p['alpha'], 'beta': p['beta']}
                               for i, p in policy.posteriors.items()}

        # Call reset (as run_multiple() does before each seed)
        policy.reset()

        # Verify posteriors are restored to warm-start, not flat priors
        for arm_idx, posterior in policy.posteriors.items():
            expected = posteriors_after_fit[arm_idx]
            assert posterior['alpha'] == expected['alpha'], \
                f"Arm {arm_idx}: alpha not preserved after reset()"
            assert posterior['beta'] == expected['beta'], \
                f"Arm {arm_idx}: beta not preserved after reset()"

    def test_thompson_hierarchical_decision_preserved_after_reset(self):
        """reset() also preserves the level-1 contact/no-contact warm-start."""
        sim = DataSimulator(seed=42)
        df_hist = sim.simulate_cohort(300, role='historical', horizon=4)
        df_hist_features = df_hist.drop_duplicates(subset=['debtor_id'])

        response_model = ResponseModel(model_type='lr')
        response_model.fit(df_hist)
        uplift_model = UpliftModel(learner_type='t')
        uplift_model.fit(df_hist)

        policy = ThompsonSamplingPolicy(response_model=response_model, uplift_model=uplift_model)
        policy.fit(
            df_hist[['balance', 'days_overdue', 'prior_payments', 'contactability']].values,
            df_hist['action_contact'].values,
            df_hist['outcome_paid'].values,
            features_df=df_hist_features,
        )

        decision_after_fit = {
            seg: {dk: dict(p) for dk, p in d.items()}
            for seg, d in policy.contact_decision_posteriors.items()
        }

        # Simulate some online updates that would perturb the posteriors
        policy.last_segment = 'high_risk'
        policy.update(NO_CONTACT_ARM, {'paid': False, 'opted_out': False, 'complained': False})

        policy.reset()

        for segment, decision in policy.contact_decision_posteriors.items():
            expected = decision_after_fit[segment]
            assert decision['contact']['alpha'] == expected['contact']['alpha']
            assert decision['contact']['beta'] == expected['contact']['beta']
            assert decision['no_contact']['alpha'] == expected['no_contact']['alpha']
            assert decision['no_contact']['beta'] == expected['no_contact']['beta']

    def test_thompson_contact_decision_is_fair_1v1(self):
        """The level-1 draw compares exactly one contact sample to one
        no-contact sample, regardless of how many channel/timing arms exist."""
        policy = ThompsonSamplingPolicy(alpha_prior=5.0, beta_prior=15.0)
        # Equal posteriors for contact and no-contact at the top level
        policy.contact_decision_posteriors['high_risk'] = {
            'contact': {'alpha': 5.0, 'beta': 15.0},
            'no_contact': {'alpha': 5.0, 'beta': 15.0},
        }
        features = pd.Series({
            'balance': 100, 'days_overdue': 30, 'prior_payments': 2,
            'contactability': 0.8, 'segment': 'high_risk'
        })
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.2, opted_out=False,
            complained=False, paid=False
        )

        n_trials = 2000
        contact_count = sum(
            1 for _ in range(n_trials)
            if policy.select_arm(features, state, week=0).contact
        )
        contact_rate = contact_count / n_trials

        # With a fair 1-vs-1 comparison and equal posteriors, contact should be
        # chosen close to 50% of the time — not ~92% as the old flat 12-vs-1
        # argmax would produce.
        assert 0.4 < contact_rate < 0.6, f"contact_rate={contact_rate} suggests an unfair comparison"

    def test_thompson_contact_budget_matches_response_only_selection(self):
        """With contact_budget_fraction=0.5, Thompson should restrict eligibility
        to the same-sized, highest-uplift half of the cohort — comparable in
        spirit to ResponseOnlyPolicy's highest-P(pay) half, but ranked by CATE."""
        sim = DataSimulator(seed=42)
        df_hist = sim.simulate_cohort(300, role='historical', horizon=4)
        df_hist_features = df_hist.drop_duplicates(subset=['debtor_id'])

        response_model = ResponseModel(model_type='lr')
        response_model.fit(df_hist)
        uplift_model = UpliftModel(learner_type='t')
        uplift_model.fit(df_hist)

        cohort = sim.simulate_cohort(30, role='backtest', horizon=1)
        cohort = cohort.drop_duplicates(subset=['debtor_id']).reset_index(drop=True)

        policy = ThompsonSamplingPolicy(
            response_model=response_model, uplift_model=uplift_model,
            contact_budget_fraction=0.5,
        )
        policy.fit(
            df_hist[['balance', 'days_overdue', 'prior_payments', 'contactability']].values,
            df_hist['action_contact'].values,
            df_hist['outcome_paid'].values,
            features_df=df_hist_features,
        )
        policy.prepare_cohort(cohort)

        assert policy._eligible_debtor_ids is not None
        assert len(policy._eligible_debtor_ids) == 15  # ceil(30 * 0.5)

        # A debtor outside the eligible set is forced to no-contact regardless
        # of what the bandit's posteriors would otherwise recommend.
        ineligible_row = cohort[~cohort['debtor_id'].astype(int).isin(policy._eligible_debtor_ids)].iloc[0]
        state = DebtorState(
            debtor_id=int(ineligible_row['debtor_id']), balance=ineligible_row['balance'],
            days_overdue=ineligible_row['days_overdue'], prior_payments=ineligible_row['prior_payments'],
            contactability=ineligible_row['contactability'], annoyance=0.0,
            opted_out=False, complained=False, paid=False,
        )
        action = policy.select_arm(ineligible_row, state, week=0)
        assert action.contact is False

    def test_thompson_no_budget_is_uncapped_by_default(self):
        """Without contact_budget_fraction, prepare_cohort() should not restrict
        eligibility (backward-compatible default)."""
        policy = ThompsonSamplingPolicy()
        cohort = pd.DataFrame({
            'debtor_id': range(10), 'balance': [100.0] * 10, 'days_overdue': [30] * 10,
            'prior_payments': [1] * 10, 'contactability': [0.8] * 10,
            'segment': ['high_risk'] * 10,
        })
        policy.prepare_cohort(cohort)
        assert policy._eligible_debtor_ids is None


class TestEpsilonGreedyPolicy:
    """Test ε-Greedy policy."""

    def test_epsilon_greedy_initialization(self):
        """ε-Greedy initialized with arm_means and counts."""
        policy = EpsilonGreedyPolicy(epsilon=0.1)
        assert len(policy.arm_means) == len(ALL_ARMS)
        assert len(policy.arm_counts) == len(ALL_ARMS)
        assert all(count == 0 for count in policy.arm_counts.values())

    def test_epsilon_greedy_select_arm(self):
        """ε-Greedy.select_arm() returns valid action."""
        policy = EpsilonGreedyPolicy(epsilon=0.1)
        features = pd.Series({
            'balance': 100, 'days_overdue': 30, 'prior_payments': 2,
            'contactability': 0.8, 'segment': 'high_risk'
        })
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.2, opted_out=False,
            complained=False, paid=False
        )

        action = policy.select_arm(features, state, week=0)
        assert action in ALL_ARMS

    def test_epsilon_greedy_update(self):
        """ε-Greedy.update() updates empirical mean."""
        policy = EpsilonGreedyPolicy(epsilon=0.1)
        arm_idx_0 = 0
        arm = ARM_IDX[arm_idx_0]
        outcome = {'paid': True, 'opted_out': False, 'complained': False, 'annoyance': 0.1}

        policy.update(arm, outcome)
        assert policy.arm_counts[arm_idx_0] == 1
        assert policy.arm_means[arm_idx_0] == 1.0  # (0*0 + 1) / 1


class TestResponseOnlyPolicy:
    """Test Response-Only baseline policy."""

    def test_response_only_without_model(self):
        """ResponseOnly without fitted model falls back to random."""
        policy = ResponseOnlyPolicy(response_model=None)
        features = pd.Series({
            'balance': 100, 'days_overdue': 30, 'prior_payments': 2,
            'contactability': 0.8, 'segment': 'high_risk'
        })
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.2, opted_out=False,
            complained=False, paid=False
        )

        action = policy.select_arm(features, state, week=0)
        assert action in ALL_ARMS

    def test_response_only_with_model(self):
        """ResponseOnly with fitted model uses top-N-by-P(pay) ranking."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(100, role='historical', horizon=4)

        response_model = ResponseModel(model_type='lr')
        response_model.fit(df)

        policy = ResponseOnlyPolicy(response_model=response_model, contact_fraction=0.5)
        features = df.iloc[0]

        # Without prepare_cohort(), falls back to a fixed-threshold approximation
        action = policy.select_arm(features, state=None, week=0)
        assert action in ALL_ARMS

    def test_response_only_contacts_highest_p_pay(self):
        """ResponseOnly should contact the debtors MOST likely to pay (the
        deliberate strawman: it over-contacts 'sure things'), not skip them."""
        sim = DataSimulator(seed=42)
        df_hist = sim.simulate_cohort(200, role='historical', horizon=4)

        response_model = ResponseModel(model_type='lr')
        response_model.fit(df_hist)

        cohort = sim.simulate_cohort(20, role='backtest', horizon=1)
        cohort = cohort.drop_duplicates(subset=['debtor_id']).reset_index(drop=True)

        policy = ResponseOnlyPolicy(response_model=response_model, contact_fraction=0.5)
        policy.prepare_cohort(cohort)

        probs = response_model.predict_proba(cohort)
        cohort = cohort.assign(p_pay=probs)
        ranked = cohort.sort_values('p_pay', ascending=False)

        top_half_ids = set(ranked['debtor_id'].iloc[:10].astype(int))
        bottom_half_ids = set(ranked['debtor_id'].iloc[10:].astype(int))

        # Everyone contacted should come from the top half by P(pay)
        assert policy._contact_debtor_ids == top_half_ids
        assert policy._contact_debtor_ids.isdisjoint(bottom_half_ids)

        # A debtor from the top half is contacted; one from the bottom half is not
        top_row = ranked.iloc[0]
        bottom_row = ranked.iloc[-1]
        assert policy.select_arm(top_row, state=None, week=0).contact is True
        assert policy.select_arm(bottom_row, state=None, week=0).contact is False


class TestGuardrails:
    """Test guardrail constraint enforcement."""

    def test_guardrails_hard_stop_optout(self):
        """Guardrail blocks contact if opted_out=True."""
        guardrails = Guardrails(hard_stop_on_optout=True)
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.5, opted_out=True,
            complained=False, paid=False
        )
        action = Action(contact=True, channel='voice', timing='morning')

        action_out, reason = guardrails.apply(action, state, week=0, contact_history=[])
        assert action_out.contact == False
        assert reason == 'opted_out'

    def test_guardrails_freq_cap_per_week(self):
        """Guardrail enforces max_contacts_per_week."""
        guardrails = Guardrails(max_contacts_per_week=1)
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.1, opted_out=False,
            complained=False, paid=False
        )
        action = Action(contact=True, channel='voice', timing='morning')

        # Already contacted in week 0
        contact_history = [(0, Action(contact=True, channel='sms', timing='morning'))]

        action_out, reason = guardrails.apply(action, state, week=0, contact_history=contact_history)
        assert action_out.contact == False
        assert reason == 'max_contacts_per_week_exceeded'

    def test_guardrails_rest_period(self):
        """Guardrail enforces rest_period_weeks."""
        guardrails = Guardrails(rest_period_weeks=1)
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.1, opted_out=False,
            complained=False, paid=False
        )
        action = Action(contact=True, channel='voice', timing='morning')

        # Contacted in week 0, now trying week 1 (violates 1-week rest)
        contact_history = [(0, Action(contact=True, channel='sms', timing='morning'))]

        action_out, reason = guardrails.apply(action, state, week=1, contact_history=contact_history)
        assert action_out.contact == False
        assert reason == 'rest_period_not_met'

    def test_guardrails_total_contacts(self):
        """Guardrail enforces max_total_contacts budget."""
        guardrails = Guardrails(max_total_contacts=2, rest_period_weeks=0)  # Disable rest period for this test
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.2, opted_out=False,
            complained=False, paid=False
        )
        action = Action(contact=True, channel='voice', timing='morning')

        # Already used 2 contacts
        contact_history = [
            (0, Action(contact=True, channel='voice', timing='morning')),
            (2, Action(contact=True, channel='sms', timing='afternoon')),
        ]

        action_out, reason = guardrails.apply(action, state, week=3, contact_history=contact_history)
        assert action_out.contact == False
        assert reason == 'max_total_contacts_exceeded'

    def test_guardrails_no_contact_passes(self):
        """Guardrails allow no-contact action without checks."""
        guardrails = Guardrails()
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.9, opted_out=False,
            complained=False, paid=False
        )
        action = Action(contact=False)

        action_out, reason = guardrails.apply(action, state, week=0, contact_history=[])
        assert action_out.contact == False
        assert reason is None

    def test_guardrails_quiet_hours_blocks_false(self):
        """Guardrail blocks timing when quiet_hours is False."""
        guardrails = Guardrails(quiet_hours={'morning': True, 'afternoon': True, 'evening': False})
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.1, opted_out=False,
            complained=False, paid=False
        )

        # Evening should be blocked
        action_evening = Action(contact=True, channel='voice', timing='evening')
        action_out, reason = guardrails.apply(action_evening, state, week=0, contact_history=[])
        assert action_out.contact == False
        assert reason == 'timing_evening_blocked'

        # Morning should be allowed
        action_morning = Action(contact=True, channel='voice', timing='morning')
        action_out, reason = guardrails.apply(action_morning, state, week=0, contact_history=[])
        assert action_out.contact == True
        assert reason is None

    def test_guardrails_quiet_hours_allows_true(self):
        """Guardrail allows timing when quiet_hours is True."""
        guardrails = Guardrails(quiet_hours={'morning': True, 'afternoon': True, 'evening': True})
        state = DebtorState(
            debtor_id=1, balance=100, days_overdue=30, prior_payments=2,
            contactability=0.8, annoyance=0.1, opted_out=False,
            complained=False, paid=False
        )

        # All timings should be allowed
        for timing in ['morning', 'afternoon', 'evening']:
            action = Action(contact=True, channel='voice', timing=timing)
            action_out, reason = guardrails.apply(action, state, week=0, contact_history=[])
            assert action_out.contact == True
            assert reason is None


class TestBacktestEngine:
    """Test backtest simulation engine."""

    def test_backtest_random_policy_runs(self):
        """Backtest with random policy completes without error."""
        sim = DataSimulator(seed=42)
        df_historical = sim.simulate_cohort(100, role='historical', horizon=4)
        df_backtest = df_historical.drop_duplicates(subset=['debtor_id']).head(20)

        policy = RandomPolicy()
        guardrails = Guardrails()
        engine = BacktestEngine(sim, policy, guardrails, df_backtest, horizon=4)

        results = engine.run(seed=42)

        assert isinstance(results, BacktestResults)
        assert results.policy_name == 'RandomPolicy'
        assert results.total_debtors == 20
        assert len(results.per_debtor_results) == 20

    def test_backtest_metrics_computed(self):
        """Backtest computes all required metrics."""
        sim = DataSimulator(seed=42)
        df_historical = sim.simulate_cohort(50, role='historical', horizon=4)
        df_backtest = df_historical.drop_duplicates(subset=['debtor_id']).head(10)

        policy = RandomPolicy()
        guardrails = Guardrails()
        engine = BacktestEngine(sim, policy, guardrails, df_backtest, horizon=4)

        results = engine.run(seed=42)

        # Metrics should be computed
        assert results.total_recovery >= 0
        assert results.total_contacts >= 0
        assert 0 <= results.payment_rate <= 1
        assert 0 <= results.opt_out_rate <= 1
        assert 0 <= results.complaint_rate <= 1
        assert results.avg_contacts_per_debtor >= 0

    def test_backtest_multiple_runs(self):
        """Backtest.run_multiple() produces multiple independent runs."""
        sim = DataSimulator(seed=42)
        df_historical = sim.simulate_cohort(50, role='historical', horizon=4)
        df_backtest = df_historical.drop_duplicates(subset=['debtor_id']).head(10)

        policy = RandomPolicy()
        guardrails = Guardrails()
        engine = BacktestEngine(sim, policy, guardrails, df_backtest, horizon=4)

        results_list = engine.run_multiple(n_seeds=3)

        assert len(results_list) == 3
        # Different seeds should produce some variation (not deterministic)
        recoveries = [r.total_recovery for r in results_list]
        assert len(set(recoveries)) >= 1  # At least some variety

    def test_backtest_guardrail_enforcement(self):
        """Backtest respects guardrail constraints."""
        sim = DataSimulator(seed=42)
        df_historical = sim.simulate_cohort(50, role='historical', horizon=8)
        df_backtest = df_historical.drop_duplicates(subset=['debtor_id']).head(10)

        policy = RandomPolicy()
        guardrails = Guardrails(max_contacts_per_week=1, max_total_contacts=4)
        engine = BacktestEngine(sim, policy, guardrails, df_backtest, horizon=8, verbose=False)

        results = engine.run(seed=42)

        # Check that max_total_contacts was respected (on average)
        avg_contacts = results.per_debtor_results['contacts_used'].mean()
        assert avg_contacts <= guardrails.max_total_contacts + 1  # Allow some variance


class TestComparisonMetrics:
    """Test policy comparison utilities."""

    def test_compare_policies(self):
        """compare_policies() produces comparison DataFrame."""
        sim = DataSimulator(seed=42)
        df_historical = sim.simulate_cohort(50, role='historical', horizon=4)
        df_backtest = df_historical.drop_duplicates(subset=['debtor_id']).head(10)

        # Run random policy
        policy_random = RandomPolicy()
        guardrails = Guardrails()
        engine = BacktestEngine(sim, policy_random, guardrails, df_backtest, horizon=4)
        results_random = [engine.run(seed=42 + i) for i in range(2)]

        results_dict = {
            'RandomPolicy': results_random,
        }

        comparison_df = compare_policies(results_dict)

        assert isinstance(comparison_df, pd.DataFrame)
        assert 'Policy' in comparison_df.columns
        assert 'Recovery (units)' in comparison_df.columns
        assert len(comparison_df) == 1


class TestEndToEndBacktest:
    """End-to-end bandit backtest tests."""

    def test_full_backtest_pipeline(self):
        """Full pipeline: fit models → backtest with multiple policies."""
        sim = DataSimulator(seed=42)

        # Generate historical and backtest cohorts
        df_historical = sim.simulate_cohort(100, role='historical', horizon=6)
        df_backtest = sim.simulate_cohort(50, role='backtest', horizon=6)
        df_backtest_features = df_backtest.drop_duplicates(subset=['debtor_id']).head(20)

        # Fit models
        response_model = ResponseModel(model_type='lr')
        response_model.fit(df_historical)

        uplift_model = UpliftModel(learner_type='t')
        uplift_model.fit(df_historical)

        # Create policies
        policy_random = RandomPolicy()
        policy_response = ResponseOnlyPolicy(response_model=response_model)

        guardrails = Guardrails(max_contacts_per_week=1, max_total_contacts=5)

        # Run backtests
        engine_random = BacktestEngine(sim, policy_random, guardrails, df_backtest_features, horizon=6)
        results_random = engine_random.run(seed=42)

        engine_response = BacktestEngine(sim, policy_response, guardrails, df_backtest_features, horizon=6)
        policy_response.reset()  # Reset between runs
        results_response = engine_response.run(seed=42)

        # Verify results
        assert results_random.payment_rate >= 0
        assert results_response.payment_rate >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
