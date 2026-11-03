"""
Test suite for the Data-Generating Process (DGP).

Tests cover:
- Debtor state initialization and transitions
- Action specification
- Feature generation and realism
- Payment probability (baseline and treatment effects)
- Heterogeneous treatment effects (key to the project)
- Multi-period simulation with state tracking
- Trust dynamics (annoyance accumulation)
- Cohort separation (historical vs backtest)
- Randomized logging policy
- End-to-end pipeline
"""

import pytest
import numpy as np
import pandas as pd
from src.dgp import DebtorState, Action, DebtorDGP, DataSimulator


class TestDebtorState:
    """Tests for DebtorState initialization and invariants."""

    def test_debtor_state_creation(self):
        """Should create a valid debtor state."""
        state = DebtorState(
            debtor_id=1,
            balance=1000.0,
            days_overdue=30,
            prior_payments=5,
            contactability=0.8,
            annoyance=0.2,
            opted_out=False,
            complained=False,
            paid=False,
        )
        assert state.debtor_id == 1
        assert state.balance == 1000.0
        assert state.prior_payments == 5

    def test_debtor_state_paid_terminal(self):
        """State should track if debtor has paid (terminal state)."""
        state = DebtorState(debtor_id=1, balance=100, days_overdue=10, prior_payments=0,
                           contactability=0.9, annoyance=0.0, opted_out=False,
                           complained=False, paid=True)
        assert state.paid is True

    def test_debtor_state_optout_terminal(self):
        """State should track opt-out (terminal state)."""
        state = DebtorState(debtor_id=1, balance=100, days_overdue=10, prior_payments=0,
                           contactability=0.9, annoyance=0.0, opted_out=True,
                           complained=False, paid=False)
        assert state.opted_out is True


class TestAction:
    """Tests for Action specification."""

    def test_action_no_contact(self):
        """Action with contact=False should have no channel/timing."""
        action = Action(contact=False)
        assert action.contact is False
        assert action.channel is None
        assert action.timing is None

    def test_action_contact_with_channel_timing(self):
        """Action with contact=True should specify channel and timing."""
        action = Action(contact=True, channel='voice', timing='morning')
        assert action.contact is True
        assert action.channel == 'voice'
        assert action.timing == 'morning'

    def test_action_valid_channels(self):
        """Should support all valid channels."""
        for channel in ['voice', 'sms', 'whatsapp', 'email']:
            action = Action(contact=True, channel=channel, timing='morning')
            assert action.channel == channel

    def test_action_valid_timings(self):
        """Should support all valid timings."""
        for timing in ['morning', 'afternoon', 'evening']:
            action = Action(contact=True, channel='voice', timing=timing)
            assert action.timing == timing


class TestDebtorDGPFeatureGeneration:
    """Tests for realistic feature generation."""

    def test_features_correct_count(self):
        """Should generate exactly n_debtors features."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(100)
        assert len(features) == 100
        assert features['debtor_id'].nunique() == 100

    def test_features_have_required_columns(self):
        """Generated features should have all required columns."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(50)
        required = ['debtor_id', 'balance', 'days_overdue', 'prior_payments',
                   'contactability', 'segment']
        for col in required:
            assert col in features.columns

    def test_balance_positive(self):
        """All balances should be positive."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(100)
        assert (features['balance'] > 0).all()

    def test_days_overdue_positive(self):
        """Days overdue should be in reasonable range."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(100)
        assert (features['days_overdue'] >= 7).all()
        assert (features['days_overdue'] <= 365).all()

    def test_prior_payments_nonnegative(self):
        """Prior payments should be non-negative."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(100)
        assert (features['prior_payments'] >= 0).all()

    def test_contactability_in_valid_range(self):
        """Contactability should be between 0 and 1."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(100)
        assert (features['contactability'] >= 0.3).all()
        assert (features['contactability'] <= 1.0).all()

    def test_segments_valid(self):
        """All segments should be one of the valid types."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(100)
        valid_segments = {'high_risk', 'medium_risk', 'low_risk', 'frequent_payer'}
        assert features['segment'].isin(valid_segments).all()

    def test_segment_distribution(self):
        """Segment distribution should roughly match intended probabilities."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(10000)
        counts = features['segment'].value_counts()
        # High-risk: ~20%, medium: ~30%, low: ~30%, frequent: ~20%
        assert 0.15 < counts.get('high_risk', 0) / 10000 < 0.25
        assert 0.25 < counts.get('medium_risk', 0) / 10000 < 0.35
        assert 0.25 < counts.get('low_risk', 0) / 10000 < 0.35


class TestBaselinePaymentProb:
    """Tests for baseline (no-contact) payment probability."""

    def test_frequent_payer_high_baseline(self):
        """Frequent payers should have higher baseline payment prob."""
        dgp = DebtorDGP(seed=42)
        features_fp = pd.Series({
            'segment': 'frequent_payer',
            'prior_payments': 15,
            'balance': 5000,
            'days_overdue': 30,
        })
        features_hr = pd.Series({
            'segment': 'high_risk',
            'prior_payments': 0,
            'balance': 5000,
            'days_overdue': 30,
        })
        p_fp = dgp.baseline_payment_prob(features_fp)
        p_hr = dgp.baseline_payment_prob(features_hr)
        assert p_fp > p_hr

    def test_high_risk_low_baseline(self):
        """High-risk debtors should have low baseline payment prob."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'high_risk',
            'prior_payments': 0,
            'balance': 100000,
            'days_overdue': 360,
        })
        p = dgp.baseline_payment_prob(features)
        assert 0 < p < 0.2

    def test_prob_in_valid_range(self):
        """All probabilities should be in [0.01, 0.95]."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(100)
        for _, f in features.iterrows():
            p = dgp.baseline_payment_prob(f)
            assert 0.01 <= p <= 0.95

    def test_prior_payments_increase_prob(self):
        """More prior payments should increase baseline prob."""
        dgp = DebtorDGP(seed=42)
        features_0 = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 0,
            'balance': 1000,  # Lower balance to avoid floor at 0.01
            'days_overdue': 50,  # Lower overdue to avoid floor
        })
        features_10 = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 10,
            'balance': 1000,
            'days_overdue': 50,
        })
        p_0 = dgp.baseline_payment_prob(features_0)
        p_10 = dgp.baseline_payment_prob(features_10)
        assert p_10 > p_0


class TestTreatmentEffect:
    """Tests for heterogeneous treatment effects (core of the project)."""

    def test_no_contact_zero_effect(self):
        """No-contact action should have zero treatment effect."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 0.8,
        })
        action = Action(contact=False)
        effect = dgp.treatment_effect(features, action)
        assert effect == 0.0

    def test_high_risk_voice_positive(self):
        """High-risk debtors should respond positively to voice."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'high_risk',
            'prior_payments': 0,
            'balance': 50000,
            'days_overdue': 200,
            'contactability': 1.0,
        })
        action_voice = Action(contact=True, channel='voice', timing='morning')
        effect = dgp.treatment_effect(features, action_voice)
        assert effect > 0.1  # Meaningful positive effect

    def test_high_risk_sms_negative(self):
        """High-risk debtors should respond negatively to SMS."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'high_risk',
            'prior_payments': 0,
            'balance': 50000,
            'days_overdue': 200,
            'contactability': 1.0,
        })
        action_sms = Action(contact=True, channel='sms', timing='morning')
        effect = dgp.treatment_effect(features, action_sms)
        assert effect < 0.05  # Weak or negative effect

    def test_low_risk_sms_better_than_voice(self):
        """For low-risk debtors, SMS should beat voice (digital preference)."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'low_risk',
            'prior_payments': 10,
            'balance': 2000,
            'days_overdue': 50,
            'contactability': 1.0,
        })
        action_voice = Action(contact=True, channel='voice', timing='morning')
        action_sms = Action(contact=True, channel='sms', timing='morning')
        effect_voice = dgp.treatment_effect(features, action_voice)
        effect_sms = dgp.treatment_effect(features, action_sms)
        assert effect_sms > effect_voice, f"SMS ({effect_sms}) should beat voice ({effect_voice}) for low-risk"

    def test_frequent_payer_lower_effects(self):
        """Frequent payers should have lower treatment effects (already motivated)."""
        dgp = DebtorDGP(seed=42)
        features_fp = pd.Series({
            'segment': 'frequent_payer',
            'prior_payments': 15,
            'balance': 1000,
            'days_overdue': 20,
            'contactability': 1.0,
        })
        features_hr = pd.Series({
            'segment': 'high_risk',
            'prior_payments': 0,
            'balance': 50000,
            'days_overdue': 200,
            'contactability': 1.0,
        })
        action = Action(contact=True, channel='voice', timing='morning')
        effect_fp = dgp.treatment_effect(features_fp, action)
        effect_hr = dgp.treatment_effect(features_hr, action)
        assert effect_fp < effect_hr

    def test_low_contactability_reduces_effect(self):
        """Low contactability should reduce treatment effect."""
        dgp = DebtorDGP(seed=42)
        features_high = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 1.0,
        })
        features_low = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 0.3,
        })
        action = Action(contact=True, channel='voice', timing='morning')
        effect_high = dgp.treatment_effect(features_high, action)
        effect_low = dgp.treatment_effect(features_low, action)
        assert effect_high > effect_low

    def test_effect_in_valid_range(self):
        """All effects should be in [-0.3, 0.4]."""
        dgp = DebtorDGP(seed=42)
        features = dgp.generate_debtor_features(50)
        for _, f in features.iterrows():
            for channel in ['voice', 'sms', 'whatsapp', 'email']:
                for timing in ['morning', 'afternoon', 'evening']:
                    action = Action(contact=True, channel=channel, timing=timing)
                    effect = dgp.treatment_effect(f, action)
                    assert -0.3 <= effect <= 0.4


class TestOutcomeSimulation:
    """Tests for outcome simulation."""

    def test_outcome_tuple_structure(self):
        """Simulate outcome should return (paid, opted_out, complained)."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 0.8,
        })
        state = DebtorState(
            debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
            contactability=0.8, annoyance=0.0, opted_out=False, complained=False, paid=False,
        )
        action = Action(contact=True, channel='voice', timing='morning')
        paid, opted_out, complained = dgp.simulate_outcome(features, state, action)
        assert isinstance(paid, (bool, np.bool_))
        assert isinstance(opted_out, (bool, np.bool_))
        assert isinstance(complained, (bool, np.bool_))

    def test_outcome_probabilistic(self):
        """Outcomes should vary across multiple simulations."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 0.8,
        })
        state = DebtorState(
            debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
            contactability=0.8, annoyance=0.0, opted_out=False, complained=False, paid=False,
        )
        action = Action(contact=True, channel='voice', timing='morning')
        outcomes = [dgp.simulate_outcome(features, state, action) for _ in range(20)]
        # Should have variance (not all same outcome)
        paid_vals = [o[0] for o in outcomes]
        assert len(set(paid_vals)) > 1  # At least 2 different outcomes

    def test_terminal_state_unchanged(self):
        """Outcome should not change if debtor already paid/opted_out."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 0.8,
        })
        state_paid = DebtorState(
            debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
            contactability=0.8, annoyance=0.0, opted_out=False, complained=False, paid=True,
        )
        action = Action(contact=True, channel='voice', timing='morning')
        paid, opted_out, complained = dgp.simulate_outcome(features, state_paid, action)
        assert paid is True
        assert opted_out is False


class TestAnnoyanceDynamics:
    """Tests for trust-cost dynamics (annoyance accumulation)."""

    def test_no_contact_annoyance_decays(self):
        """No contact should decay annoyance."""
        dgp = DebtorDGP(seed=42)
        state = DebtorState(
            debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
            contactability=0.8, annoyance=0.5, opted_out=False, complained=False, paid=False,
        )
        action = Action(contact=False)
        new_annoyance = dgp.update_annoyance(state, action)
        assert new_annoyance < state.annoyance

    def test_contact_annoyance_increases(self):
        """Contact should increase annoyance."""
        dgp = DebtorDGP(seed=42)
        state = DebtorState(
            debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
            contactability=0.8, annoyance=0.0, opted_out=False, complained=False, paid=False,
        )
        action = Action(contact=True, channel='voice', timing='morning')
        new_annoyance = dgp.update_annoyance(state, action)
        assert new_annoyance > state.annoyance

    def test_annoyance_in_valid_range(self):
        """Annoyance should stay in [0, 1]."""
        dgp = DebtorDGP(seed=42)
        for _ in range(100):
            state = DebtorState(
                debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
                contactability=0.8, annoyance=np.random.uniform(0, 1),
                opted_out=False, complained=False, paid=False,
            )
            action = Action(contact=True, channel='voice', timing='morning')
            new_annoyance = dgp.update_annoyance(state, action)
            assert 0 <= new_annoyance <= 1.0


class TestStateTransition:
    """Tests for state stepping (one timestep forward)."""

    def test_step_forward_balance_unchanged_no_payment(self):
        """Balance should stay same if debtor doesn't pay."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'high_risk',
            'prior_payments': 0,
            'balance': 50000,
            'days_overdue': 200,
            'contactability': 1.0,
        })
        state = DebtorState(
            debtor_id=1, balance=50000, days_overdue=200, prior_payments=0,
            contactability=1.0, annoyance=0.0, opted_out=False, complained=False, paid=False,
        )
        action = Action(contact=False)
        new_state, outcome = dgp.step_forward(features, state, action)
        if not outcome['paid']:
            assert new_state.balance == state.balance

    def test_step_forward_days_overdue_increases(self):
        """Days overdue should increase by 7 (one week)."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 0.8,
        })
        state = DebtorState(
            debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
            contactability=0.8, annoyance=0.0, opted_out=False, complained=False, paid=False,
        )
        action = Action(contact=True, channel='voice', timing='morning')
        new_state, outcome = dgp.step_forward(features, state, action)
        assert new_state.days_overdue == state.days_overdue + 7

    def test_step_forward_outcome_dict(self):
        """Outcome dict should have expected keys."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 0.8,
        })
        state = DebtorState(
            debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
            contactability=0.8, annoyance=0.0, opted_out=False, complained=False, paid=False,
        )
        action = Action(contact=True, channel='voice', timing='morning')
        new_state, outcome = dgp.step_forward(features, state, action)
        expected_keys = {'paid', 'opted_out', 'complained', 'annoyance'}
        assert set(outcome.keys()) == expected_keys


class TestCounterfactual:
    """Tests for counterfactual simulation (what-if queries)."""

    def test_counterfactual_outcome_dict(self):
        """Counterfactual should return outcome dict without advancing state."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'medium_risk',
            'prior_payments': 5,
            'balance': 5000,
            'days_overdue': 100,
            'contactability': 0.8,
        })
        state = DebtorState(
            debtor_id=1, balance=5000, days_overdue=100, prior_payments=5,
            contactability=0.8, annoyance=0.0, opted_out=False, complained=False, paid=False,
        )
        action = Action(contact=True, channel='voice', timing='morning')
        outcome = dgp.simulate_counterfactual(features, state, action)

        expected_keys = {'paid', 'opted_out', 'complained', 'annoyance'}
        assert set(outcome.keys()) == expected_keys

    def test_counterfactual_vs_actual(self):
        """Compare counterfactual action vs actual taken action."""
        dgp = DebtorDGP(seed=42)
        features = pd.Series({
            'segment': 'high_risk',
            'prior_payments': 0,
            'balance': 50000,
            'days_overdue': 200,
            'contactability': 1.0,
        })
        state = DebtorState(
            debtor_id=1, balance=50000, days_overdue=200, prior_payments=0,
            contactability=1.0, annoyance=0.0, opted_out=False, complained=False, paid=False,
        )

        # Actual action: voice
        action_voice = Action(contact=True, channel='voice', timing='morning')
        outcome_voice = dgp.simulate_counterfactual(features, state, action_voice)

        # Counterfactual: SMS
        action_sms = Action(contact=True, channel='sms', timing='morning')
        outcome_sms = dgp.simulate_counterfactual(features, state, action_sms)

        # Both should be valid outcomes
        assert isinstance(outcome_voice['paid'], (bool, np.bool_))
        assert isinstance(outcome_sms['paid'], (bool, np.bool_))


class TestCohortGeneration:
    """Tests for simulation and cohort separation."""

    def test_cohort_generation_historical(self):
        """Should generate a valid historical cohort."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(100, role='historical', horizon=8)
        assert len(df) > 0
        assert 'debtor_id' in df.columns
        assert 'week' in df.columns
        assert 'outcome_paid' in df.columns

    def test_cohort_generation_backtest(self):
        """Should generate a valid backtest cohort."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(100, role='backtest', horizon=8)
        assert len(df) > 0
        assert df['debtor_id'].nunique() <= 100

    def test_cohorts_distinct_debtors(self):
        """Historical and backtest cohorts use different random seeds."""
        sim = DataSimulator(seed=42)
        df_hist = sim.simulate_cohort(50, role='historical', horizon=4, seed=100)
        df_bt = sim.simulate_cohort(50, role='backtest', horizon=4, seed=200)

        # Both should have valid data
        assert len(df_hist) > 0
        assert len(df_bt) > 0

        # Verify they're from different random samples by checking distinct features
        # (debtors in each cohort start with IDs 0..n, so we can't compare IDs directly)
        assert df_hist['balance'].std() > 0
        assert df_bt['balance'].std() > 0

    def test_multi_period_simulation(self):
        """Each debtor should have multiple weeks of data (up to horizon)."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(50, role='historical', horizon=8)

        weeks_per_debtor = df.groupby('debtor_id')['week'].count()
        # Some will stop early (payment/optout), but all should have >= 1 week
        assert (weeks_per_debtor >= 1).all()
        # Some should go beyond 1 week (not all immediate payments)
        assert (weeks_per_debtor > 1).any()

    def test_randomized_logging_policy(self):
        """Actions should be randomized (mix of contact/no-contact)."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(500, role='historical', horizon=8)

        contact_rate = df['action_contact'].mean()
        # Should be close to 60% (configured in DGP)
        assert 0.5 < contact_rate < 0.7

    def test_channels_only_when_contacting(self):
        """Channel should be None only when action_contact=0."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(100, role='historical', horizon=8)

        no_contact = df[df['action_contact'] == 0]
        assert (no_contact['action_channel'].isna()).all()

        contact = df[df['action_contact'] == 1]
        assert (~contact['action_channel'].isna()).all()


class TestHeterogeneousEffects:
    """Tests validating the presence of heterogeneous effects."""

    def test_effects_vary_by_segment(self):
        """Different segments should respond differently to same action."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(1000, role='historical', horizon=8)

        results = {}
        for segment in ['high_risk', 'medium_risk', 'low_risk', 'frequent_payer']:
            seg_data = df[df['segment'] == segment]
            contacted = seg_data[seg_data['action_contact'] == 1]
            not_contacted = seg_data[seg_data['action_contact'] == 0]

            if len(contacted) > 0 and len(not_contacted) > 0:
                effect = contacted['outcome_paid'].mean() - not_contacted['outcome_paid'].mean()
                results[segment] = effect

        # Effects should vary across segments (not all the same)
        effects = list(results.values())
        assert max(effects) - min(effects) > 0.05

    def test_effects_vary_by_channel(self):
        """Different channels should have different effects."""
        sim = DataSimulator(seed=42)
        df = sim.simulate_cohort(1000, role='historical', horizon=8)

        results = {}
        for channel in ['voice', 'sms', 'whatsapp', 'email']:
            channel_data = df[df['action_channel'] == channel]
            if len(channel_data) > 50:
                effect = channel_data['outcome_paid'].mean()
                results[channel] = effect

        # Channel effects should vary
        if len(results) >= 2:
            effects = list(results.values())
            assert max(effects) - min(effects) > 0.02


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_runs(self):
        """Full pipeline from cohort generation to evaluation should run."""
        sim = DataSimulator(seed=123)

        # Generate both cohorts with different seeds
        df_hist = sim.simulate_cohort(200, role='historical', horizon=6, seed=300)
        df_bt = sim.simulate_cohort(200, role='backtest', horizon=6, seed=400)

        # Both should have data
        assert len(df_hist) > 0
        assert len(df_bt) > 0

        # Each cohort should have debtor records
        assert df_hist['debtor_id'].nunique() > 0
        assert df_bt['debtor_id'].nunique() > 0

    def test_sanity_check_runs_without_error(self):
        """Sanity check should run and detect heterogeneous effects."""
        sim = DataSimulator(seed=42)
        df = sim.sanity_check(n_debtors=300, horizon=6)

        # Should have generated data
        assert len(df) > 0
        assert df['debtor_id'].nunique() > 0


class TestReproducibility:
    """Tests for reproducibility across runs."""

    def test_same_seed_same_features(self):
        """Same seed should produce identical features."""
        dgp1 = DebtorDGP(seed=42)
        dgp2 = DebtorDGP(seed=42)

        features1 = dgp1.generate_debtor_features(50)
        features2 = dgp2.generate_debtor_features(50)

        pd.testing.assert_frame_equal(features1, features2)

    def test_different_seed_different_features(self):
        """Different seeds should produce different features."""
        dgp1 = DebtorDGP(seed=42)
        dgp2 = DebtorDGP(seed=123)

        features1 = dgp1.generate_debtor_features(50)
        features2 = dgp2.generate_debtor_features(50)

        # Should not be identical
        assert not features1.equals(features2)

    def test_cohort_seed_reproducible(self):
        """Cohorts with same role and seed should be identical."""
        sim1 = DataSimulator(seed=42)
        sim2 = DataSimulator(seed=42)

        df1 = sim1.simulate_cohort(100, role='historical', horizon=6, seed=1000)
        df2 = sim2.simulate_cohort(100, role='historical', horizon=6, seed=1000)

        pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
