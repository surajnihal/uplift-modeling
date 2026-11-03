"""
Tests for the stronger baselines in src/baselines.py:
UpliftArgmaxPolicy and SegmentBestActionPolicy.

Validates: BanditPolicy interface, contact budget respected, guardrails respected
(through BacktestEngine), and the policies' core selection logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.baselines import SegmentBestActionPolicy, UpliftArgmaxPolicy
from src.backtest import BacktestEngine
from src.bandit import BanditPolicy, CHANNELS, TIMINGS, NO_CONTACT_ARM
from src.dgp import Action, DataSimulator, DebtorState
from src.guardrails import Guardrails
from src.modeling import UpliftModel


class StubUpliftModel:
    """Deterministic CATE: one clearly best arm (sms/evening); debtor-level scale = contactability."""

    def predict_cate(self, df, channel='voice', timing='morning'):
        base = 0.20 if (channel, timing) == ('sms', 'evening') else 0.05
        return base * df['contactability'].values


@pytest.fixture(scope='module')
def data():
    sim = DataSimulator(seed=7)
    hist = sim.simulate_cohort(600, role='historical', horizon=8)
    cohort = (sim.simulate_cohort(60, role='backtest', horizon=8)
              .drop_duplicates(subset=['debtor_id']).reset_index(drop=True))
    return sim, hist, cohort


@pytest.fixture(scope='module')
def fitted_uplift(data):
    _, hist, _ = data
    return UpliftModel(learner_type='t').fit(hist)


def _state(row):
    return DebtorState(
        debtor_id=int(row['debtor_id']), balance=row['balance'], days_overdue=row['days_overdue'],
        prior_payments=row['prior_payments'], contactability=row['contactability'],
        annoyance=0.0, opted_out=False, complained=False, paid=False,
    )


def _policies(data, fitted_uplift, fraction=0.5):
    _, hist, _ = data
    return [
        UpliftArgmaxPolicy(uplift_model=fitted_uplift, contact_budget_fraction=fraction),
        SegmentBestActionPolicy(contact_budget_fraction=fraction).fit(hist),
    ]


class TestInterface:
    def test_subclasses_banditpolicy(self):
        assert issubclass(UpliftArgmaxPolicy, BanditPolicy)
        assert issubclass(SegmentBestActionPolicy, BanditPolicy)

    def test_select_arm_returns_valid_action(self, data, fitted_uplift):
        _, _, cohort = data
        for policy in _policies(data, fitted_uplift):
            policy.prepare_cohort(cohort)
            for _, row in cohort.iterrows():
                a = policy.select_arm(row, _state(row), week=0)
                assert isinstance(a, Action)
                if a.contact:
                    assert a.channel in CHANNELS and a.timing in TIMINGS
                else:
                    assert a == NO_CONTACT_ARM

    def test_update_fit_reset_summary_dont_crash(self, data, fitted_uplift):
        for policy in _policies(data, fitted_uplift):
            policy.update(NO_CONTACT_ARM, {'paid': True, 'opted_out': False,
                                           'complained': False, 'annoyance': 0.1})
            policy.reset()
            assert 'policy' in policy.get_summary()

    def test_no_online_learning(self, data, fitted_uplift):
        """Static policies: update() must not change decisions."""
        _, _, cohort = data
        for policy in _policies(data, fitted_uplift):
            policy.prepare_cohort(cohort)
            row = cohort.iloc[0]
            before = policy.select_arm(row, _state(row), 0)
            for _ in range(50):
                policy.update(Action(contact=True, channel='email', timing='morning'),
                              {'paid': False, 'opted_out': True, 'complained': True, 'annoyance': 1.0})
            assert policy.select_arm(row, _state(row), 0) == before

    def test_uplift_policy_requires_model(self, data):
        _, _, cohort = data
        with pytest.raises(ValueError):
            UpliftArgmaxPolicy(uplift_model=None).prepare_cohort(cohort)


class TestBudget:
    @pytest.mark.parametrize('fraction', [0.1, 0.5, 1.0])
    def test_eligible_set_size_matches_budget(self, data, fitted_uplift, fraction):
        _, _, cohort = data
        expected = int(np.ceil(len(cohort) * fraction))
        for policy in _policies(data, fitted_uplift, fraction):
            policy.prepare_cohort(cohort)
            n_contact = sum(
                policy.select_arm(row, _state(row), 0).contact for _, row in cohort.iterrows()
            )
            assert n_contact == expected

    def test_zero_budget_never_contacts(self, data, fitted_uplift):
        _, _, cohort = data
        for policy in _policies(data, fitted_uplift, fraction=0.0):
            policy.prepare_cohort(cohort)
            assert all(not policy.select_arm(row, _state(row), 0).contact for _, row in cohort.iterrows())

    def test_backtest_contacts_only_budgeted_debtors(self, data, fitted_uplift):
        sim, _, cohort = data
        guardrails = Guardrails(max_contacts_per_week=1, max_total_contacts=4, rest_period_weeks=1)
        for policy in _policies(data, fitted_uplift, 0.5):
            res = BacktestEngine(sim, policy, guardrails, cohort, horizon=8).run(seed=1)
            contacted = (res.per_debtor_results['contacts_used'] > 0).sum()
            assert contacted <= int(np.ceil(len(cohort) * 0.5))
            assert res.total_contacts > 0


class TestSelectionLogic:
    def test_uplift_argmax_ranks_by_cate_and_picks_best_arm(self):
        cohort = pd.DataFrame({
            'debtor_id': range(10),
            'contactability': np.linspace(0.1, 1.0, 10),
            'segment': 'high_risk',
        })
        policy = UpliftArgmaxPolicy(uplift_model=StubUpliftModel(), contact_budget_fraction=0.3)
        policy.prepare_cohort(cohort)
        acts = {i: policy.select_arm(cohort.iloc[i], None, 0) for i in range(10)}
        contacted = {i for i, a in acts.items() if a.contact}
        assert contacted == {7, 8, 9}  # highest contactability -> highest CATE
        for i in contacted:
            assert (acts[i].channel, acts[i].timing) == ('sms', 'evening')

    def test_uplift_argmax_min_uplift_floor(self):
        cohort = pd.DataFrame({'debtor_id': range(10),
                               'contactability': np.linspace(0.1, 1.0, 10), 'segment': 'high_risk'})
        policy = UpliftArgmaxPolicy(uplift_model=StubUpliftModel(),
                                    contact_budget_fraction=1.0, min_uplift=0.15)
        policy.prepare_cohort(cohort)
        contacted = {i for i in range(10) if policy.select_arm(cohort.iloc[i], None, 0).contact}
        assert contacted == {i for i in range(10) if 0.2 * cohort.contactability[i] >= 0.15}

    def test_segment_policy_picks_empirical_best_and_ranks_segments(self):
        rows = []
        # seg A: baseline 0.1; voice/morning pays 0.6 (lift 0.5); sms/evening pays 0.2
        # seg B: baseline 0.3; email/morning pays 0.4 (lift 0.1)
        def add(seg, contact, ch, t, n_paid, n):
            for k in range(n):
                rows.append({'segment': seg, 'action_contact': contact, 'action_channel': ch,
                             'action_timing': t, 'outcome_paid': int(k < n_paid)})
        add('A', 0, None, None, 1, 10)
        add('A', 1, 'voice', 'morning', 6, 10)
        add('A', 1, 'sms', 'evening', 2, 10)
        add('B', 0, None, None, 3, 10)
        add('B', 1, 'email', 'morning', 4, 10)
        add('B', 1, 'voice', 'evening', 1, 10)
        policy = SegmentBestActionPolicy(contact_budget_fraction=0.5).fit(pd.DataFrame(rows))

        assert policy.segment_best_action['A'] == Action(True, 'voice', 'morning')
        assert policy.segment_best_action['B'] == Action(True, 'email', 'morning')
        assert policy.segment_lift['A'] == pytest.approx(0.5)
        assert policy.segment_lift['B'] == pytest.approx(0.1)

        cohort = pd.DataFrame({'debtor_id': range(8), 'segment': ['A', 'B'] * 4})
        policy.prepare_cohort(cohort)
        acts = [policy.select_arm(cohort.iloc[i], None, 0) for i in range(8)]
        # Budget 4 of 8 -> exactly the four segment-A debtors (higher lift), voice/morning.
        assert [a.contact for a in acts] == [True, False] * 4
        assert all(a == Action(True, 'voice', 'morning') for a in acts[::2])

    def test_segment_policy_unseen_segment_not_contacted(self):
        policy = SegmentBestActionPolicy(contact_budget_fraction=1.0)
        policy.segment_best_action = {'A': Action(True, 'sms', 'morning')}
        policy.segment_lift = {'A': 0.2}
        cohort = pd.DataFrame({'debtor_id': [0, 1], 'segment': ['A', 'Z']})
        policy.prepare_cohort(cohort)
        assert policy.select_arm(cohort.iloc[0], None, 0).contact
        assert not policy.select_arm(cohort.iloc[1], None, 0).contact


class TestGuardrails:
    def test_guardrails_override_baseline_recommendations(self, data, fitted_uplift):
        """Baselines recommend contact every week; guardrails must still cap/space contacts."""
        sim, _, cohort = data
        guardrails = Guardrails(max_contacts_per_week=1, max_total_contacts=2, rest_period_weeks=2)
        for policy in _policies(data, fitted_uplift, 1.0):
            res = BacktestEngine(sim, policy, guardrails, cohort, horizon=8).run(seed=3)
            assert res.per_debtor_results['contacts_used'].max() <= 2

            dec = res.policy_decisions
            contacted = dec[dec['final_action'].str.contains('contact=True')]
            # Rest period of 2 weeks: consecutive contacts for a debtor >= 3 weeks apart
            for _, g in contacted.groupby('debtor_id'):
                weeks = sorted(g['week'])
                assert all(b - a >= 3 for a, b in zip(weeks, weeks[1:]))
            # Blocks were actually exercised
            assert dec['guardrail_block_reason'].notna().any()

    def test_no_contact_after_optout(self, data, fitted_uplift):
        sim, _, cohort = data
        guardrails = Guardrails(hard_stop_on_optout=True)
        for policy in _policies(data, fitted_uplift, 1.0):
            policy.prepare_cohort(cohort)
            row = cohort.iloc[0]
            state = _state(row)
            state.opted_out = True
            recommended = policy.select_arm(row, state, week=1)
            final, reason = guardrails.apply(recommended, state, 1, [])
            assert not final.contact
            assert reason == 'opted_out'

    def test_quiet_hours_respected(self, data):
        sim, _, cohort = data
        # Block the timing the stub model prefers; guardrails must turn it into no-contact.
        guardrails = Guardrails(quiet_hours={'morning': True, 'afternoon': True, 'evening': False})
        policy = UpliftArgmaxPolicy(uplift_model=StubUpliftModel(), contact_budget_fraction=1.0)
        res = BacktestEngine(sim, policy, guardrails, cohort, horizon=4).run(seed=2)
        assert res.total_contacts == 0
        assert (res.policy_decisions['guardrail_block_reason'] == 'timing_evening_blocked').any()
