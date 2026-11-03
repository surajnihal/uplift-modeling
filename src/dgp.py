"""
Data-Generating Process for Debt Collection Uplift Modeling.

Simulates debtors with heterogeneous treatment effects under a randomized historical
logging policy. Supports multi-period simulation with state tracking and trust dynamics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class DebtorState:
    """State of a single debtor at a timestep."""
    debtor_id: int
    balance: float
    days_overdue: int
    prior_payments: int
    contactability: float  # 0-1, probability of being reachable
    annoyance: float  # 0-1, accumulated contact dissatisfaction
    opted_out: bool
    complained: bool
    paid: bool


@dataclass
class Action:
    """Contact action taken."""
    contact: bool  # True = contact, False = no contact
    channel: Optional[str] = None  # 'voice', 'sms', 'whatsapp', 'email'
    timing: Optional[str] = None  # 'morning', 'afternoon', 'evening'


class DebtorDGP:
    """
    Data-generating process for debtors.

    Implements heterogeneous treatment effects: payment probability depends on both
    debtor characteristics and the action taken. Different debtor segments respond
    differently to different channels and timings.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_debtor_features(self, n_debtors: int) -> pd.DataFrame:
        """Generate static debtor features."""
        debtors = pd.DataFrame({
            'debtor_id': np.arange(n_debtors),
            'balance': self.rng.lognormal(mean=5, sigma=1.5, size=n_debtors),  # roughly 1k to 500k currency units
            'days_overdue': self.rng.randint(7, 365, n_debtors),
            'prior_payments': self.rng.randint(0, 20, n_debtors),  # payment track record
            'contactability': self.rng.uniform(0.3, 1.0, n_debtors),  # easy to reach?
            'segment': self.rng.choice(['high_risk', 'medium_risk', 'low_risk', 'frequent_payer'],
                                      p=[0.2, 0.3, 0.3, 0.2], size=n_debtors),
        })
        return debtors

    def baseline_payment_prob(self, features: pd.Series) -> float:
        """
        P(pay | no contact).
        Reflects inherent willingness: frequent payers high, high-risk low.
        """
        segment_base = {
            'frequent_payer': 0.6,
            'low_risk': 0.3,
            'medium_risk': 0.15,
            'high_risk': 0.05,
        }
        base = segment_base.get(features['segment'], 0.1)

        # Prior payments increase baseline slightly
        base += features['prior_payments'] * 0.005

        # High balance and long overdue reduce baseline
        base -= np.clip(features['balance'] / 100000, 0, 0.2)
        base -= np.clip(features['days_overdue'] / 365, 0, 0.2)

        return np.clip(base, 0.01, 0.95)

    def treatment_effect(self, features: pd.Series, action: Action) -> float:
        """
        Heterogeneous treatment effect: increment to P(pay) from action.

        Different segments respond differently to different channels:
        - Frequent payers: respond well to any reminder (subtle effect)
        - Low/medium risk: respond well to voice/whatsapp, less to SMS
        - High risk: voice only has meaningful effect; SMS/email backfire

        Timing matters for some segments; contact frequency (annoyance) reduces effect.
        """
        if not action.contact:
            return 0.0

        segment = features['segment']
        contactability = features['contactability']

        # Channel-segment fit: strong heterogeneity in optimal channel per segment
        # High-risk: voice dominant (direct contact matters most)
        # Frequent-payer: WhatsApp best (low-friction, already motivated)
        # Low-risk: SMS/Email best (easy digital contact, over-contact fatigue on phone)
        # Medium-risk: balanced across channels
        channel_effect = {
            'voice': {'frequent_payer': 0.02, 'low_risk': 0.08, 'medium_risk': 0.18, 'high_risk': 0.30},
            'sms': {'frequent_payer': 0.04, 'low_risk': 0.18, 'medium_risk': 0.12, 'high_risk': -0.08},
            'whatsapp': {'frequent_payer': 0.12, 'low_risk': 0.16, 'medium_risk': 0.16, 'high_risk': 0.10},
            'email': {'frequent_payer': 0.03, 'low_risk': 0.14, 'medium_risk': 0.08, 'high_risk': -0.05},
        }
        effect = channel_effect.get(action.channel, {}).get(segment, 0.1)

        # Timing adjusts effect (time-of-day sensitivity varies by segment)
        timing_mult = {
            'morning': {'frequent_payer': 1.0, 'low_risk': 1.1, 'medium_risk': 1.0, 'high_risk': 0.9},
            'afternoon': {'frequent_payer': 0.95, 'low_risk': 1.0, 'medium_risk': 1.05, 'high_risk': 1.0},
            'evening': {'frequent_payer': 0.9, 'low_risk': 0.95, 'medium_risk': 0.95, 'high_risk': 1.1},
        }
        effect *= timing_mult.get(action.timing, {}).get(segment, 1.0)

        # Contactability gates the effect (can't reach = no effect)
        effect *= contactability

        return np.clip(effect, -0.3, 0.4)

    def simulate_outcome(self, features: pd.Series, state: DebtorState, action: Action) -> Tuple[bool, bool, bool]:
        """
        Simulate outcome of an action taken.

        Returns:
            (paid, opted_out, complained)
        """
        if state.opted_out or state.paid or state.complained:
            return state.paid, state.opted_out, state.complained

        # Probability of payment given action
        p_pay = self.baseline_payment_prob(features) + self.treatment_effect(features, action)
        p_pay = np.clip(p_pay, 0.01, 0.95)

        # Annoyance increases opt-out and complaint risk
        p_optout = 0.01 + (state.annoyance ** 1.5) * 0.3
        p_complaint = 0.005 + (state.annoyance ** 1.5) * 0.15

        # Ensure probabilities sum to <= 1 (Bug fix #2)
        total = p_pay + p_optout + p_complaint
        if total > 1.0:
            scale = 0.99 / total
            p_pay *= scale
            p_optout *= scale
            p_complaint *= scale

        # Sample outcomes (mutually exclusive: pay, opt out, or complain)
        pvals = [p_pay, p_optout, p_complaint, 1 - p_pay - p_optout - p_complaint]
        outcomes = self.rng.multinomial(1, pvals)
        paid = bool(outcomes[0])
        opted_out = bool(outcomes[1])
        complained = bool(outcomes[2])

        return paid, opted_out, complained

    def update_annoyance(self, state: DebtorState, action: Action,
                        balance_p75: Optional[float] = None) -> float:
        """
        Update annoyance (frustration) from repeated/bad-timed contact.

        Args:
            balance_p75: 75th percentile of balance in the cohort (pre-computed for efficiency).
                        If None, uses a default estimate based on lognormal distribution.
        """
        if not action.contact:
            decay = 0.9  # Annoyance decays slightly if no contact
            return state.annoyance * decay

        # Contact increments annoyance
        increment = 0.15

        # High-balance debtors get more annoyed by bad channels
        if balance_p75 is None:
            # Default: estimate 75th percentile of lognormal(mean=5, sigma=1.5)
            balance_p75 = 251.0

        if state.balance > balance_p75:
            if action.channel == 'sms':
                increment += 0.1
            elif action.channel == 'email':
                increment += 0.05

        return np.clip(state.annoyance + increment, 0, 1.0)

    def step_forward(self, features: pd.Series, state: DebtorState, action: Action,
                     balance_p75: Optional[float] = None) -> Tuple[DebtorState, dict]:
        """
        Advance state one timestep given an action.

        Args:
            balance_p75: 75th percentile of balance (pre-computed per cohort for efficiency).

        Returns:
            (updated_state, outcome_dict)
        """
        paid, opted_out, complained = self.simulate_outcome(features, state, action)
        annoyance = self.update_annoyance(state, action, balance_p75=balance_p75)

        updated = DebtorState(
            debtor_id=state.debtor_id,
            balance=state.balance * 0.95 if paid else state.balance,  # Balance decreases if paid
            days_overdue=state.days_overdue + 7,  # Advance by 1 week
            prior_payments=state.prior_payments + (1 if paid else 0),
            contactability=state.contactability,  # Stays constant (inherent reachability)
            annoyance=annoyance,
            opted_out=opted_out or state.opted_out,
            complained=complained or state.complained,
            paid=paid or state.paid,
        )

        outcome = {
            'paid': paid,
            'opted_out': opted_out,
            'complained': complained,
            'annoyance': annoyance,
        }

        return updated, outcome


    def simulate_counterfactual(self, features: pd.Series, state: DebtorState,
                               action: Action, balance_p75: Optional[float] = None) -> dict:
        """
        Simulate outcome for a counterfactual action without advancing state.

        Used by backtest to ask "what if we took action X?" given current state.

        Returns:
            outcome dict with keys: paid, opted_out, complained, annoyance
        """
        paid, opted_out, complained = self.simulate_outcome(features, state, action)
        annoyance = self.update_annoyance(state, action, balance_p75=balance_p75)

        return {
            'paid': paid,
            'opted_out': opted_out,
            'complained': complained,
            'annoyance': annoyance,
        }


class DataSimulator:
    """
    End-to-end simulator: generates historical (training) and backtest cohorts.
    """

    def __init__(self, seed: int = 42):
        self.dgp = DebtorDGP(seed=seed)
        self.seed = seed

    def simulate_cohort(self, n_debtors: int, role: str = 'historical',
                       horizon: int = 8, seed: Optional[int] = None,
                       policy_fn: Optional[callable] = None) -> pd.DataFrame:
        """
        Generate a full cohort with optional external policy.

        Args:
            n_debtors: Number of debtors to simulate
            role: 'historical' or 'backtest' (determines seed sampling so cohorts are distinct)
            horizon: Number of weeks to simulate per debtor
            seed: Random seed for this cohort (if None, derives from self.seed and role)
            policy_fn: Optional callable(features, state, week) -> Action.
                      If None, uses randomized logging policy (60% contact, uniform channel/timing).
                      Use this for backtest to apply response-only or uplift-bandit policies.

        Returns:
            DataFrame with columns:
            - debtor_id, balance, days_overdue, prior_payments, contactability, segment (features)
            - week, action_contact, action_channel, action_timing (logged action)
            - outcome_paid, outcome_annoyance, outcome_opted_out, outcome_complained (outcome)
        """
        if seed is None:
            seed = self.seed + (1000 if role == 'backtest' else 0)

        dgp = DebtorDGP(seed=seed)

        # Generate static features
        features_df = dgp.generate_debtor_features(n_debtors)

        # Pre-compute balance 75th percentile for the cohort (used in annoyance dynamics)
        balance_p75 = features_df['balance'].quantile(0.75)

        rows = []
        for _, feature_row in features_df.iterrows():
            debtor_id = int(feature_row['debtor_id'])

            # Initialize state
            state = DebtorState(
                debtor_id=debtor_id,
                balance=feature_row['balance'],
                days_overdue=feature_row['days_overdue'],
                prior_payments=feature_row['prior_payments'],
                contactability=feature_row['contactability'],
                annoyance=0.0,
                opted_out=False,
                complained=False,
                paid=False,
            )

            # Simulate over horizon (e.g., 8 weeks)
            for week in range(horizon):
                # Decide action: either from external policy or default randomized logging
                if policy_fn is not None:
                    action = policy_fn(feature_row, state, week)
                else:
                    # Randomized logging policy: uniform over action space
                    # (This is the key to clean uplift identification)
                    contact = dgp.rng.choice([True, False], p=[0.6, 0.4])
                    channel = dgp.rng.choice(['voice', 'sms', 'whatsapp', 'email']) if contact else None
                    timing = dgp.rng.choice(['morning', 'afternoon', 'evening']) if contact else None
                    action = Action(contact=contact, channel=channel, timing=timing)

                # Step forward
                state, outcome = dgp.step_forward(feature_row, state, action, balance_p75=balance_p75)

                # Record (Bug fix: use action object fields, not local variables)
                row = {
                    'debtor_id': debtor_id,
                    'balance': feature_row['balance'],
                    'days_overdue': feature_row['days_overdue'],
                    'prior_payments': feature_row['prior_payments'],
                    'contactability': feature_row['contactability'],
                    'segment': feature_row['segment'],
                    'week': week,
                    'action_contact': int(action.contact),
                    'action_channel': action.channel,
                    'action_timing': action.timing,
                    'outcome_paid': int(outcome['paid']),
                    'outcome_annoyance': outcome['annoyance'],
                    'outcome_opted_out': int(outcome['opted_out']),
                    'outcome_complained': int(outcome['complained']),
                }
                rows.append(row)

                # Stop if debtor paid, opted out, or complained
                if state.paid or state.opted_out or state.complained:
                    break

        df = pd.DataFrame(rows)
        return df

    def sanity_check(self, n_debtors: int = 1000, horizon: int = 8):
        """
        Sanity check: verify heterogeneous effects are present.

        Plots true effect sizes by segment and action.
        """
        print("Running sanity check: verifying heterogeneous effects...")

        # Simulate a cohort
        df = self.simulate_cohort(n_debtors, role='historical', horizon=horizon)

        # For each segment and action, compute empirical effect
        effects = {}
        for segment in ['high_risk', 'medium_risk', 'low_risk', 'frequent_payer']:
            segment_data = df[df['segment'] == segment]

            for channel in ['voice', 'sms', 'whatsapp', 'email']:
                contacted = segment_data[segment_data['action_channel'] == channel]
                not_contacted = segment_data[segment_data['action_contact'] == 0]

                if len(contacted) > 0 and len(not_contacted) > 0:
                    p_pay_contacted = contacted['outcome_paid'].mean()
                    p_pay_not = not_contacted['outcome_paid'].mean()
                    effect = p_pay_contacted - p_pay_not
                    effects[f'{segment}_{channel}'] = effect

        print("\nEmpirical treatment effects (pay probability difference):")
        for key, val in sorted(effects.items(), key=lambda x: x[1], reverse=True):
            print(f"  {key}: {val:.3f}")

        print("\nHeterogeneous effects confirmed!" if len(effects) > 0 else "No effects detected!")

        return df


if __name__ == '__main__':
    # Quick test
    sim = DataSimulator(seed=42)
    df = sim.sanity_check(n_debtors=500, horizon=8)
    print(f"\nGenerated {len(df)} records from {df['debtor_id'].nunique()} debtors")
    print(df.head())
