"""
Bandit Policy Layer for Debt Collection Contact Strategy.

Implements Thompson Sampling, ε-Greedy, Response-Only, and Random policies
for arm (action) selection during backtest. Each policy learns from outcomes
to optimize contact strategy per debtor.

Arms: 4 channels × 3 timings = 12 contact actions + 1 no-contact = 13 total.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from scipy.stats import beta as beta_dist

from src.dgp import Action, DebtorState


# Define arm space: 4 channels × 3 timings + no-contact
CHANNELS = ['voice', 'sms', 'whatsapp', 'email']
TIMINGS = ['morning', 'afternoon', 'evening']
CONTACT_ARMS = [
    Action(contact=True, channel=ch, timing=t)
    for ch in CHANNELS for t in TIMINGS
]  # 12 arms
NO_CONTACT_ARM = Action(contact=False)  # 1 arm
ALL_ARMS = CONTACT_ARMS + [NO_CONTACT_ARM]  # 13 arms

ARM_IDX = {i: arm for i, arm in enumerate(ALL_ARMS)}
IDX_ARM = {str(arm): i for i, arm in enumerate(ALL_ARMS)}


class BanditPolicy(ABC):
    """
    Abstract base class for bandit policies.

    Policies learn which arm (contact action) maximizes expected reward per debtor.
    Thompson Sampling uses Bayesian posteriors; ε-Greedy uses empirical means.
    """

    @abstractmethod
    def select_arm(self, features: pd.Series, state: DebtorState, week: int) -> Action:
        """
        Select an arm (action) for this debtor at this week.

        Args:
            features: Debtor static features (segment, balance, etc.)
            state: Current debtor state (annoyance, balance, opted_out, etc.)
            week: Week number (0-indexed)

        Returns:
            Action (contact or no-contact, with channel/timing if contact)
        """
        pass

    @abstractmethod
    def update(self, arm: Action, outcome: dict) -> None:
        """
        Update policy after observing outcome of an arm pull.

        Args:
            arm: Action that was taken
            outcome: Result dict with keys: paid (bool), opted_out (bool), complained (bool), annoyance (float)
        """
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, T: np.ndarray, y: np.ndarray) -> 'BanditPolicy':
        """
        Warm-start policy from historical training data (e.g., CATE estimates).

        Args:
            X: Feature matrix (debtor features + action encodings)
            T: Treatment indicator (1 if contacted, 0 if not)
            y: Outcome (1 if paid, 0 if not)

        Returns:
            self
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset per-debtor state (e.g., clear outcome history)."""
        pass

    def prepare_cohort(self, cohort_df: pd.DataFrame) -> None:
        """
        Optional hook: called by BacktestEngine with the full backtest cohort
        before a run starts. Policies that need cohort-wide information (e.g.,
        ranking debtors for a fixed contact budget) override this; default is
        a no-op since most policies decide purely per-debtor.
        """
        pass

    def get_summary(self) -> dict:
        """Return summary of policy state for logging/inspection."""
        return {'policy': self.__class__.__name__}


class RandomPolicy(BanditPolicy):
    """
    Baseline: select arm uniformly at random.
    No learning; used for sanity check and comparison.
    """

    def __init__(self):
        self.rng = np.random.RandomState(seed=42)

    def select_arm(self, features: pd.Series, state: DebtorState, week: int) -> Action:
        """Uniform random arm selection."""
        return self.rng.choice(ALL_ARMS)

    def update(self, arm: Action, outcome: dict) -> None:
        """No-op: random policy doesn't learn."""
        pass

    def fit(self, X: pd.DataFrame, T: np.ndarray, y: np.ndarray) -> 'RandomPolicy':
        """No-op: random policy needs no warm-start."""
        return self

    def reset(self) -> None:
        """No state to reset."""
        pass


class ThompsonSamplingPolicy(BanditPolicy):
    """
    Hierarchical Thompson Sampling bandit for arm selection.

    Two decision levels, matching the action space in the project overview:
    1. Contact or not — a single 1-vs-1 Beta comparison per segment between an
       aggregate "contact" posterior and a "no-contact" posterior.
    2. Channel + timing — if contact wins, a second Thompson Sampling draw
       picks among the 12 channel/timing combinations, each with its own
       per-segment posterior.

    A flat 13-way argmax (all channel/timing arms plus no-contact competing
    directly) is NOT used deliberately: with 12 contact arms racing a single
    no-contact arm, the maximum of 12 independent samples beats a single
    sample far more than half the time even when the arms are equally good in
    expectation (an order-statistics artifact, not a real preference for
    contact) — empirically ~92% of the time for posteriors typical of this
    project. Separating the contact/no-contact decision into its own fair
    1-vs-1 comparison removes that bias; the 12-way race only decides *which*
    channel/timing to use once contact has already been chosen on its own
    merits.

    Updates posteriors with Bernoulli likelihood (reward = paid AND NOT
    opted_out/complained). Warm-start: initialize Beta posteriors from
    historical CATE estimates.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0, seed: int = 42,
                 response_model=None, uplift_model=None, warmstart_pseudo_obs: int = 20,
                 contact_budget_fraction: Optional[float] = None):
        """
        Args:
            alpha_prior: Beta prior shape parameter (alpha = successes + 1)
            beta_prior: Beta prior shape parameter (beta = failures + 1)
            seed: Random seed for posterior sampling
            response_model: Fitted ResponseModel for baseline P(pay)
            uplift_model: Fitted UpliftModel for CATE estimates per arm
            warmstart_pseudo_obs: Pseudo-observation count used to translate the
                CATE-based warm-start success rate into Beta(alpha, beta) shape
                parameters. Larger values anchor the policy more strongly to the
                offline model and make it slower to update from real backtest
                outcomes; smaller values let online learning matter sooner.
                Kept modest (relative to the ~1-2 real pulls per (segment, arm)
                cell a single backtest run typically produces) so the bandit is
                genuinely online-updating rather than effectively frozen.
            contact_budget_fraction: If set, caps the policy to only ever contact
                (across the whole horizon) the top `contact_budget_fraction` of
                the cohort — ranked by estimated best-arm CATE (uplift), highest
                first — mirroring ResponseOnlyPolicy's `contact_fraction` so the
                two policies can be compared at an equal contact budget. None
                (default) leaves the bandit uncapped, deciding contact/no-contact
                per debtor-week on its own.
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.rng = np.random.RandomState(seed=seed)
        self.response_model = response_model
        self.uplift_model = uplift_model
        self.warmstart_pseudo_obs = warmstart_pseudo_obs
        self.contact_budget_fraction = contact_budget_fraction

        # Debtor_ids eligible for contact under the budget cap; None = uncapped.
        # Set by prepare_cohort(), called automatically by BacktestEngine.
        self._eligible_debtor_ids = None

        self.segments = ['high_risk', 'medium_risk', 'low_risk', 'frequent_payer']

        # Level 2: Beta(alpha, beta) per (segment, arm) — contact arms only
        # (channel/timing selection, conditional on contact already chosen).
        self.posteriors = {
            (seg, arm_idx): {'alpha': alpha_prior, 'beta': beta_prior}
            for seg in self.segments
            for arm_idx, arm in ARM_IDX.items() if arm.contact
        }

        # Level 1: Beta(alpha, beta) per segment for the binary contact/no-contact
        # decision — a fair 1-vs-1 comparison, independent of how many channel
        # arms exist underneath "contact".
        self.contact_decision_posteriors = {
            seg: {
                'contact': {'alpha': alpha_prior, 'beta': beta_prior},
                'no_contact': {'alpha': alpha_prior, 'beta': beta_prior},
            }
            for seg in self.segments
        }

        # Warm-start posteriors (preserved across reset() calls in multi-seed backtest)
        self.warmstart_posteriors = None
        self.warmstart_contact_decision_posteriors = None

        # Track last selected segment for update() call
        self.last_segment = None

        # Outcome history (for logging/diagnostics)
        self.outcome_history = []

    def prepare_cohort(self, cohort_df: pd.DataFrame) -> None:
        """
        Under a contact budget, rank the cohort by estimated best-arm uplift
        (max CATE across the 12 channel/timing arms) and restrict contact
        eligibility to the top `contact_budget_fraction` — i.e., spend the
        limited budget on the debtors the uplift model expects to benefit
        from contact the most, a direct uplift-prioritization analogue of
        ResponseOnlyPolicy's P(pay)-ranked budget.

        No-op (uncapped) if `contact_budget_fraction` wasn't set or the models
        needed to score debtors aren't available.
        """
        if self.contact_budget_fraction is None or self.uplift_model is None \
                or self.response_model is None or len(cohort_df) == 0:
            self._eligible_debtor_ids = None
            return

        best_uplift = np.full(len(cohort_df), -np.inf)
        for ch in CHANNELS:
            for t in TIMINGS:
                cate = self.uplift_model.predict_cate(cohort_df, channel=ch, timing=t)
                best_uplift = np.maximum(best_uplift, cate)

        n_eligible = int(np.ceil(len(cohort_df) * self.contact_budget_fraction))
        order = np.argsort(-best_uplift, kind='stable')  # Highest uplift first
        top_idx = order[:n_eligible]
        self._eligible_debtor_ids = set(cohort_df.iloc[top_idx]['debtor_id'].astype(int).tolist())

    def select_arm(self, features: pd.Series, state: DebtorState, week: int) -> Action:
        """
        Hierarchical Thompson Sampling: decide contact/no-contact first (fair
        1-vs-1 draw), then, only if contacting, sample among channel/timing arms.

        If a contact budget is active and this debtor didn't make the cut,
        skip the bandit draw and return no-contact directly.

        Returns:
            Action chosen for this debtor's segment
        """
        segment = features.get('segment', 'frequent_payer')  # Default fallback
        self.last_segment = segment  # Store for update() call

        if self._eligible_debtor_ids is not None and int(features['debtor_id']) not in self._eligible_debtor_ids:
            return NO_CONTACT_ARM

        decision = self.contact_decision_posteriors.get(segment, {
            'contact': {'alpha': self.alpha_prior, 'beta': self.beta_prior},
            'no_contact': {'alpha': self.alpha_prior, 'beta': self.beta_prior},
        })
        contact_sample = self.rng.beta(decision['contact']['alpha'], decision['contact']['beta'])
        no_contact_sample = self.rng.beta(decision['no_contact']['alpha'], decision['no_contact']['beta'])

        if no_contact_sample >= contact_sample:
            return NO_CONTACT_ARM

        # Contact won the top-level draw: sample among the 12 channel/timing arms
        samples = {}
        for arm_idx, arm in ARM_IDX.items():
            if not arm.contact:
                continue
            key = (segment, arm_idx)
            if key in self.posteriors:
                alpha = self.posteriors[key]['alpha']
                beta = self.posteriors[key]['beta']
                samples[arm_idx] = self.rng.beta(alpha, beta)
            else:
                samples[arm_idx] = 0  # Unknown segment, fallback

        best_arm_idx = max(samples, key=samples.get)
        return ARM_IDX[best_arm_idx]

    def update(self, arm: Action, outcome: dict) -> None:
        """
        Update Beta posteriors with outcome, accounting for trust cost.

        Reward = payment AND not (opt-out or complained). Always updates the
        level-1 contact/no-contact posterior for this segment; if the final
        action contacted, also updates the level-2 channel/timing posterior.
        """
        if self.last_segment is None:
            return  # No segment recorded (select_arm wasn't called first)

        paid = int(outcome.get('paid', 0))
        opted_out = int(outcome.get('opted_out', 0))
        complained = int(outcome.get('complained', 0))

        # Reward: payment is good, opt-out/complaint is bad
        # Treat any negative outcome as failure (conservative learning)
        reward = paid and not (opted_out or complained)

        # Level 1: contact/no-contact decision posterior
        decision = self.contact_decision_posteriors.setdefault(self.last_segment, {
            'contact': {'alpha': self.alpha_prior, 'beta': self.beta_prior},
            'no_contact': {'alpha': self.alpha_prior, 'beta': self.beta_prior},
        })
        decision_key = 'contact' if arm.contact else 'no_contact'
        if reward:
            decision[decision_key]['alpha'] += 1
        else:
            decision[decision_key]['beta'] += 1

        # Level 2: channel/timing posterior (only meaningful if contacted)
        arm_idx = IDX_ARM.get(str(arm)) if arm.contact else None
        if arm_idx is not None:
            key = (self.last_segment, arm_idx)
            if key in self.posteriors:
                if reward:
                    self.posteriors[key]['alpha'] += 1
                else:
                    self.posteriors[key]['beta'] += 1

        self.outcome_history.append((self.last_segment, arm_idx, reward))

    def fit(self, X: pd.DataFrame, T: np.ndarray, y: np.ndarray, features_df: Optional[pd.DataFrame] = None) -> 'ThompsonSamplingPolicy':
        """
        Warm-start posteriors from historical data and model predictions.

        Initializes both decision levels per segment:
        - Level 1 (contact vs. no-contact): "contact" warm-starts from the
          average expected success across all 12 channel/timing arms (mirroring
          the historical logging policy's uniform-random channel/timing choice
          conditional on contact); "no-contact" warm-starts from the response
          model's baseline P(pay) directly (its CATE is 0 by definition — no
          artificial adjustment).
        - Level 2 (channel/timing): each arm warm-starts from baseline + its
          own CATE estimate, as before.

        Args:
            X: Feature matrix (debtor features)
            T: Treatment indicator
            y: Outcome (paid or not)
            features_df: Optional DataFrame with segment/balance/etc for CATE computation
        """
        overall_success_rate = y.mean()

        # If models available, use CATE-based warm-start per segment
        if self.response_model is not None and self.uplift_model is not None and features_df is not None:
            for segment in self.segments:
                seg_features = features_df[features_df['segment'] == segment]
                if len(seg_features) == 0:
                    continue  # Skip empty segments

                baseline_probs = self.response_model.predict_proba(seg_features)
                baseline_mean = np.clip(baseline_probs.mean(), 0.01, 0.99)

                # Level 2: expected success per channel/timing arm = baseline + CATE
                contact_expected_success = {}
                for arm_idx, arm in ARM_IDX.items():
                    if not arm.contact:
                        continue
                    cate_effects = self.uplift_model.predict_cate(seg_features, channel=arm.channel, timing=arm.timing)
                    expected_probs = np.clip(baseline_probs + cate_effects, 0.01, 0.99)
                    contact_expected_success[arm_idx] = expected_probs.mean()

                    alpha_init = self.alpha_prior + int(self.warmstart_pseudo_obs * contact_expected_success[arm_idx])
                    beta_init = self.beta_prior + int(self.warmstart_pseudo_obs * (1 - contact_expected_success[arm_idx]))
                    self.posteriors[(segment, arm_idx)] = {'alpha': alpha_init, 'beta': beta_init}

                # Level 1: "contact" = average over channel/timing arms (matches
                # the uniform-random channel/timing choice in the historical log);
                # "no_contact" = baseline P(pay), unadjusted.
                contact_mean = np.clip(
                    np.mean(list(contact_expected_success.values())) if contact_expected_success else baseline_mean,
                    0.01, 0.99
                )
                self.contact_decision_posteriors[segment] = {
                    'contact': {
                        'alpha': self.alpha_prior + int(self.warmstart_pseudo_obs * contact_mean),
                        'beta': self.beta_prior + int(self.warmstart_pseudo_obs * (1 - contact_mean)),
                    },
                    'no_contact': {
                        'alpha': self.alpha_prior + int(self.warmstart_pseudo_obs * baseline_mean),
                        'beta': self.beta_prior + int(self.warmstart_pseudo_obs * (1 - baseline_mean)),
                    },
                }
        else:
            # Fallback: simple warm-start based on overall success rate
            rate = np.clip(overall_success_rate, 0.01, 0.99)
            for segment in self.segments:
                for arm_idx, arm in ARM_IDX.items():
                    if not arm.contact:
                        continue
                    alpha_init = self.alpha_prior + int(self.warmstart_pseudo_obs * rate)
                    beta_init = self.beta_prior + int(self.warmstart_pseudo_obs * (1 - rate))
                    self.posteriors[(segment, arm_idx)] = {'alpha': alpha_init, 'beta': beta_init}

                self.contact_decision_posteriors[segment] = {
                    'contact': {
                        'alpha': self.alpha_prior + int(self.warmstart_pseudo_obs * rate),
                        'beta': self.beta_prior + int(self.warmstart_pseudo_obs * (1 - rate)),
                    },
                    'no_contact': {
                        'alpha': self.alpha_prior + int(self.warmstart_pseudo_obs * rate),
                        'beta': self.beta_prior + int(self.warmstart_pseudo_obs * (1 - rate)),
                    },
                }

        # Save warm-start posteriors (preserved across reset() in multi-seed backtest)
        self.warmstart_posteriors = {k: {'alpha': p['alpha'], 'beta': p['beta']}
                                     for k, p in self.posteriors.items()}
        self.warmstart_contact_decision_posteriors = {
            seg: {dk: {'alpha': p['alpha'], 'beta': p['beta']} for dk, p in d.items()}
            for seg, d in self.contact_decision_posteriors.items()
        }

        return self

    def reset(self) -> None:
        """
        Reset posteriors to warm-start (or priors if not fitted).

        Preserves the CATE-informed warm-start across multiple backtest runs,
        allowing each seed to benefit from the same model-based initialization.
        """
        if self.warmstart_posteriors is not None:
            self.posteriors = {k: {'alpha': p['alpha'], 'beta': p['beta']}
                             for k, p in self.warmstart_posteriors.items()}
        else:
            self.posteriors = {
                (seg, arm_idx): {'alpha': self.alpha_prior, 'beta': self.beta_prior}
                for seg in self.segments
                for arm_idx, arm in ARM_IDX.items() if arm.contact
            }

        if self.warmstart_contact_decision_posteriors is not None:
            self.contact_decision_posteriors = {
                seg: {dk: {'alpha': p['alpha'], 'beta': p['beta']} for dk, p in d.items()}
                for seg, d in self.warmstart_contact_decision_posteriors.items()
            }
        else:
            self.contact_decision_posteriors = {
                seg: {
                    'contact': {'alpha': self.alpha_prior, 'beta': self.beta_prior},
                    'no_contact': {'alpha': self.alpha_prior, 'beta': self.beta_prior},
                }
                for seg in self.segments
            }

        self.last_segment = None
        self.outcome_history = []

    def get_summary(self) -> dict:
        """Return posterior means (sample summary)."""
        # Show summary for first segment only (contextual)
        means = {}
        sample_seg = self.segments[0]
        for arm_idx, arm in ARM_IDX.items():
            if not arm.contact:
                continue
            key = (sample_seg, arm_idx)
            if key in self.posteriors:
                alpha = self.posteriors[key]['alpha']
                beta = self.posteriors[key]['beta']
                mean = alpha / (alpha + beta)
                means[f'arm_{arm_idx}'] = mean

        decision = self.contact_decision_posteriors.get(sample_seg, {})
        decision_means = {
            dk: p['alpha'] / (p['alpha'] + p['beta'])
            for dk, p in decision.items()
        }

        return {
            'policy': 'ThompsonSampling (hierarchical, contextual)',
            'sample_segment': sample_seg,
            'contact_decision_means': decision_means,
            'arm_means': means,
            'num_updates': len(self.outcome_history),
        }


class EpsilonGreedyPolicy(BanditPolicy):
    """
    ε-Greedy bandit: exploit best arm with probability 1-ε, explore random with probability ε.

    Simpler than Thompson; no posteriors, just empirical means.
    """

    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        """
        Args:
            epsilon: Exploration probability
            seed: Random seed
        """
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed=seed)

        # Empirical means and counts for each arm
        self.arm_means = {i: 0.0 for i in range(len(ALL_ARMS))}
        self.arm_counts = {i: 0 for i in range(len(ALL_ARMS))}

    def select_arm(self, features: pd.Series, state: DebtorState, week: int) -> Action:
        """
        ε-Greedy: exploit with prob 1-ε, explore with prob ε.
        """
        if self.rng.uniform() < self.epsilon:
            # Explore: random arm
            arm_idx = self.rng.choice(list(self.arm_means.keys()))
        else:
            # Exploit: best arm
            arm_idx = max(self.arm_means, key=self.arm_means.get)

        return ARM_IDX[arm_idx]

    def update(self, arm: Action, outcome: dict) -> None:
        """
        Update empirical mean and count.
        Running average: mean = (old_mean * count + reward) / (count + 1)
        """
        arm_idx = IDX_ARM.get(str(arm))
        if arm_idx is None:
            return

        paid = int(outcome.get('paid', 0))
        count = self.arm_counts[arm_idx]
        old_mean = self.arm_means[arm_idx]

        self.arm_counts[arm_idx] += 1
        self.arm_means[arm_idx] = (old_mean * count + paid) / (count + 1)

    def fit(self, X: pd.DataFrame, T: np.ndarray, y: np.ndarray) -> 'EpsilonGreedyPolicy':
        """Warm-start empirical means from historical data."""
        overall_success_rate = y.mean()
        for arm_idx in range(len(ALL_ARMS)):
            self.arm_means[arm_idx] = overall_success_rate
            self.arm_counts[arm_idx] = 100  # Assume 100 historical pulls (scaled)

        return self

    def reset(self) -> None:
        """Reset counts and means."""
        self.arm_means = {i: 0.0 for i in range(len(ALL_ARMS))}
        self.arm_counts = {i: 0 for i in range(len(ALL_ARMS))}

    def get_summary(self) -> dict:
        return {
            'policy': 'EpsilonGreedy',
            'epsilon': self.epsilon,
            'arm_means': self.arm_means.copy(),
            'arm_counts': self.arm_counts.copy(),
        }


class ResponseOnlyPolicy(BanditPolicy):
    """
    Baseline: Response model only (deliberate strawman).

    Ranks the cohort by predicted P(pay) and contacts the top `contact_fraction`
    of debtors — mirroring the real-world instinct to prioritize contacting
    whoever looks likeliest to pay. Since a response model has no action
    input, everyone contacted gets the same fixed default action (voice/morning).

    This deliberately over-contacts "sure things" (high P(pay) regardless of
    action) and cannot answer the channel/timing question at all — the gap
    the uplift model fills. No learning; represents current state-of-practice.
    """

    def __init__(self, response_model=None, contact_fraction: float = 0.5, seed: int = 42):
        """
        Args:
            response_model: Fitted ResponseModel for P(pay|features)
            contact_fraction: Fraction of the cohort (ranked by P(pay), highest
                first) to contact. E.g. 0.5 contacts the top half.
            seed: Random seed
        """
        self.response_model = response_model
        self.contact_fraction = contact_fraction
        self.rng = np.random.RandomState(seed=seed)

        # Default action: voice/morning
        self.default_action = Action(contact=True, channel='voice', timing='morning')
        self.no_contact_action = Action(contact=False)

        # Set of debtor_ids to contact, computed by prepare_cohort() from the
        # full backtest cohort's P(pay) ranking.
        self._contact_debtor_ids = None

    def prepare_cohort(self, cohort_df: pd.DataFrame) -> None:
        """
        Rank the cohort by predicted P(pay) (descending) and mark the top
        `contact_fraction` for contact. Must be called before select_arm()
        can use the ranking; BacktestEngine calls this automatically.
        """
        if self.response_model is None or len(cohort_df) == 0:
            self._contact_debtor_ids = None
            return

        probs = self.response_model.predict_proba(cohort_df)
        n_contact = int(np.ceil(len(cohort_df) * self.contact_fraction))
        # Highest P(pay) first; stable sort keeps ties in original cohort order
        order = np.argsort(-probs, kind='stable')
        top_idx = order[:n_contact]
        self._contact_debtor_ids = set(cohort_df.iloc[top_idx]['debtor_id'].astype(int).tolist())

    def select_arm(self, features: pd.Series, state: DebtorState, week: int) -> Action:
        """
        Contact iff this debtor is in the top-`contact_fraction` by P(pay).
        """
        if self.response_model is None:
            # No model fitted; default to random for safety
            return self.rng.choice(ALL_ARMS)

        if self._contact_debtor_ids is not None:
            debtor_id = int(features['debtor_id'])
            contact = debtor_id in self._contact_debtor_ids
        else:
            # prepare_cohort() wasn't called (e.g., single-debtor use outside
            # a backtest run); approximate top-N ranking with a fixed cutoff.
            p_pay = self.response_model.predict_proba(pd.DataFrame([features]))[0]
            contact = p_pay > 0.5

        return self.default_action if contact else self.no_contact_action

    def update(self, arm: Action, outcome: dict) -> None:
        """No-op: response-only policy doesn't learn."""
        pass

    def fit(self, X: pd.DataFrame, T: np.ndarray, y: np.ndarray) -> 'ResponseOnlyPolicy':
        """No-op: response model already fitted externally."""
        return self

    def reset(self) -> None:
        """No per-run state to reset; the cohort ranking is not seed-dependent."""
        pass

    def get_summary(self) -> dict:
        return {
            'policy': 'ResponseOnly',
            'contact_fraction': self.contact_fraction,
            'default_action': str(self.default_action),
        }
