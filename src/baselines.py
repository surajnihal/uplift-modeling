"""
Stronger baselines for the contact-strategy backtest.

`ResponseOnlyPolicy` is a deliberate strawman (it ranks by P(pay) and has no
channel/timing logic), so beating it says little about whether the Thompson
Sampling bandit in `src/bandit.py` actually adds value. The two policies here
are the honest comparison points:

- `UpliftArgmaxPolicy`: static. Uses the *same* fitted UpliftModel as the bandit
  and the *same* uplift-ranked contact budget, but picks each debtor's
  (channel, timing) by plain argmax of predicted CATE. No Thompson sampling, no
  online learning. If the bandit can't beat this, the online-learning layer
  isn't earning its complexity.
- `SegmentBestActionPolicy`: no model at all. Uses only the randomized
  historical log: per segment, the empirical best (channel, timing) vs. no
  contact. Contacts the segments with the largest empirical lift first.

Both implement the `BanditPolicy` interface and the same
`contact_budget_fraction` semantics as `ThompsonSamplingPolicy` /
`ResponseOnlyPolicy`: at most ceil(fraction * cohort_size) distinct debtors are
ever eligible for contact. Guardrails are applied by `BacktestEngine` exactly as
for every other policy (these classes never bypass them).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.bandit import BanditPolicy, CHANNELS, TIMINGS, NO_CONTACT_ARM
from src.dgp import Action, DebtorState


def _top_budget_ids(cohort_df: pd.DataFrame, score: np.ndarray, fraction: float) -> np.ndarray:
    """Row positions of the top ceil(fraction * n) rows by score (stable, descending)."""
    n_eligible = int(np.ceil(len(cohort_df) * fraction))
    order = np.argsort(-score, kind='stable')
    return order[:n_eligible]


class UpliftArgmaxPolicy(BanditPolicy):
    """
    Static uplift-argmax baseline (no bandit, no online learning).

    prepare_cohort(): for every debtor, score all 12 (channel, timing) arms with
    `uplift_model.predict_cate`, take the argmax arm and its CATE, rank debtors
    by that best-arm CATE (highest first) and make the top
    `contact_budget_fraction` eligible — the same ranking rule the bandit uses to
    spend its budget. select_arm() then returns each eligible debtor's argmax arm
    every week (guardrails decide when a contact is actually allowed) and
    no-contact for everyone else.
    """

    def __init__(self, uplift_model=None, contact_budget_fraction: float = 0.5,
                 min_uplift: Optional[float] = None):
        """
        Args:
            uplift_model: Fitted UpliftModel (provides predict_cate per channel/timing).
            contact_budget_fraction: Fraction of the cohort eligible for contact,
                ranked by best-arm estimated uplift.
            min_uplift: Optional floor; eligible debtors whose best-arm CATE is
                below it are not contacted. None (default) = pure top-fraction,
                matching the bandit's budget semantics.
        """
        self.uplift_model = uplift_model
        self.contact_budget_fraction = contact_budget_fraction
        self.min_uplift = min_uplift
        self._plan: Optional[Dict[int, Action]] = None  # debtor_id -> argmax action

    def _score_arms(self, df: pd.DataFrame):
        """Return (best_cate, best_arm_per_row) over the 12 channel/timing arms."""
        if self.uplift_model is None:
            raise ValueError("UpliftArgmaxPolicy requires a fitted uplift_model.")
        arms = [(ch, t) for ch in CHANNELS for t in TIMINGS]
        cate = np.column_stack([
            self.uplift_model.predict_cate(df, channel=ch, timing=t) for ch, t in arms
        ])
        best_idx = cate.argmax(axis=1)
        best_cate = cate[np.arange(len(df)), best_idx]
        best_arms = [Action(contact=True, channel=arms[i][0], timing=arms[i][1]) for i in best_idx]
        return best_cate, best_arms

    def prepare_cohort(self, cohort_df: pd.DataFrame) -> None:
        if len(cohort_df) == 0:
            self._plan = {}
            return
        best_cate, best_arms = self._score_arms(cohort_df)
        top_idx = _top_budget_ids(cohort_df, best_cate, self.contact_budget_fraction)
        ids = cohort_df['debtor_id'].astype(int).values
        self._plan = {
            int(ids[i]): best_arms[i] for i in top_idx
            if self.min_uplift is None or best_cate[i] >= self.min_uplift
        }

    def select_arm(self, features: pd.Series, state: DebtorState, week: int) -> Action:
        if self._plan is None:
            # prepare_cohort() not called (single-debtor use): no budget to apply,
            # so contact only if the best arm has positive estimated uplift.
            best_cate, best_arms = self._score_arms(pd.DataFrame([features]))
            floor = 0.0 if self.min_uplift is None else self.min_uplift
            return best_arms[0] if best_cate[0] > floor else NO_CONTACT_ARM
        return self._plan.get(int(features['debtor_id']), NO_CONTACT_ARM)

    def update(self, arm: Action, outcome: dict) -> None:
        """No-op: static policy."""
        pass

    def fit(self, X=None, T=None, y=None) -> 'UpliftArgmaxPolicy':
        """No-op: the uplift model is fitted externally."""
        return self

    def reset(self) -> None:
        """No per-run state (the cohort plan is not seed-dependent)."""
        pass

    def get_summary(self) -> dict:
        return {
            'policy': 'UpliftArgmax',
            'contact_budget_fraction': self.contact_budget_fraction,
            'min_uplift': self.min_uplift,
        }


class SegmentBestActionPolicy(BanditPolicy):
    """
    Model-free baseline: per-segment empirical best action from the randomized log.

    fit(log_df): for each segment, compute the observed payment rate under each
    of the 12 (channel, timing) arms and under no contact (rows of the historical
    log are debtor-weeks). A segment's best action is the arm with the highest
    rate; its lift is (best arm rate - no-contact rate). No smoothing, no
    covariates — deliberately simple.

    prepare_cohort(): rank debtors by their segment's lift (highest first; ties
    within a segment keep cohort order) and make the top
    `contact_budget_fraction` eligible, each assigned their segment's best arm.
    """

    def __init__(self, contact_budget_fraction: float = 0.5, min_lift: Optional[float] = None):
        """
        Args:
            contact_budget_fraction: Fraction of the cohort eligible for contact.
            min_lift: Optional floor; debtors in segments whose best-arm lift is
                below it are not contacted. None (default) = pure top-fraction.
        """
        self.contact_budget_fraction = contact_budget_fraction
        self.min_lift = min_lift
        self.segment_best_action: Dict[str, Action] = {}
        self.segment_lift: Dict[str, float] = {}
        self._plan: Optional[Dict[int, Action]] = None

    def fit(self, X: pd.DataFrame, T=None, y=None) -> 'SegmentBestActionPolicy':
        """
        Args:
            X: Historical log DataFrame with columns `segment`, `action_contact`,
               `action_channel`, `action_timing`, `outcome_paid`
               (as returned by DataSimulator.simulate_cohort). T and y are
               accepted for interface compatibility and ignored.
        """
        log = X
        self.segment_best_action, self.segment_lift = {}, {}
        for seg, g in log.groupby('segment'):
            no_contact = g[g['action_contact'] == 0]['outcome_paid']
            base = float(no_contact.mean()) if len(no_contact) else 0.0
            contacted = g[g['action_contact'] == 1]
            rates = contacted.groupby(['action_channel', 'action_timing'])['outcome_paid'].mean()
            if rates.empty:
                continue
            (ch, t), best_rate = rates.idxmax(), float(rates.max())
            self.segment_best_action[seg] = Action(contact=True, channel=ch, timing=t)
            self.segment_lift[seg] = best_rate - base
        return self

    def prepare_cohort(self, cohort_df: pd.DataFrame) -> None:
        if len(cohort_df) == 0:
            self._plan = {}
            return
        # Segments never seen in the log get -inf lift: ranked last, never contacted.
        score = cohort_df['segment'].map(self.segment_lift).fillna(-np.inf).values.astype(float)
        top_idx = _top_budget_ids(cohort_df, score, self.contact_budget_fraction)
        ids = cohort_df['debtor_id'].astype(int).values
        segs = cohort_df['segment'].values
        self._plan = {
            int(ids[i]): self.segment_best_action[segs[i]] for i in top_idx
            if segs[i] in self.segment_best_action
            and (self.min_lift is None or score[i] >= self.min_lift)
        }

    def select_arm(self, features: pd.Series, state: DebtorState, week: int) -> Action:
        if self._plan is None:
            # prepare_cohort() not called: no budget to apply; contact iff the
            # debtor's segment has a positive empirical lift.
            seg = features.get('segment')
            floor = 0.0 if self.min_lift is None else self.min_lift
            if seg in self.segment_best_action and self.segment_lift[seg] > floor:
                return self.segment_best_action[seg]
            return NO_CONTACT_ARM
        return self._plan.get(int(features['debtor_id']), NO_CONTACT_ARM)

    def update(self, arm: Action, outcome: dict) -> None:
        """No-op: static policy."""
        pass

    def reset(self) -> None:
        """No per-run state."""
        pass

    def get_summary(self) -> dict:
        return {
            'policy': 'SegmentBestAction',
            'contact_budget_fraction': self.contact_budget_fraction,
            'segment_best_action': {s: str(a) for s, a in self.segment_best_action.items()},
            'segment_lift': dict(self.segment_lift),
        }
