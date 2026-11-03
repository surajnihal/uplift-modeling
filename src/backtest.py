"""
Backtest Engine: Run contact policies on held-out cohort.

Simulates multi-period interaction: policy selects actions, guardrails enforce
constraints, DGP generates outcomes, bandit updates. Collects metrics and
compares policies (random vs. response-only vs. uplift-bandit).
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from src.dgp import DataSimulator, DebtorState, Action
from src.bandit import BanditPolicy
from src.guardrails import Guardrails


@dataclass
class BacktestResults:
    """Results from one backtest run of a single policy."""

    policy_name: str
    total_recovery: float  # Total recovered across cohort (currency units)
    total_contacts: int  # Total contacts used
    total_debtors: int  # Cohort size
    payment_rate: float  # Share of debtors who paid
    opt_out_rate: float  # Share who opted out
    complaint_rate: float  # Share who complained
    avg_recovery_per_contact: float  # Efficiency: units recovered per contact
    avg_contacts_per_debtor: float  # Average contacts used per debtor

    # Per-debtor breakdown
    per_debtor_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Columns: debtor_id, paid, recovery, contacts_used, opted_out, complained, final_annoyance

    # Per-action tracking (for diagnostics)
    policy_decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Columns: debtor_id, week, policy_recommendation, guardrail_block_reason, final_action, outcome_paid

    seed: int = 42

    def __repr__(self) -> str:
        """Pretty-print results summary."""
        lines = [
            f"\n{'='*60}",
            f"Policy: {self.policy_name}",
            f"{'='*60}",
            f"Cohort Size: {self.total_debtors}",
            f"Total Contacts: {self.total_contacts}",
            f"Total Recovery: {self.total_recovery:,.0f}",
            f"Payment Rate: {self.payment_rate:.1%}",
            f"Opt-Out Rate: {self.opt_out_rate:.1%}",
            f"Complaint Rate: {self.complaint_rate:.1%}",
            f"Avg Recovery/Contact: {self.avg_recovery_per_contact:,.0f}",
            f"Avg Contacts/Debtor: {self.avg_contacts_per_debtor:.2f}",
            f"{'='*60}",
        ]
        return '\n'.join(lines)


class BacktestEngine:
    """
    Run a bandit policy on backtest cohort with guardrails and outcome tracking.
    """

    def __init__(self, dgp: DataSimulator, policy: BanditPolicy, guardrails: Guardrails,
                 cohort_df: pd.DataFrame, horizon: int = 8, verbose: bool = False):
        """
        Args:
            dgp: Data-generating process (for outcome simulation)
            policy: Bandit policy (to select arms)
            guardrails: Constraint enforcement
            cohort_df: Backtest cohort (features only; one row per debtor)
            horizon: Weeks per debtor
            verbose: Print progress
        """
        self.dgp = dgp
        self.policy = policy
        self.guardrails = guardrails
        self.cohort_df = cohort_df.reset_index(drop=True)
        self.horizon = horizon
        self.verbose = verbose

    def run(self, seed: Optional[int] = None) -> BacktestResults:
        """
        Run backtest: for each debtor, simulate policy over horizon.

        Returns:
            BacktestResults with metrics and per-debtor tracking
        """
        if seed is not None:
            dgp = DataSimulator(seed=seed)
        else:
            dgp = self.dgp

        # Let cohort-aware policies (e.g., ResponseOnlyPolicy's top-N ranking)
        # see the full backtest cohort before deciding per-debtor actions.
        self.policy.prepare_cohort(self.cohort_df)

        policy_name = self.policy.get_summary().get('policy', 'Unknown')

        # Pre-compute balance 75th percentile for annoyance calculations
        balance_p75 = self.cohort_df['balance'].quantile(0.75)

        # Tracking
        per_debtor_results = []
        policy_decisions = []

        total_recovery = 0.0
        total_contacts = 0
        debtors_paid = 0
        debtors_opted_out = 0
        debtors_complained = 0

        # For each debtor
        for debtor_idx, (_, features) in enumerate(self.cohort_df.iterrows()):
            debtor_id = int(features['debtor_id'])

            # Initialize debtor state
            state = DebtorState(
                debtor_id=debtor_id,
                balance=features['balance'],
                days_overdue=features['days_overdue'],
                prior_payments=features['prior_payments'],
                contactability=features['contactability'],
                annoyance=0.0,
                opted_out=False,
                complained=False,
                paid=False,
            )

            contact_history = []  # List of (week, action) tuples
            recovery_this_debtor = 0.0

            # Simulate over horizon
            for week in range(self.horizon):
                # Policy selects arm
                action_policy = self.policy.select_arm(features, state, week)

                # Guardrails apply constraints
                action_final, block_reason = self.guardrails.apply(
                    action_policy, state, week, contact_history
                )

                # Step forward (outcome simulation)
                state, outcome = dgp.dgp.step_forward(features, state, action_final, balance_p75=balance_p75)

                # Update policy (online learning)
                self.policy.update(action_final, outcome)

                # Track
                contact_history.append((week, action_final))
                if outcome['paid']:
                    # Simplification: "paid" is a single terminal event (DebtorDGP
                    # stops simulating a debtor once paid) always worth a flat 5%
                    # of their original balance — not partial/incremental repayment,
                    # and the same 5% regardless of channel or how long it took.
                    # Recovery is therefore ~payment_rate weighted by balance, not
                    # an independently-varying signal. Stated simplification, not
                    # a realistic recovery model.
                    recovery_this_debtor += features['balance'] * 0.05
                    total_recovery += recovery_this_debtor

                # Record decision
                policy_decisions.append({
                    'debtor_id': debtor_id,
                    'week': week,
                    'policy_recommendation': str(action_policy),
                    'guardrail_block_reason': block_reason,
                    'final_action': str(action_final),
                    'outcome_paid': int(outcome['paid']),
                    'outcome_opted_out': int(outcome['opted_out']),
                    'outcome_complained': int(outcome['complained']),
                    'annoyance': outcome['annoyance'],
                })

                # Stop if terminal state
                if state.paid or state.opted_out or state.complained:
                    break

            # Record per-debtor result
            per_debtor_results.append({
                'debtor_id': debtor_id,
                'paid': int(state.paid),
                'recovery': recovery_this_debtor,
                'contacts_used': sum(1 for w, act in contact_history if act.contact),
                'opted_out': int(state.opted_out),
                'complained': int(state.complained),
                'final_annoyance': state.annoyance,
            })

            total_contacts += sum(1 for w, act in contact_history if act.contact)
            if state.paid:
                debtors_paid += 1
            if state.opted_out:
                debtors_opted_out += 1
            if state.complained:
                debtors_complained += 1

            if self.verbose and (debtor_idx + 1) % 100 == 0:
                print(f"  [{policy_name}] Processed {debtor_idx + 1}/{len(self.cohort_df)} debtors")

        # Compute aggregate metrics
        n_debtors = len(self.cohort_df)
        payment_rate = debtors_paid / n_debtors if n_debtors > 0 else 0
        opt_out_rate = debtors_opted_out / n_debtors if n_debtors > 0 else 0
        complaint_rate = debtors_complained / n_debtors if n_debtors > 0 else 0
        avg_recovery_per_contact = (total_recovery / total_contacts) if total_contacts > 0 else 0
        avg_contacts_per_debtor = total_contacts / n_debtors if n_debtors > 0 else 0

        # Create results object
        results = BacktestResults(
            policy_name=policy_name,
            total_recovery=total_recovery,
            total_contacts=total_contacts,
            total_debtors=n_debtors,
            payment_rate=payment_rate,
            opt_out_rate=opt_out_rate,
            complaint_rate=complaint_rate,
            avg_recovery_per_contact=avg_recovery_per_contact,
            avg_contacts_per_debtor=avg_contacts_per_debtor,
            per_debtor_results=pd.DataFrame(per_debtor_results),
            policy_decisions=pd.DataFrame(policy_decisions),
            seed=seed or self.dgp.seed,
        )

        return results

    def run_multiple(self, n_seeds: int = 5, seed_base: int = 42) -> List[BacktestResults]:
        """
        Run backtest multiple times with different seeds for CI estimation.

        Args:
            n_seeds: Number of runs
            seed_base: Base seed for run i = seed_base + i

        Returns:
            List of BacktestResults, one per seed
        """
        results_list = []
        for i in range(n_seeds):
            seed = seed_base + i
            self.policy.reset()  # Reset policy state between runs
            results = self.run(seed=seed)
            results_list.append(results)
            if self.verbose:
                print(f"Completed run {i+1}/{n_seeds}")

        return results_list


def mean_ci(values, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Mean and 95%-CI half-width across independent seed runs.

    Uses the t-distribution (not a fixed 1.96) so the half-width is exact at
    any sample size, including the small n_seeds runs used for quick checks.
    Half-width shrinks with sqrt(n_seeds) — this is a confidence interval on
    the *mean*, not the raw spread across seeds (which doesn't shrink as
    n_seeds grows; it estimates population variability instead).

    Returns:
        (mean, half_width). half_width is 0.0 for n < 2 (can't estimate spread).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n > 0 else float('nan')
    if n < 2:
        return mean, 0.0
    sem = values.std(ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, float(t_crit * sem)


def compare_policies(results_dict: Dict[str, List[BacktestResults]]) -> pd.DataFrame:
    """
    Compare multiple policies across multiple runs.

    Args:
        results_dict: Dict of {policy_name: [BacktestResults, ...]}
                      (multiple runs per policy for CI)

    Returns:
        Comparison DataFrame with metrics and 95% CI on the mean (half-width
        via mean_ci — shrinks as more seeds are run, unlike raw ± std).
    """
    rows = []
    for policy_name, results_list in results_dict.items():
        recoveries = [r.total_recovery for r in results_list]
        contacts = [r.total_contacts for r in results_list]
        payments = [r.payment_rate for r in results_list]
        opt_outs = [r.opt_out_rate for r in results_list]
        complaints = [r.complaint_rate for r in results_list]

        recovery_mean, recovery_ci = mean_ci(recoveries)
        contacts_mean, contacts_ci = mean_ci(contacts)
        payment_mean, payment_ci = mean_ci(payments)
        opt_out_mean, opt_out_ci = mean_ci(opt_outs)
        complaint_mean, complaint_ci = mean_ci(complaints)

        row = {
            'Policy': policy_name,
            'Seeds': len(results_list),
            'Recovery (units)': f"{recovery_mean:,.0f} ± {recovery_ci:,.0f}",
            'Contacts': f"{contacts_mean:.1f} ± {contacts_ci:.1f}",
            'Payment Rate': f"{payment_mean:.1%} ± {payment_ci:.1%}",
            'Opt-Out Rate': f"{opt_out_mean:.1%} ± {opt_out_ci:.1%}",
            'Complaint Rate': f"{complaint_mean:.1%} ± {complaint_ci:.1%}",
            'Recovery/Contact': f"{np.mean([r.avg_recovery_per_contact for r in results_list]):,.0f}",
        }
        rows.append(row)

    return pd.DataFrame(rows)
