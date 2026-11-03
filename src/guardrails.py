"""
Guardrails: Constraint enforcement layer for contact strategy.

Applied after policy recommendation to enforce business rules:
- Frequency caps (max contacts per week, total contacts per horizon)
- Opt-out hard stop (no contacts after opt-out)
- Rest periods (min weeks between contacts)
- Quiet hours (allowed timings per segment)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from src.dgp import Action, DebtorState


@dataclass
class Guardrails:
    """
    Constraint rules applied to policy recommendations.

    All thresholds default to "permissive" (rarely block) to prioritize
    exploration in backtest; can be tightened for production.
    """

    # Frequency constraints
    max_contacts_per_week: int = 1  # Max 1 contact per week
    max_total_contacts: int = 10    # Max 10 total across horizon
    rest_period_weeks: int = 1      # Min 1 week between contacts

    # Opt-out / Complaint handling
    hard_stop_on_optout: bool = True  # No contacts after opt-out

    # Timing constraints (per segment; can customize)
    quiet_hours: dict = None  # E.g., {"voice": False, "sms": True} means SMS allowed in evening, voice not

    def __post_init__(self):
        """Initialize quiet_hours if not provided."""
        if self.quiet_hours is None:
            self.quiet_hours = {
                'morning': True,
                'afternoon': True,
                'evening': True,
            }

    def apply(self, action: Action, state: DebtorState, week: int,
              contact_history: List[Tuple[int, Action]]) -> Tuple[Action, Optional[str]]:
        """
        Apply guardrails to policy recommendation.

        Args:
            action: Action recommended by policy
            state: Current debtor state
            week: Current week
            contact_history: List of (week, action) tuples from previous weeks

        Returns:
            (action_modified, reason)
            - action_modified: Either the original action or a no-contact override
            - reason: None if approved, or string explaining why blocked/modified
        """

        # Check 1: Hard stop on opt-out
        if self.hard_stop_on_optout and state.opted_out:
            return Action(contact=False), "opted_out"

        # If action is no-contact, nothing more to check
        if not action.contact:
            return action, None

        # Check 2: Frequency cap (max per week)
        contacts_this_week = sum(1 for w, act in contact_history if w == week and act.contact)
        if contacts_this_week >= self.max_contacts_per_week:
            return Action(contact=False), "max_contacts_per_week_exceeded"

        # Check 3: Rest period (min weeks between contacts)
        if self.rest_period_weeks > 0 and contact_history:
            last_contact_week = max(w for w, act in contact_history if act.contact) if any(act.contact for _, act in contact_history) else -np.inf
            if week - last_contact_week <= self.rest_period_weeks:
                return Action(contact=False), "rest_period_not_met"

        # Check 4: Total contact budget
        total_contacts = sum(1 for _, act in contact_history if act.contact)
        if total_contacts >= self.max_total_contacts:
            return Action(contact=False), "max_total_contacts_exceeded"

        # Check 5: Quiet hours (timing allowed?)
        # quiet_hours is dict where True = allowed, False = blocked
        if action.timing and not self.quiet_hours.get(action.timing, True):
            # Timing is blocked (explicitly set to False); block
            return Action(contact=False), f"timing_{action.timing}_blocked"

        # All checks passed
        return action, None

    def apply_batch(self, actions: List[Action], states: List[DebtorState],
                   weeks: List[int], contact_histories: List[List[Tuple[int, Action]]]) \
            -> Tuple[List[Action], List[Optional[str]]]:
        """
        Apply guardrails to multiple recommendations (vectorized).

        Args:
            actions: List of actions from policy
            states: List of debtor states
            weeks: List of weeks
            contact_histories: List of contact histories (one per debtor)

        Returns:
            (actions_modified, reasons)
        """
        actions_out = []
        reasons_out = []

        for action, state, week, history in zip(actions, states, weeks, contact_histories):
            act_mod, reason = self.apply(action, state, week, history)
            actions_out.append(act_mod)
            reasons_out.append(reason)

        return actions_out, reasons_out

    def get_summary(self) -> dict:
        """Return guardrail configuration for logging."""
        return {
            'max_contacts_per_week': self.max_contacts_per_week,
            'max_total_contacts': self.max_total_contacts,
            'rest_period_weeks': self.rest_period_weeks,
            'hard_stop_on_optout': self.hard_stop_on_optout,
            'quiet_hours': self.quiet_hours,
        }
