"""
Response Model vs. Uplift Model for Debt Collection.

Response Model: P(pay | debtor features) — predicts who is likely to pay.
Uplift Model: CATE (Conditional Average Treatment Effect) — estimates how much
  an action changes payment probability for a given debtor profile.

Key insight: Response model ranks debtors by baseline willingness but cannot answer
"which action works best for this debtor." Uplift model answers both questions.

Uplift model uses T-learner: fit separate models on treated (contacted) vs.
control (not contacted) groups, then CATE = E[outcome | contacted] - E[outcome | control].
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# Default base model for T-learner: GradientBoosting for better capacity
DEFAULT_BASE_MODEL = GradientBoostingClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
)


class ResponseModel:
    """
    Response model: P(pay | features).

    Predicts the probability a debtor pays, given only their profile (no action).
    This is what a naive approach would optimize: contact debtors with highest P(pay).

    Problem: Ignores heterogeneous treatment effects. May over-contact debtors who'd
    pay anyway (high P(pay) regardless of action) while missing opportunities to
    move fence-sitters with the right action.
    """

    def __init__(self, model_type: str = 'lr'):
        """
        Args:
            model_type: 'lr' (logistic regression) or 'gb' (gradient boosting)
        """
        self.model_type = model_type
        self.model = None
        self.feature_cols = None
        self.le_segment = LabelEncoder()

    def fit(self, df: pd.DataFrame) -> 'ResponseModel':
        """
        Fit response model on historical cohort data.

        Args:
            df: DataFrame with columns from DGP (balance, days_overdue, prior_payments,
                contactability, segment, outcome_paid)
        """
        # Feature engineering
        X = df[['balance', 'days_overdue', 'prior_payments', 'contactability']].copy()
        X['segment_encoded'] = self.le_segment.fit_transform(df['segment'])

        y = df['outcome_paid'].values

        # Fit model
        if self.model_type == 'lr':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'gb':
            self.model = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=3)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X, y)
        self.feature_cols = X.columns.tolist()

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict P(pay) for new debtors.

        Args:
            df: DataFrame with required features

        Returns:
            1D array of probabilities P(pay)
        """
        X = df[['balance', 'days_overdue', 'prior_payments', 'contactability']].copy()
        # Bug fix #6: Handle unseen categories per-row (not batch-wide)
        segments = df['segment'].values
        known_mask = np.isin(segments, self.le_segment.classes_)

        X['segment_encoded'] = 0  # Default for unseen
        if known_mask.any():
            known_indices = np.where(known_mask)[0]
            X.iloc[known_indices, X.columns.get_loc('segment_encoded')] = self.le_segment.transform(segments[known_mask])

        # Warn if any unseen segments
        if not known_mask.all():
            unseen = np.unique(segments[~known_mask])
            warnings.warn(f"Unseen segments {unseen} in predict_proba; defaulting to category 0")

        return self.model.predict_proba(X)[:, 1]

    def score(self, df: pd.DataFrame) -> float:
        """
        Evaluate model on held-out data (e.g., backtest cohort).

        Returns:
            Accuracy score
        """
        X = df[['balance', 'days_overdue', 'prior_payments', 'contactability']].copy()
        # Bug fix #6: Handle unseen categories per-row
        segments = df['segment'].values
        known_mask = np.isin(segments, self.le_segment.classes_)

        X['segment_encoded'] = 0  # Default for unseen
        if known_mask.any():
            known_indices = np.where(known_mask)[0]
            X.iloc[known_indices, X.columns.get_loc('segment_encoded')] = self.le_segment.transform(segments[known_mask])

        # Warn if any unseen segments
        if not known_mask.all():
            unseen = np.unique(segments[~known_mask])
            warnings.warn(f"Unseen segments {unseen} in score; defaulting to category 0")

        y = df['outcome_paid'].values

        return self.model.score(X, y)


class TLearner:
    """
    Custom T-learner (Two-Model Learner) for uplift estimation.

    Fits separate outcome models on treated and control groups:
    - Model 1 (Y1): trained on observations where T=1 (contacted)
    - Model 0 (Y0): trained on observations where T=0 (not contacted)

    CATE = E[Y1(X) | X] - E[Y0(X) | X]
    """

    def __init__(self, base_model=None):
        """
        Args:
            base_model: sklearn classifier to use. Defaults to GradientBoostingClassifier.
        """
        if base_model is None:
            self.base_model = clone(DEFAULT_BASE_MODEL)
        else:
            self.base_model = base_model
        self.model_1 = None  # Model trained on treated (T=1)
        self.model_0 = None  # Model trained on control (T=0)
        self.le_segment = LabelEncoder()

    def fit(self, X: pd.DataFrame, T: np.ndarray, y: np.ndarray) -> 'TLearner':
        """
        Fit T-learner on treatment assignment and outcomes.

        Args:
            X: Features (treatment-specific features included)
            T: Binary treatment indicator (1=contacted, 0=not contacted)
            y: Binary outcome (1=paid, 0=didn't pay)

        Raises:
            ValueError: If treatment or control group is empty
        """
        # Fit on plain arrays so fit and predict_cate see the same input type
        X = np.asarray(X)
        T = np.asarray(T)
        y = np.asarray(y)

        # Separate by treatment group
        idx_1 = T == 1
        idx_0 = T == 0

        X_1, y_1 = X[idx_1], y[idx_1]
        X_0, y_0 = X[idx_0], y[idx_0]

        # Bug fix #5: Check for empty groups
        if len(X_1) == 0 or len(X_0) == 0:
            raise ValueError("Treatment or control group is empty; cannot fit T-learner")

        # Bug fix #4: Use base_model parameter instead of hardcoding
        self.model_1 = clone(self.base_model)
        self.model_0 = clone(self.base_model)

        self.model_1.fit(X_1, y_1)
        self.model_0.fit(X_0, y_0)

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE (treatment effect) for new samples.

        Args:
            X: Feature matrix (numpy array)

        Returns:
            E[Y | T=1, X] - E[Y | T=0, X]

        Raises:
            ValueError: If models are not fitted
        """
        # Bug fix #5: Better error detection
        if self.model_1 is None or self.model_0 is None:
            raise ValueError("Models not fitted. Call fit() first.")

        if len(X) == 0:
            return np.array([])

        X = np.asarray(X)

        # Predict probability of paying under treatment and control
        y1_pred = self.model_1.predict_proba(X)[:, 1]
        y0_pred = self.model_0.predict_proba(X)[:, 1]

        # CATE = effect of treatment
        cate = y1_pred - y0_pred

        return cate


class UpliftModel:
    """
    Uplift Model: CATE (Conditional Average Treatment Effect).

    Estimates how much a specific action changes payment probability for a given debtor.
    Uses T-learner (trains two separate models: one for treated, one for control).

    Requires:
    - Randomized treatment assignment in training data (to avoid confounding)
    - Action features (channel, timing) in addition to debtor features

    Note: the T-learner here splits only on contact vs. not (see TLearner below) —
    channel/timing are features inside the treated model, not separate treatment
    arms. The "CATE" it estimates is "contacted (with these channel/timing values)
    vs. not contacted," which the treated model relies on channel/timing being
    randomized to decompose correctly. It is not a strict multi-arm T-learner
    (that would fit one model per channel/timing combination).
    """

    def __init__(self, learner_type: str = 't'):
        """
        Args:
            learner_type: 't' for T-learner (currently only one supported)
        """
        if learner_type != 't':
            raise ValueError("Only T-learner ('t') is supported in this implementation")

        self.learner_type = learner_type
        self.learner = TLearner()
        self.feature_cols = None
        self.le_segment = LabelEncoder()
        self.le_channel = LabelEncoder()
        self.le_timing = LabelEncoder()

    def fit(self, df: pd.DataFrame) -> 'UpliftModel':
        """
        Fit uplift model on historical cohort data with randomized actions.

        Args:
            df: DataFrame with debtor features, action (channel, timing), and outcome.
                Must have randomized action assignment for clean identification.
        """
        # Feature engineering
        X = df[['balance', 'days_overdue', 'prior_payments', 'contactability']].copy()
        X['segment_encoded'] = self.le_segment.fit_transform(df['segment'])
        X['channel_encoded'] = self.le_channel.fit_transform(df['action_channel'].fillna('none'))
        X['timing_encoded'] = self.le_timing.fit_transform(df['action_timing'].fillna('none'))

        # Treatment: contacted (1) vs. not (0)
        T = df['action_contact'].values

        # Outcome: paid
        y = df['outcome_paid'].values

        # Fit T-learner
        self.learner.fit(X, T, y)
        self.feature_cols = X.columns.tolist()

        return self

    def predict_cate(self, df: pd.DataFrame, channel: str = 'voice', timing: str = 'morning') -> np.ndarray:
        """
        Predict CATE (treatment effect) for new debtors.

        Args:
            df: DataFrame with required features
            channel: Action channel to evaluate ('voice', 'sms', 'whatsapp', 'email')
            timing: Action timing to evaluate ('morning', 'afternoon', 'evening')

        Returns:
            1D array of estimated treatment effects for the given action
        """
        X = df[['balance', 'days_overdue', 'prior_payments', 'contactability']].copy()

        # Bug fix #6: Handle unseen categories per-row (not batch-wide)
        segments = df['segment'].values
        known_seg_mask = np.isin(segments, self.le_segment.classes_)

        X['segment_encoded'] = 0  # Default for unseen
        if known_seg_mask.any():
            known_indices = np.where(known_seg_mask)[0]
            X.iloc[known_indices, X.columns.get_loc('segment_encoded')] = self.le_segment.transform(segments[known_seg_mask])

        # Warn if any unseen segments
        if not known_seg_mask.all():
            unseen = np.unique(segments[~known_seg_mask])
            warnings.warn(f"Unseen segments {unseen} in predict_cate; defaulting to category 0")

        # Channel and timing should always be known (fixed set), but handle gracefully
        X['channel_encoded'] = self.le_channel.transform([channel] * len(df))
        X['timing_encoded'] = self.le_timing.transform([timing] * len(df))

        return self.learner.predict_cate(X.values)


class ModelEvaluator:
    """
    Validate uplift model against DGP ground truth.

    Since we built the DGP with known heterogeneous effects, we can:
    1. Compute true treatment effect per debtor × channel combination
    2. Compare estimated CATE vs. true effect
    3. Report calibration and recovery rate
    """

    def __init__(self, dgp, response_model: ResponseModel, uplift_model: UpliftModel):
        """
        Args:
            dgp: DebtorDGP instance with known ground-truth effects
            response_model: Fitted response model
            uplift_model: Fitted uplift model
        """
        self.dgp = dgp
        self.response_model = response_model
        self.uplift_model = uplift_model

    def compute_ground_truth_effects(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute true treatment effect (from DGP) for each debtor × channel.

        Uses DGP's known treatment_effect() method to get ground truth.
        """
        from src.dgp import Action

        effects = {}
        for idx, row in features_df.iterrows():
            debtor_id = row['debtor_id']
            baseline = self.dgp.baseline_payment_prob(row)

            # Effect of voice contact in morning (most common in training)
            action_voice = Action(contact=True, channel='voice', timing='morning')
            effect_voice = self.dgp.treatment_effect(row, action_voice)

            effects[debtor_id] = {
                'baseline': baseline,
                'effect_voice': effect_voice,
                'true_p_pay': baseline + effect_voice,
            }

        return effects

    def calibration_curve(self, features_df: pd.DataFrame, ground_truth: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare estimated effects vs. true effects.

        Returns:
            (true_effects, estimated_effects) for plotting
        """
        # Batch predict all rows at once (vectorized)
        est_effects_all = self.uplift_model.predict_cate(features_df)

        true_effects = []
        estimated_effects = []

        for pos, (idx, row) in enumerate(features_df.iterrows()):
            debtor_id = int(row['debtor_id'])
            if debtor_id in ground_truth:
                true_effects.append(ground_truth[debtor_id]['effect_voice'])
                estimated_effects.append(est_effects_all[pos])

        return np.array(true_effects), np.array(estimated_effects)

    def report(self, features_df: pd.DataFrame) -> Dict:
        """
        Generate evaluation report.
        """
        ground_truth = self.compute_ground_truth_effects(features_df)
        true_effects, est_effects = self.calibration_curve(features_df, ground_truth)

        if len(true_effects) == 0:
            return {
                'mae': np.nan,
                'rmse': np.nan,
                'correlation': np.nan,
                'true_effects': np.array([]),
                'estimated_effects': np.array([]),
            }

        # Metrics
        mae = np.mean(np.abs(true_effects - est_effects))
        rmse = np.sqrt(np.mean((true_effects - est_effects) ** 2))

        if np.std(true_effects) > 0 and np.std(est_effects) > 0:
            correlation = np.corrcoef(true_effects, est_effects)[0, 1]
        else:
            correlation = np.nan

        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'true_effects': true_effects,
            'estimated_effects': est_effects,
        }


if __name__ == '__main__':
    # Quick test
    from src.dgp import DataSimulator

    print("Response vs. Uplift Modeling\n")

    # Generate historical cohort for training
    sim = DataSimulator(seed=42)
    df_hist = sim.simulate_cohort(500, role='historical', horizon=8)

    print(f"Historical cohort: {len(df_hist)} records from {df_hist['debtor_id'].nunique()} debtors")

    # Fit response model
    response_model = ResponseModel(model_type='gb')
    response_model.fit(df_hist)
    print(f"Response model fitted (GB). Training accuracy: {response_model.score(df_hist):.3f}")

    # Fit uplift model
    uplift_model = UpliftModel(learner_type='t')
    uplift_model.fit(df_hist)
    print(f"Uplift model fitted (T-learner). Ready for CATE prediction.\n")

    print("Models fitted and ready for use in bandit policies.")
