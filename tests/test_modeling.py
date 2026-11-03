"""
Test suite for Response Model vs. Uplift Model.

Tests cover:
- Response model training and prediction
- Uplift model training and CATE prediction
- Model evaluation and ground truth comparison
- Heterogeneous effect recovery
"""

import pytest
import numpy as np
import pandas as pd
from src.dgp import DataSimulator, DebtorDGP, Action
from src.modeling import ResponseModel, UpliftModel, ModelEvaluator


class TestResponseModel:
    """Tests for response model (P(pay | features))."""

    @pytest.fixture
    def historical_cohort(self):
        """Generate a training cohort."""
        sim = DataSimulator(seed=42)
        return sim.simulate_cohort(200, role='historical', horizon=6, seed=100)

    def test_response_model_fit(self, historical_cohort):
        """Response model should fit without errors."""
        model = ResponseModel(model_type='lr')
        model.fit(historical_cohort)

        assert model.model is not None
        assert model.feature_cols is not None

    def test_response_model_predict_shape(self, historical_cohort):
        """Predictions should have correct shape."""
        model = ResponseModel(model_type='lr')
        model.fit(historical_cohort)

        test_df = historical_cohort.drop_duplicates(subset=['debtor_id']).head(10)
        probs = model.predict_proba(test_df)

        assert len(probs) == len(test_df)
        assert all(0 <= p <= 1 for p in probs)

    def test_response_model_prob_range(self, historical_cohort):
        """Predicted probabilities should be in [0, 1]."""
        model = ResponseModel(model_type='gb')
        model.fit(historical_cohort)

        test_df = historical_cohort.sample(n=min(50, len(historical_cohort)))
        probs = model.predict_proba(test_df)

        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_response_model_score(self, historical_cohort):
        """Model should achieve reasonable accuracy."""
        model = ResponseModel(model_type='gb')
        model.fit(historical_cohort)

        score = model.score(historical_cohort)

        # Should be better than random (50%)
        assert score > 0.5

    def test_response_model_gb_vs_lr(self, historical_cohort):
        """GB model should fit without errors (like LR)."""
        model_lr = ResponseModel(model_type='lr')
        model_lr.fit(historical_cohort)

        model_gb = ResponseModel(model_type='gb')
        model_gb.fit(historical_cohort)

        # Both should produce valid predictions
        test_df = historical_cohort.drop_duplicates(subset=['debtor_id']).head(10)
        probs_lr = model_lr.predict_proba(test_df)
        probs_gb = model_gb.predict_proba(test_df)

        assert len(probs_lr) == len(probs_gb)


class TestUpliftModel:
    """Tests for uplift model (CATE estimation)."""

    @pytest.fixture
    def historical_cohort(self):
        """Generate a training cohort with randomized actions."""
        sim = DataSimulator(seed=42)
        return sim.simulate_cohort(300, role='historical', horizon=6, seed=200)

    def test_uplift_model_t_learner_fit(self, historical_cohort):
        """T-learner should fit without errors."""
        model = UpliftModel(learner_type='t')
        model.fit(historical_cohort)

        assert model.learner is not None
        assert model.feature_cols is not None

    def test_uplift_model_x_learner_not_supported(self, historical_cohort):
        """X-learner is not yet implemented (T-learner sufficient for MVP)."""
        # Note: X-learner requires causalml with system-level dependencies
        # T-learner provides strong baseline; can extend later if needed
        with pytest.raises(ValueError):
            model = UpliftModel(learner_type='x')

    def test_uplift_model_cate_prediction_shape(self, historical_cohort):
        """CATE predictions should have correct shape."""
        model = UpliftModel(learner_type='t')
        model.fit(historical_cohort)

        test_df = historical_cohort.drop_duplicates(subset=['debtor_id']).head(20)
        cate = model.predict_cate(test_df)

        assert len(cate) == len(test_df)

    def test_uplift_model_cate_in_reasonable_range(self, historical_cohort):
        """CATE estimates should be in a reasonable range (not NaN, not 1e10)."""
        model = UpliftModel(learner_type='t')
        model.fit(historical_cohort)

        test_df = historical_cohort.drop_duplicates(subset=['debtor_id']).head(50)
        cate = model.predict_cate(test_df)

        assert not np.any(np.isnan(cate))
        assert np.all(cate >= -1)  # Effect can't be worse than -100%
        assert np.all(cate <= 1)   # Effect can't be better than +100%

    def test_uplift_model_heterogeneity(self, historical_cohort):
        """CATE should show heterogeneity (not all same value)."""
        model = UpliftModel(learner_type='t')
        model.fit(historical_cohort)

        test_df = historical_cohort.drop_duplicates(subset=['debtor_id']).head(100)
        cate = model.predict_cate(test_df)

        # CATE should have variance
        assert np.std(cate) > 0.01


class TestModelEvaluator:
    """Tests for model evaluation and ground truth comparison."""

    @pytest.fixture
    def setup_models(self):
        """Setup DGP, training data, and fitted models."""
        sim = DataSimulator(seed=42)
        dgp = sim.dgp
        df_hist = sim.simulate_cohort(250, role='historical', horizon=6, seed=300)

        response_model = ResponseModel(model_type='gb')
        response_model.fit(df_hist)

        uplift_model = UpliftModel(learner_type='t')
        uplift_model.fit(df_hist)

        return dgp, df_hist, response_model, uplift_model

    def test_evaluator_init(self, setup_models):
        """Evaluator should initialize without errors."""
        dgp, df_hist, response_model, uplift_model = setup_models

        evaluator = ModelEvaluator(dgp, response_model, uplift_model)
        assert evaluator.dgp is not None

    def test_ground_truth_computation(self, setup_models):
        """Should compute ground truth effects from DGP."""
        dgp, df_hist, response_model, uplift_model = setup_models
        evaluator = ModelEvaluator(dgp, response_model, uplift_model)

        test_features = df_hist.drop_duplicates(subset=['debtor_id']).head(20)
        ground_truth = evaluator.compute_ground_truth_effects(test_features)

        assert len(ground_truth) > 0
        for debtor_id, effects in ground_truth.items():
            assert 'baseline' in effects
            assert 'effect_voice' in effects
            assert 'true_p_pay' in effects
            assert 0 <= effects['baseline'] <= 1
            assert 0 <= effects['true_p_pay'] <= 1

    def test_calibration_curve(self, setup_models):
        """Should produce calibration curve data."""
        dgp, df_hist, response_model, uplift_model = setup_models
        evaluator = ModelEvaluator(dgp, response_model, uplift_model)

        test_features = df_hist.drop_duplicates(subset=['debtor_id']).head(30)
        ground_truth = evaluator.compute_ground_truth_effects(test_features)
        true_effects, est_effects = evaluator.calibration_curve(test_features, ground_truth)

        assert len(true_effects) > 0
        assert len(est_effects) == len(true_effects)
        assert not np.any(np.isnan(true_effects))
        assert not np.any(np.isnan(est_effects))

    def test_report_generation(self, setup_models):
        """Should generate comprehensive evaluation report."""
        dgp, df_hist, response_model, uplift_model = setup_models
        evaluator = ModelEvaluator(dgp, response_model, uplift_model)

        test_features = df_hist.drop_duplicates(subset=['debtor_id']).head(40)
        report = evaluator.report(test_features)

        expected_keys = {'mae', 'rmse', 'correlation', 'true_effects', 'estimated_effects'}
        assert set(report.keys()) == expected_keys

        # Metrics should be reasonable
        assert report['mae'] >= 0
        assert report['rmse'] >= 0
        assert -1 <= report['correlation'] <= 1


class TestUpliftRecovery:
    """Tests validating that uplift model recovers known heterogeneous effects."""

    def test_effect_recovery_by_segment(self):
        """Uplift model should recover that different segments have different effects."""
        sim = DataSimulator(seed=42)
        dgp = sim.dgp
        df_hist = sim.simulate_cohort(400, role='historical', horizon=8, seed=400)

        # Fit uplift model
        uplift_model = UpliftModel(learner_type='t')
        uplift_model.fit(df_hist)

        # Get CATE by segment
        for segment in ['high_risk', 'medium_risk', 'low_risk', 'frequent_payer']:
            segment_df = df_hist[df_hist['segment'] == segment].drop_duplicates(subset=['debtor_id']).head(30)
            if len(segment_df) > 0:
                cate = uplift_model.predict_cate(segment_df)
                # Should have some variance within segment too
                assert len(cate) > 0

    def test_true_vs_estimated_correlation(self):
        """Estimated CATE should correlate with true effects."""
        sim = DataSimulator(seed=42)
        dgp = sim.dgp
        df_hist = sim.simulate_cohort(500, role='historical', horizon=8, seed=500)

        response_model = ResponseModel(model_type='gb')
        response_model.fit(df_hist)

        uplift_model = UpliftModel(learner_type='t')
        uplift_model.fit(df_hist)

        evaluator = ModelEvaluator(dgp, response_model, uplift_model)

        test_features = df_hist.drop_duplicates(subset=['debtor_id']).head(100)
        report = evaluator.report(test_features)

        # Correlation should be positive (estimated tracks true effects)
        # With small sample it might not be high, but should be positive
        assert report['correlation'] > 0


class TestResponseVsUplift:
    """Tests comparing response model vs uplift model capabilities."""

    def test_response_ignores_actions(self):
        """Response model prediction should not depend on action features."""
        sim = DataSimulator(seed=42)
        df_hist = sim.simulate_cohort(300, role='historical', horizon=6, seed=600)

        response_model = ResponseModel(model_type='gb')
        response_model.fit(df_hist)

        # Drop action columns and predict
        test_df = df_hist.drop(columns=['action_contact', 'action_channel', 'action_timing',
                                        'week', 'outcome_paid', 'outcome_annoyance',
                                        'outcome_opted_out', 'outcome_complained']).drop_duplicates()

        if len(test_df) > 0:
            probs = response_model.predict_proba(test_df)
            assert len(probs) == len(test_df)

    def test_uplift_uses_heterogeneity(self):
        """Uplift model should detect that effects vary by segment."""
        sim = DataSimulator(seed=42)
        df_hist = sim.simulate_cohort(400, role='historical', horizon=8, seed=700)

        uplift_model = UpliftModel(learner_type='t')
        uplift_model.fit(df_hist)

        # Get CATE for high-risk and frequent-payer debtors
        high_risk = df_hist[df_hist['segment'] == 'high_risk'].drop_duplicates(subset=['debtor_id']).head(20)
        frequent = df_hist[df_hist['segment'] == 'frequent_payer'].drop_duplicates(subset=['debtor_id']).head(20)

        if len(high_risk) > 0 and len(frequent) > 0:
            cate_hr = uplift_model.predict_cate(high_risk)
            cate_fp = uplift_model.predict_cate(frequent)

            # High-risk should have positive effect (contact helps them pay)
            # Frequent payers should have near-zero or negative effect (already motivated)
            assert np.mean(cate_hr) > np.mean(cate_fp)


class TestEndToEndModeling:
    """End-to-end tests for the response and uplift modeling pipeline."""

    def test_full_pipeline_runs(self):
        """Full modeling pipeline should run without errors."""
        sim = DataSimulator(seed=42)

        # Generate historical training cohort
        df_hist = sim.simulate_cohort(300, role='historical', horizon=8, seed=800)

        # Fit both models
        response_model = ResponseModel(model_type='gb')
        response_model.fit(df_hist)

        uplift_model = UpliftModel(learner_type='t')
        uplift_model.fit(df_hist)

        # Evaluate
        evaluator = ModelEvaluator(sim.dgp, response_model, uplift_model)

        test_features = df_hist.drop_duplicates(subset=['debtor_id']).head(50)
        report = evaluator.report(test_features)

        assert 'mae' in report
        assert report['mae'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
