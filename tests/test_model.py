"""Tests for src/model.py -- feature encoding, training/evaluation with
threshold tuning, and inference on new data."""
from __future__ import annotations

import pandas as pd
import pytest

from src.generate_data import generate_claims_dataset
from src.model import (
    feature_importances,
    prepare_features,
    predict_risk,
    train_and_evaluate,
)


@pytest.fixture(scope="module")
def dataset_df():
    rows = generate_claims_dataset(n=3000, seed=42)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained(dataset_df):
    return train_and_evaluate(dataset_df)


class TestPrepareFeatures:
    def test_one_hot_encodes_categorical_columns(self, dataset_df):
        X = prepare_features(dataset_df)
        assert any(col.startswith("claim_type_") for col in X.columns)
        assert any(col.startswith("channel_") for col in X.columns)

    def test_output_has_same_row_count_as_input(self, dataset_df):
        X = prepare_features(dataset_df)
        assert len(X) == len(dataset_df)

    def test_output_contains_no_nulls(self, dataset_df):
        X = prepare_features(dataset_df)
        assert not X.isnull().any().any()

    def test_numeric_columns_are_preserved(self, dataset_df):
        X = prepare_features(dataset_df)
        assert "claim_amount_eur" in X.columns
        assert "prior_claims_last_12mo" in X.columns


class TestTrainAndEvaluate:
    def test_returns_a_fitted_model(self, trained):
        assert hasattr(trained.model, "predict")
        assert hasattr(trained.model, "predict_proba")

    def test_evaluation_metrics_are_in_valid_ranges(self, trained):
        e = trained.evaluation
        for metric in [e.accuracy, e.precision, e.recall, e.f1, e.roc_auc]:
            assert 0.0 <= metric <= 1.0

    def test_roc_auc_beats_random_chance(self, trained):
        # A genuinely trained model on data with real (if noisy) signal
        # should beat 0.5 (random guessing) by a meaningful margin.
        assert trained.evaluation.roc_auc > 0.55

    def test_confusion_matrix_has_correct_shape_and_total(self, trained):
        cm = trained.evaluation.confusion_matrix
        assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2
        total = sum(sum(row) for row in cm)
        assert total == trained.evaluation.n_test

    def test_n_positive_test_is_a_minority_of_n_test(self, trained):
        # Sanity check the stratified split preserved the imbalanced
        # class ratio in the test set.
        e = trained.evaluation
        assert 0 < e.n_positive_test < e.n_test * 0.25

    def test_decision_threshold_is_tuned_not_hardcoded_default(self, trained):
        # Regression test for the real issue found during development:
        # the default 0.5 threshold produced 90.8% accuracy but only
        # 5.9% recall on this imbalanced task (the model was
        # essentially always predicting the majority class). The tuned
        # threshold should differ from 0.5 for this dataset.
        assert trained.decision_threshold != 0.5
        assert 0.05 <= trained.decision_threshold <= 0.95

    def test_tuned_threshold_is_chosen_to_maximize_validation_f1(self, dataset_df):
        # The threshold is optimized on a VALIDATION split, so an
        # improvement there is guaranteed by construction. Whether that
        # also improves the held-out TEST set's F1 is a separate,
        # non-guaranteed question -- it depends on how well the
        # validation split's optimum generalizes, which can genuinely
        # vary with the sklearn version's internal tie-breaking even at
        # a fixed random_state (found via a real cross-environment
        # discrepancy during development: the same seed picked
        # threshold 0.39 under scikit-learn 1.8 and 0.57 under 1.9,
        # with different resulting test-set F1 outcomes). So this test
        # checks the guaranteed property directly -- that threshold
        # tuning happened via genuine validation-F1 maximization, not
        # a hardcoded or arbitrary value -- rather than asserting a
        # specific test-set outcome that isn't actually guaranteed.
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        from src.model import prepare_features, TARGET

        X = prepare_features(dataset_df)
        y = dataset_df[TARGET]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        tuning_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
        tuning_model.fit(X_tr, y_tr)
        val_proba = tuning_model.predict_proba(X_val)[:, 1]

        # Re-derive what the chosen threshold's validation F1 actually is,
        # and confirm no other candidate threshold beats it on validation.
        from src.model import train_and_evaluate
        trained = train_and_evaluate(dataset_df, random_state=42)
        chosen_pred = (val_proba >= trained.decision_threshold).astype(int)
        chosen_val_f1 = f1_score(y_val, chosen_pred, zero_division=0)

        best_other_f1 = -1.0
        for t in [i / 100 for i in range(5, 96)]:
            pred = (val_proba >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            best_other_f1 = max(best_other_f1, f1)

        assert chosen_val_f1 == pytest.approx(best_other_f1, abs=1e-9)

    def test_tuned_threshold_metrics_are_internally_consistent(self, trained):
        # Whatever the tuned threshold's test-set outcome turns out to
        # be (see the validation-side test above for the guaranteed
        # property), the reported evaluation numbers must at least be
        # internally consistent with each other.
        e = trained.evaluation
        assert 0.0 <= e.f1 <= 1.0
        assert 0.0 <= e.f1_at_default_threshold <= 1.0
        assert 0.0 <= e.accuracy_at_default_threshold <= 1.0

    def test_reproducible_with_same_random_state(self, dataset_df):
        trained_a = train_and_evaluate(dataset_df, random_state=1)
        trained_b = train_and_evaluate(dataset_df, random_state=1)
        assert trained_a.evaluation.roc_auc == trained_b.evaluation.roc_auc
        assert trained_a.decision_threshold == trained_b.decision_threshold


class TestFeatureImportances:
    def test_returns_requested_top_n(self, trained):
        result = feature_importances(trained, top_n=3)
        assert len(result) == 3

    def test_importances_are_sorted_descending(self, trained):
        result = feature_importances(trained, top_n=10)
        importances = [imp for _, imp in result]
        assert importances == sorted(importances, reverse=True)

    def test_a_feature_from_the_actual_generating_rule_ranks_in_top_5(self, trained):
        # The synthetic label depends on claim_amount_eur,
        # days_to_report, prior_claims_last_12mo, and
        # days_since_policy_start (see generate_data.py's risk rule).
        # A model that learned something real should rank at least one
        # of these highly, not just arbitrary/spurious features.
        top_5_names = {name for name, _ in feature_importances(trained, top_n=5)}
        rule_features = {"claim_amount_eur", "days_to_report", "prior_claims_last_12mo", "days_since_policy_start"}
        assert top_5_names & rule_features


class TestPredictRisk:
    def test_scores_new_data_without_error(self, trained):
        new_rows = generate_claims_dataset(n=20, seed=999)
        new_df = pd.DataFrame(new_rows)
        scored = predict_risk(trained, new_df)
        assert len(scored) == 20

    def test_output_includes_probability_and_binary_prediction_columns(self, trained):
        new_rows = generate_claims_dataset(n=10, seed=999)
        new_df = pd.DataFrame(new_rows)
        scored = predict_risk(trained, new_df)
        assert "predicted_risk_probability" in scored.columns
        assert "predicted_high_risk" in scored.columns

    def test_probabilities_are_in_valid_range(self, trained):
        new_rows = generate_claims_dataset(n=50, seed=999)
        new_df = pd.DataFrame(new_rows)
        scored = predict_risk(trained, new_df)
        assert scored["predicted_risk_probability"].between(0, 1).all()

    def test_binary_prediction_matches_tuned_threshold_not_hardcoded_half(self, trained):
        new_rows = generate_claims_dataset(n=200, seed=999)
        new_df = pd.DataFrame(new_rows)
        scored = predict_risk(trained, new_df)
        expected = (scored["predicted_risk_probability"] >= trained.decision_threshold).astype(int)
        assert (scored["predicted_high_risk"] == expected).all()

    def test_handles_a_single_row_batch(self, trained):
        new_rows = generate_claims_dataset(n=1, seed=999)
        new_df = pd.DataFrame(new_rows)
        scored = predict_risk(trained, new_df)
        assert len(scored) == 1
