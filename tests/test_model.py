"""Tests for model training and evaluation."""
import pytest
import os
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.model import (
    apply_smote, train_random_forest, train_logistic_regression,
    evaluate_model, save_model, load_model,
)


class TestSMOTE:

    def test_increases_minority_class(self, processed_data):
        X, y, _ = processed_data
        fraud_before = (y == 1).sum()

        X_res, y_res = apply_smote(X, y, sampling_ratio=0.5)
        fraud_after = (y_res == 1).sum()

        assert fraud_after > fraud_before

    def test_preserves_majority_class(self, processed_data):
        X, y, _ = processed_data
        legit_before = (y == 0).sum()

        X_res, y_res = apply_smote(X, y)
        legit_after = (y_res == 0).sum()

        # SMOTE shouldn't change the number of majority samples
        assert legit_after == legit_before

    def test_output_shapes_match(self, processed_data):
        X, y, _ = processed_data
        X_res, y_res = apply_smote(X, y)

        assert len(X_res) == len(y_res)
        assert X_res.shape[1] == X.shape[1]  # same number of features


class TestTraining:

    def test_random_forest_returns_model(self, processed_data):
        X, y, _ = processed_data
        model = train_random_forest(X, y)
        assert isinstance(model, RandomForestClassifier)

    def test_model_can_predict(self, processed_data):
        X, y, _ = processed_data
        model = train_random_forest(X, y, params={
            "n_estimators": 10, "random_state": 42, "max_depth": 5
        })
        predictions = model.predict(X.head(5))
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)

    def test_model_predict_proba(self, processed_data):
        X, y, _ = processed_data
        model = train_random_forest(X, y, params={
            "n_estimators": 10, "random_state": 42, "max_depth": 5
        })
        probas = model.predict_proba(X.head(5))
        assert probas.shape == (5, 2)
        # probabilities should sum to 1
        for row in probas:
            assert abs(sum(row) - 1.0) < 0.001

    def test_logistic_regression_baseline(self, processed_data):
        X, y, _ = processed_data
        model = train_logistic_regression(X, y)
        preds = model.predict(X.head(10))
        assert len(preds) == 10


class TestEvaluation:

    def test_evaluate_returns_metrics(self, processed_data):
        X, y, _ = processed_data
        model = train_random_forest(X, y, params={
            "n_estimators": 10, "random_state": 42, "max_depth": 5
        })
        metrics = evaluate_model(model, X, y, "Test Model")

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics

    def test_metrics_are_valid_range(self, processed_data):
        X, y, _ = processed_data
        model = train_random_forest(X, y, params={
            "n_estimators": 10, "random_state": 42, "max_depth": 5
        })
        metrics = evaluate_model(model, X, y)

        for key in ["precision", "recall", "f1_score", "auc_roc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


class TestSaveLoad:

    def test_save_and_load_model(self, trained_model):
        model, scaler = trained_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            scaler_path = os.path.join(tmpdir, "scaler.joblib")

            save_model(model, scaler, model_path, scaler_path)

            assert os.path.exists(model_path)
            assert os.path.exists(scaler_path)

            loaded_model, loaded_scaler = load_model(model_path, scaler_path)
            assert loaded_model is not None
            assert loaded_scaler is not None

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/tmp/does_not_exist_model.joblib")

    @pytest.mark.skip(reason="need to figure out model versioning first")
    def test_model_version_tracking(self):
        # TODO: save model metadata (training date, params, metrics) alongside model
        pass
