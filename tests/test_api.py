"""Tests for the Flask API endpoints."""
import pytest
import json
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier

from src.api import app, EXPECTED_FEATURES
from src.data_loader import generate_sample_data, preprocess, get_feature_target_split


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_model():
    """Create a quick model and patch the load function."""
    df = generate_sample_data(n_samples=200, fraud_ratio=0.05)
    df_proc, scaler = preprocess(df)
    X, y = get_feature_target_split(df_proc)

    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X, y)
    return model, scaler


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"


class TestPredictEndpoint:

    def test_predict_returns_fraud_probability(self, client, mock_model):
        model, scaler = mock_model

        with patch("src.api.get_model", return_value=(model, scaler)):
            # build a transaction with all V1-V28 features
            tx = {f"V{i}": 0.1 for i in range(1, 29)}
            tx["Amount"] = 50.0

            response = client.post(
                "/predict",
                data=json.dumps(tx),
                content_type="application/json"
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert "fraud" in data
            assert "fraud_probability" in data
            assert isinstance(data["fraud"], bool)
            assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_missing_features_returns_400(self, client, mock_model):
        model, scaler = mock_model

        with patch("src.api.get_model", return_value=(model, scaler)):
            # only send V1, missing V2-V28
            response = client.post(
                "/predict",
                data=json.dumps({"V1": 0.5, "Amount": 10}),
                content_type="application/json"
            )

            assert response.status_code == 400
            data = json.loads(response.data)
            assert "error" in data

    def test_predict_empty_body_returns_400(self, client):
        response = client.post(
            "/predict",
            data="{}",
            content_type="application/json"
        )
        assert response.status_code == 400

    def test_predict_invalid_json_returns_400(self, client):
        response = client.post(
            "/predict",
            data="not json at all",
            content_type="application/json"
        )
        # flask might return 400 or our validation catches it
        assert response.status_code in [400, 500]


class TestBatchEndpoint:

    def test_batch_predict(self, client, mock_model):
        model, scaler = mock_model

        with patch("src.api.get_model", return_value=(model, scaler)):
            tx1 = {f"V{i}": 0.1 for i in range(1, 29)}
            tx1["Amount"] = 30.0
            tx2 = {f"V{i}": -0.5 for i in range(1, 29)}
            tx2["Amount"] = 500.0

            response = client.post(
                "/predict/batch",
                data=json.dumps([tx1, tx2]),
                content_type="application/json"
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["processed"] == 2
            assert len(data["results"]) == 2

    def test_batch_not_array_returns_400(self, client):
        response = client.post(
            "/predict/batch",
            data=json.dumps({"not": "array"}),
            content_type="application/json"
        )
        assert response.status_code == 400

    # FIXME: this test is kinda slow with 1000 items, maybe mock more aggressively
    def test_batch_limit(self, client):
        """Batch should reject more than 1000 transactions."""
        huge_batch = [{"V1": 0} for _ in range(1001)]
        response = client.post(
            "/predict/batch",
            data=json.dumps(huge_batch),
            content_type="application/json"
        )
        assert response.status_code == 400
