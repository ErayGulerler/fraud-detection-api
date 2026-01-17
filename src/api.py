"""
Flask REST API for fraud detection.
Serves the trained model as a simple prediction endpoint.

Usage:
    python -m src.api          # run the dev server
    gunicorn src.api:app       # production

Endpoints:
    POST /predict       - classify a single transaction
    POST /predict/batch - classify multiple transactions at once
    GET  /health        - health check
"""
import os
import traceback
import numpy as np
from flask import Flask, request, jsonify

from src.config import API_HOST, API_PORT, FRAUD_THRESHOLD
from src.model import load_model

app = Flask(__name__)

# load model on startup
# using global vars here - not ideal but fine for a single-model API
_model = None
_scaler = None


def get_model():
    """Lazy-load the model so tests can run without a saved model file."""
    global _model, _scaler
    if _model is None:
        _model, _scaler = load_model()
    return _model, _scaler


# expected feature names in the right order
# V1-V28 from PCA + our engineered features
EXPECTED_FEATURES = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount_scaled", "tp_night", "tp_morning", "tp_afternoon", "tp_evening", "tx_velocity"]
)


def validate_transaction(data):
    """
    Validate input data. Returns (cleaned_data, error_message).
    If error_message is not None, the request is bad.
    """
    if not data:
        return None, "Request body is empty"

    if not isinstance(data, dict):
        return None, "Expected a JSON object"

    # check for required fields - at minimum we need V1-V28 and Amount
    missing = []
    for i in range(1, 29):
        key = f"V{i}"
        if key not in data:
            missing.append(key)

    if "Amount" not in data and "Amount_scaled" not in data:
        missing.append("Amount")

    if missing:
        return None, f"Missing required fields: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"

    # try to build feature vector
    try:
        features = []
        for feat in EXPECTED_FEATURES:
            if feat in data:
                features.append(float(data[feat]))
            elif feat == "Amount_scaled" and "Amount" in data:
                # if they sent raw Amount, we'll just scale it simply
                # not perfect without the fitted scaler but good enough for demo
                features.append(float(data["Amount"]) / 100.0)
            else:
                # default to 0 for missing optional features
                features.append(0.0)

        return features, None

    except (ValueError, TypeError) as e:
        return None, f"Invalid data type in fields: {str(e)}"


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a single transaction as fraud or legit.
    
    Expects JSON with V1-V28, Amount (or Amount_scaled), and optionally
    the engineered features. Returns fraud probability and classification.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    features, error = validate_transaction(data)
    if error:
        return jsonify({"error": error}), 400

    try:
        model, scaler = get_model()

        X = np.array(features).reshape(1, -1)
        proba = model.predict_proba(X)[0][1]
        is_fraud = proba >= FRAUD_THRESHOLD

        return jsonify({
            "fraud": bool(is_fraud),
            "fraud_probability": round(float(proba), 4),
            "threshold": FRAUD_THRESHOLD,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        # TODO: replace with proper logging
        print(f"Prediction error: {traceback.format_exc()}")
        return jsonify({"error": "Internal prediction error"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Classify multiple transactions at once.
    Expects JSON array of transaction objects.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400

    if len(data) > 1000:
        return jsonify({"error": "Batch size limit is 1000 transactions"}), 400

    results = []
    errors = []

    model, scaler = get_model()

    for i, tx in enumerate(data):
        features, error = validate_transaction(tx)
        if error:
            errors.append({"index": i, "error": error})
            continue

        X = np.array(features).reshape(1, -1)
        proba = model.predict_proba(X)[0][1]
        is_fraud = proba >= FRAUD_THRESHOLD

        results.append({
            "index": i,
            "fraud": bool(is_fraud),
            "fraud_probability": round(float(proba), 4),
        })

    response = {
        "results": results,
        "processed": len(results),
        "errors": errors,
        "threshold": FRAUD_THRESHOLD,
    }

    return jsonify(response)


if __name__ == "__main__":
    print(f"Starting fraud detection API on {API_HOST}:{API_PORT}")
    print(f"Swagger/docs not set up yet - use curl or Postman")
    app.run(host=API_HOST, port=API_PORT, debug=True)
