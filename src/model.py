"""
Model training and evaluation.
Tried logistic regression, random forest, and xgboost.
RF gave the best precision/recall balance so that's the default.
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE

from src.config import (
    RANDOM_STATE, TEST_SIZE, SMOTE_SAMPLING_RATIO,
    RF_PARAMS, MODEL_DIR, MODEL_PATH, SCALER_PATH,
)


def apply_smote(X_train, y_train, sampling_ratio=None):
    """
    Apply SMOTE to handle class imbalance.
    
    The fraud class is only ~0.17% of the data, so without this the model
    just predicts "not fraud" for everything and gets 99.8% accuracy lol
    
    Using sampling_strategy < 1.0 because full oversampling was causing overfitting.
    """
    if sampling_ratio is None:
        sampling_ratio = SMOTE_SAMPLING_RATIO

    smote = SMOTE(sampling_strategy=sampling_ratio, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"After SMOTE:  {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

    return X_resampled, y_resampled


def train_random_forest(X_train, y_train, params=None):
    """Train a random forest classifier."""
    if params is None:
        params = RF_PARAMS

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    """
    Logistic regression baseline.
    Not as good as RF but useful for comparison and it's way faster.
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",  # handles imbalance without SMOTE
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate and print metrics.
    
    For fraud detection we care more about recall (catching actual fraud)
    than precision (false alarms are annoying but not as costly as missing fraud).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "model": model_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return metrics


def cross_validate(model, X, y, cv=5):
    """Quick cross-validation to check for overfitting."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"CV F1 scores: {scores.round(4)}")
    print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def save_model(model, scaler, model_path=None, scaler_path=None):
    """Save trained model and scaler to disk."""
    if model_path is None:
        model_path = MODEL_PATH
    if scaler_path is None:
        scaler_path = SCALER_PATH

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def load_model(model_path=None, scaler_path=None):
    """Load model and scaler from disk."""
    if model_path is None:
        model_path = MODEL_PATH
    if scaler_path is None:
        scaler_path = SCALER_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}. Run train.py first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler
