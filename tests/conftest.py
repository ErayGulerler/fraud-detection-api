"""
Shared test fixtures.
Using generated sample data so tests don't need the 150MB Kaggle CSV.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import generate_sample_data, preprocess, get_feature_target_split


@pytest.fixture
def sample_df():
    """Raw sample dataframe mimicking the credit card dataset."""
    return generate_sample_data(n_samples=500, fraud_ratio=0.05)


@pytest.fixture
def processed_data(sample_df):
    """Preprocessed data ready for training."""
    df_processed, scaler = preprocess(sample_df)
    X, y = get_feature_target_split(df_processed)
    return X, y, scaler


@pytest.fixture
def trained_model(processed_data):
    """A quick trained model for testing predictions."""
    X, y, scaler = processed_data
    # small RF for fast tests
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X, y)
    return model, scaler


@pytest.fixture
def sample_transaction():
    """A single transaction dict for API testing."""
    # just fill V1-V28 with small random values, looks like a legit transaction
    tx = {f"V{i}": float(np.random.normal(0, 0.5)) for i in range(1, 29)}
    tx["Amount"] = 25.50
    return tx
