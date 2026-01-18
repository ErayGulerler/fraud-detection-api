"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np

from src.data_loader import generate_sample_data, preprocess, get_feature_target_split


class TestGenerateSampleData:
    """Tests for the sample data generator."""

    def test_returns_dataframe(self):
        df = generate_sample_data(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        df = generate_sample_data(n_samples=200)
        assert len(df) == 200
        # V1-V28 + Time + Amount + Class = 31 columns
        assert len(df.columns) == 31

    def test_has_required_columns(self):
        df = generate_sample_data()
        required = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_fraud_ratio(self):
        df = generate_sample_data(n_samples=1000, fraud_ratio=0.1)
        actual_ratio = df["Class"].mean()
        # allow some tolerance because of rounding
        assert 0.08 <= actual_ratio <= 0.12

    def test_class_values_are_binary(self):
        df = generate_sample_data()
        assert set(df["Class"].unique()).issubset({0, 1})

    def test_no_nulls(self):
        df = generate_sample_data()
        assert df.isnull().sum().sum() == 0


class TestPreprocess:
    """Tests for preprocessing pipeline."""

    def test_returns_tuple(self, sample_df):
        result = preprocess(sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 2  # (df, scaler)

    def test_adds_engineered_features(self, sample_df):
        df_processed, _ = preprocess(sample_df)
        # should have Amount_scaled and time period dummies
        assert "Amount_scaled" in df_processed.columns
        assert "tx_velocity" in df_processed.columns

    def test_removes_original_columns(self, sample_df):
        df_processed, _ = preprocess(sample_df)
        # Time and Amount should be replaced by engineered versions
        assert "Time" not in df_processed.columns
        assert "Amount" not in df_processed.columns

    def test_keeps_class_column(self, sample_df):
        df_processed, _ = preprocess(sample_df)
        assert "Class" in df_processed.columns

    def test_no_nulls_after_processing(self, sample_df):
        df_processed, _ = preprocess(sample_df)
        assert df_processed.isnull().sum().sum() == 0

    def test_removes_duplicates(self):
        """Create data with duplicates and check they're removed."""
        df = generate_sample_data(n_samples=100)
        # add duplicates
        df = pd.concat([df, df.head(5)], ignore_index=True)
        assert len(df) == 105

        df_processed, _ = preprocess(df)
        assert len(df_processed) <= 100  # duplicates should be gone


class TestFeatureTargetSplit:

    def test_split_shapes(self, sample_df):
        df_processed, _ = preprocess(sample_df)
        X, y = get_feature_target_split(df_processed)

        assert len(X) == len(y)
        assert "Class" not in X.columns
        assert len(X.columns) > 0

    def test_target_is_series(self, sample_df):
        df_processed, _ = preprocess(sample_df)
        X, y = get_feature_target_split(df_processed)
        assert isinstance(y, pd.Series)
