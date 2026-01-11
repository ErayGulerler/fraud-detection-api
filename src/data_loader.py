"""
Data loading and preprocessing for the credit card fraud dataset.
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset has 284,807 transactions with 30 features (V1-V28 from PCA, plus Time and Amount).
Only 492 are fraud (~0.17%) so it's extremely imbalanced.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import RAW_DATA_PATH


def load_raw_data(path=None):
    """Load the raw CSV. Returns a dataframe."""
    if path is None:
        path = RAW_DATA_PATH

    df = pd.read_csv(path)

    # basic sanity checks
    assert "Class" in df.columns, "Missing 'Class' column - wrong dataset?"
    assert len(df) > 0, "Empty dataset"

    return df


def preprocess(df):
    """
    Clean and prepare the data for training.
    
    Steps:
    1. Drop duplicates (there are a few in the dataset)
    2. Scale Amount and Time (V1-V28 are already PCA-transformed)
    3. Add some engineered features
    """
    df = df.copy()
    df = df.drop_duplicates()

    # drop nulls if any (there shouldn't be but just in case)
    if df.isnull().sum().sum() > 0:
        print(f"Warning: found {df.isnull().sum().sum()} null values, dropping rows")
        df = df.dropna()

    # scale Amount - it has a huge range (0 to 25,000+)
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    # Time is seconds from first transaction in dataset
    # convert to hour of day (mod 24*3600 then divide by 3600)
    # not perfect since we don't know the actual start time but gives some cyclical pattern
    df["Hour"] = (df["Time"] % 86400) / 3600

    # bin into time periods - found this helps the model
    df["Time_period"] = pd.cut(
        df["Hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        include_lowest=True
    )
    # one-hot encode time period
    time_dummies = pd.get_dummies(df["Time_period"], prefix="tp", dtype=int)
    df = pd.concat([df, time_dummies], axis=1)

    # transaction velocity: rolling count of transactions in nearby time window
    # using a simple approach - just group by rounded time
    df["Time_bin"] = (df["Time"] // 3600).astype(int)  # hourly bins
    tx_counts = df.groupby("Time_bin").size().rename("tx_velocity")
    df = df.merge(tx_counts, on="Time_bin", how="left")

    # drop columns we don't need anymore
    drop_cols = ["Time", "Amount", "Hour", "Time_period", "Time_bin"]
    df = df.drop(columns=drop_cols)

    return df, scaler


def get_feature_target_split(df):
    """Split dataframe into X (features) and y (target)."""
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def generate_sample_data(n_samples=1000, fraud_ratio=0.02):
    """
    Generate fake data for testing when we don't have the real dataset.
    Mimics the structure of the credit card fraud dataset.
    
    Not for actual training - just so tests can run without the 150MB CSV.
    """
    rng = np.random.RandomState(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # generate PCA-like features (V1-V28)
    legit_features = rng.normal(0, 1, (n_legit, 28))
    # fraud transactions have slightly different distribution
    fraud_features = rng.normal(0.5, 1.5, (n_fraud, 28))

    features = np.vstack([legit_features, fraud_features])

    # Time: seconds, spread over ~2 days
    time_col = rng.uniform(0, 172800, n_samples)

    # Amount: legit usually small, fraud sometimes large
    legit_amounts = rng.exponential(50, n_legit)
    fraud_amounts = rng.exponential(200, n_fraud)
    amounts = np.concatenate([legit_amounts, fraud_amounts])

    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # build dataframe
    col_names = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(features, columns=col_names)
    df["Time"] = time_col
    df["Amount"] = amounts
    df["Class"] = labels.astype(int)

    # shuffle so fraud isn't all at the bottom
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df
