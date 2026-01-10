"""
Configuration for the fraud detection project.
Keeping everything in one place so I don't have magic numbers scattered around.
"""
import os

# paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")

# model params - these gave the best results after tuning
RANDOM_STATE = 42
TEST_SIZE = 0.2

# the dataset is super imbalanced (~0.17% fraud) so we need SMOTE
# tried different ratios, 0.3 works better than 1.0 (full oversampling was overfitting)
SMOTE_SAMPLING_RATIO = 0.3

# random forest params (from grid search, see notebooks/01_eda.py)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# xgboost gave slightly better recall but RF is more interpretable
# keeping both for comparison
XGBOOST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

# flask
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5000))
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# threshold for fraud classification
# default 0.5 but we lower it because missing fraud is worse than false alarm
FRAUD_THRESHOLD = 0.35
