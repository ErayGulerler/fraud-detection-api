#!/usr/bin/env python3
"""
Train the fraud detection model.

Usage:
    python train.py                         # train with default params
    python train.py --data path/to/data.csv # use custom data path
    python train.py --compare               # train and compare all models

Requires the Kaggle credit card fraud dataset in data/creditcard.csv
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""
import argparse
import sys
import os
from sklearn.model_selection import train_test_split

# add project root to path so imports work when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TEST_SIZE, RANDOM_STATE
from src.data_loader import load_raw_data, preprocess, get_feature_target_split
from src.model import (
    apply_smote, train_random_forest, train_logistic_regression,
    evaluate_model, save_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    args = parser.parse_args()

    # 1. load data
    print("Loading data...")
    df = load_raw_data(args.data)
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Fraud rate: {df['Class'].mean()*100:.2f}%")

    # 2. preprocess
    print("\nPreprocessing...")
    df_processed, scaler = preprocess(df)
    X, y = get_feature_target_split(df_processed)
    print(f"Features: {X.shape[1]}")

    # 3. split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. handle imbalance
    print("\nApplying SMOTE...")
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # 5. train
    if args.compare:
        # compare logistic regression vs random forest
        print("\n--- Training Logistic Regression ---")
        lr_model = train_logistic_regression(X_train_balanced, y_train_balanced)
        lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

        print("\n--- Training Random Forest ---")
        rf_model = train_random_forest(X_train_balanced, y_train_balanced)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

        # pick the better one based on recall (catching fraud matters most)
        if rf_metrics["recall"] >= lr_metrics["recall"]:
            print("\n>> Random Forest wins on recall, saving it")
            best_model = rf_model
        else:
            print("\n>> Logistic Regression wins on recall, saving it")
            best_model = lr_model
    else:
        print("\nTraining Random Forest...")
        best_model = train_random_forest(X_train_balanced, y_train_balanced)
        evaluate_model(best_model, X_test, y_test, "Random Forest")

    # 6. save
    print("\nSaving model...")
    save_model(best_model, scaler)

    print("\nDone! Run the API with: python -m src.api")


if __name__ == "__main__":
    main()
