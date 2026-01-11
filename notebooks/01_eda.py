"""
Exploratory Data Analysis for the credit card fraud dataset.
Originally did this in a Jupyter notebook but saving as .py for version control.
The actual notebook with plots is in my local files, this just has the analysis code.

Key findings:
- Dataset is highly imbalanced: 492 fraud out of 284,807 (~0.17%)
- V1-V28 are PCA components, already scaled
- Amount and Time need scaling
- Fraud transactions tend to have higher absolute values for V1, V3, V10, V14
- Transaction velocity (how many txns in the same hour) is a good feature
- Time of day matters: more fraud happens at night (makes sense)
"""
import sys
import os
import pandas as pd
import numpy as np

# matplotlib and seaborn for plots - won't crash if not installed (CI)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False
    print("matplotlib/seaborn not available, skipping plots")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_raw_data


def run_eda(data_path=None):
    """Run the full EDA pipeline."""

    print("=" * 60)
    print("  Credit Card Fraud - Exploratory Data Analysis")
    print("=" * 60)

    # load
    df = load_raw_data(data_path)

    # basic stats
    print(f"\nShape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["Class"].value_counts())
    fraud_pct = df["Class"].mean() * 100
    print(f"Fraud rate: {fraud_pct:.3f}%")

    print(f"\nAmount stats:")
    print(df["Amount"].describe())

    # compare fraud vs legit amounts
    print(f"\nFraud transactions - Amount:")
    print(df[df["Class"] == 1]["Amount"].describe())
    print(f"\nLegit transactions - Amount:")
    print(df[df["Class"] == 0]["Amount"].describe())

    # check for nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\nNull values found:")
        print(null_counts[null_counts > 0])
    else:
        print(f"\nNo null values - nice")

    # check for duplicates
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes}")

    # correlation with target
    print(f"\nTop features correlated with fraud (absolute):")
    corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    print(corr.head(10))

    # --- plots ---
    if HAS_PLOTS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. class distribution
        df["Class"].value_counts().plot(kind="bar", ax=axes[0, 0], color=["#2ecc71", "#e74c3c"])
        axes[0, 0].set_title("Class Distribution")
        axes[0, 0].set_xticklabels(["Legit", "Fraud"], rotation=0)

        # 2. amount distribution (log scale because of outliers)
        axes[0, 1].hist(df[df["Class"] == 0]["Amount"], bins=50, alpha=0.7, label="Legit", color="#2ecc71")
        axes[0, 1].hist(df[df["Class"] == 1]["Amount"], bins=50, alpha=0.7, label="Fraud", color="#e74c3c")
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Amount Distribution (log scale)")
        axes[0, 1].legend()

        # 3. time distribution
        axes[1, 0].hist(df[df["Class"] == 0]["Time"], bins=48, alpha=0.7, label="Legit", color="#2ecc71")
        axes[1, 0].hist(df[df["Class"] == 1]["Time"], bins=48, alpha=0.7, label="Fraud", color="#e74c3c")
        axes[1, 0].set_title("Transaction Time Distribution")
        axes[1, 0].legend()

        # 4. top correlated features boxplot
        top_features = corr.head(5).index.tolist()
        fraud_data = df[df["Class"] == 1][top_features]
        axes[1, 1].boxplot([fraud_data[f].values for f in top_features], labels=top_features)
        axes[1, 1].set_title("Top 5 Correlated Features (Fraud only)")

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/eda_overview.png", dpi=150)
        print("\nPlots saved to plots/eda_overview.png")
    else:
        print("\nSkipping plots (no matplotlib)")

    return df


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_eda(data_path)
