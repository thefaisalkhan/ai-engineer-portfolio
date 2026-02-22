"""
Pandas Data Pipeline: Cleaning, Feature Engineering, EDA
Real-world patterns used before training ML/LLM models.
"""

import pandas as pd
import numpy as np
from io import StringIO


# ─── Synthetic Dataset (avoids file dependency) ───────────────────────────────

RAW_CSV = """user_id,age,tenure_days,monthly_spend,plan,last_active,churned
1,25,120,49.99,basic,2024-01-15,0
2,,340,89.99,premium,2024-01-20,0
3,34,45,29.99,basic,2023-11-01,1
4,45,700,149.99,enterprise,2024-01-22,0
5,28,30,,basic,2024-01-10,1
6,52,500,99.99,premium,2023-09-15,1
7,-1,200,59.99,basic,2024-01-18,0
8,31,150,79.99,premium,2024-01-19,0
9,29,90,39.99,basic,2023-12-01,1
10,41,600,129.99,enterprise,2024-01-21,0
"""


# ─── Step 1: Load & Inspect ───────────────────────────────────────────────────

def load_and_inspect(csv: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv))
    print("=== Raw Data ===")
    print(df.to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDtypes:\n{df.dtypes}")
    return df


# ─── Step 2: Clean ────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix invalid age (negative)
    df.loc[df["age"] < 0, "age"] = np.nan

    # Impute missing numerics
    df["age"] = df["age"].fillna(df["age"].median())
    df["monthly_spend"] = df["monthly_spend"].fillna(
        df.groupby("plan")["monthly_spend"].transform("median")
    )

    # Parse dates
    df["last_active"] = pd.to_datetime(df["last_active"])
    df["days_since_active"] = (pd.Timestamp("2024-01-22") - df["last_active"]).dt.days

    # Drop duplicates
    df = df.drop_duplicates()

    print("\n=== After Cleaning ===")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df


# ─── Step 3: Feature Engineering ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric features
    df["spend_per_day"] = df["monthly_spend"] / 30
    df["tenure_months"] = df["tenure_days"] / 30
    df["value_score"] = df["monthly_spend"] * df["tenure_months"]

    # Encode categorical
    plan_dummies = pd.get_dummies(df["plan"], prefix="plan", drop_first=True)
    df = pd.concat([df, plan_dummies], axis=1)

    # Engagement bucket
    df["engagement"] = pd.cut(
        df["days_since_active"],
        bins=[0, 7, 30, 90, float("inf")],
        labels=["active", "recent", "lapsed", "churned_risk"],
    )

    print("\n=== Engineered Features ===")
    new_cols = ["spend_per_day", "tenure_months", "value_score", "engagement"]
    print(df[new_cols].to_string())
    return df


# ─── Step 4: Aggregations ────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("plan")
        .agg(
            users=("user_id", "count"),
            avg_spend=("monthly_spend", "mean"),
            avg_tenure=("tenure_months", "mean"),
            churn_rate=("churned", "mean"),
        )
        .round(2)
        .sort_values("churn_rate", ascending=False)
    )
    print("\n=== Plan-Level Summary ===")
    print(summary.to_string())
    return summary


# ─── Step 5: Export Features ─────────────────────────────────────────────────

def export_ml_ready(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "age", "tenure_months", "monthly_spend", "spend_per_day",
        "value_score", "days_since_active",
        "plan_enterprise", "plan_premium",  # one-hot encoded
    ]
    target = "churned"

    # Only keep columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    ml_df = df[feature_cols + [target]].dropna()

    print(f"\n=== ML-Ready Dataset ===")
    print(f"Shape: {ml_df.shape}")
    print(f"Churn rate: {ml_df[target].mean():.1%}")
    print(ml_df.head().to_string())
    return ml_df


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_inspect(RAW_CSV)
    df = clean(df)
    df = engineer_features(df)
    aggregate(df)
    export_ml_ready(df)
