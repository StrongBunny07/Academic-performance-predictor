from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "combined_finals_with_attendance.xlsx"

TARGET_SEMESTER = 4
THRESHOLD = 61

SELECTED_FEATURES = [
    "Internal_Avg_Sem1",
    "Internal_Avg_Sem2",
    "Internal_Avg_Sem3",
    "Internal_Avg_Sem4",
    "Ext_Avg_Sem1",
    "Ext_Avg_Sem2",
    "Ext_Avg_Sem3",
    "arrears",
    "Absent Percentage (%)",
    "Internal_trend_Sem2",
    "Internal_trend_Sem3",
    "Ext_trend_Sem3",
    "Int_Ext_gap_Sem1",
    "Int_Ext_gap_Sem2",
    "Int_Ext_gap_Sem3",
    "Internal_roll2_Sem3",
    "Internal_std_Sem2_3",
    "Internal_roll2_Sem4",
    "Internal_std_Sem3_4",
    "Weighted_Internal",
    "Weighted_External",
    "Int_Ext_ratio_Sem2",
    "Int_Ext_ratio_Sem3",
    "arrears_x_Absent",
    "absent_bin_absent_15_20_x_Weighted_External",
]

RAW_NUMERIC_COLUMNS = [
    "Internal_Avg_Sem1",
    "Internal_Avg_Sem2",
    "Internal_Avg_Sem3",
    "Internal_Avg_Sem4",
    "Ext_Avg_Sem1",
    "Ext_Avg_Sem2",
    "Ext_Avg_Sem3",
    "arrears",
    "Absent Percentage (%)",
]


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, SimpleImputer, KMeans]:
    if "Has_Failure_Before_Sem5" in df.columns and "arrears" not in df.columns:
        df = df.rename(columns={"Has_Failure_Before_Sem5": "arrears"})

    for column in RAW_NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    target_col = f"Ext_Avg_Sem{TARGET_SEMESTER}"
    if target_col not in df.columns:
        raise ValueError(f"Required column missing: {target_col}")

    y = (df[target_col] <= THRESHOLD).astype(int)

    X_base = df[RAW_NUMERIC_COLUMNS].copy()
    imputer = SimpleImputer(strategy="median")
    X_base[:] = imputer.fit_transform(X_base)

    for sem in range(2, TARGET_SEMESTER + 1):
        X_base[f"Internal_trend_Sem{sem}"] = (
            X_base[f"Internal_Avg_Sem{sem}"] - X_base[f"Internal_Avg_Sem{sem - 1}"]
        )

    for sem in range(2, TARGET_SEMESTER):
        X_base[f"Ext_trend_Sem{sem}"] = X_base[f"Ext_Avg_Sem{sem}"] - X_base[f"Ext_Avg_Sem{sem - 1}"]

    for sem in range(1, TARGET_SEMESTER):
        X_base[f"Int_Ext_gap_Sem{sem}"] = X_base[f"Internal_Avg_Sem{sem}"] - X_base[f"Ext_Avg_Sem{sem}"]

    for sem in range(3, TARGET_SEMESTER + 1):
        X_base[f"Internal_roll2_Sem{sem}"] = (
            X_base[f"Internal_Avg_Sem{sem - 1}"] + X_base[f"Internal_Avg_Sem{sem}"]
        ) / 2
        X_base[f"Internal_std_Sem{sem - 1}_{sem}"] = X_base[
            [f"Internal_Avg_Sem{sem - 1}", f"Internal_Avg_Sem{sem}"]
        ].std(axis=1)

    X_base["Weighted_Internal"] = sum(
        X_base[f"Internal_Avg_Sem{sem}"] for sem in range(1, TARGET_SEMESTER + 1)
    ) / TARGET_SEMESTER

    X_base["Weighted_External"] = sum(
        X_base[f"Ext_Avg_Sem{sem}"] for sem in range(1, TARGET_SEMESTER)
    ) / (TARGET_SEMESTER - 1)

    for sem in range(1, TARGET_SEMESTER):
        X_base[f"Int_Ext_ratio_Sem{sem}"] = X_base[f"Internal_Avg_Sem{sem}"] / (
            X_base[f"Ext_Avg_Sem{sem}"] + 1e-5
        )

    X_base["arrears_x_Weighted_Internal"] = X_base["arrears"] * X_base["Weighted_Internal"]
    X_base["arrears_x_Absent"] = X_base["arrears"] * X_base["Absent Percentage (%)"]

    bins = [0, 5, 10, 15, 20, 25, 100]
    labels = [f"absent_{i}_{j}" for i, j in zip(bins[:-1], bins[1:])]
    X_base["Absent_Bin"] = pd.cut(X_base["Absent Percentage (%)"], bins=bins, labels=labels)
    absent_dummies = pd.get_dummies(X_base["Absent_Bin"], prefix="absent_bin")
    X_base = pd.concat([X_base, absent_dummies], axis=1)
    for col in absent_dummies.columns:
        X_base[f"{col}_x_Weighted_Internal"] = X_base[col] * X_base["Weighted_Internal"]
        X_base[f"{col}_x_Weighted_External"] = X_base[col] * X_base["Weighted_External"]
    X_base = X_base.drop(columns=["Absent_Bin"])

    traj_cols = [f"Internal_Avg_Sem{i}" for i in range(1, TARGET_SEMESTER + 1)]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    X_base["trajectory_cluster"] = kmeans.fit_predict(X_base[traj_cols])

    X = X_base.loc[:, X_base.var() > 1e-10].copy()
    X = X.reindex(columns=SELECTED_FEATURES, fill_value=0)
    return X, y, imputer, kmeans


def main() -> None:
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
    X, y, imputer, kmeans = _build_feature_matrix(df)

    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        max_depth=5,
        min_samples_split=10,
    )
    model.fit(scaler.transform(X), y)

    artifacts = {
        "risk_rf_model.pkl": model,
        "imputer.pkl": imputer,
        "scaler.pkl": scaler,
        "feature_names.pkl": SELECTED_FEATURES,
        "original_numeric_cols.pkl": RAW_NUMERIC_COLUMNS,
        "kmeans_model.pkl": kmeans,
    }

    for filename, obj in artifacts.items():
        joblib.dump(obj, BASE_DIR / filename)

    print("Saved semester-4 model artifacts:")
    for filename in artifacts:
        print(f"  - {filename}")
    print(f"Training rows: {len(X)}")
    print(f"Positive class count: {int(y.sum())}")


if __name__ == "__main__":
    main()
