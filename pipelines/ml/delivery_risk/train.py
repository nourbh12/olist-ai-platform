"""
Delivery Risk Prediction - V7

Model comparison:
1. Logistic Regression
2. Random Forest
3. XGBoost

Evaluation strategy:
- Chronological 60/20/20 split
- Threshold tuning on validation set
- Final evaluation on future test set

Primary metric:
- PR-AUC

Secondary metrics:
- ROC-AUC
- Precision
- Recall
- F1
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from preprocessing import (
    CORE_FEATURES,
    TARGET_COLUMN,
    create_logistic_preprocessor,
    create_tree_preprocessor,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DELIVERY_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "delivery_features"
)

FACT_ORDERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "fact_orders"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20


# ============================================================
# Load dataset
# ============================================================

def load_dataset():
    """
    Load delivery features and purchase timestamps.

    The timestamp is required only to create the chronological
    train/validation/test split.

    The timestamp itself is NOT used as a model feature.
    """

    delivery_path = str(
        DELIVERY_FEATURES_PATH / "*.parquet"
    )

    fact_orders_path = str(
        FACT_ORDERS_PATH / "*.parquet"
    )

    query = f"""
        SELECT
            d.*,
            o.order_purchase_timestamp
        FROM read_parquet('{delivery_path}') AS d
        LEFT JOIN read_parquet('{fact_orders_path}') AS o
            ON d.order_id = o.order_id
    """

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = duckdb.query(query).to_df()

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(df):
    """
    Clean the dataset and prepare it for chronological splitting.
    """

    print()
    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Convert target to numeric
    # --------------------------------------------------------

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------------

    missing_target = df[TARGET_COLUMN].isna().sum()

    print(f"Missing target rows: {missing_target:,}")

    df = df.dropna(
        subset=[
            TARGET_COLUMN,
            "order_purchase_timestamp",
        ]
    ).copy()

    # --------------------------------------------------------
    # Target as integer
    # --------------------------------------------------------

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    print(f"Rows after filtering: {len(df):,}")

    print()
    print("Target distribution:")

    target_counts = df[TARGET_COLUMN].value_counts().sort_index()

    for value, count in target_counts.items():
        percentage = count / len(df) * 100

        print(
            f"  {value}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    return df


# ============================================================
# Chronological split
# ============================================================

def chronological_split(df):
    """
    Split data chronologically:

    60% train
    20% validation
    20% test

    No shuffling.
    """

    n = len(df)

    train_end = int(n * TRAIN_RATIO)

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    print()
    print("=" * 60)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 60)

    print(
        f"Training rows:   {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Testing rows:    {len(test_df):,}"
    )

    print()

    print("Training period:")
    print(
        f"  {train_df['order_purchase_timestamp'].min()}"
        f" → "
        f"{train_df['order_purchase_timestamp'].max()}"
    )

    print()

    print("Validation period:")
    print(
        f"  {validation_df['order_purchase_timestamp'].min()}"
        f" → "
        f"{validation_df['order_purchase_timestamp'].max()}"
    )

    print()

    print("Testing period:")
    print(
        f"  {test_df['order_purchase_timestamp'].min()}"
        f" → "
        f"{test_df['order_purchase_timestamp'].max()}"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# Prepare X / y
# ============================================================

def prepare_xy(df):
    """
    Extract model features and target.
    """

    X = df[CORE_FEATURES].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# Threshold tuning
# ============================================================

def find_best_threshold(y_true, probabilities):
    """
    Find the probability threshold that maximizes F1
    on the validation set.

    The test set is NOT used here.
    """

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# Model evaluation
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    """
    Train one model, tune threshold on validation,
    then evaluate once on the future test set.
    """

    print()
    print("=" * 60)
    print(model_name.upper())
    print("=" * 60)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Tune threshold ONLY on validation
    # --------------------------------------------------------

    best_threshold, validation_f1 = (
        find_best_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    print(
        f"Best validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Apply validation threshold to test
    # --------------------------------------------------------

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    print()
    print("Test metrics:")
    print(
        f"  Precision: {precision:.4f}"
    )
    print(
        f"  Recall:    {recall:.4f}"
    )
    print(
        f"  F1:        {f1:.4f}"
    )
    print(
        f"  ROC-AUC:   {roc_auc:.4f}"
    )
    print(
        f"  PR-AUC:    {pr_auc:.4f}"
    )

    return {
        "model": model_name,
        "threshold": best_threshold,
        "validation_f1": validation_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("DELIVERY RISK PREDICTION - V7")
    print("=" * 60)

    print()
    print("Core features:")
    for feature in CORE_FEATURES:
        print(f"  - {feature}")

    print()
    print("Models:")
    print("  - Logistic Regression")
    print("  - Random Forest")
    print("  - XGBoost")

    print()
    print("Evaluation:")
    print("  - Chronological 60/20/20 split")
    print("  - Threshold tuned on validation")
    print("  - Final metrics on future test set")
    print("  - Primary metric: PR-AUC")

    # ========================================================
    # Load
    # ========================================================

    df = load_dataset()

    # ========================================================
    # Prepare
    # ========================================================

    df = prepare_dataset(df)

    # ========================================================
    # Split
    # ========================================================

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(df)

    # ========================================================
    # X / y
    # ========================================================

    X_train, y_train = prepare_xy(
        train_df
    )

    X_validation, y_validation = prepare_xy(
        validation_df
    )

    X_test, y_test = prepare_xy(
        test_df
    )

    # ========================================================
    # Class imbalance
    # ========================================================

    positive_count = (
        y_train == 1
    ).sum()

    negative_count = (
        y_train == 0
    ).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print()
    print("=" * 60)
    print("CLASS IMBALANCE")
    print("=" * 60)

    print(
        f"Positive class: {positive_count:,}"
    )

    print(
        f"Negative class: {negative_count:,}"
    )

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    # ========================================================
    # Models
    # ========================================================

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_logistic_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_tree_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgboost_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_tree_preprocessor(),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ========================================================
    # Train + evaluate
    # ========================================================

    results = []

    results.append(
        evaluate_model(
            "logistic_regression",
            logistic_model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        )
    )

    results.append(
        evaluate_model(
            "random_forest",
            random_forest_model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        )
    )

    results.append(
        evaluate_model(
            "xgboost",
            xgboost_model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        )
    )

    # ========================================================
    # Results table
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    print()
    print()
    print("=" * 60)
    print("V7 RESULTS")
    print("=" * 60)

    display_columns = [
        "model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # Best model
    # ========================================================

    best_model = results_df.iloc[0]

    print()
    print("=" * 60)
    print("BEST MODEL BY PR-AUC")
    print("=" * 60)

    print(
        f"Model: {best_model['model']}"
    )

    print(
        f"PR-AUC: {best_model['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC: {best_model['roc_auc']:.4f}"
    )

    print(
        f"F1: {best_model['f1']:.4f}"
    )

    print(
        f"Precision: {best_model['precision']:.4f}"
    )

    print(
        f"Recall: {best_model['recall']:.4f}"
    )

    print(
        f"Threshold: {best_model['threshold']:.2f}"
    )

    print()
    print("=" * 60)
    print("V7 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()