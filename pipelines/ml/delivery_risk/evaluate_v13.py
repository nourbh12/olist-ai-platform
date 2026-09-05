import duckdb
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "pipelines"
    / "ml"
    / "delivery_risk"
    / "output"
)


# ============================================================
# Configuration
# ============================================================

CORE_FEATURES = [
    "order_item_count",
    "order_total_price",
    "order_total_freight",
    "purchase_hour",
    "purchase_day_of_week",
    "estimated_delivery_duration_days",
]

TARGET = "is_late"

# Selected by V12 threshold stability analysis
FINAL_THRESHOLD = 0.07

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

def load_dataset() -> pd.DataFrame:

    query = f"""
        SELECT
            d.*,
            o.order_purchase_timestamp
        FROM read_parquet(
            '{DELIVERY_FEATURES_PATH.as_posix()}/*.parquet'
        ) d
        LEFT JOIN read_parquet(
            '{FACT_ORDERS_PATH.as_posix()}/*.parquet'
        ) o
            ON d.order_id = o.order_id
    """

    df = duckdb.query(query).df()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    return df


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:

    columns = CORE_FEATURES + [
        TARGET,
        "order_purchase_timestamp",
    ]

    df = df[columns].copy()

    # Remove rows where target or timestamp is unavailable
    df = df.dropna(
        subset=[
            TARGET,
            "order_purchase_timestamp",
        ]
    )

    df[TARGET] = df[TARGET].astype(int)

    # Critical: chronological ordering
    df = (
        df
        .sort_values("order_purchase_timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Build final model
# ============================================================

def build_model() -> Pipeline:

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                CORE_FEATURES,
            )
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        class_weight=None,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessing,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# Basic dataset information
# ============================================================

def print_dataset_summary(df: pd.DataFrame):

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Late orders: "
        f"{df[TARGET].sum():,}"
    )

    print(
        f"Late rate: "
        f"{df[TARGET].mean():.2%}"
    )

    print(
        f"Period: "
        f"{df['order_purchase_timestamp'].min()} "
        f"→ "
        f"{df['order_purchase_timestamp'].max()}"
    )


# ============================================================
# Chronological split
# ============================================================

def chronological_split(
    df: pd.DataFrame,
):

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# Print split information
# ============================================================

def print_split_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 60)

    splits = [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]

    for name, split in splits:

        print(
            f"\n{name}"
        )

        print(
            f"  Rows: "
            f"{len(split):,}"
        )

        print(
            f"  Period: "
            f"{split['order_purchase_timestamp'].min().date()} "
            f"→ "
            f"{split['order_purchase_timestamp'].max().date()}"
        )

        print(
            f"  Late orders: "
            f"{split[TARGET].sum():,}"
        )

        print(
            f"  Late rate: "
            f"{split[TARGET].mean():.2%}"
        )


# ============================================================
# Evaluate final test set
# ============================================================

def evaluate_test_set(
    model: Pipeline,
    test_df: pd.DataFrame,
):

    X_test = test_df[
        CORE_FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities
        >= FINAL_THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    flagged_rate = predictions.mean()

    actual_late_rate = y_test.mean()

    lift = (
        precision / actual_late_rate
        if actual_late_rate > 0
        else np.nan
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    # --------------------------------------------------------
    # Print final metrics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("V13 - FINAL TEST EVALUATION")
    print("=" * 60)

    print(
        f"\nFinal threshold: "
        f"{FINAL_THRESHOLD:.2f}"
    )

    print("\nClassification metrics:")

    print(
        f"  Precision: "
        f"{precision:.4f} "
        f"({precision:.2%})"
    )

    print(
        f"  Recall:    "
        f"{recall:.4f} "
        f"({recall:.2%})"
    )

    print(
        f"  F1:        "
        f"{f1:.4f}"
    )

    print(
        f"  ROC-AUC:   "
        f"{roc_auc:.4f}"
    )

    print(
        f"  PR-AUC:    "
        f"{pr_auc:.4f}"
    )

    print(
        f"  Brier:     "
        f"{brier:.4f}"
    )

    print("\nOperational metrics:")

    print(
        f"  Actual late rate: "
        f"{actual_late_rate:.2%}"
    )

    print(
        f"  Flagged rate:     "
        f"{flagged_rate:.2%}"
    )

    print(
        f"  Flagged orders:   "
        f"{predictions.sum():,}"
    )

    print(
        f"  Precision lift:   "
        f"{lift:.2f}x"
    )

    print("\nConfusion matrix:")

    print(
        f"  TN: {tn:,}"
    )

    print(
        f"  FP: {fp:,}"
    )

    print(
        f"  FN: {fn:,}"
    )

    print(
        f"  TP: {tp:,}"
    )

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    metrics = {
        "threshold": FINAL_THRESHOLD,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "actual_late_rate": actual_late_rate,
        "flagged_rate": flagged_rate,
        "flagged_orders": int(
            predictions.sum()
        ),
        "lift": lift,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return (
        probabilities,
        predictions,
        metrics,
    )


# ============================================================
# Performance by delivery duration
# ============================================================

def evaluate_duration_buckets(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
):

    result = test_df.copy()

    result["probability"] = probabilities
    result["prediction"] = predictions

    result["duration_bucket"] = pd.cut(
        result[
            "estimated_delivery_duration_days"
        ],
        bins=[
            -np.inf,
            14,
            30,
            np.inf,
        ],
        labels=[
            "short_<=14",
            "medium_15_30",
            "long_>30",
        ],
    )

    rows = []

    for bucket, group in result.groupby(
        "duration_bucket",
        observed=False,
    ):

        y_true = group[TARGET]
        y_pred = group["prediction"]

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        rows.append(
            {
                "duration_bucket": str(bucket),
                "rows": len(group),
                "late_orders": int(
                    y_true.sum()
                ),
                "late_rate": y_true.mean(),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "flagged_rate": y_pred.mean(),
            }
        )

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("PERFORMANCE BY DELIVERY DURATION")
    print("=" * 60)

    display_df = result_df.copy()

    for column in [
        "late_rate",
        "precision",
        "recall",
        "f1",
        "flagged_rate",
    ]:
        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    print(
        display_df.to_string(
            index=False
        )
    )

    return result_df


# ============================================================
# Performance by month
# ============================================================

def evaluate_monthly(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
):

    result = test_df.copy()

    result["probability"] = probabilities
    result["prediction"] = predictions

    result["month"] = (
        result[
            "order_purchase_timestamp"
        ]
        .dt.to_period("M")
        .astype(str)
    )

    rows = []

    for month, group in result.groupby(
        "month"
    ):

        y_true = group[TARGET]
        y_pred = group["prediction"]

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        rows.append(
            {
                "month": month,
                "rows": len(group),
                "late_orders": int(
                    y_true.sum()
                ),
                "late_rate": y_true.mean(),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "flagged_rate": y_pred.mean(),
            }
        )

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("MONTHLY TEST PERFORMANCE")
    print("=" * 60)

    display_df = result_df.copy()

    for column in [
        "late_rate",
        "precision",
        "recall",
        "f1",
        "flagged_rate",
    ]:
        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    print(
        display_df.to_string(
            index=False
        )
    )

    return result_df


# ============================================================
# Save results
# ============================================================

def save_results(
    metrics: dict,
    duration_results: pd.DataFrame,
    monthly_results: pd.DataFrame,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        OUTPUT_DIR
        / "v13_final_metrics.csv"
    )

    duration_path = (
        OUTPUT_DIR
        / "v13_duration_performance.csv"
    )

    monthly_path = (
        OUTPUT_DIR
        / "v13_monthly_performance.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_path,
        index=False,
    )

    duration_results.to_csv(
        duration_path,
        index=False,
    )

    monthly_results.to_csv(
        monthly_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)

    print(metrics_path)
    print(duration_path)
    print(monthly_path)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("DELIVERY RISK - V13")
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print_dataset_summary(df)

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_dataset(df)

    print(
        f"\nRows after cleaning: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(df)

    print_split_summary(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Train ONLY on training data
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    X_train = train_df[
        CORE_FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Training late rate: "
        f"{y_train.mean():.2%}"
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "\nModel training complete."
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    (
        probabilities,
        predictions,
        metrics,
    ) = evaluate_test_set(
        model,
        test_df,
    )

    # --------------------------------------------------------
    # Duration analysis
    # --------------------------------------------------------

    duration_results = (
        evaluate_duration_buckets(
            test_df,
            probabilities,
            predictions,
        )
    )

    # --------------------------------------------------------
    # Monthly analysis
    # --------------------------------------------------------

    monthly_results = (
        evaluate_monthly(
            test_df,
            probabilities,
            predictions,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        metrics,
        duration_results,
        monthly_results,
    )

    # --------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("V13 COMPLETE")
    print("=" * 60)

    print(
        f"\nFinal threshold: "
        f"{FINAL_THRESHOLD:.2f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Test F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Test recall: "
        f"{metrics['recall']:.2%}"
    )

    print(
        f"Test precision: "
        f"{metrics['precision']:.2%}"
    )

    print(
        f"Test flagged rate: "
        f"{metrics['flagged_rate']:.2%}"
    )

    print("\nThe test set was used only for final evaluation.")


if __name__ == "__main__":
    main()