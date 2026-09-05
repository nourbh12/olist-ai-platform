import duckdb
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

TARGET_COLUMN = "is_late"

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    print("=" * 60)
    print("V9 - LOGISTIC REGRESSION PROBABILITY CALIBRATION")
    print("=" * 60)

    con = duckdb.connect()

    query = f"""
        SELECT
            d.*,
            o.order_purchase_timestamp
        FROM read_parquet(
            '{DELIVERY_FEATURES_PATH.as_posix()}/*.parquet'
        ) AS d
        LEFT JOIN read_parquet(
            '{FACT_ORDERS_PATH.as_posix()}/*.parquet'
        ) AS o
        ON d.order_id = o.order_id
    """

    df = con.execute(query).df()

    con.close()

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    missing_target = df[TARGET_COLUMN].isna().sum()
    missing_timestamp = df["order_purchase_timestamp"].isna().sum()

    print(f"Missing target rows: {missing_target:,}")
    print(f"Missing timestamp rows: {missing_timestamp:,}")

    df = df.dropna(
        subset=[
            TARGET_COLUMN,
            "order_purchase_timestamp",
        ]
    )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Chronological ordering is essential.
    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    print(f"Rows after filtering: {len(df):,}")

    print("\nTarget distribution:")

    target_counts = df[TARGET_COLUMN].value_counts().sort_index()

    for value, count in target_counts.items():
        percentage = count / len(df) * 100
        print(
            f"{value}: {count:,} "
            f"({percentage:.2f}%)"
        )

    return df


# ============================================================
# Chronological split
# ============================================================

def chronological_split(
    df: pd.DataFrame,
):
    """
    Split:

        60% train
        10% calibration
        10% threshold validation
        20% test
    """

    n = len(df)

    train_end = int(n * 0.60)
    calibration_end = int(n * 0.70)
    threshold_end = int(n * 0.80)

    train_df = df.iloc[:train_end].copy()

    calibration_df = df.iloc[
        train_end:calibration_end
    ].copy()

    threshold_df = df.iloc[
        calibration_end:threshold_end
    ].copy()

    test_df = df.iloc[
        threshold_end:
    ].copy()

    return (
        train_df,
        calibration_df,
        threshold_df,
        test_df,
    )


# ============================================================
# Print period information
# ============================================================

def print_period(
    name: str,
    df: pd.DataFrame,
):
    print(f"\n{name}:")
    print(f"Rows: {len(df):,}")

    print(
        f"Period: "
        f"{df['order_purchase_timestamp'].min()} "
        f"→ "
        f"{df['order_purchase_timestamp'].max()}"
    )

    late_rate = df[TARGET_COLUMN].mean()

    print(
        f"Late rate: {late_rate:.4f} "
        f"({late_rate * 100:.2f}%)"
    )


# ============================================================
# Build Logistic Regression pipeline
# ============================================================

def build_model() -> Pipeline:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                CORE_FEATURES,
            )
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# Platt calibration
# ============================================================

def fit_platt_calibrator(
    raw_probabilities: np.ndarray,
    y_true: np.ndarray,
):
    """
    Learn a calibration model using validation probabilities.

    We use the logit of the raw probability as input to a
    Logistic Regression calibrator.

    This is Platt scaling.
    """

    epsilon = 1e-7

    clipped_probabilities = np.clip(
        raw_probabilities,
        epsilon,
        1 - epsilon,
    )

    logits = np.log(
        clipped_probabilities
        / (1 - clipped_probabilities)
    )

    logits = logits.reshape(-1, 1)

    calibrator = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        logits,
        y_true,
    )

    return calibrator


# ============================================================
# Apply Platt calibration
# ============================================================

def calibrate_probabilities(
    calibrator,
    raw_probabilities: np.ndarray,
) -> np.ndarray:

    epsilon = 1e-7

    clipped_probabilities = np.clip(
        raw_probabilities,
        epsilon,
        1 - epsilon,
    )

    logits = np.log(
        clipped_probabilities
        / (1 - clipped_probabilities)
    )

    logits = logits.reshape(-1, 1)

    calibrated_probabilities = (
        calibrator.predict_proba(logits)[:, 1]
    )

    return calibrated_probabilities


# ============================================================
# Find best threshold
# ============================================================

def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        predicted_positive_rate = (
            predictions.mean()
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive_rate":
                    predicted_positive_rate,
            }
        )

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    return (
        float(best_row["threshold"]),
        results_df,
    )


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Threshold: {threshold:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    print(
        f"Brier:     {brier:.4f}"
    )

    print("\nConfusion matrix:")

    print(
        f"TN: {tn:,}"
    )

    print(
        f"FP: {fp:,}"
    )

    print(
        f"FN: {fn:,}"
    )

    print(
        f"TP: {tp:,}"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# Calibration summary
# ============================================================

def calibration_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
):

    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    calibration_df = pd.DataFrame(
        {
            "actual": y_true,
            "probability": probabilities,
        }
    )

    calibration_df["bin"] = pd.qcut(
        calibration_df["probability"],
        q=n_bins,
        duplicates="drop",
    )

    summary = (
        calibration_df
        .groupby(
            "bin",
            observed=True,
        )
        .agg(
            mean_predicted_probability=(
                "probability",
                "mean",
            ),
            actual_positive_rate=(
                "actual",
                "mean",
            ),
            count=(
                "actual",
                "count",
            ),
        )
        .reset_index(drop=True)
    )

    summary["difference"] = (
        summary["actual_positive_rate"]
        - summary["mean_predicted_probability"]
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_dataset(df)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        calibration_df,
        threshold_df,
        test_df,
    ) = chronological_split(df)

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 60)

    print_period(
        "TRAIN",
        train_df,
    )

    print_period(
        "CALIBRATION",
        calibration_df,
    )

    print_period(
        "THRESHOLD VALIDATION",
        threshold_df,
    )

    print_period(
        "TEST",
        test_df,
    )

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

    X_train = train_df[CORE_FEATURES]
    y_train = train_df[TARGET_COLUMN]

    X_calibration = calibration_df[CORE_FEATURES]
    y_calibration = calibration_df[TARGET_COLUMN]

    X_threshold = threshold_df[CORE_FEATURES]
    y_threshold = threshold_df[TARGET_COLUMN]

    X_test = test_df[CORE_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Train base Logistic Regression
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING BASE LOGISTIC REGRESSION")
    print("=" * 60)

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Raw probabilities
    # --------------------------------------------------------

    calibration_raw_probabilities = (
        model.predict_proba(
            X_calibration
        )[:, 1]
    )

    threshold_raw_probabilities = (
        model.predict_proba(
            X_threshold
        )[:, 1]
    )

    test_raw_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Raw probability statistics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RAW PROBABILITY DISTRIBUTION")
    print("=" * 60)

    print(
        f"Calibration mean: "
        f"{calibration_raw_probabilities.mean():.4f}"
    )

    print(
        f"Threshold validation mean: "
        f"{threshold_raw_probabilities.mean():.4f}"
    )

    print(
        f"Test mean: "
        f"{test_raw_probabilities.mean():.4f}"
    )

    print(
        f"Test actual late rate: "
        f"{y_test.mean():.4f}"
    )

    # --------------------------------------------------------
    # Fit calibrator
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FITTING PLATT CALIBRATOR")
    print("=" * 60)

    calibrator = fit_platt_calibrator(
        calibration_raw_probabilities,
        y_calibration.to_numpy(),
    )

    # --------------------------------------------------------
    # Calibrate probabilities
    # --------------------------------------------------------

    threshold_calibrated_probabilities = (
        calibrate_probabilities(
            calibrator,
            threshold_raw_probabilities,
        )
    )

    test_calibrated_probabilities = (
        calibrate_probabilities(
            calibrator,
            test_raw_probabilities,
        )
    )

    # --------------------------------------------------------
    # Probability statistics after calibration
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CALIBRATED PROBABILITY DISTRIBUTION")
    print("=" * 60)

    print(
        f"Threshold validation mean: "
        f"{threshold_calibrated_probabilities.mean():.4f}"
    )

    print(
        f"Test mean: "
        f"{test_calibrated_probabilities.mean():.4f}"
    )

    print(
        f"Test actual late rate: "
        f"{y_test.mean():.4f}"
    )

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    (
        best_threshold,
        threshold_results,
    ) = find_best_threshold(
        y_threshold.to_numpy(),
        threshold_calibrated_probabilities,
    )

    print("\n" + "=" * 60)
    print("THRESHOLD SELECTION")
    print("=" * 60)

    print(
        f"Best calibrated threshold: "
        f"{best_threshold:.4f}"
    )

    best_threshold_row = threshold_results.loc[
        threshold_results["f1"].idxmax()
    ]

    print(
        f"Validation Precision: "
        f"{best_threshold_row['precision']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{best_threshold_row['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{best_threshold_row['f1']:.4f}"
    )

    print(
        f"Validation predicted-positive rate: "
        f"{best_threshold_row['predicted_positive_rate']:.4f}"
    )

    # --------------------------------------------------------
    # Compare raw vs calibrated on TEST
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RAW MODEL ON TEST")
    print("=" * 60)

    raw_metrics = evaluate_model(
        name="RAW LOGISTIC REGRESSION",
        y_true=y_test.to_numpy(),
        probabilities=test_raw_probabilities,
        threshold=0.50,
    )

    print("\n" + "=" * 60)
    print("CALIBRATED MODEL ON TEST")
    print("=" * 60)

    calibrated_metrics = evaluate_model(
        name="CALIBRATED LOGISTIC REGRESSION",
        y_true=y_test.to_numpy(),
        probabilities=test_calibrated_probabilities,
        threshold=best_threshold,
    )

    # --------------------------------------------------------
    # Calibration summary
    # --------------------------------------------------------

    print("\n")
    calibration_summary(
        y_true=y_test.to_numpy(),
        probabilities=test_calibrated_probabilities,
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "raw",
                **raw_metrics,
            },
            {
                "model": "calibrated",
                **calibrated_metrics,
            },
        ]
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        comparison[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "brier",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Important interpretation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print(
        """
V9 evaluates whether probability calibration improves
the usefulness of the Logistic Regression model.

Important:

- Calibration should improve the reliability of probabilities.
- PR-AUC and ROC-AUC should remain approximately similar
  because calibration is mainly a probability transformation.
- Brier score should generally improve if calibration helps.
- The selected threshold is learned on a separate temporal
  validation period.
- The test set remains untouched until final evaluation.
- No Gold-layer changes are required.
"""
    )


if __name__ == "__main__":
    main()