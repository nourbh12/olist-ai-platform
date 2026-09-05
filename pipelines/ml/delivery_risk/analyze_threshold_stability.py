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

TARGET = "is_late"

# Thresholds to evaluate
THRESHOLDS = [
    0.05,
    0.07,
    0.10,
    0.12,
    0.15,
    0.17,
    0.20,
    0.25,
    0.30,
]

# Same chronological split used in V11
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

# Number of chronological windows inside validation
N_VALIDATION_WINDOWS = 4

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

def load_dataset() -> pd.DataFrame:

    query = f"""
        SELECT
            d.*,
            o.order_purchase_timestamp
        FROM read_parquet('{DELIVERY_FEATURES_PATH.as_posix()}/*.parquet') d
        LEFT JOIN read_parquet('{FACT_ORDERS_PATH.as_posix()}/*.parquet') o
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

    # Target must exist
    df = df.dropna(subset=[TARGET])

    # Timestamp is required for chronological evaluation
    df = df.dropna(subset=["order_purchase_timestamp"])

    # Make sure target is integer
    df[TARGET] = df[TARGET].astype(int)

    # Strict chronological ordering
    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    return df


# ============================================================
# Train model
# ============================================================

def build_model() -> Pipeline:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False,
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
        class_weight=None,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

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

    flagged_rate = predictions.mean()

    actual_late_rate = y_true.mean()

    if flagged_rate > 0:
        precision_lift = (
            precision / actual_late_rate
            if actual_late_rate > 0
            else np.nan
        )
    else:
        precision_lift = np.nan

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_rate": flagged_rate,
        "lift": precision_lift,
    }


# ============================================================
# Analyze validation windows
# ============================================================

def analyze_validation_windows(
    df: pd.DataFrame,
    model: Pipeline,
    train_end: int,
    validation_end: int,
) -> pd.DataFrame:

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    X_train = train_df[CORE_FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[CORE_FEATURES]
    y_validation = validation_df[TARGET]

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Training late rate: "
        f"{y_train.mean():.2%}"
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # Split validation into chronological windows
    window_size = len(validation_df) // N_VALIDATION_WINDOWS

    results = []

    for window_idx in range(N_VALIDATION_WINDOWS):

        start = window_idx * window_size

        if window_idx == N_VALIDATION_WINDOWS - 1:
            end = len(validation_df)
        else:
            end = (window_idx + 1) * window_size

        window_df = validation_df.iloc[
            start:end
        ]

        y_window = window_df[TARGET]

        window_probabilities = probabilities[
            start:end
        ]

        window_start_date = (
            window_df["order_purchase_timestamp"].min()
        )

        window_end_date = (
            window_df["order_purchase_timestamp"].max()
        )

        print("\n" + "-" * 60)
        print(
            f"VALIDATION WINDOW {window_idx + 1}"
        )
        print("-" * 60)

        print(
            f"Rows: {len(window_df):,}"
        )

        print(
            f"Period: "
            f"{window_start_date.date()} → "
            f"{window_end_date.date()}"
        )

        print(
            f"Late rate: "
            f"{y_window.mean():.2%}"
        )

        for threshold in THRESHOLDS:

            metrics = calculate_metrics(
                y_window,
                window_probabilities,
                threshold,
            )

            results.append(
                {
                    "window": window_idx + 1,
                    "start_date": window_start_date,
                    "end_date": window_end_date,
                    "late_rate": y_window.mean(),
                    "threshold": threshold,
                    **metrics,
                }
            )

    return pd.DataFrame(results)


# ============================================================
# Stability summary
# ============================================================

def create_stability_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        results
        .groupby("threshold")
        .agg(
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            min_precision=("precision", "min"),
            max_precision=("precision", "max"),

            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            min_recall=("recall", "min"),
            max_recall=("recall", "max"),

            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            min_f1=("f1", "min"),
            max_f1=("f1", "max"),

            mean_flagged_rate=("flagged_rate", "mean"),
            std_flagged_rate=("flagged_rate", "std"),

            mean_lift=("lift", "mean"),
            std_lift=("lift", "std"),
            min_lift=("lift", "min"),
            max_lift=("lift", "max"),
        )
        .reset_index()
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("V12 - THRESHOLD STABILITY ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nLoaded rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_dataset(df)

    print(
        f"Rows after cleaning: {len(df):,}"
    )

    print(
        f"Overall late rate: "
        f"{df[TARGET].mean():.2%}"
    )

    print(
        f"Period: "
        f"{df['order_purchase_timestamp'].min()} "
        f"→ "
        f"{df['order_purchase_timestamp'].max()}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = df.iloc[:train_end]
    validation_df = df.iloc[
        train_end:validation_end
    ]
    test_df = df.iloc[
        validation_end:
    ]

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 60)

    print(
        f"Train:      {len(train_df):,} "
        f"({len(train_df) / n:.1%})"
    )

    print(
        f"Validation: {len(validation_df):,} "
        f"({len(validation_df) / n:.1%})"
    )

    print(
        f"Test:       {len(test_df):,} "
        f"({len(test_df) / n:.1%})"
    )

    print("\nLate rates:")

    print(
        f"  Train:      "
        f"{train_df[TARGET].mean():.2%}"
    )

    print(
        f"  Validation: "
        f"{validation_df[TARGET].mean():.2%}"
    )

    print(
        f"  Test:       "
        f"{test_df[TARGET].mean():.2%}"
    )

    # --------------------------------------------------------
    # Train model and analyze validation windows
    # --------------------------------------------------------

    model = build_model()

    results = analyze_validation_windows(
        df,
        model,
        train_end,
        validation_end,
    )

    # --------------------------------------------------------
    # Display detailed results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS BY WINDOW")
    print("=" * 60)

    display_columns = [
        "window",
        "threshold",
        "late_rate",
        "precision",
        "recall",
        "f1",
        "flagged_rate",
        "lift",
    ]

    display_df = results[
        display_columns
    ].copy()

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

    display_df["lift"] = (
        display_df["lift"]
        .round(2)
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Stability summary
    # --------------------------------------------------------

    summary = create_stability_summary(
        results
    )

    print("\n" + "=" * 60)
    print("THRESHOLD STABILITY SUMMARY")
    print("=" * 60)

    summary_display = summary.copy()

    percentage_columns = [
        "mean_precision",
        "std_precision",
        "min_precision",
        "max_precision",
        "mean_recall",
        "std_recall",
        "min_recall",
        "max_recall",
        "mean_f1",
        "std_f1",
        "min_f1",
        "max_f1",
        "mean_flagged_rate",
        "std_flagged_rate",
    ]

    for column in percentage_columns:
        summary_display[column] = (
            summary_display[column] * 100
        ).round(2)

    for column in [
        "mean_lift",
        "std_lift",
        "min_lift",
        "max_lift",
    ]:
        summary_display[column] = (
            summary_display[column]
            .round(2)
        )

    print(
        summary_display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Candidate thresholds
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ROBUST THRESHOLD CANDIDATES")
    print("=" * 60)

    # Candidate rule:
    # - average recall >= 50%
    # - average flagged rate <= 60%
    # - positive average lift
    candidates = summary[
        (summary["mean_recall"] >= 0.50)
        & (summary["mean_flagged_rate"] <= 0.60)
        & (summary["mean_lift"] > 1.0)
    ].copy()

    if candidates.empty:

        print(
            "No threshold satisfies the "
            "current robustness criteria."
        )

    else:

        candidates = candidates.sort_values(
            by=[
                "mean_f1",
                "mean_lift",
            ],
            ascending=False,
        )

        candidate_display = candidates[
            [
                "threshold",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "mean_flagged_rate",
                "mean_lift",
                "std_f1",
            ]
        ].copy()

        for column in [
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "mean_flagged_rate",
            "std_f1",
        ]:
            candidate_display[column] *= 100

        candidate_display[
            "mean_precision"
        ] = candidate_display[
            "mean_precision"
        ].round(2)

        candidate_display[
            "mean_recall"
        ] = candidate_display[
            "mean_recall"
        ].round(2)

        candidate_display[
            "mean_f1"
        ] = candidate_display[
            "mean_f1"
        ].round(2)

        candidate_display[
            "mean_flagged_rate"
        ] = candidate_display[
            "mean_flagged_rate"
        ].round(2)

        candidate_display[
            "std_f1"
        ] = candidate_display[
            "std_f1"
        ].round(2)

        candidate_display[
            "mean_lift"
        ] = candidate_display[
            "mean_lift"
        ].round(2)

        print(
            candidate_display.to_string(
                index=False
            )
        )

        best_threshold = candidates.iloc[
            0
        ]["threshold"]

        print(
            f"\nRecommended robust threshold: "
            f"{best_threshold:.2f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "pipelines"
        / "ml"
        / "delivery_risk"
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = (
        output_dir
        / "threshold_stability_by_window.csv"
    )

    summary_path = (
        output_dir
        / "threshold_stability_summary.csv"
    )

    results.to_csv(
        detailed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(detailed_path)
    print(summary_path)


if __name__ == "__main__":
    main()