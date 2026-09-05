import duckdb
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
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

    return df


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            TARGET_COLUMN,
            "order_purchase_timestamp",
        ]
    )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Critical:
    # preserve temporal ordering.
    df = (
        df
        .sort_values("order_purchase_timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Chronological split
# ============================================================

def chronological_split(
    df: pd.DataFrame,
):

    n = len(df)

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    train_df = df.iloc[:train_end].copy()

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
# Build model
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

    # IMPORTANT:
    # No class_weight="balanced".
    model = LogisticRegression(
        class_weight=None,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
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


# ============================================================
# Analyze thresholds
# ============================================================

def analyze_thresholds(
    y_true,
    probabilities,
):

    thresholds = np.arange(
        0.01,
        0.51,
        0.01,
    )

    rows = []

    total_orders = len(y_true)
    total_late = int(y_true.sum())

    baseline_rate = (
        total_late / total_orders
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        flagged_orders = int(
            predictions.sum()
        )

        true_positives = int(
            ((predictions == 1) & (y_true == 1)).sum()
        )

        false_positives = int(
            ((predictions == 1) & (y_true == 0)).sum()
        )

        false_negatives = int(
            ((predictions == 0) & (y_true == 1)).sum()
        )

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

        flagged_rate = (
            flagged_orders / total_orders
        )

        # Lift:
        #
        # precision of flagged orders
        # --------------------------------
        # overall late-order prevalence
        #
        lift = (
            precision / baseline_rate
            if baseline_rate > 0
            else 0
        )

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "flagged_orders": flagged_orders,
                "flagged_rate": flagged_rate,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "lift": lift,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Print selected thresholds
# ============================================================

def print_selected_thresholds(
    results: pd.DataFrame,
):

    selected = results[
        results["threshold"].isin(
            [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
            ]
        )
    ].copy()

    print("\n" + "=" * 100)
    print("THRESHOLD ANALYSIS")
    print("=" * 100)

    display_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "flagged_orders",
        "flagged_rate",
        "true_positives",
        "false_positives",
        "false_negatives",
        "lift",
    ]

    print(
        selected[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# Find best thresholds
# ============================================================

def print_best_thresholds(
    results: pd.DataFrame,
):

    print("\n" + "=" * 100)
    print("BEST THRESHOLDS")
    print("=" * 100)

    # Best F1
    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    print("\nBest F1:")
    print(
        f"Threshold: {best_f1['threshold']:.2f}"
    )
    print(
        f"Precision: {best_f1['precision']:.4f}"
    )
    print(
        f"Recall: {best_f1['recall']:.4f}"
    )
    print(
        f"F1: {best_f1['f1']:.4f}"
    )
    print(
        f"Flagged orders: "
        f"{int(best_f1['flagged_orders']):,}"
    )
    print(
        f"Flagged rate: "
        f"{best_f1['flagged_rate'] * 100:.2f}%"
    )
    print(
        f"Lift: {best_f1['lift']:.2f}x"
    )

    # Best precision among thresholds
    # that still catch at least 50% of late orders.
    recall_50 = results[
        results["recall"] >= 0.50
    ]

    if not recall_50.empty:

        best_precision_50 = recall_50.loc[
            recall_50["precision"].idxmax()
        ]

        print(
            "\nBest precision with recall >= 50%:"
        )

        print(
            f"Threshold: "
            f"{best_precision_50['threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_precision_50['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_precision_50['recall']:.4f}"
        )

        print(
            f"F1: "
            f"{best_precision_50['f1']:.4f}"
        )

        print(
            f"Flagged rate: "
            f"{best_precision_50['flagged_rate'] * 100:.2f}%"
        )

        print(
            f"Lift: "
            f"{best_precision_50['lift']:.2f}x"
        )

    # Best precision among thresholds
    # that flag at most 20% of orders.
    flag_20 = results[
        results["flagged_rate"] <= 0.20
    ]

    if not flag_20.empty:

        best_precision_20 = flag_20.loc[
            flag_20["precision"].idxmax()
        ]

        print(
            "\nBest precision with flagged rate <= 20%:"
        )

        print(
            f"Threshold: "
            f"{best_precision_20['threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_precision_20['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_precision_20['recall']:.4f}"
        )

        print(
            f"F1: "
            f"{best_precision_20['f1']:.4f}"
        )

        print(
            f"Flagged rate: "
            f"{best_precision_20['flagged_rate'] * 100:.2f}%"
        )

        print(
            f"Lift: "
            f"{best_precision_20['lift']:.2f}x"
        )


# ============================================================
# Evaluate threshold on test
# ============================================================

def evaluate_test_threshold(
    threshold,
    y_true,
    probabilities,
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

    flagged_orders = int(
        predictions.sum()
    )

    flagged_rate = (
        flagged_orders / len(y_true)
    )

    baseline_rate = y_true.mean()

    lift = (
        precision / baseline_rate
        if baseline_rate > 0
        else 0
    )

    tp = int(
        ((predictions == 1) & (y_true == 1)).sum()
    )

    fp = int(
        ((predictions == 1) & (y_true == 0)).sum()
    )

    fn = int(
        ((predictions == 0) & (y_true == 1)).sum()
    )

    print("\n" + "=" * 100)
    print(
        f"TEST PERFORMANCE AT THRESHOLD {threshold:.2f}"
    )
    print("=" * 100)

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
        f"Flagged orders: {flagged_orders:,}"
    )

    print(
        f"Flagged rate: {flagged_rate * 100:.2f}%"
    )

    print(
        f"Lift: {lift:.2f}x"
    )

    print(
        f"True positives: {tp:,}"
    )

    print(
        f"False positives: {fp:,}"
    )

    print(
        f"False negatives: {fn:,}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print("V11 - DELIVERY RISK THRESHOLD / BUSINESS ANALYSIS")
    print("=" * 100)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nRows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_dataset(df)

    print(
        f"Rows after filtering: {len(df):,}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(df)

    print("\n" + "=" * 100)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 100)

    print(
        f"\nTrain:      {len(train_df):,}"
    )

    print(
        f"{train_df['order_purchase_timestamp'].min()} "
        f"→ "
        f"{train_df['order_purchase_timestamp'].max()}"
    )

    print(
        f"Late rate: "
        f"{train_df[TARGET_COLUMN].mean() * 100:.2f}%"
    )

    print(
        f"\nValidation: {len(validation_df):,}"
    )

    print(
        f"{validation_df['order_purchase_timestamp'].min()} "
        f"→ "
        f"{validation_df['order_purchase_timestamp'].max()}"
    )

    print(
        f"Late rate: "
        f"{validation_df[TARGET_COLUMN].mean() * 100:.2f}%"
    )

    print(
        f"\nTest:       {len(test_df):,}"
    )

    print(
        f"{test_df['order_purchase_timestamp'].min()} "
        f"→ "
        f"{test_df['order_purchase_timestamp'].max()}"
    )

    print(
        f"Late rate: "
        f"{test_df[TARGET_COLUMN].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = train_df[CORE_FEATURES]
    y_train = train_df[TARGET_COLUMN]

    X_validation = validation_df[CORE_FEATURES]
    y_validation = validation_df[TARGET_COLUMN]

    X_test = test_df[CORE_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 100)

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Ranking performance
    # --------------------------------------------------------

    validation_roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    print("\n" + "=" * 100)
    print("RANKING PERFORMANCE")
    print("=" * 100)

    print(
        f"\nValidation ROC-AUC: "
        f"{validation_roc_auc:.4f}"
    )

    print(
        f"Validation PR-AUC:  "
        f"{validation_pr_auc:.4f}"
    )

    print(
        f"\nTest ROC-AUC: "
        f"{test_roc_auc:.4f}"
    )

    print(
        f"Test PR-AUC:  "
        f"{test_pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Threshold analysis on validation
    # --------------------------------------------------------

    validation_results = analyze_thresholds(
        y_validation.to_numpy(),
        validation_probabilities,
    )

    print_selected_thresholds(
        validation_results
    )

    print_best_thresholds(
        validation_results
    )

    # --------------------------------------------------------
    # Choose threshold based on F1
    # --------------------------------------------------------

    best_f1_row = validation_results.loc[
        validation_results["f1"].idxmax()
    ]

    best_f1_threshold = float(
        best_f1_row["threshold"]
    )

    # --------------------------------------------------------
    # Test using validation-selected threshold
    # --------------------------------------------------------

    evaluate_test_threshold(
        threshold=best_f1_threshold,
        y_true=y_test.to_numpy(),
        probabilities=test_probabilities,
    )

    # --------------------------------------------------------
    # Test several practical thresholds
    # --------------------------------------------------------

    practical_thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    ]

    print("\n" + "=" * 100)
    print("PRACTICAL THRESHOLDS ON TEST")
    print("=" * 100)

    test_rows = []

    for threshold in practical_thresholds:

        predictions = (
            test_probabilities >= threshold
        ).astype(int)

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

        flagged_orders = int(
            predictions.sum()
        )

        flagged_rate = (
            flagged_orders / len(y_test)
        )

        lift = (
            precision / y_test.mean()
            if y_test.mean() > 0
            else 0
        )

        test_rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "flagged_orders": flagged_orders,
                "flagged_rate": flagged_rate,
                "lift": lift,
            }
        )

    practical_df = pd.DataFrame(
        test_rows
    )

    print(
        practical_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("V11 INTERPRETATION")
    print("=" * 100)

    print(
        """
The objective of V11 is not to improve the underlying
ranking model.

Instead, it determines how the model should be operated.

Important metrics:

- Precision = percentage of flagged orders that are actually late.
- Recall = percentage of late orders successfully detected.
- F1 = balance between precision and recall.
- Flagged rate = percentage of all orders requiring intervention.
- Lift = how much more likely a flagged order is to be late
  compared with a random order.

The final threshold should be selected according to the
business cost of false positives versus false negatives.

The test set remains untouched when selecting the threshold.
"""
    )


if __name__ == "__main__":
    main()