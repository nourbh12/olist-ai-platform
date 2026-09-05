import duckdb
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

from preprocessing import create_preprocessor


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

FEATURE_COLUMNS = [
    "order_item_count",
    "order_total_price",
    "order_total_freight",
    "purchase_hour",
    "purchase_day_of_week",
    "estimated_delivery_duration_days",
    "seller_count",
    "seller_avg_order_count",
    "seller_max_order_count",
    "seller_avg_late_rate",
    "seller_max_late_rate",
    "seller_avg_delivery_delay_days",
    "seller_max_delivery_delay_days",
]

TARGET_COLUMN = "is_late"

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

def load_dataset():
    print("=" * 60)
    print("LOADING DELIVERY RISK DATASET")
    print("=" * 60)

    connection = duckdb.connect()

    query = f"""
        SELECT
            d.*,
            f.order_purchase_timestamp
        FROM read_parquet('{DELIVERY_FEATURES_PATH.as_posix()}/*.parquet') AS d
        INNER JOIN read_parquet('{FACT_ORDERS_PATH.as_posix()}/*.parquet') AS f
            ON d.order_id = f.order_id
    """

    df = connection.execute(query).df()

    connection.close()

    return df


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(df):
    print("\nPreparing dataset...")

    initial_rows = len(df)

    missing_target = df[TARGET_COLUMN].isna().sum()

    print(f"Total rows before filtering: {initial_rows:,}")
    print(f"Missing target rows: {missing_target:,}")

    # Target must never be imputed.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    print(f"Total rows after filtering: {len(df):,}")

    return df


# ============================================================
# Chronological train / validation / test split
# ============================================================

def chronological_split(df):
    """
    60% train
    20% validation
    20% test

    The validation set is used only to select the probability
    threshold.

    The test set is used only for final evaluation.
    """

    n = len(df)

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 60)

    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Testing rows:    {len(test_df):,}")

    print("\nTraining period:")
    print(
        train_df["order_purchase_timestamp"].min(),
        "→",
        train_df["order_purchase_timestamp"].max(),
    )

    print("\nValidation period:")
    print(
        validation_df["order_purchase_timestamp"].min(),
        "→",
        validation_df["order_purchase_timestamp"].max(),
    )

    print("\nTesting period:")
    print(
        test_df["order_purchase_timestamp"].min(),
        "→",
        test_df["order_purchase_timestamp"].max(),
    )

    print("\nTarget distribution:")

    print("\nTraining:")
    print(train_df[TARGET_COLUMN].value_counts())

    print("\nValidation:")
    print(validation_df[TARGET_COLUMN].value_counts())

    print("\nTesting:")
    print(test_df[TARGET_COLUMN].value_counts())

    print("\nTarget rates:")

    print(
        f"Train late rate:       "
        f"{train_df[TARGET_COLUMN].mean():.2%}"
    )

    print(
        f"Validation late rate:  "
        f"{validation_df[TARGET_COLUMN].mean():.2%}"
    )

    print(
        f"Test late rate:        "
        f"{test_df[TARGET_COLUMN].mean():.2%}"
    )

    return train_df, validation_df, test_df


# ============================================================
# Build models
# ============================================================

def build_models(y_train):
    """
    Build baseline models and XGBoost.

    scale_pos_weight is calculated from the training data only.
    """

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE")
    print("=" * 60)

    print(f"Negative samples: {negative_count:,}")
    print(f"Positive samples: {positive_count:,}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    models = {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "xgboost": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=500,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


# ============================================================
# Find best threshold
# ============================================================

def find_best_threshold(y_true, probabilities):
    """
    Find the probability threshold that maximizes F1
    on the validation set.
    """

    thresholds = [x / 100 for x in range(5, 96)]

    best_threshold = 0.5
    best_f1 = 0.0

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

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    results_df = pd.DataFrame(results)

    return best_threshold, best_f1, results_df


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    threshold=0.5,
):
    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    return {
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

    df = load_dataset()

    df = prepare_dataset(df)

    train_df, validation_df, test_df = chronological_split(df)

    # --------------------------------------------------------
    # Features / target
    # --------------------------------------------------------

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)

    # --------------------------------------------------------
    # Build models
    # --------------------------------------------------------

    models = build_models(y_train)

    results = []

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"TRAINING: {model_name.upper()}")
        print("=" * 60)

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Validation probabilities
        # ----------------------------------------------------

        validation_probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        # ----------------------------------------------------
        # Threshold tuning
        # ----------------------------------------------------

        best_threshold, best_validation_f1, _ = (
            find_best_threshold(
                y_validation,
                validation_probabilities,
            )
        )

        print(
            f"\nBest validation threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Best validation F1: "
            f"{best_validation_f1:.4f}"
        )

        # ----------------------------------------------------
        # Final test evaluation
        # ----------------------------------------------------

        metrics = evaluate_model(
            model=model,
            X=X_test,
            y=y_test,
            threshold=best_threshold,
        )

        print("\nFINAL TEST RESULTS")

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall:    {metrics['recall']:.4f}"
        )

        print(
            f"F1:        {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:   {metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC:    {metrics['pr_auc']:.4f}"
        )

        results.append(
            {
                "model": model_name,
                "threshold": best_threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "pr_auc",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = results_df.iloc[0]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Model:     {best_model['model']}"
    )

    print(
        f"Threshold: {best_model['threshold']:.2f}"
    )

    print(
        f"Precision: {best_model['precision']:.4f}"
    )

    print(
        f"Recall:    {best_model['recall']:.4f}"
    )

    print(
        f"F1:        {best_model['f1']:.4f}"
    )

    print(
        f"ROC-AUC:   {best_model['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:    {best_model['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()