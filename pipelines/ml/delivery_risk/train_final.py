import json
import duckdb
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "delivery_risk"
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

# Selected through V12 threshold stability analysis
FINAL_THRESHOLD = 0.07

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

def load_dataset() -> pd.DataFrame:

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

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

    print(
        f"Loaded rows: {len(df):,}"
    )

    return df


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)

    columns = CORE_FEATURES + [
        TARGET,
        "order_purchase_timestamp",
    ]

    df = df[columns].copy()

    # The target is required for training
    before_target_drop = len(df)

    df = df.dropna(
        subset=[TARGET]
    )

    removed_target = (
        before_target_drop - len(df)
    )

    print(
        f"Removed rows with missing target: "
        f"{removed_target:,}"
    )

    # Timestamp is kept for metadata / auditability.
    # It is NOT used as a model feature.
    before_timestamp_drop = len(df)

    df = df.dropna(
        subset=[
            "order_purchase_timestamp"
        ]
    )

    removed_timestamp = (
        before_timestamp_drop - len(df)
    )

    print(
        f"Removed rows with missing timestamp: "
        f"{removed_timestamp:,}"
    )

    df[TARGET] = df[TARGET].astype(int)

    # Keep chronological ordering for reproducibility
    df = (
        df
        .sort_values(
            "order_purchase_timestamp"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Build production pipeline
# ============================================================

def build_model() -> Pipeline:

    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

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

    classifier = LogisticRegression(
        class_weight=None,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessing,
            ),
            (
                "model",
                classifier,
            ),
        ]
    )

    return pipeline


# ============================================================
# Train production model
# ============================================================

def train_model(
    model: Pipeline,
    df: pd.DataFrame,
) -> Pipeline:

    print("\n" + "=" * 60)
    print("TRAINING FINAL PRODUCTION MODEL")
    print("=" * 60)

    X = df[
        CORE_FEATURES
    ]

    y = df[
        TARGET
    ]

    print(
        f"Training rows: {len(X):,}"
    )

    print(
        f"Number of features: "
        f"{len(CORE_FEATURES)}"
    )

    print(
        f"Late orders: {y.sum():,}"
    )

    print(
        f"On-time orders: "
        f"{(y == 0).sum():,}"
    )

    print(
        f"Late rate: {y.mean():.2%}"
    )

    print("\nFeatures:")

    for feature in CORE_FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        "\nClass weighting: None"
    )

    model.fit(
        X,
        y,
    )

    print(
        "\nTraining complete."
    )

    return model


# ============================================================
# Inspect trained model
# ============================================================

def inspect_model(
    model: Pipeline,
):

    classifier = model.named_steps[
        "model"
    ]

    coefficients = (
        classifier.coef_[0]
    )

    coefficient_df = pd.DataFrame(
        {
            "feature": CORE_FEATURES,
            "coefficient": coefficients,
        }
    )

    coefficient_df[
        "abs_coefficient"
    ] = coefficient_df[
        "coefficient"
    ].abs()

    coefficient_df = (
        coefficient_df
        .sort_values(
            "abs_coefficient",
            ascending=False,
        )
        .drop(
            columns=["abs_coefficient"]
        )
    )

    print("\n" + "=" * 60)
    print("MODEL COEFFICIENTS")
    print("=" * 60)

    print(
        coefficient_df.to_string(
            index=False
        )
    )

    print(
        "\nPositive coefficient → "
        "higher predicted late risk."
    )

    print(
        "Negative coefficient → "
        "lower predicted late risk."
    )

    return coefficient_df


# ============================================================
# Save model
# ============================================================

def save_model(
    model: Pipeline,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "delivery_risk_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(model_path)

    return model_path


# ============================================================
# Save metadata
# ============================================================

def save_metadata(
    df: pd.DataFrame,
    model_path: Path,
    coefficient_df: pd.DataFrame,
):

    metadata = {
        "model_name": "delivery_risk_model",
        "model_type": "LogisticRegression",
        "task": "binary_classification",
        "target": TARGET,
        "threshold": FINAL_THRESHOLD,
        "features": CORE_FEATURES,
        "preprocessing": {
            "missing_values": "median_imputation",
            "scaling": "StandardScaler",
        },
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "training": {
            "rows": int(len(df)),
            "late_orders": int(
                df[TARGET].sum()
            ),
            "on_time_orders": int(
                (df[TARGET] == 0).sum()
            ),
            "late_rate": float(
                df[TARGET].mean()
            ),
            "start_date": (
                df[
                    "order_purchase_timestamp"
                ]
                .min()
                .isoformat()
            ),
            "end_date": (
                df[
                    "order_purchase_timestamp"
                ]
                .max()
                .isoformat()
            ),
        },
        "validation": {
            "evaluation_script": "evaluate_v13.py",
            "threshold_selection": "analyze_threshold_stability.py",
            "test_roc_auc": 0.6984,
            "test_pr_auc": 0.0633,
            "test_precision": 0.0489,
            "test_recall": 0.8309,
            "test_f1": 0.0924,
            "test_brier_score": 0.0352,
            "test_flagged_rate": 0.5931,
            "test_lift": 1.40,
        },
        "model_path": str(
            model_path
        ),
        "coefficients": (
            coefficient_df
            .to_dict(orient="records")
        ),
    }

    metadata_path = (
        MODEL_DIR
        / "delivery_risk_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        metadata_path
    )

    return metadata_path


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("DELIVERY RISK - FINAL PRODUCTION MODEL")
    print("V14")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_dataset(
        df
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TRAINING DATASET")
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

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Train on ALL historical data
    # --------------------------------------------------------

    model = train_model(
        model,
        df,
    )

    # --------------------------------------------------------
    # Inspect
    # --------------------------------------------------------

    coefficient_df = inspect_model(
        model
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model_path = save_model(
        model
    )

    metadata_path = save_metadata(
        df,
        model_path,
        coefficient_df,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("V14 COMPLETE")
    print("=" * 60)

    print(
        "\nProduction model:"
    )

    print(
        f"  {model_path}"
    )

    print(
        "\nMetadata:"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        "\nFeatures:"
    )

    for feature in CORE_FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        f"\nDecision threshold: "
        f"{FINAL_THRESHOLD:.2f}"
    )

    print(
        "\nThe production model was trained "
        "on all available labeled historical data."
    )

    print(
        "\nV13 remains the official untouched-test "
        "benchmark."
    )


if __name__ == "__main__":
    main()