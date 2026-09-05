from pathlib import Path

import duckdb
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

DELIVERY_RISK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DELIVERY_RISK_DIR.parents[2]

DELIVERY_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "delivery_features"
)


# ============================================================
# Features
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


# ============================================================
# Load dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load delivery risk features from the Gold Parquet dataset.
    """

    query = f"""
        SELECT
            {", ".join(FEATURE_COLUMNS)},
            {TARGET_COLUMN}
        FROM read_parquet('{DELIVERY_FEATURES_PATH}/*.parquet')
    """

    with duckdb.connect() as conn:
        df = conn.execute(query).df()

    return df


# ============================================================
# Split features / target
# ============================================================

def prepare_data(df: pd.DataFrame):
    """
    Separate input features X from target y.
    """

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# Preprocessing pipeline
# ============================================================

def create_preprocessor() -> ColumnTransformer:
    """
    Create the preprocessing pipeline for numerical features.

    Steps:
    1. Impute missing values using the median.
    2. Standardize numerical features.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("DELIVERY RISK PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

    X, y = prepare_data(df)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\n=== TARGET DISTRIBUTION ===")

    print(
        y.value_counts()
        .sort_index()
    )

    print("\nTarget percentages:")

    print(
        (y.value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n=== MISSING VALUES ===")

    missing = X.isnull().sum()

    print(
        missing[missing > 0]
        .sort_values(ascending=False)
    )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = create_preprocessor()

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    print("\n=== PROCESSED DATA ===")

    print(f"Processed shape: {X_processed.shape}")

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()