"""
Delivery Risk - V7 Preprocessing

Defines:
- Core features
- Target
- Preprocessing pipelines
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Features
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


# ============================================================
# Logistic Regression preprocessing
# ============================================================

def create_logistic_preprocessor():
    """
    Preprocessing for Logistic Regression.

    - Median imputation handles missing values.
    - StandardScaler is important for Logistic Regression.
    """

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

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                CORE_FEATURES,
            )
        ],
        remainder="drop",
    )


# ============================================================
# Tree-based preprocessing
# ============================================================

def create_tree_preprocessor():
    """
    Preprocessing for Random Forest and XGBoost.

    Tree-based models do not require feature scaling.

    We still use median imputation so that the models receive
    a complete numeric matrix.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                CORE_FEATURES,
            )
        ],
        remainder="drop",
    )