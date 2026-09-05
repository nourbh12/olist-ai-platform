import duckdb
import pandas as pd

from pathlib import Path


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

TARGET_COLUMN = "is_late"

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


# ============================================================
# Load delivery risk dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load delivery risk features from the Gold layer.
    """

    query = f"""
        SELECT
            *
        FROM read_parquet(
            '{DELIVERY_FEATURES_PATH.as_posix()}/*.parquet'
        )
    """

    with duckdb.connect() as conn:
        df = conn.execute(query).df()

    return df


# ============================================================
# Basic dataset information
# ============================================================

def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")

    for column in df.columns:
        print(f"- {column}")


# ============================================================
# Missing values
# ============================================================

def inspect_missing_values(df: pd.DataFrame) -> None:
    """
    Display missing values and percentages for ML features
    and the target.
    """

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = (
        df[columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_pct = (
        missing
        / len(df)
        * 100
    )

    missing_df = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": missing_pct,
        }
    )

    print(
        missing_df.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )


# ============================================================
# Target distribution
# ============================================================

def inspect_target(df: pd.DataFrame) -> None:
    """
    Display target class distribution.
    """

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    target = df[TARGET_COLUMN].dropna()

    counts = (
        target
        .value_counts()
        .sort_index()
    )

    percentages = (
        target
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    target_df = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    print(
        target_df.to_string(
            float_format=lambda x: f"{x:.2f}%"
        )
    )

    late_rate = (
        (target == 1).mean()
        * 100
    )

    print(f"\nLate rate: {late_rate:.2f}%")


# ============================================================
# Feature statistics
# ============================================================

def inspect_feature_statistics(
    df: pd.DataFrame,
) -> None:
    """
    Display descriptive statistics for ML features.
    """

    print("\n" + "=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)

    statistics = (
        df[FEATURE_COLUMNS]
        .describe()
        .T
    )

    selected_columns = [
        "count",
        "mean",
        "std",
        "min",
        "50%",
        "max",
    ]

    print(
        statistics[selected_columns].to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# Compare on-time vs late orders
# ============================================================

def compare_classes(
    df: pd.DataFrame,
) -> None:
    """
    Compare average feature values between on-time and
    late orders.
    """

    print("\n" + "=" * 60)
    print("FEATURE COMPARISON: ON-TIME vs LATE")
    print("=" * 60)

    comparison = (
        df
        .groupby(TARGET_COLUMN)[FEATURE_COLUMNS]
        .mean()
        .T
    )

    # Make sure both target classes exist.
    if 0 not in comparison.columns:
        comparison[0] = pd.NA

    if 1 not in comparison.columns:
        comparison[1] = pd.NA

    comparison = comparison[[0, 1]]

    comparison.columns = [
        "on_time",
        "late",
    ]

    comparison["difference"] = (
        comparison["late"]
        - comparison["on_time"]
    )

    comparison["difference_pct"] = (
        comparison["difference"]
        / comparison["on_time"].replace(0, pd.NA)
        * 100
    )

    print(
        comparison.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# Correlation analysis
# ============================================================

def correlation_analysis(
    df: pd.DataFrame,
) -> None:
    """
    Calculate Pearson correlation between each feature
    and the target.
    """

    print("\n" + "=" * 60)
    print("FEATURE CORRELATION WITH TARGET")
    print("=" * 60)

    correlation = (
        df[
            FEATURE_COLUMNS
            + [TARGET_COLUMN]
        ]
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    result = pd.DataFrame(
        {
            "correlation": correlation,
            "absolute_correlation": correlation.abs(),
        }
    )

    print(
        result.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# Seller risk analysis
# ============================================================

def seller_risk_analysis(
    df: pd.DataFrame,
) -> None:
    """
    Analyze seller historical risk features.
    """

    print("\n" + "=" * 60)
    print("SELLER RISK FEATURES")
    print("=" * 60)

    seller_features = [
        "seller_avg_late_rate",
        "seller_max_late_rate",
        "seller_avg_delivery_delay_days",
        "seller_max_delivery_delay_days",
    ]

    available_features = [
        feature
        for feature in seller_features
        if feature in df.columns
    ]

    if not available_features:
        print("No seller risk features found.")
        return

    seller_summary = (
        df
        .groupby(TARGET_COLUMN)[available_features]
        .mean()
        .T
    )

    if 0 not in seller_summary.columns:
        seller_summary[0] = pd.NA

    if 1 not in seller_summary.columns:
        seller_summary[1] = pd.NA

    seller_summary = seller_summary[[0, 1]]

    seller_summary.columns = [
        "on_time",
        "late",
    ]

    seller_summary["difference"] = (
        seller_summary["late"]
        - seller_summary["on_time"]
    )

    print(
        seller_summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# Temporal target analysis
# ============================================================

def temporal_target_analysis(
    df: pd.DataFrame,
) -> None:
    """
    Analyze how the late-order rate changes over time.

    The order purchase timestamp comes from fact_orders and is
    used only for temporal analysis.
    """

    print("\n" + "=" * 60)
    print("TEMPORAL TARGET ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load purchase timestamps
    # --------------------------------------------------------

    query = f"""
        SELECT
            order_id,
            order_purchase_timestamp
        FROM read_parquet(
            '{FACT_ORDERS_PATH.as_posix()}/*.parquet'
        )
    """

    with duckdb.connect() as conn:
        orders = conn.execute(query).df()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    # --------------------------------------------------------
    # Join timestamp with delivery features
    # --------------------------------------------------------

    temporal_df = df.merge(
        orders,
        on="order_id",
        how="inner",
    )

    # Keep only labeled orders.
    temporal_df = temporal_df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    # --------------------------------------------------------
    # Create month
    # --------------------------------------------------------

    temporal_df["month"] = (
        temporal_df["order_purchase_timestamp"]
        .dt.to_period("M")
    )

    # --------------------------------------------------------
    # Monthly statistics
    # --------------------------------------------------------

    monthly = (
        temporal_df
        .groupby("month")
        .agg(
            total_orders=(
                TARGET_COLUMN,
                "count",
            ),
            late_orders=(
                TARGET_COLUMN,
                "sum",
            ),
        )
    )

    monthly["on_time_orders"] = (
        monthly["total_orders"]
        - monthly["late_orders"]
    )

    monthly["late_rate"] = (
        monthly["late_orders"]
        / monthly["total_orders"]
        * 100
    )

    # --------------------------------------------------------
    # Display monthly statistics
    # --------------------------------------------------------

    print("\nMonthly delivery risk:")

    print(
        monthly.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------------
    # Highest late-rate months
    # --------------------------------------------------------

    print("\nHighest late-rate months:")

    highest = (
        monthly
        .sort_values(
            by="late_rate",
            ascending=False,
        )
        .head(10)
    )

    print(
        highest.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------------
    # Lowest late-rate months
    # --------------------------------------------------------

    print("\nLowest late-rate months:")

    lowest = (
        monthly
        .sort_values(
            by="late_rate",
            ascending=True,
        )
        .head(10)
    )

    print(
        lowest.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # --------------------------------------------------------
    # First vs last month
    # --------------------------------------------------------

    if len(monthly) >= 2:

        first_month = monthly.iloc[0]
        last_month = monthly.iloc[-1]

        print("\nTemporal change:")

        print(
            f"First month "
            f"({monthly.index[0]}): "
            f"{first_month['late_rate']:.2f}% late"
        )

        print(
            f"Last month "
            f"({monthly.index[-1]}): "
            f"{last_month['late_rate']:.2f}% late"
        )

        change = (
            last_month["late_rate"]
            - first_month["late_rate"]
        )

        print(
            f"Change: {change:+.2f} percentage points"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 60)
    print("DELIVERY RISK FEATURE EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Dataset overview
    # --------------------------------------------------------

    inspect_dataset(df)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    inspect_missing_values(df)

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    inspect_target(df)

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    inspect_feature_statistics(df)

    # --------------------------------------------------------
    # Keep only labeled orders for target-based analysis
    # --------------------------------------------------------

    labeled_df = (
        df
        .dropna(
            subset=[TARGET_COLUMN]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Compare on-time vs late
    # --------------------------------------------------------

    compare_classes(labeled_df)

    # --------------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------------

    correlation_analysis(labeled_df)

    # --------------------------------------------------------
    # Seller risk analysis
    # --------------------------------------------------------

    seller_risk_analysis(labeled_df)

    # --------------------------------------------------------
    # Temporal target analysis
    # --------------------------------------------------------

    temporal_target_analysis(df)

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE EVALUATION COMPLETED")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()