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
# Load dataset
# ============================================================

def load_dataset():
    print("=" * 60)
    print("LOADING DELIVERY RISK DATASET")
    print("=" * 60)

    connection = duckdb.connect()

    try:
        query = f"""
        SELECT
            d.*,
            f.order_purchase_timestamp
        FROM read_parquet(
            '{DELIVERY_FEATURES_PATH.as_posix()}/*.parquet'
        ) AS d
        INNER JOIN read_parquet(
            '{FACT_ORDERS_PATH.as_posix()}/*.parquet'
        ) AS f
            ON d.order_id = f.order_id
        """

        df = connection.execute(query).df()

    finally:
        connection.close()

    print(f"Rows loaded: {len(df):,}")

    return df


# ============================================================
# Prepare dataset
# ============================================================

def prepare_dataset(df):
    print("\n" + "=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    initial_rows = len(df)

    missing_target = df[TARGET_COLUMN].isna().sum()

    df = df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    print(f"Initial rows:         {initial_rows:,}")
    print(f"Missing target rows:  {missing_target:,}")
    print(f"Rows after filtering: {len(df):,}")

    # Convert timestamp
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    # Remove invalid timestamps
    invalid_timestamps = (
        df["order_purchase_timestamp"].isna().sum()
    )

    if invalid_timestamps > 0:
        print(
            f"Invalid timestamp rows removed: "
            f"{invalid_timestamps:,}"
        )

        df = df.dropna(
            subset=["order_purchase_timestamp"]
        ).copy()

    # Convert target to integer
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(int)
    )

    # Sort chronologically
    df = (
        df.sort_values(
            "order_purchase_timestamp"
        )
        .reset_index(drop=True)
    )

    # Create month
    df["purchase_month"] = (
        df["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    return df


# ============================================================
# Add chronological period labels
# ============================================================

def add_period_labels(df):
    """
    Reproduce the same chronological 60/20/20
    split used by train.py.
    """

    df = df.copy()

    n = len(df)

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    df["period"] = "test"

    df.loc[
        :train_end - 1,
        "period"
    ] = "train"

    df.loc[
        train_end:validation_end - 1,
        "period"
    ] = "validation"

    return df


# ============================================================
# Period summary
# ============================================================

def analyze_period_summary(df):
    print("\n" + "=" * 60)
    print("PERIOD SUMMARY")
    print("=" * 60)

    summary = (
        df.groupby("period")
        .agg(
            rows=(TARGET_COLUMN, "size"),
            late_orders=(TARGET_COLUMN, "sum"),
            late_rate=(TARGET_COLUMN, "mean"),
            start_date=(
                "order_purchase_timestamp",
                "min",
            ),
            end_date=(
                "order_purchase_timestamp",
                "max",
            ),
        )
        .reindex(
            ["train", "validation", "test"]
        )
        .reset_index()
    )

    summary["late_rate"] = (
        summary["late_rate"] * 100
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "late_rate": "{:.2f}%".format
            },
        )
    )

    return summary


# ============================================================
# Monthly target stability
# ============================================================

def analyze_monthly_target(df):
    print("\n" + "=" * 60)
    print("MONTHLY TARGET STABILITY")
    print("=" * 60)

    monthly = (
        df.groupby("purchase_month")
        .agg(
            total_orders=(TARGET_COLUMN, "size"),
            late_orders=(TARGET_COLUMN, "sum"),
            late_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )

    monthly["late_rate"] = (
        monthly["late_rate"] * 100
    )

    print(
        monthly.to_string(
            index=False,
            formatters={
                "late_rate": "{:.2f}%".format
            },
        )
    )

    return monthly


# ============================================================
# Feature stability by period
# ============================================================

def analyze_feature_period_stability(df):
    print("\n" + "=" * 60)
    print("FEATURE STABILITY BY PERIOD")
    print("=" * 60)

    rows = []

    periods = [
        "train",
        "validation",
        "test",
    ]

    for feature in FEATURE_COLUMNS:

        for period in periods:

            values = df.loc[
                df["period"] == period,
                feature,
            ]

            non_null_values = values.dropna()

            if non_null_values.empty:
                continue

            rows.append(
                {
                    "feature": feature,
                    "period": period,
                    "count": len(non_null_values),
                    "missing_pct": (
                        values.isna().mean() * 100
                    ),
                    "mean": non_null_values.mean(),
                    "median": non_null_values.median(),
                    "std": non_null_values.std(),
                    "min": non_null_values.min(),
                    "max": non_null_values.max(),
                }
            )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


# ============================================================
# Train vs test distribution shift
# ============================================================

def analyze_distribution_shift(df):
    print("\n" + "=" * 60)
    print("TRAIN VS TEST DISTRIBUTION SHIFT")
    print("=" * 60)

    train_df = df[
        df["period"] == "train"
    ]

    test_df = df[
        df["period"] == "test"
    ]

    rows = []

    for feature in FEATURE_COLUMNS:

        train_values = (
            train_df[feature]
            .dropna()
        )

        test_values = (
            test_df[feature]
            .dropna()
        )

        if train_values.empty or test_values.empty:
            continue

        train_mean = train_values.mean()
        test_mean = test_values.mean()

        train_median = train_values.median()
        test_median = test_values.median()

        mean_diff_pct = (
            abs(test_mean - train_mean)
            / max(abs(train_mean), 1e-8)
            * 100
        )

        median_diff_pct = (
            abs(test_median - train_median)
            / max(abs(train_median), 1e-8)
            * 100
        )

        rows.append(
            {
                "feature": feature,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "mean_diff_pct": mean_diff_pct,
                "train_median": train_median,
                "test_median": test_median,
                "median_diff_pct": median_diff_pct,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        print("No valid features for distribution analysis.")
        return result

    result = (
        result
        .sort_values(
            "mean_diff_pct",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


# ============================================================
# Feature / target relationship
# ============================================================

def analyze_feature_target_relationship(df):
    print("\n" + "=" * 60)
    print("FEATURE / TARGET RELATIONSHIP BY PERIOD")
    print("=" * 60)

    rows = []

    periods = [
        "train",
        "validation",
        "test",
    ]

    for feature in FEATURE_COLUMNS:

        for period in periods:

            subset = (
                df.loc[
                    df["period"] == period,
                    [feature, TARGET_COLUMN],
                ]
                .dropna()
            )

            if len(subset) < 10:
                continue

            correlation = subset[feature].corr(
                subset[TARGET_COLUMN]
            )

            on_time = subset.loc[
                subset[TARGET_COLUMN] == 0,
                feature,
            ]

            late = subset.loc[
                subset[TARGET_COLUMN] == 1,
                feature,
            ]

            on_time_mean = (
                on_time.mean()
                if not on_time.empty
                else float("nan")
            )

            late_mean = (
                late.mean()
                if not late.empty
                else float("nan")
            )

            rows.append(
                {
                    "feature": feature,
                    "period": period,
                    "correlation": correlation,
                    "on_time_mean": on_time_mean,
                    "late_mean": late_mean,
                    "difference": (
                        late_mean - on_time_mean
                    ),
                }
            )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


# ============================================================
# Seller risk stability
# ============================================================

def analyze_seller_risk(df):
    print("\n" + "=" * 60)
    print("SELLER RISK STABILITY")
    print("=" * 60)

    seller_features = [
        "seller_avg_late_rate",
        "seller_max_late_rate",
        "seller_avg_delivery_delay_days",
        "seller_max_delivery_delay_days",
    ]

    rows = []

    periods = [
        "train",
        "validation",
        "test",
    ]

    for feature in seller_features:

        for period in periods:

            subset = (
                df.loc[
                    df["period"] == period,
                    [feature, TARGET_COLUMN],
                ]
                .dropna()
            )

            if subset.empty:
                continue

            on_time = subset.loc[
                subset[TARGET_COLUMN] == 0,
                feature,
            ]

            late = subset.loc[
                subset[TARGET_COLUMN] == 1,
                feature,
            ]

            on_time_mean = (
                on_time.mean()
                if not on_time.empty
                else float("nan")
            )

            late_mean = (
                late.mean()
                if not late.empty
                else float("nan")
            )

            correlation = subset[feature].corr(
                subset[TARGET_COLUMN]
            )

            rows.append(
                {
                    "feature": feature,
                    "period": period,
                    "on_time_mean": on_time_mean,
                    "late_mean": late_mean,
                    "difference": (
                        late_mean - on_time_mean
                    ),
                    "correlation": correlation,
                }
            )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


# ============================================================
# Monthly seller risk
# ============================================================

def analyze_monthly_seller_risk(df):
    print("\n" + "=" * 60)
    print("MONTHLY SELLER RISK")
    print("=" * 60)

    result = (
        df.groupby("purchase_month")
        .agg(
            orders=(TARGET_COLUMN, "size"),
            late_orders=(TARGET_COLUMN, "sum"),
            late_rate=(TARGET_COLUMN, "mean"),
            avg_seller_late_rate=(
                "seller_avg_late_rate",
                "mean",
            ),
            avg_seller_delay=(
                "seller_avg_delivery_delay_days",
                "mean",
            ),
        )
        .reset_index()
    )

    result["late_rate"] = (
        result["late_rate"] * 100
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "late_rate": "{:.2f}%".format
            },
        )
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    # Load data
    df = load_dataset()

    # Prepare data
    df = prepare_dataset(df)

    # Reproduce chronological split
    df = add_period_labels(df)

    # Run analyses
    analyze_period_summary(df)

    analyze_monthly_target(df)

    analyze_feature_period_stability(df)

    analyze_distribution_shift(df)

    analyze_feature_target_relationship(df)

    analyze_seller_risk(df)

    analyze_monthly_seller_risk(df)

    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()