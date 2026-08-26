from pathlib import Path

from pyspark.sql import SparkSession

from checks import (
    check_allowed_values,
    check_between,
    check_columns,
    check_non_negative,
    check_not_null,
    check_unique,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldDataQuality")
        .master("local[*]")
        .getOrCreate()
    )


def validate_fact_orders(spark):

    dataset_name = "fact_orders"

    df = spark.read.parquet(
        str(GOLD_DATA_PATH / dataset_name)
    )

    print(f"\nValidating {dataset_name}...")
    print(f"Rows: {df.count()}")

    check_columns(
        df,
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "delivery_duration_days",
            "delivery_delay_days",
            "is_late",
        ],
        dataset_name,
    )

    check_not_null(
        df,
        "order_id",
        dataset_name,
    )

    check_unique(
        df,
        ["order_id"],
        dataset_name,
    )

    check_allowed_values(
        df,
        "is_late",
        [0, 1],
        dataset_name,
    )

    check_non_negative(
        df,
        "delivery_duration_days",
        dataset_name,
    )

    print(f"✓ {dataset_name} passed")


def validate_fact_order_items(spark):

    dataset_name = "fact_order_items"

    df = spark.read.parquet(
        str(GOLD_DATA_PATH / dataset_name)
    )

    print(f"\nValidating {dataset_name}...")
    print(f"Rows: {df.count()}")

    check_columns(
        df,
        [
            "order_id",
            "seller_id",
            "price",
            "freight_value",
        ],
        dataset_name,
    )

    check_not_null(
        df,
        "order_id",
        dataset_name,
    )

    check_not_null(
        df,
        "seller_id",
        dataset_name,
    )

    check_non_negative(
        df,
        "price",
        dataset_name,
    )

    check_non_negative(
        df,
        "freight_value",
        dataset_name,
    )

    print(f"✓ {dataset_name} passed")


def validate_seller_performance(spark):

    dataset_name = "seller_performance"

    df = spark.read.parquet(
        str(GOLD_DATA_PATH / dataset_name)
    )

    print(f"\nValidating {dataset_name}...")
    print(f"Rows: {df.count()}")

    check_columns(
        df,
        [
            "order_id",
            "seller_id",
            "seller_order_count",
            "seller_late_order_count",
            "seller_late_rate",
            "seller_avg_delivery_delay_days",
        ],
        dataset_name,
    )

    check_not_null(
        df,
        "order_id",
        dataset_name,
    )

    check_not_null(
        df,
        "seller_id",
        dataset_name,
    )

    check_unique(
        df,
        ["order_id", "seller_id"],
        dataset_name,
    )

    check_non_negative(
        df,
        "seller_order_count",
        dataset_name,
    )

    check_non_negative(
        df,
        "seller_late_order_count",
        dataset_name,
    )

    check_between(
        df,
        "seller_late_rate",
        0.0,
        1.0,
        dataset_name,
    )

    print(f"✓ {dataset_name} passed")


def validate_delivery_features(spark):

    dataset_name = "delivery_features"

    df = spark.read.parquet(
        str(GOLD_DATA_PATH / dataset_name)
    )

    print(f"\nValidating {dataset_name}...")
    print(f"Rows: {df.count()}")

    check_columns(
        df,
        [
            "order_id",
            "customer_id",
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
            "is_late",
        ],
        dataset_name,
    )

    # One row per order
    check_not_null(
        df,
        "order_id",
        dataset_name,
    )

    check_unique(
        df,
        ["order_id"],
        dataset_name,
    )

    # Numerical sanity checks
    check_non_negative(
        df,
        "order_item_count",
        dataset_name,
    )

    check_non_negative(
        df,
        "order_total_price",
        dataset_name,
    )

    check_non_negative(
        df,
        "order_total_freight",
        dataset_name,
    )

    check_non_negative(
        df,
        "seller_count",
        dataset_name,
    )

    check_non_negative(
        df,
        "seller_avg_order_count",
        dataset_name,
    )

    check_non_negative(
        df,
        "seller_max_order_count",
        dataset_name,
    )

    # Seller late rates must be probabilities
    check_between(
        df,
        "seller_avg_late_rate",
        0.0,
        1.0,
        dataset_name,
    )

    check_between(
        df,
        "seller_max_late_rate",
        0.0,
        1.0,
        dataset_name,
    )

    # Target can legitimately be NULL because some orders
    # don't have a known delivery outcome.
    check_allowed_values(
        df,
        "is_late",
        [0, 1],
        dataset_name,
    )

    print(f"✓ {dataset_name} passed")


def main():

    spark = create_spark_session()

    try:

        print("=" * 60)
        print("GOLD DATA QUALITY VALIDATION")
        print("=" * 60)

        validate_fact_orders(spark)

        validate_fact_order_items(spark)

        validate_seller_performance(spark)

        validate_delivery_features(spark)

        print("\n" + "=" * 60)
        print("ALL GOLD DATA QUALITY CHECKS PASSED ✓")
        print("=" * 60)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()