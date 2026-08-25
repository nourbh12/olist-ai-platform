from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    lag,
    round,
    sum,
)
from pyspark.sql.window import Window


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldSellerPerformance")
        .master("local[*]")
        .getOrCreate()
    )


def main():

    spark = create_spark_session()

    try:

        # --------------------------------------------------
        # Read gold tables
        # --------------------------------------------------

        fact_orders = spark.read.parquet(
            str(GOLD_DATA_PATH / "fact_orders")
        )

        fact_order_items = spark.read.parquet(
            str(GOLD_DATA_PATH / "fact_order_items")
        )

        # --------------------------------------------------
        # Build seller-order relationship
        #
        # One row = one seller associated with one order
        # --------------------------------------------------

        seller_orders = (
            fact_order_items
            .select(
                "order_id",
                "seller_id",
            )
            .dropDuplicates(
                ["order_id", "seller_id"]
            )
        )

        # --------------------------------------------------
        # Join order delivery information
        # --------------------------------------------------

        seller_orders = (
            seller_orders
            .join(
                fact_orders.select(
                    "order_id",
                    "order_purchase_timestamp",
                    "delivery_delay_days",
                    "is_late",
                ),
                on="order_id",
                how="inner",
            )
        )

        # --------------------------------------------------
        # Historical seller features
        #
        # IMPORTANT:
        # We only use orders BEFORE the current order.
        #
        # This prevents target leakage.
        # --------------------------------------------------

        seller_window = (
            Window
            .partitionBy("seller_id")
            .orderBy("order_purchase_timestamp")
            .rowsBetween(
                Window.unboundedPreceding,
                -1,
            )
        )

        seller_performance = (
            seller_orders

            # Number of previous orders
            .withColumn(
                "seller_order_count",
                count("*").over(seller_window),
            )

            # Number of previous late orders
            .withColumn(
                "seller_late_order_count",
                sum(
                    col("is_late")
                ).over(seller_window),
            )

            # Historical late rate
            .withColumn(
                "seller_late_rate",
                col("seller_late_order_count")
                / col("seller_order_count"),
            )

            # Historical average delivery delay
            .withColumn(
                "seller_avg_delivery_delay_days",
                avg(
                    col("delivery_delay_days")
                ).over(seller_window),
            )

            .select(
                "order_id",
                "seller_id",
                "order_purchase_timestamp",
                "seller_order_count",
                "seller_late_order_count",
                "seller_late_rate",
                "seller_avg_delivery_delay_days",
            )
        )

        # --------------------------------------------------
        # Round numerical features
        # --------------------------------------------------

        seller_performance = (
            seller_performance
            .withColumn(
                "seller_late_rate",
                round(
                    col("seller_late_rate"),
                    4,
                ),
            )
            .withColumn(
                "seller_avg_delivery_delay_days",
                round(
                    col("seller_avg_delivery_delay_days"),
                    2,
                ),
            )
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        print(
            f"Seller-order rows: "
            f"{seller_performance.count()}"
        )

        seller_performance.printSchema()

        seller_performance.orderBy(
            "seller_id",
            "order_purchase_timestamp",
        ).show(
            20,
            truncate=False,
        )

        # --------------------------------------------------
        # Write
        # --------------------------------------------------

        output_path = (
            GOLD_DATA_PATH
            / "seller_performance"
        )

        (
            seller_performance.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        print(
            f"Written to: {output_path}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()