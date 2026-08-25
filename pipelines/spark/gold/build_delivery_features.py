from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    datediff,
    dayofweek,
    hour,
    max,
    round,
    sum,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldDeliveryFeatures")
        .master("local[*]")
        .getOrCreate()
    )


def validate_one_row_per_order(df, name):
    rows = df.count()
    distinct_orders = df.select("order_id").distinct().count()

    print(
        f"{name}: rows={rows}, "
        f"distinct_orders={distinct_orders}"
    )

    if rows != distinct_orders:
        raise ValueError(
            f"{name} contains duplicate order_id values! "
            f"Rows={rows}, "
            f"Distinct orders={distinct_orders}"
        )


def main():

    spark = create_spark_session()

    try:

        # ==================================================
        # READ INPUT TABLES
        # ==================================================

        fact_orders = spark.read.parquet(
            str(GOLD_DATA_PATH / "fact_orders")
        )

        fact_order_items = spark.read.parquet(
            str(GOLD_DATA_PATH / "fact_order_items")
        )

        seller_performance = spark.read.parquet(
            str(GOLD_DATA_PATH / "seller_performance")
        )

        # ==================================================
        # VALIDATE FACT ORDERS
        # ==================================================

        validate_one_row_per_order(
            fact_orders,
            "fact_orders",
        )

        # ==================================================
        # 1. ORDER ITEM FEATURES
        #
        # One row per order
        # ==================================================

        order_item_features = (
            fact_order_items
            .groupBy("order_id")
            .agg(
                count("*").alias(
                    "order_item_count"
                ),

                round(
                    sum("price"),
                    2,
                ).alias(
                    "order_total_price"
                ),

                round(
                    sum("freight_value"),
                    2,
                ).alias(
                    "order_total_freight"
                ),
            )
        )

        validate_one_row_per_order(
            order_item_features,
            "order_item_features",
        )

        # ==================================================
        # 2. ORDER -> SELLER RELATIONSHIP
        #
        # One order may contain multiple sellers.
        #
        # We keep this table at order + seller level here.
        # ==================================================

        order_sellers = (
            fact_order_items
            .select(
                "order_id",
                "seller_id",
            )
            .dropDuplicates(
                ["order_id", "seller_id"]
            )
        )

        print(
            "Order-seller rows:",
            order_sellers.count(),
        )

        print(
            "Orders represented:",
            order_sellers
            .select("order_id")
            .distinct()
            .count(),
        )

        # ==================================================
        # 3. JOIN HISTORICAL SELLER PERFORMANCE
        #
        # One row per order + seller
        # ==================================================

        order_seller_features = (
            order_sellers
            .join(
                seller_performance.select(
                    "order_id",
                    "seller_id",
                    "seller_order_count",
                    "seller_late_order_count",
                    "seller_late_rate",
                    "seller_avg_delivery_delay_days",
                ),
                on=[
                    "order_id",
                    "seller_id",
                ],
                how="left",
            )
        )

        # Validate order + seller uniqueness
        order_seller_duplicates = (
            order_seller_features
            .groupBy(
                "order_id",
                "seller_id",
            )
            .count()
            .filter(
                col("count") > 1
            )
            .count()
        )

        if order_seller_duplicates > 0:
            raise ValueError(
                "Duplicate order_id + seller_id "
                "combinations detected."
            )

        # ==================================================
        # 4. AGGREGATE SELLER FEATURES TO ORDER LEVEL
        #
        # IMPORTANT:
        # After this point there MUST be exactly
        # one row per order.
        # ==================================================

        order_seller_features = (
            order_seller_features
            .groupBy("order_id")
            .agg(

                # Number of sellers involved in order
                count(
                    "seller_id"
                ).alias(
                    "seller_count"
                ),

                # Historical seller experience
                avg(
                    "seller_order_count"
                ).alias(
                    "seller_avg_order_count"
                ),

                max(
                    "seller_order_count"
                ).alias(
                    "seller_max_order_count"
                ),

                # Historical late rate
                avg(
                    "seller_late_rate"
                ).alias(
                    "seller_avg_late_rate"
                ),

                max(
                    "seller_late_rate"
                ).alias(
                    "seller_max_late_rate"
                ),

                # Historical delivery delay
                avg(
                    "seller_avg_delivery_delay_days"
                ).alias(
                    "seller_avg_delivery_delay_days"
                ),

                max(
                    "seller_avg_delivery_delay_days"
                ).alias(
                    "seller_max_delivery_delay_days"
                ),
            )
        )

        validate_one_row_per_order(
            order_seller_features,
            "order_seller_features",
        )

        # ==================================================
        # 5. FINAL DELIVERY FEATURES
        # ==================================================

        delivery_features = (
            fact_orders

            # ------------------------------
            # Order item features
            # ------------------------------

            .join(
                order_item_features,
                on="order_id",
                how="left",
            )

            # ------------------------------
            # Seller features
            # ------------------------------

            .join(
                order_seller_features,
                on="order_id",
                how="left",
            )

            # ------------------------------
            # Temporal features
            # ------------------------------

            .withColumn(
                "purchase_hour",
                hour(
                    "order_purchase_timestamp"
                ),
            )

            .withColumn(
                "purchase_day_of_week",
                dayofweek(
                    "order_purchase_timestamp"
                ),
            )

            # ------------------------------
            # Estimated delivery duration
            #
            # estimated date - purchase date
            # ------------------------------

            .withColumn(
                "estimated_delivery_duration_days",
                datediff(
                    col(
                        "order_estimated_delivery_date"
                    ),
                    col(
                        "order_purchase_timestamp"
                    ),
                ),
            )

            # ------------------------------
            # Round numerical features
            # ------------------------------

            .withColumn(
                "seller_avg_order_count",
                round(
                    col(
                        "seller_avg_order_count"
                    ),
                    2,
                ),
            )

            .withColumn(
                "seller_avg_late_rate",
                round(
                    col(
                        "seller_avg_late_rate"
                    ),
                    4,
                ),
            )

            .withColumn(
                "seller_max_late_rate",
                round(
                    col(
                        "seller_max_late_rate"
                    ),
                    4,
                ),
            )

            .withColumn(
                "seller_avg_delivery_delay_days",
                round(
                    col(
                        "seller_avg_delivery_delay_days"
                    ),
                    2,
                ),
            )

            .withColumn(
                "seller_max_delivery_delay_days",
                round(
                    col(
                        "seller_max_delivery_delay_days"
                    ),
                    2,
                ),
            )

            # ==================================================
            # FINAL COLUMNS
            # ==================================================

            .select(
                "order_id",
                "customer_id",

                # Order features
                "order_status",
                "order_item_count",
                "order_total_price",
                "order_total_freight",

                # Temporal features
                "purchase_hour",
                "purchase_day_of_week",

                # Delivery features
                "estimated_delivery_duration_days",

                # Seller features
                "seller_count",
                "seller_avg_order_count",
                "seller_max_order_count",
                "seller_avg_late_rate",
                "seller_max_late_rate",
                "seller_avg_delivery_delay_days",
                "seller_max_delivery_delay_days",

                # Target
                "is_late",
            )
        )

        # ==================================================
        # FINAL VALIDATION
        # ==================================================

        validate_one_row_per_order(
            delivery_features,
            "FINAL delivery_features",
        )

        # ==================================================
        # SHOW DATA
        # ==================================================

        delivery_features.printSchema()

        delivery_features.show(
            10,
            truncate=False,
        )

        # ==================================================
        # WRITE
        # ==================================================

        output_path = (
            GOLD_DATA_PATH
            / "delivery_features"
        )

        (
            delivery_features
            .write
            .mode("overwrite")
            .parquet(
                str(output_path)
            )
        )

        print(
            f"Written to: {output_path}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()