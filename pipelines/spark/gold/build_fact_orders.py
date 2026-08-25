from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    datediff,
    when,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SILVER_DATA_PATH = PROJECT_ROOT / "data" / "silver"
GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldFactOrders")
        .master("local[*]")
        .getOrCreate()
    )


def main():

    spark = create_spark_session()

    try:
        orders = spark.read.parquet(
            str(SILVER_DATA_PATH / "orders")
        )

        fact_orders = (
            orders
            .withColumn(
                "delivery_delay_days",
                datediff(
                    col("order_delivered_customer_date"),
                    col("order_estimated_delivery_date"),
                ),
            )
            .withColumn(
                "is_late",
                when(col("delivery_delay_days").isNull(), None)
                .when(col("delivery_delay_days") > 0, 1)
                .otherwise(0),
            )
            .withColumn(
                "delivery_duration_days",
                datediff(
                    col("order_delivered_customer_date"),
                    col("order_purchase_timestamp"),
                ),
            )
            .select(
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "delivery_duration_days",
                "delivery_delay_days",
                "is_late",
            )
        )

        print(f"Rows: {fact_orders.count()}")

        output_path = GOLD_DATA_PATH / "fact_orders"

        (
            fact_orders.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        print(f"Written to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()