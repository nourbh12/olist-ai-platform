from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SILVER_DATA_PATH = PROJECT_ROOT / "data" / "silver"
GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldFactOrderItems")
        .master("local[*]")
        .getOrCreate()
    )


def main():

    spark = create_spark_session()

    try:

        orders = spark.read.parquet(
            str(SILVER_DATA_PATH / "orders")
        )

        order_items = spark.read.parquet(
            str(SILVER_DATA_PATH / "order_items")
        )

        fact_order_items = (
            order_items
            .join(
                orders.select(
                    "order_id",
                    "customer_id",
                    "order_status",
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ),
                on="order_id",
                how="left",
            )
            .select(
                "order_id",
                "order_item_id",
                "product_id",
                "seller_id",
                "customer_id",
                "order_status",
                "shipping_limit_date",
                "price",
                "freight_value",
                "total_item_value",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            )
        )

        print(f"Rows: {fact_order_items.count()}")

        output_path = GOLD_DATA_PATH / "fact_order_items"

        (
            fact_order_items.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        print(f"Written to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()