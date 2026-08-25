from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze"
SILVER_PATH = PROJECT_ROOT / "data" / "silver"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("OlistSilverOrders")
        .master("local[*]")
        .getOrCreate()
    )


def transform_orders(spark: SparkSession):
    input_path = BRONZE_PATH / "orders"
    output_path = SILVER_PATH / "orders"

    print(f"Reading Bronze orders from: {input_path}")

    orders = spark.read.parquet(str(input_path))

    # 1. Convert timestamp columns

    timestamp_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in timestamp_columns:
        orders = orders.withColumn(
            column,
            F.to_timestamp(F.col(column))
        )


    # 2. Remove duplicate order IDs

    orders = orders.dropDuplicates(["order_id"])

  
    # 3. Delivery duration

    orders = orders.withColumn(
        "delivery_duration_days",
        F.datediff(
            F.col("order_delivered_customer_date"),
            F.col("order_purchase_timestamp")
        )
    )


    # 4. Delivery delay

    orders = orders.withColumn(
        "delivery_delay_days",
        F.datediff(
            F.col("order_delivered_customer_date"),
            F.col("order_estimated_delivery_date")
        )
    )


    # 5. Late delivery flag

    orders = orders.withColumn(
        "is_late",
        F.when(
            F.col("delivery_delay_days") > 0,
            1
        ).otherwise(0)
    )


    # 6. Write Silver

    (
        orders.write
        .mode("overwrite")
        .parquet(str(output_path))
    )

    print(f"Silver orders written to: {output_path}")

    return orders


def main():
    spark = create_spark_session()

    try:
        orders = transform_orders(spark)

        print("\nSilver schema:")
        orders.printSchema()

        print("\nSample:")
        orders.select(
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "delivery_duration_days",
            "delivery_delay_days",
            "is_late",
        ).show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()