from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "order_items"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "order_items"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("OlistSilverOrderItems")
        .master("local[*]")
        .getOrCreate()
    )


def transform_order_items(spark: SparkSession):

    print(f"Reading Bronze order_items from: {BRONZE_PATH}")

    df = spark.read.parquet(str(BRONZE_PATH))

    # ---------------------------------------------------------
    # 1. Convert shipping date
    # ---------------------------------------------------------

    df = df.withColumn(
        "shipping_limit_date",
        F.to_timestamp(F.col("shipping_limit_date"))
    )

    # ---------------------------------------------------------
    # 2. Cast numeric columns
    # ---------------------------------------------------------

    df = (
        df
        .withColumn("price", F.col("price").cast("double"))
        .withColumn("freight_value", F.col("freight_value").cast("double"))
        .withColumn("order_item_id", F.col("order_item_id").cast("integer"))
    )

    # ---------------------------------------------------------
    # 3. Remove exact duplicates
    # ---------------------------------------------------------

    df = df.dropDuplicates()

    # ---------------------------------------------------------
    # 4. Basic validation / cleaning
    # ---------------------------------------------------------

    # Price and freight should not be negative
    df = df.filter(
        (F.col("price") >= 0) &
        (F.col("freight_value") >= 0)
    )

    # ---------------------------------------------------------
    # 5. Add total item value
    # ---------------------------------------------------------

    df = df.withColumn(
        "total_item_value",
        F.col("price") + F.col("freight_value")
    )

    # ---------------------------------------------------------
    # 6. Write Silver
    # ---------------------------------------------------------

    (
        df.write
        .mode("overwrite")
        .parquet(str(SILVER_PATH))
    )

    print(f"Silver order_items written to: {SILVER_PATH}")

    return df


def main():

    spark = create_spark_session()

    try:

        df = transform_order_items(spark)

        print("\nSilver schema:")
        df.printSchema()

        print("\nSample:")
        df.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()