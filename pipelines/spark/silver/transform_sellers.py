from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "sellers"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "sellers"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("OlistSilverSellers")
        .master("local[*]")
        .getOrCreate()
    )


def transform_sellers(spark):

    df = spark.read.parquet(str(BRONZE_PATH))

    df = (
        df
        .withColumn(
            "seller_city",
            F.trim(F.lower(F.col("seller_city")))
        )
        .withColumn(
            "seller_state",
            F.upper(F.trim(F.col("seller_state")))
        )
        .withColumn(
            "seller_zip_code_prefix",
            F.col("seller_zip_code_prefix").cast("integer")
        )
    )

    df = df.dropDuplicates(["seller_id"])

    df.write.mode("overwrite").parquet(str(SILVER_PATH))

    return df


def main():

    spark = create_spark_session()

    try:
        df = transform_sellers(spark)

        print("Silver sellers:")
        df.printSchema()
        df.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()