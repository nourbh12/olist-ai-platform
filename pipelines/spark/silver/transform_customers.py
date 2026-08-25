from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "customers"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "customers"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("OlistSilverCustomers")
        .master("local[*]")
        .getOrCreate()
    )


def transform_customers(spark):

    df = spark.read.parquet(str(BRONZE_PATH))

    df = (
        df
        .withColumn(
            "customer_city",
            F.trim(F.lower(F.col("customer_city")))
        )
        .withColumn(
            "customer_state",
            F.upper(F.trim(F.col("customer_state")))
        )
        .withColumn(
            "customer_zip_code_prefix",
            F.col("customer_zip_code_prefix").cast("integer")
        )
    )

    df = df.dropDuplicates(["customer_id"])

    df.write.mode("overwrite").parquet(str(SILVER_PATH))

    return df


def main():

    spark = create_spark_session()

    try:
        df = transform_customers(spark)

        print("Silver customers:")
        df.printSchema()
        df.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()