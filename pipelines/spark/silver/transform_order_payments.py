from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "order_payments"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "order_payments"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("OlistSilverPayments")
        .master("local[*]")
        .getOrCreate()
    )


def transform_payments(spark):

    df = spark.read.parquet(str(BRONZE_PATH))

    df = (
        df
        .withColumn(
            "payment_sequential",
            F.col("payment_sequential").cast("integer")
        )
        .withColumn(
            "payment_installments",
            F.col("payment_installments").cast("integer")
        )
        .withColumn(
            "payment_value",
            F.col("payment_value").cast("double")
        )
        .withColumn(
            "payment_type",
            F.trim(F.lower(F.col("payment_type")))
        )
    )

    df = df.dropDuplicates()

    df.write.mode("overwrite").parquet(str(SILVER_PATH))

    return df


def main():

    spark = create_spark_session()

    try:
        df = transform_payments(spark)

        print("Silver payments:")
        df.printSchema()
        df.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()