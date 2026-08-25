from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "geolocation"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "geolocation"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("OlistSilverGeolocation")
        .master("local[*]")
        .getOrCreate()
    )


def transform_geolocation(spark):

    df = spark.read.parquet(str(BRONZE_PATH))

    df = (
        df
        .withColumn(
            "geolocation_zip_code_prefix",
            F.col("geolocation_zip_code_prefix").cast("integer")
        )
        .withColumn(
            "geolocation_lat",
            F.col("geolocation_lat").cast("double")
        )
        .withColumn(
            "geolocation_lng",
            F.col("geolocation_lng").cast("double")
        )
        .withColumn(
            "geolocation_city",
            F.trim(F.lower(F.col("geolocation_city")))
        )
        .withColumn(
            "geolocation_state",
            F.upper(F.trim(F.col("geolocation_state")))
        )
    )

    # Keep unique geographic points
    df = df.dropDuplicates([
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ])

    df.write.mode("overwrite").parquet(str(SILVER_PATH))

    return df


def main():

    spark = create_spark_session()

    try:
        df = transform_geolocation(spark)

        print("Silver geolocation:")
        df.printSchema()
        df.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()