from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "products"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "products"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("OlistSilverProducts")
        .master("local[*]")
        .getOrCreate()
    )


def transform_products(spark):

    df = spark.read.parquet(str(BRONZE_PATH))

    # Numeric columns
    numeric_columns = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    for column in numeric_columns:
        df = df.withColumn(
            column,
            F.col(column).cast("double")
        )

    # Standardize category
    df = df.withColumn(
        "product_category_name",
        F.trim(F.lower(F.col("product_category_name")))
    )

    # Remove exact duplicates
    df = df.dropDuplicates(["product_id"])

    # Product volume
    df = df.withColumn(
        "product_volume_cm3",
        F.col("product_length_cm")
        * F.col("product_height_cm")
        * F.col("product_width_cm")
    )

    df.write.mode("overwrite").parquet(str(SILVER_PATH))

    return df


def main():

    spark = create_spark_session()

    try:
        df = transform_products(spark)

        print("Silver products:")
        df.printSchema()

        df.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()