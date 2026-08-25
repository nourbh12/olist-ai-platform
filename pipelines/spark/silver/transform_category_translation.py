from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "product_category_name_translation"
)

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "product_category_name_translation"
)


def create_spark_session():

    return (
        SparkSession.builder
        .appName("OlistSilverCategoryTranslation")
        .master("local[*]")
        .getOrCreate()
    )


def transform_categories(spark):

    df = spark.read.parquet(str(BRONZE_PATH))

    df = (
        df
        .withColumn(
            "product_category_name",
            F.trim(F.lower(F.col("product_category_name")))
        )
        .withColumn(
            "product_category_name_english",
            F.trim(F.lower(
                F.col("product_category_name_english")
            ))
        )
    )

    df = df.dropDuplicates(["product_category_name"])

    df.write.mode("overwrite").parquet(str(SILVER_PATH))

    return df


def main():

    spark = create_spark_session()

    try:

        df = transform_categories(spark)

        print("Silver category translation:")
        df.printSchema()
        df.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()