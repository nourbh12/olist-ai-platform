from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "order_reviews"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "order_reviews"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("OlistSilverReviews")
        .master("local[*]")
        .getOrCreate()
    )


def transform_reviews(spark):

    df = spark.read.parquet(str(BRONZE_PATH))

    df = (
        df
        .withColumn(
            "review_score",
            F.col("review_score").cast("integer")
        )
        .withColumn(
            "review_creation_date",
            F.to_timestamp(F.col("review_creation_date"))
        )
        .withColumn(
            "review_answer_timestamp",
            F.to_timestamp(F.col("review_answer_timestamp"))
        )
        .withColumn(
            "review_comment_title",
            F.trim(F.col("review_comment_title"))
        )
        .withColumn(
            "review_comment_message",
            F.trim(F.col("review_comment_message"))
        )
    )

    df = df.dropDuplicates(
        ["review_id", "order_id"]
    )

    df.write.mode("overwrite").parquet(str(SILVER_PATH))

    return df


def main():

    spark = create_spark_session()

    try:
        df = transform_reviews(spark)

        print("Silver reviews:")
        df.printSchema()
        df.show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
    
    