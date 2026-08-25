from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SILVER_DATA_PATH = PROJECT_ROOT / "data" / "silver"
GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldFactReviews")
        .master("local[*]")
        .getOrCreate()
    )


def main():

    spark = create_spark_session()

    try:
        reviews = spark.read.parquet(
            str(SILVER_DATA_PATH / "order_reviews")
        )

        fact_reviews = (
            reviews
            .select(
                "review_id",
                "order_id",
                "review_score",
                "review_comment_title",
                "review_comment_message",
                "review_creation_date",
                "review_answer_timestamp",
            )
        )

        print(f"Rows: {fact_reviews.count()}")

        output_path = GOLD_DATA_PATH / "fact_reviews"

        (
            fact_reviews.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        print(f"Written to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()