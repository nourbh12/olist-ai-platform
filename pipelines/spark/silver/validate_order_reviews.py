from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "order_reviews"


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverReviews")
        .master("local[*]")
        .getOrCreate()
    )

    try:

        df = spark.read.parquet(str(SILVER_PATH))

        print("\n========== REVIEWS QUALITY ==========\n")

        print(f"Rows: {df.count()}")

        invalid_scores = df.filter(
            (F.col("review_score") < 1) |
            (F.col("review_score") > 5)
        ).count()

        null_review_ids = df.filter(
            F.col("review_id").isNull()
        ).count()

        null_order_ids = df.filter(
            F.col("order_id").isNull()
        ).count()

        checks = {
            "invalid_review_scores": invalid_scores,
            "null_review_ids": null_review_ids,
            "null_order_ids": null_order_ids,
        }

        for name, value in checks.items():
            if value == 0:
                print(f"PASS: {name}")
            else:
                print(f"FAIL: {name} -> {value}")

        print("\nReview score distribution:")

        df.groupBy("review_score").count().orderBy(
            "review_score"
        ).show()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()