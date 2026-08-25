from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "product_category_name_translation"
)


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverCategories")
        .master("local[*]")
        .getOrCreate()
    )

    try:

        df = spark.read.parquet(str(SILVER_PATH))

        print("\n========== CATEGORY QUALITY ==========\n")

        print(f"Rows: {df.count()}")

        duplicate_categories = (
            df.groupBy("product_category_name")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        null_original = df.filter(
            F.col("product_category_name").isNull()
        ).count()

        null_english = df.filter(
            F.col("product_category_name_english").isNull()
        ).count()

        checks = {
            "duplicate_categories": duplicate_categories,
            "null_original_category": null_original,
            "null_english_category": null_english,
        }

        for name, value in checks.items():

            if value == 0:
                print(f"PASS: {name}")
            else:
                print(f"FAIL: {name} -> {value}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()