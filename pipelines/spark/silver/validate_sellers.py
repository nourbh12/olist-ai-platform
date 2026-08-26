from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "sellers"


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverSellers")
        .master("local[*]")
        .getOrCreate()
    )

    try:

        df = spark.read.parquet(str(SILVER_PATH))

        print("\n========== SELLERS QUALITY ==========\n")

        print(f"Rows: {df.count()}")

        duplicate_ids = (
            df.groupBy("seller_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        null_ids = df.filter(
            F.col("seller_id").isNull()
        ).count()

        invalid_zip = df.filter(
            F.col("seller_zip_code_prefix") < 0
        ).count()

        checks = {
            "duplicate_seller_ids": duplicate_ids,
            "null_seller_ids": null_ids,
            "invalid_zip_codes": invalid_zip,
        }

        all_passed = True

        for name, value in checks.items():
            if value == 0:
                print(f"PASS: {name}")
            else:
                print(f"FAIL: {name} -> {value}")
                all_passed = False

        if all_passed:
            print("\nAll Silver sellers quality checks passed.")
        else:
            print("\nSome Silver sellers quality checks failed.")
            raise ValueError("Silver sellers quality checks failed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()