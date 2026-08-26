from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "customers"


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverCustomers")
        .master("local[*]")
        .getOrCreate()
    )

    try:

        df = spark.read.parquet(str(SILVER_PATH))

        print("\n========== CUSTOMERS QUALITY ==========\n")

        print(f"Rows: {df.count()}")

        duplicate_ids = (
            df.groupBy("customer_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        null_ids = df.filter(
            F.col("customer_id").isNull()
        ).count()

        invalid_zip = df.filter(
            F.col("customer_zip_code_prefix") < 0
        ).count()

        checks = {
            "duplicate_customer_ids": duplicate_ids,
            "null_customer_ids": null_ids,
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
            print("\nAll Silver customers quality checks passed.")
        else:
            print("\nSome Silver customers quality checks failed.")
            raise ValueError("Silver customers quality checks failed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()