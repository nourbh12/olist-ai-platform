from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "order_items"


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverOrderItems")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        df = spark.read.parquet(str(SILVER_PATH))

        print("\n========== SILVER ORDER ITEMS QUALITY CHECKS ==========\n")

        # --------------------------------------------------
        # 1. Row count
        # --------------------------------------------------

        total_rows = df.count()

        print(f"Total rows: {total_rows}")

        # --------------------------------------------------
        # 2. Null checks for critical columns
        # --------------------------------------------------

        critical_columns = [
            "order_id",
            "product_id",
            "seller_id",
            "order_item_id",
            "price",
            "freight_value",
            "shipping_limit_date",
        ]

        print("\nNull values:")

        null_counts = {}

        for column in critical_columns:

            count = (
                df
                .filter(F.col(column).isNull())
                .count()
            )

            null_counts[column] = count

            print(f"{column}: {count}")

        # --------------------------------------------------
        # 3. Duplicate records
        # --------------------------------------------------

        duplicate_rows = (
            df
            .groupBy(
                "order_id",
                "order_item_id",
            )
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        print(f"\nDuplicate order-item records: {duplicate_rows}")

        # --------------------------------------------------
        # 4. Negative prices
        # --------------------------------------------------

        negative_prices = (
            df
            .filter(F.col("price") < 0)
            .count()
        )

        print(f"Negative prices: {negative_prices}")

        # --------------------------------------------------
        # 5. Negative freight values
        # --------------------------------------------------

        negative_freight = (
            df
            .filter(F.col("freight_value") < 0)
            .count()
        )

        print(f"Negative freight values: {negative_freight}")

        # --------------------------------------------------
        # 6. Invalid order_item_id
        # --------------------------------------------------

        invalid_item_ids = (
            df
            .filter(F.col("order_item_id") < 1)
            .count()
        )

        print(f"Invalid order_item_id values: {invalid_item_ids}")

        # --------------------------------------------------
        # 7. Validate total_item_value
        # --------------------------------------------------

        incorrect_total_value = (
            df
            .filter(
                F.abs(
                    F.col("total_item_value")
                    - (
                        F.col("price")
                        + F.col("freight_value")
                    )
                ) >= 0.001
            )
            .count()
        )

        print(
            f"Incorrect total_item_value calculations: "
            f"{incorrect_total_value}"
        )

        # --------------------------------------------------
        # 8. Price statistics
        # --------------------------------------------------

        print("\nPrice statistics:")

        (
            df
            .select(
                F.min("price").alias("min_price"),
                F.max("price").alias("max_price"),
                F.avg("price").alias("avg_price"),
            )
            .show()
        )

        # --------------------------------------------------
        # 9. Freight statistics
        # --------------------------------------------------

        print("Freight statistics:")

        (
            df
            .select(
                F.min("freight_value").alias("min_freight"),
                F.max("freight_value").alias("max_freight"),
                F.avg("freight_value").alias("avg_freight"),
            )
            .show()
        )

        # --------------------------------------------------
        # 10. Quality summary
        # --------------------------------------------------

        print("\n========== QUALITY SUMMARY ==========\n")

        all_passed = True

        # Null checks
        for column, count in null_counts.items():

            if count == 0:
                print(f"PASS: no null values in {column}")
            else:
                print(f"FAIL: {column} contains {count} null values")
                all_passed = False

        # Other checks
        checks = {
            "duplicate_order_item_records": duplicate_rows,
            "negative_prices": negative_prices,
            "negative_freight": negative_freight,
            "invalid_order_item_ids": invalid_item_ids,
            "incorrect_total_item_value": incorrect_total_value,
        }

        for check, value in checks.items():

            if value == 0:
                print(f"PASS: {check}")
            else:
                print(f"FAIL: {check} -> {value}")
                all_passed = False

        print()

        if all_passed:
            print("All Silver order_items quality checks passed.")
        else:
            print("Some Silver order_items quality checks failed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()