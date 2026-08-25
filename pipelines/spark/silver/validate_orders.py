from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "orders"


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverOrders")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        orders = spark.read.parquet(str(SILVER_PATH))

        print("\n========== SILVER ORDERS QUALITY CHECKS ==========\n")

        # --------------------------------------------------
        # 1. Row count
        # --------------------------------------------------

        total_rows = orders.count()

        print(f"Total rows: {total_rows}")

        # --------------------------------------------------
        # 2. Duplicate order IDs
        # --------------------------------------------------

        duplicate_orders = (
            orders
            .groupBy("order_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        print(f"Duplicate order IDs: {duplicate_orders}")

        # --------------------------------------------------
        # 3. Null order IDs
        # --------------------------------------------------

        null_order_ids = (
            orders
            .filter(F.col("order_id").isNull())
            .count()
        )

        print(f"Null order IDs: {null_order_ids}")

        # --------------------------------------------------
        # 4. Invalid delivery duration
        # --------------------------------------------------

        negative_duration = (
            orders
            .filter(F.col("delivery_duration_days") < 0)
            .count()
        )

        print(f"Negative delivery durations: {negative_duration}")

        # --------------------------------------------------
        # 5. Invalid delivery delay
        # --------------------------------------------------

        invalid_delay = (
            orders
            .filter(F.col("delivery_delay_days") < -365)
            .count()
        )

        print(f"Suspicious delivery delays: {invalid_delay}")

        # --------------------------------------------------
        # 6. Unknown order statuses
        # --------------------------------------------------

        print("\nOrder statuses:")

        (
            orders
            .groupBy("order_status")
            .count()
            .orderBy(F.desc("count"))
            .show()
        )

        # --------------------------------------------------
        # 7. Late delivery distribution
        # --------------------------------------------------

        print("\nLate delivery distribution:")

        (
            orders
            .groupBy("is_late")
            .count()
            .show()
        )

        # --------------------------------------------------
        # 8. Summary
        # --------------------------------------------------

        print("\n========== QUALITY SUMMARY ==========")

        checks = {
            "duplicate_order_ids": duplicate_orders,
            "null_order_ids": null_order_ids,
            "negative_delivery_duration": negative_duration,
            "suspicious_delivery_delay": invalid_delay,
        }

        all_passed = True

        for check, value in checks.items():

            if value == 0:
                print(f"PASS: {check}")
            else:
                print(f"FAIL: {check} -> {value}")
                all_passed = False

        print()

        if all_passed:
            print("All Silver orders quality checks passed.")
        else:
            print("Some Silver orders quality checks failed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()