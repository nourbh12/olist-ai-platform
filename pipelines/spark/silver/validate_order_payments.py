from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "order_payments"


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverPayments")
        .master("local[*]")
        .getOrCreate()
    )

    try:

        df = spark.read.parquet(str(SILVER_PATH))

        print("\n========== PAYMENTS QUALITY ==========\n")

        print(f"Rows: {df.count()}")

        negative_values = df.filter(
            F.col("payment_value") < 0
        ).count()

        invalid_installments = df.filter(
            F.col("payment_installments") < 1
        ).count()

        invalid_sequential = df.filter(
            F.col("payment_sequential") < 1
        ).count()

        null_order_ids = df.filter(
            F.col("order_id").isNull()
        ).count()

        checks = {
            "negative_payment_values": negative_values,
            "invalid_installments": invalid_installments,
            "invalid_payment_sequence": invalid_sequential,
            "null_order_ids": null_order_ids,
        }

        for name, value in checks.items():
            if value == 0:
                print(f"PASS: {name}")
            else:
                print(f"FAIL: {name} -> {value}")

        print("\nPayment types:")

        df.groupBy("payment_type").count().show()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()