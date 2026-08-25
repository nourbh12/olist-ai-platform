from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "products"


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverProducts")
        .master("local[*]")
        .getOrCreate()
    )

    try:

        df = spark.read.parquet(str(SILVER_PATH))

        print("\n========== PRODUCTS QUALITY ==========\n")

        print(f"Rows: {df.count()}")

        duplicate_ids = (
            df.groupBy("product_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        null_product_ids = df.filter(
            F.col("product_id").isNull()
        ).count()

        negative_weight = df.filter(
            F.col("product_weight_g") < 0
        ).count()

        negative_volume = df.filter(
            F.col("product_volume_cm3") < 0
        ).count()

        invalid_photos = df.filter(
            F.col("product_photos_qty") < 0
        ).count()

        checks = {
            "duplicate_product_ids": duplicate_ids,
            "null_product_ids": null_product_ids,
            "negative_weight": negative_weight,
            "negative_volume": negative_volume,
            "invalid_photo_quantity": invalid_photos,
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