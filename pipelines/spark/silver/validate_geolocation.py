from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "geolocation"


def main():

    spark = (
        SparkSession.builder
        .appName("ValidateSilverGeolocation")
        .master("local[*]")
        .getOrCreate()
    )

    try:

        df = spark.read.parquet(str(SILVER_PATH))

        print("\n========== GEOLOCATION QUALITY ==========\n")

        print(f"Rows: {df.count()}")

        invalid_latitude = df.filter(
            (F.col("geolocation_lat") < -90) |
            (F.col("geolocation_lat") > 90)
        ).count()

        invalid_longitude = df.filter(
            (F.col("geolocation_lng") < -180) |
            (F.col("geolocation_lng") > 180)
        ).count()

        null_zip = df.filter(
            F.col("geolocation_zip_code_prefix").isNull()
        ).count()

        checks = {
            "invalid_latitude": invalid_latitude,
            "invalid_longitude": invalid_longitude,
            "null_zip_prefix": null_zip,
        }

        all_passed = True

        for name, value in checks.items():
            if value == 0:
                print(f"PASS: {name}")
            else:
                print(f"FAIL: {name} -> {value}")
                all_passed = False

        if all_passed:
            print("\nAll Silver geolocation quality checks passed.")
        else:
            print("\nSome Silver geolocation quality checks failed.")
            raise ValueError(
                "Silver geolocation quality checks failed."
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()