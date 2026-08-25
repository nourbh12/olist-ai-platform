from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SILVER_DATA_PATH = PROJECT_ROOT / "data" / "silver"
GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldDimSellers")
        .master("local[*]")
        .getOrCreate()
    )


def main():

    spark = create_spark_session()

    try:

        input_path = SILVER_DATA_PATH / "sellers"
        output_path = GOLD_DATA_PATH / "dim_sellers"

        sellers = spark.read.parquet(str(input_path))

        dim_sellers = (
            sellers
            .select(
                col("seller_id"),
                col("seller_zip_code_prefix"),
                col("seller_city"),
                col("seller_state"),
            )
            .dropDuplicates(["seller_id"])
        )

        print(f"Rows: {dim_sellers.count()}")

        (
            dim_sellers.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        print(f"Written to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()