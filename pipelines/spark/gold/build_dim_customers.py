from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SILVER_DATA_PATH = PROJECT_ROOT / "data" / "silver"
GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldDimCustomers")
        .master("local[*]")
        .getOrCreate()
    )


def main():

    spark = create_spark_session()

    try:

        input_path = SILVER_DATA_PATH / "customers"
        output_path = GOLD_DATA_PATH / "dim_customers"

        print(f"Reading: {input_path}")

        customers = spark.read.parquet(str(input_path))

        dim_customers = (
            customers
            .select(
                col("customer_id"),
                col("customer_unique_id"),
                col("customer_zip_code_prefix"),
                col("customer_city"),
                col("customer_state"),
            )
            .dropDuplicates(["customer_id"])
        )

        print(f"Rows: {dim_customers.count()}")

        (
            dim_customers.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        print(f"Written to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()