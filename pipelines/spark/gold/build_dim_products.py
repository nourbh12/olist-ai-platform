from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SILVER_DATA_PATH = PROJECT_ROOT / "data" / "silver"
GOLD_DATA_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("GoldDimProducts")
        .master("local[*]")
        .getOrCreate()
    )


def main():

    spark = create_spark_session()

    try:

        input_path = SILVER_DATA_PATH / "products"
        output_path = GOLD_DATA_PATH / "dim_products"

        products = spark.read.parquet(str(input_path))

        dim_products = (
            products
            .select(
                col("product_id"),
                col("product_category_name"),
                col("product_name_lenght"),
                col("product_description_lenght"),
                col("product_photos_qty"),
                col("product_weight_g"),
                col("product_length_cm"),
                col("product_height_cm"),
                col("product_width_cm"),
            )
            .dropDuplicates(["product_id"])
        )

        print(f"Rows: {dim_products.count()}")

        (
            dim_products.write
            .mode("overwrite")
            .parquet(str(output_path))
        )

        print(f"Written to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()