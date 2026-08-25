from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
BRONZE_DATA_PATH = PROJECT_ROOT / "data" / "bronze"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("OlistBronzeIngestion")
        .master("local[*]")
        .getOrCreate()
    )


def ingest_csv_to_bronze(
    spark: SparkSession,
    input_path: Path,
    output_path: Path,
) -> None:

    print(f"Reading: {input_path}")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("mode", "PERMISSIVE")
        .csv(str(input_path))
    )

    row_count = df.count()

    print(f"Rows: {row_count}")
    print(f"Columns: {len(df.columns)}")
    print(f"Schema:")
    df.printSchema()

    (
        df.write
        .mode("overwrite")
        .parquet(str(output_path))
    )

    print(f"Written to: {output_path}")

def main():
    spark = create_spark_session()

    try:
        for csv_file in RAW_DATA_PATH.glob("*.csv"):

            dataset_name = (
                csv_file.stem
                .replace("olist_", "")
                .replace("_dataset", "")
            )

            output_path = BRONZE_DATA_PATH / dataset_name

            ingest_csv_to_bronze(
                spark,
                csv_file,
                output_path,
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()