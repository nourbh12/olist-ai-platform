from typing import Iterable

from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def check_columns(
    df: DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
):
    """
    Check that all required columns exist.
    """
    actual_columns = set(df.columns)

    missing = [
        column
        for column in required_columns
        if column not in actual_columns
    ]

    if missing:
        raise ValueError(
            f"[{dataset_name}] Missing columns: {missing}"
        )


def check_not_null(
    df: DataFrame,
    column_name: str,
    dataset_name: str,
):
    """
    Check that a column contains no NULL values.
    """
    null_count = (
        df.filter(col(column_name).isNull())
        .count()
    )

    if null_count > 0:
        raise ValueError(
            f"[{dataset_name}] "
            f"{column_name} contains {null_count} NULL values"
        )


def check_unique(
    df: DataFrame,
    columns: list[str],
    dataset_name: str,
):
    """
    Check uniqueness of one or multiple columns.
    """
    duplicate_count = (
        df.groupBy(*columns)
        .count()
        .filter(col("count") > 1)
        .count()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"[{dataset_name}] "
            f"Found {duplicate_count} duplicate groups "
            f"for columns {columns}"
        )


def check_non_negative(
    df: DataFrame,
    column_name: str,
    dataset_name: str,
):
    """
    Check that numeric values are >= 0.
    NULL values are allowed.
    """
    invalid_count = (
        df.filter(
            col(column_name).isNotNull()
            & (col(column_name) < 0)
        )
        .count()
    )

    if invalid_count > 0:
        raise ValueError(
            f"[{dataset_name}] "
            f"{column_name} contains "
            f"{invalid_count} negative values"
        )


def check_between(
    df: DataFrame,
    column_name: str,
    minimum: float,
    maximum: float,
    dataset_name: str,
):
    """
    Check that non-null values are within [minimum, maximum].
    """
    invalid_count = (
        df.filter(
            col(column_name).isNotNull()
            & (
                (col(column_name) < minimum)
                | (col(column_name) > maximum)
            )
        )
        .count()
    )

    if invalid_count > 0:
        raise ValueError(
            f"[{dataset_name}] "
            f"{column_name} contains "
            f"{invalid_count} values outside "
            f"[{minimum}, {maximum}]"
        )


def check_allowed_values(
    df: DataFrame,
    column_name: str,
    allowed_values: list,
    dataset_name: str,
):
    """
    Check that non-null values belong to an allowed set.
    """
    invalid_count = (
        df.filter(
            col(column_name).isNotNull()
            & (~col(column_name).isin(allowed_values))
        )
        .count()
    )

    if invalid_count > 0:
        raise ValueError(
            f"[{dataset_name}] "
            f"{column_name} contains "
            f"{invalid_count} invalid values"
        )